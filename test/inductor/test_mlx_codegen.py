import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import codecache, config, exc as inductor_exc
import torch._dynamo.config as dynamo_config
from torch._inductor.compile_fx import compile_fx
from unittest import mock

import pytest

MLX_IMPORT_ERROR = None
try:
    import mlx.core as _mx  # noqa: F401
except Exception as exc:  # pragma: no cover - import-time failure
    MLX_IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(
    MLX_IMPORT_ERROR is not None,
    reason=f"mlx.core unavailable: {MLX_IMPORT_ERROR}",
)


@pytest.fixture(autouse=True, params=[False, True], ids=["static_shapes", "dynamic_shapes"])
def _toggle_dynamic_shapes(request):
    prev_dynamic = dynamo_config.dynamic_shapes
    dynamo_config.dynamic_shapes = request.param
    try:
        yield
    finally:
        dynamo_config.dynamic_shapes = prev_dynamic
def test_mlx_backend_torch_compile_matches_eager():
    x = torch.randn(16, 16)
    y = torch.randn(16, 16)

    def fn(a, b):
        return torch.sin(a) + torch.cos(a) + b

    expected = fn(x, y)

    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        result = compiled(x, y)

    torch.testing.assert_close(result, expected)


def test_mlx_backend_torch_compile_matmul_add():
    torch.manual_seed(0)
    a = torch.randn(8, 12)
    b = torch.randn(12, 6)
    c = torch.randn(8, 6)

    def fn(x, y, bias):
        return torch.matmul(x, y) + bias

    expected = fn(a, b, c)

    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        result = compiled(a, b, c)

    torch.testing.assert_close(result, expected)


def test_mlx_backend_torch_compile_attention_block():
    torch.manual_seed(1)
    batch_size, seq_len, hidden_dim = 2, 4, 8
    q = torch.randn(batch_size, seq_len, hidden_dim)
    k = torch.randn(batch_size, seq_len, hidden_dim)
    v = torch.randn(batch_size, seq_len, hidden_dim)

    def fn(query, key, value):
        scale = 1.0 / (hidden_dim**0.5)
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, value)

    expected = fn(q, k, v)

    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        result = compiled(q, k, v)

    torch.testing.assert_close(result, expected)


def test_mlx_backend_torch_compile_backward_simple():
    torch.manual_seed(2)
    base = torch.randn(6)

    def fn(x):
        return (x * x).sum()

    eager_input = base.clone().requires_grad_(True)
    compiled_input = base.clone().requires_grad_(True)

    expected = fn(eager_input)
    expected.backward()
    expected_grad = eager_input.grad.detach().clone()

    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        result = compiled(compiled_input)
        result.backward()

    compiled_grad = compiled_input.grad.detach().clone()

    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(compiled_grad, expected_grad)


def test_mlx_backend_torch_compile_training_loop():
    torch.manual_seed(3)

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            hidden = F.relu(self.fc1(x))
            return self.fc2(hidden).squeeze(-1)

    inputs = torch.randn(10, 4)
    targets = torch.randn(10)
    base_model = TinyNet()

    eager_model = TinyNet()
    eager_model.load_state_dict(base_model.state_dict())

    compiled_model = TinyNet()
    compiled_model.load_state_dict(base_model.state_dict())

    def run_training(model: nn.Module, forward_fn):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        losses = []
        model.train()
        for _ in range(5):
            optimizer.zero_grad()
            output = forward_fn()
            loss = F.mse_loss(output, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        return torch.stack(losses)

    eager_losses = run_training(eager_model, lambda: eager_model(inputs))

    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(compiled_model, backend="inductor", fullgraph=True)
        compiled_losses = run_training(compiled_model, lambda: compiled(inputs))

    torch.testing.assert_close(compiled_losses, eager_losses)


def test_mlx_backend_torch_compile_attention_training_loop():
    torch.manual_seed(4)

    class TinyAttention(nn.Module):
        def __init__(self, embed_dim: int = 16, num_heads: int = 4):
            super().__init__()
            assert embed_dim % num_heads == 0
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.scale = self.head_dim**-0.5

            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)

        def _shape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
            batch, seq_len, _ = tensor.shape
            return tensor.reshape(batch, seq_len, self.num_heads, self.head_dim).permute(
                0, 2, 1, 3
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            q = self._shape_heads(self.q_proj(x))
            k = self._shape_heads(self.k_proj(x))
            v = self._shape_heads(self.v_proj(x))

            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = torch.softmax(scores, dim=-1)
            context = torch.matmul(attn, v)
            context = context.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.embed_dim)
            return self.out_proj(context)

    batch_size, seq_len, embed_dim = 3, 6, 16
    inputs = torch.randn(batch_size, seq_len, embed_dim)
    targets = torch.randn(batch_size, seq_len, embed_dim)

    base_model = TinyAttention(embed_dim=embed_dim, num_heads=4)

    eager_model = TinyAttention(embed_dim=embed_dim, num_heads=4)
    eager_model.load_state_dict(base_model.state_dict())

    compiled_model = TinyAttention(embed_dim=embed_dim, num_heads=4)
    compiled_model.load_state_dict(base_model.state_dict())

    def run_training(model: nn.Module, forward_fn):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        losses = []
        model.train()
        for _ in range(4):
            optimizer.zero_grad()
            predictions = forward_fn()
            loss = F.mse_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        return torch.stack(losses)

    eager_losses = run_training(eager_model, lambda: eager_model(inputs))

    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(compiled_model, backend="inductor", fullgraph=True)
        compiled_losses = run_training(compiled_model, lambda: compiled(inputs))

    torch.testing.assert_close(compiled_losses, eager_losses)


def test_mlx_backend_inplace_mutation_propagates():
    torch.manual_seed(5)

    def fn(t):
        t.add_(2.0)
        return t

    eager_input = torch.randn(6)
    eager_expected = fn(eager_input.clone())

    compiled_input = eager_input.clone()

    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_output = compiled(compiled_input)

    torch.testing.assert_close(compiled_input, eager_expected)
    torch.testing.assert_close(compiled_output, eager_expected)
    assert compiled_output.data_ptr() == compiled_input.data_ptr()


def test_mlx_backend_clone_creates_copy():
    torch.manual_seed(6)

    def fn(t):
        cloned = t.clone()
        cloned.add_(3.0)
        return cloned

    x = torch.randn(4, 4)
    eager = fn(x.clone())

    compiled_input = x.clone()
    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_output = compiled(compiled_input)

    torch.testing.assert_close(compiled_output, eager)
    assert compiled_output.data_ptr() != compiled_input.data_ptr()
    torch.testing.assert_close(compiled_input, x)


def test_mlx_backend_expand_with_negative_one():
    torch.manual_seed(7)
    base = torch.randn(3, 2)

    def fn(t):
        expanded = t.expand(-1, t.shape[1])
        return expanded * 2.0

    eager = fn(base)

    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_result = compiled(base)

    torch.testing.assert_close(compiled_result, eager)


def test_mlx_backend_full_respects_dtype():
    def fn():
        return torch.full((2, 2), 7, dtype=torch.int64)

    eager = fn()

    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_result = compiled()

    torch.testing.assert_close(compiled_result, eager)
    assert compiled_result.dtype == torch.int64


def test_mlx_backend_sdpa_mps_op():
    if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
        pytest.skip("MPS backend required for SDPA reference")
    torch.manual_seed(8)
    batch, heads, seq_len, head_dim = 2, 2, 4, 8
    device = torch.device("mps")
    q = torch.randn(batch, heads, seq_len, head_dim, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device)
    mask = torch.zeros(batch, 1, seq_len, seq_len, device=device)
    scale = head_dim ** -0.5

    def fn(query, key, value, attn_mask):
        return torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(
            query,
            key,
            value,
            attn_mask,
            0.0,
            False,
            None,
            scale=scale,
        )

    eager_out, eager_attn = fn(q, k, v, mask)

    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_out, compiled_attn = compiled(q, k, v, mask)

    torch.testing.assert_close(compiled_out, eager_out)
    torch.testing.assert_close(compiled_attn, eager_attn)


def test_mlx_backend_add_tensor():
    torch.manual_seed(9)
    a = torch.randn(5, 3)
    b = torch.randn(5, 3)

    def fn(x, y):
        return torch.add(x, y, alpha=2.0)

    eager = fn(a, b)
    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_out = compiled(a, b)

    torch.testing.assert_close(compiled_out, eager)


def test_mlx_backend_bitwise_and_tensor():
    x = torch.randint(0, 8, (4, 4), dtype=torch.int32)
    y = torch.randint(0, 8, (4, 4), dtype=torch.int32)

    def fn(a, b):
        return torch.bitwise_and(a, b)

    eager = fn(x, y)
    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_out = compiled(x, y)

    torch.testing.assert_close(compiled_out, eager)


def test_mlx_backend_cat_default():
    torch.manual_seed(10)
    tensors = [torch.randn(2, 3) for _ in range(3)]

    def fn(a, b, c):
        return torch.cat([a, b, c], dim=0)

    eager = fn(*tensors)
    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_out = compiled(*tensors)

    torch.testing.assert_close(compiled_out, eager)


def test_mlx_backend_add_tensor_with_alpha():
    torch.manual_seed(10)
    x = torch.randn(7, 3)
    y = torch.randn(7, 3)

    def fn(a, b):
        return torch.add(a, b, alpha=0.5)

    eager = fn(x, y)
    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_out = compiled(x, y)

    torch.testing.assert_close(compiled_out, eager)


def test_mlx_backend_add_scalar():
    torch.manual_seed(11)
    x = torch.randn(4, 5)

    def fn(a):
        return torch.add(a, 2.5)

    eager = fn(x)
    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_out = compiled(x)

    torch.testing.assert_close(compiled_out, eager)


def test_mlx_backend_python_add_scalar_tensor():
    torch.manual_seed(11)
    x = torch.randn(2, 3)

    def fn(a):
        return (a + 1.25, 2.0 + a)

    eager = fn(x)
    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_out = compiled(x)

    torch.testing.assert_close(compiled_out, eager)


def test_mlx_backend_symbolic_shape_full():
    torch.manual_seed(12)
    class SymShapeNet(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            seq = x.shape[1]
            filled = torch.full((seq, 1), 1.0, dtype=x.dtype, device=x.device)
            return filled.reshape(seq, 1)

    model = SymShapeNet()
    inputs = torch.randn(2, 5)
    eager = model(inputs)
    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        compiled_out = compiled(inputs)
    torch.testing.assert_close(compiled_out, eager)


def test_mlx_backend_symbolic_iota():
    torch.manual_seed(13)
    class MaskNet(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            seq = x.shape[1]
            mask = torch.arange(seq, device=x.device)
            return mask.to(x.dtype)

    model = MaskNet()
    inputs = torch.randn(1, 7)
    eager = model(inputs)
    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        compiled_out = compiled(inputs)
    torch.testing.assert_close(compiled_out, eager)


def test_mlx_backend_sym_sum_fx_graph():
    torch.manual_seed(14)

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    sym = graph.call_function(torch.ops.aten.sym_size.int, args=(x, 1))
    sym_sum = graph.call_function(torch.sym_sum, args=((1, sym),))
    full = graph.call_function(
        torch.ops.aten.full.default,
        args=([sym_sum], 0.0),
        kwargs={
            "dtype": torch.float32,
            "device": torch.device("cpu"),
            "layout": torch.strided,
            "pin_memory": False,
        },
    )
    graph.output(full)

    module = torch.fx.GraphModule(torch.nn.Module(), graph)
    inputs = (torch.randn(1, 4),)
    eager = module(*inputs)

    with config.patch({"mlx_codegen": True}):
        compiled = compile_fx(module, inputs)
        compiled_out = compiled(*inputs)

    torch.testing.assert_close(compiled_out, eager)


def test_mlx_backend_cos_default():
    torch.manual_seed(11)
    x = torch.randn(6, 6)

    def fn(t):
        return torch.cos(t)

    eager = fn(x)
    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_out = compiled(x)

    torch.testing.assert_close(compiled_out, eager)


def test_mlx_backend_embedding_default():
    torch.manual_seed(12)
    weight = torch.randn(6, 4)
    indices = torch.tensor([[0, 2, 4], [3, 5, 0]], dtype=torch.long)

    def fn(w, idx):
        return torch.embedding(w, idx)

    eager = fn(weight, indices)
    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_out = compiled(weight, indices)

    torch.testing.assert_close(compiled_out, eager)


def test_mlx_backend_index_tensor():
    torch.manual_seed(13)
    data = torch.randn(5, 3)
    idx = torch.tensor([4, 1, 3], dtype=torch.long)

    def fn(x, index):
        return x[index]

    eager = fn(data, idx)
    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_out = compiled(data, idx)

    torch.testing.assert_close(compiled_out, eager)


def test_mlx_backend_index_multi_tensor():
    torch.manual_seed(14)
    data = torch.randn(4, 6, 2)
    row_idx = torch.tensor([[0], [3]], dtype=torch.long)
    col_idx = torch.tensor([[1, 4, 2]], dtype=torch.long)

    def fn(x, r, c):
        return x[r, c]

    eager = fn(data, row_idx, col_idx)
    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_out = compiled(data, row_idx, col_idx)

    torch.testing.assert_close(compiled_out, eager)


def test_mlx_backend_pow_tensor_scalar():
    torch.manual_seed(16)
    x = torch.rand(4, 4).clamp_min(1e-3)

    def fn(t):
        return torch.pow(t, 1.5)

    eager = fn(x)
    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_out = compiled(x)

    torch.testing.assert_close(compiled_out, eager)


def test_mlx_backend_slice_tensor():
    torch.manual_seed(17)
    x = torch.randn(5, 6, 7)

    def fn(t):
        return t[:, 1:4, :]

    eager = fn(x)
    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_out = compiled(x)

    torch.testing.assert_close(compiled_out, eager)


def test_mlx_backend_slice_tensor_dynamic_resize():
    if not dynamo_config.dynamic_shapes:
        pytest.skip("dynamic shapes required")

    torch.manual_seed(170)

    def fn(t):
        return t[:, :, :, 48:]

    short = torch.randn(1, 2, 15, 96)
    long = torch.randn(1, 2, 16, 96)

    def _disable_cpp(*_args, **_kwargs):
        raise inductor_exc.InvalidCxxCompiler()

    with config.patch({"mlx_codegen": True}):
        with mock.patch.object(codecache.CppCodeCache, "load", side_effect=_disable_cpp), mock.patch.object(
            codecache.CppCodeCache, "load_async", side_effect=_disable_cpp
        ):
            compiled = torch.compile(fn, backend="inductor", fullgraph=True)
            short_out = compiled(short)
            long_out = compiled(long)

    torch.testing.assert_close(short_out, fn(short))
    torch.testing.assert_close(long_out, fn(long))


def test_mlx_backend_prims_convert_element_type():
    torch.manual_seed(18)
    x = torch.randn(4, 4, dtype=torch.float64)

    def fn(t):
        return torch.ops.prims.convert_element_type.default(t, torch.float32)

    eager = fn(x)
    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_out = compiled(x)

    torch.testing.assert_close(compiled_out, eager)


def test_mlx_backend_prims_iota():
    def fn():
        return torch.ops.prims.iota.default(
            6,
            start=2,
            step=3,
            dtype=torch.int32,
            device=torch.device("cpu"),
            requires_grad=False,
        )

    eager = fn()

    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_out = compiled()

    torch.testing.assert_close(compiled_out, eager)


def test_mlx_backend_le_tensor():
    torch.manual_seed(14)
    a = torch.randn(3, 3)
    b = torch.randn(3, 3)

    def fn(x, y):
        return torch.le(x, y)

    eager = fn(a, b)
    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_out = compiled(a, b)

    torch.testing.assert_close(compiled_out, eager)


def test_mlx_backend_mean_dim():
    torch.manual_seed(15)
    x = torch.randn(2, 4, 6)

    def fn(t):
        return torch.mean(t, dim=1, keepdim=True)

    eager = fn(x)
    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_out = compiled(x)

    torch.testing.assert_close(compiled_out, eager)
