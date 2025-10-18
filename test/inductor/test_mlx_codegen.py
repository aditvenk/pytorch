import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config


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
