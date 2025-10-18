# Demo script to time forward passes of a large multi-head attention module
# comparing eager (MPS) vs Inductor+MLX (CPU).
from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn as nn
from torch._inductor import config

WARMUP_STEPS = 5
BATCH_SIZE = 32
SEQ_LEN = 256
EMBED_DIM = 1024
NUM_HEADS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLX vs eager attention inference demo"
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Inference steps to measure"
    )
    return parser.parse_args()


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, num_heads: int = NUM_HEADS):
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

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        x = x.reshape(b, s, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(x))
        v = self._split(self.v_proj(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.permute(0, 2, 1, 3).reshape(
            x.shape[0], x.shape[1], self.embed_dim
        )
        return self.out_proj(context)


def time_forward(
    model: nn.Module, inputs: torch.Tensor, steps: int
) -> tuple[list[float], torch.Tensor]:
    durations: list[float] = []
    out: torch.Tensor | None = None
    with torch.no_grad():
        for _ in range(steps):
            start = time.perf_counter()
            out = model(inputs)
            if inputs.device.type == "mps":
                torch.mps.synchronize()
            durations.append(time.perf_counter() - start)
    if WARMUP_STEPS:
        durations = durations[WARMUP_STEPS:]
    assert out is not None
    return durations, out


def main() -> None:
    args = parse_args()
    torch.manual_seed(2025)

    eager_device = torch.device("mps")
    mlx_device = torch.device("cpu")

    inputs_eager = torch.randn(
        BATCH_SIZE, SEQ_LEN, EMBED_DIM, device=eager_device
    )
    inputs_mlx = inputs_eager.to(mlx_device)

    base_model = MultiHeadAttention()
    eager_model = MultiHeadAttention().to(eager_device)
    eager_model.load_state_dict(base_model.state_dict())
    eager_model.eval()

    compiled_model = MultiHeadAttention().to(mlx_device)
    compiled_model.load_state_dict(base_model.state_dict())
    compiled_model.eval()

    eager_times, eager_out = time_forward(
        eager_model, inputs_eager, args.steps
    )

    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(
            compiled_model, backend="inductor", fullgraph=True
        )
        mlx_times, mlx_out = time_forward(compiled, inputs_mlx, args.steps)

    eager_avg = statistics.mean(eager_times)
    mlx_avg = statistics.mean(mlx_times)
    speedup = eager_avg / mlx_avg if mlx_avg else float("inf")

    diff = (eager_out.to(mlx_device) - mlx_out).abs().max().item()

    print("=== Attention Inference Timing ===")
    print(f"Steps measured: {args.steps} (warmup {WARMUP_STEPS})")
    print(f"Eager device: {eager_device}")
    print(f"MLX device:   {mlx_device}")
    print(f"Avg eager step: {eager_avg * 1000:.3f} ms")
    print(f"Avg MLX step:   {mlx_avg * 1000:.3f} ms")
    print(f"Speedup (eager/MLX): {speedup:.2f}x")
    print(f"Max |eager-MLX| diff: {diff:.4e}")


if __name__ == "__main__":
    main()
