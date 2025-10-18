# Demo script to compare eager MPS vs Inductor+MLX on Apple Silicon.
from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config


WARMUP_STEPS = 5
BATCH_SIZE = 32
SEQ_LEN = 256
EMBED_DIM = 1024
NUM_HEADS = 16
LEARNING_RATE = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLX vs eager attention training demo"
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Training steps to measure"
    )
    return parser.parse_args()


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int = 1024, num_heads: int = 16):
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
        b, s, d = x.shape
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


def train_steps(
    model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, steps: int
) -> tuple[list[float], list[float]]:
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    durations: list[float] = []
    losses: list[float] = []
    for step in range(steps):
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        preds = model(inputs)
        loss = F.mse_loss(preds, targets)
        loss.backward()
        opt.step()
        if inputs.device.type == "mps":
            torch.mps.synchronize()
        end = time.perf_counter()
        durations.append(end - start)
        losses.append(loss.item())
    if WARMUP_STEPS:
        durations = durations[WARMUP_STEPS:]
        losses = losses[WARMUP_STEPS:]
    return durations, losses


def main() -> None:
    args = parse_args()
    torch.manual_seed(2024)

    eager_device = torch.device("mps")
    mlx_device = torch.device("cpu")

    inputs_eager = torch.randn(
        BATCH_SIZE, SEQ_LEN, EMBED_DIM, device=eager_device
    )
    targets_eager = torch.randn(
        BATCH_SIZE, SEQ_LEN, EMBED_DIM, device=eager_device
    )

    inputs_mlx = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, device=mlx_device)
    targets_mlx = torch.randn(
        BATCH_SIZE, SEQ_LEN, EMBED_DIM, device=mlx_device
    )

    base_model = MultiHeadAttention(EMBED_DIM, NUM_HEADS)
    eager_model = MultiHeadAttention(EMBED_DIM, NUM_HEADS).to(eager_device)
    eager_model.load_state_dict(base_model.state_dict())

    compiled_model = MultiHeadAttention(EMBED_DIM, NUM_HEADS).to(mlx_device)
    compiled_model.load_state_dict(base_model.state_dict())

    eager_durations, eager_losses = train_steps(
        eager_model, inputs_eager, targets_eager, args.steps
    )

    with config.patch({"mlx_codegen": True}):
        compiled = torch.compile(
            compiled_model, backend="inductor", fullgraph=True
        )
        mlx_durations, mlx_losses = train_steps(
            compiled, inputs_mlx, targets_mlx, args.steps
        )

    eager_avg = statistics.mean(eager_durations)
    mlx_avg = statistics.mean(mlx_durations)
    speedup = eager_avg / mlx_avg if mlx_avg else float("inf")

    print("=== Attention Training Timing ===")
    print(f"Steps measured: {args.steps} (warmup {WARMUP_STEPS})")
    print(f"Eager device: {eager_device}")
    print(f"MLX device:   {mlx_device}")
    print(f"Avg eager step: {eager_avg * 1000:.3f} ms")
    print(f"Avg MLX step:   {mlx_avg * 1000:.3f} ms")
    print(f"Speedup (eager/MLX): {speedup:.2f}x")
    print(f"Final eager loss: {eager_losses[-1]:.6f}")
    print(f"Final MLX loss:   {mlx_losses[-1]:.6f}")


if __name__ == "__main__":
    main()
