#!/usr/bin/env python3
"""
HF generate() smoke test that compares eager vs MLX-compiled decoding.

Usage (from repo root, pytorch-mlx env):
    TORCH_TRACE=./traces python3 scripts/test2.py
"""

from __future__ import annotations

import argparse
import contextlib
import time

import torch
from torch._inductor import config as inductor_config
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM-135M"
PROMPT = "# A function that computes the Fibonnaci sequence"
DEFAULT_MAX_NEW_TOKENS = 100


def resolve_device(backend: str) -> torch.device:
    if backend == "inductor":
        if (
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        raise RuntimeError(
            "MPS device is required for --backend inductor, but MPS is unavailable."
        )
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare eager vs MLX-compiled HF generate() output."
    )
    parser.add_argument(
        "--compiled-only",
        action="store_true",
        help="Skip the eager run and only execute the compiled model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Number of tokens to ask generate() for.",
    )
    parser.add_argument(
        "--backend",
        choices=["mlx", "inductor", "aot_eager"],
        default="mlx",
        help="torch.compile backend to use for the compiled model.",
    )
    return parser.parse_args()


def load_model(device: torch.device) -> AutoModelForCausalLM:
    dtype = torch.float16 if device.type != "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=None,
    ).to(device=device, dtype=dtype)
    model.eval()
    return model


def run_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
) -> tuple[torch.Tensor, float]:
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    duration = time.perf_counter() - start
    return output, duration


def main() -> None:
    args = parse_args()
    device = resolve_device(args.backend)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eager_model = None if args.compiled_only else load_model(device)
    compiled_model = load_model(device)
    if eager_model is not None:
        compiled_model.load_state_dict(eager_model.state_dict())

    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

    if eager_model is not None:
        eager_tokens, eager_secs = run_generate(
            eager_model, tokenizer, inputs, args.max_new_tokens
        )
    else:
        eager_tokens = None

    compile_kwargs = {
        "backend": (
            "inductor" if args.backend in {"mlx", "inductor"} else args.backend
        ),
        "fullgraph": True,
    }
    compile_ctx: contextlib.AbstractContextManager
    if args.backend == "mlx":
        compile_ctx = inductor_config.patch({"mlx_codegen": True})
    else:
        compile_ctx = contextlib.nullcontext()
    with compile_ctx:
        compiled_forward = torch.compile(
            compiled_model.forward,
            **compile_kwargs,
        )
        compiled_model.forward = compiled_forward  # type: ignore[method-assign]
        with torch.no_grad():
            compiled_model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        compiled_tokens, compiled_secs = run_generate(
            compiled_model, tokenizer, inputs, args.max_new_tokens
        )

    if eager_tokens is not None:
        eager_tps = args.max_new_tokens / max(eager_secs, 1e-9)
        eager_text = tokenizer.decode(
            eager_tokens[0], skip_special_tokens=True
        )
        print(
            f"\nEager completion (tokens/sec={eager_tps:.2f}):\n{eager_text}"
        )

    compiled_tps = args.max_new_tokens / max(compiled_secs, 1e-9)
    compiled_text = tokenizer.decode(
        compiled_tokens[0], skip_special_tokens=True
    )
    print(
        f"\nCompiled completion (tokens/sec={compiled_tps:.2f}):\n{compiled_text}"
    )


if __name__ == "__main__":
    main()
