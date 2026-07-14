#!/usr/bin/env python3
"""Managed GPU validation for the canonical rwkv-lm state-passing kernel.

Run with cwd=src/train/rwkv-lm through helicopter-dev remote run. Required
RWKV_* settings are intentionally supplied by the managed command contract.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from any2rwkv.kernel import NativeRwkv7Kernel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    required = {
        "RWKV_JIT_ON": "0",
        "RWKV_HEAD_SIZE": "64",
        "RWKV_MY_TESTING": "x070",
        "RWKV_TRAIN_TYPE": "infctx",
    }
    mismatch = {key: (os.environ.get(key), value) for key, value in required.items() if os.environ.get(key) != value}
    if mismatch:
        raise SystemExit(f"managed RWKV kernel environment mismatch: {mismatch}")
    from src.model import RWKV7_STATEPASSING_CLAMPW_CUDA

    kernel = NativeRwkv7Kernel(RWKV7_STATEPASSING_CLAMPW_CUDA)
    generator = torch.Generator(device="cuda").manual_seed(20260714)
    signals = [
        torch.randn(1, 32, 64, generator=generator, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
        for _ in range(6)
    ]
    state = torch.randn(1, 1, 64, 64, generator=generator, device="cuda", dtype=torch.float32).requires_grad_(True)
    full_output, full_state = kernel(state, *signals)
    first_output, middle_state = kernel(state, *(value[:, :16] for value in signals))
    second_output, final_state = kernel(middle_state, *(value[:, 16:] for value in signals))
    chunk_output = torch.cat((first_output, second_output), dim=1)
    output_max_abs = float((full_output - chunk_output).abs().max())
    state_max_abs = float((full_state - final_state).abs().max())
    loss = full_output.float().square().mean() + full_state.square().mean() * 0.01
    gradients = torch.autograd.grad(loss, (state, *signals))
    result = {
        "schema_version": 1,
        "status": "pass" if output_max_abs <= 0.02 and state_max_abs <= 0.003 else "fail",
        "kernel": "rwkv-lm/RWKV7_STATEPASSING_CLAMPW_CUDA",
        "state_dtype": str(state.dtype),
        "signal_dtype": str(signals[0].dtype),
        "output_max_abs": output_max_abs,
        "state_max_abs": state_max_abs,
        "gradient_finite": [bool(torch.isfinite(gradient).all()) for gradient in gradients],
        "gpu": torch.cuda.get_device_name(0),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" and all(result["gradient_finite"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
