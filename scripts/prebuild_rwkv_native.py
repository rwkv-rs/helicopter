#!/usr/bin/env python3
"""Prebuild the strict MaxRL native RWKV extension profile without occupying GPUs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

NATIVE_PROFILE = {
    "RWKV_MY_TESTING": "x070",
    "RWKV_KERNEL": "",
    "RWKV_HEAD_SIZE": "64",
    "RWKV_HEAD_L2WRAP_CE_CHUNK": "0",
    "RWKV_FLOAT_MODE": "bf16",
    "RWKV_JIT_ON": "1",
    "RWKV_TRAIN_TYPE": "infctx",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rwkv-lm-path", type=Path, required=True)
    parser.add_argument("--ctx-len", type=int, default=10240)
    parser.add_argument("--chunk-ctx", type=int, default=2048)
    parser.add_argument("--print-manifest", action="store_true")
    args = parser.parse_args()
    if args.chunk_ctx <= 0 or args.chunk_ctx >= args.ctx_len or args.chunk_ctx % 16:
        raise SystemExit("chunk_ctx must be positive, smaller than ctx_len, and divisible by 16")

    native_env = {
        **NATIVE_PROFILE,
        "RWKV_CTXLEN": str(args.ctx_len),
        "RWKV_CHUNK_CTX": str(args.chunk_ctx),
    }
    if args.print_manifest:
        print(json.dumps(native_env, sort_keys=True, separators=(",", ":")))
        return

    from verl.models.rwkv.native_imports import import_rwkv_lm

    import_rwkv_lm(rwkv_lm_path=str(args.rwkv_lm_path), native_env=native_env)
    print(
        "RWKV native extension profile ready: "
        f"ctx_len={args.ctx_len} chunk_ctx={args.chunk_ctx} "
        f"cache={os.environ.get('TORCH_EXTENSIONS_DIR', 'torch-default')}"
    )


if __name__ == "__main__":
    main()
