#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from any2rwkv.checkpoint import read_checkpoint
from any2rwkv.distill import normalized_mse
from any2rwkv.distill_runner import read_packed_token_rows
from any2rwkv.mixer_store import RWKV7MixerLayerStore
from any2rwkv.streamed_teacher import (
    StreamedQwen35HybridExecutor,
    StreamedQwen35Teacher,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, action="append", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--burn-in-tokens", type=int, default=128)
    parser.add_argument("--supervised-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.rows < 1 or args.layer < 0:
        raise SystemExit("--rows must be positive and --layer must be non-negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    source = read_checkpoint(args.source, require_final_layers=False)
    rows = read_packed_token_rows(
        args.dataset_manifest,
        split="validation",
        burn_in_tokens=args.burn_in_tokens,
        supervised_tokens=args.supervised_tokens,
    )[: args.rows]
    input_ids = torch.tensor(rows, dtype=torch.long, device=device)
    teacher = StreamedQwen35Teacher(
        source,
        device=device,
        dtype=dtype,
        cache_layers=False,
        load_output_head=False,
    )
    executor = StreamedQwen35HybridExecutor(teacher)
    hidden_states = teacher.embed_input_ids(input_ids)
    loaded = teacher.loader.load_layer(args.layer, device=device, dtype=dtype)
    results = []
    for checkpoint in args.checkpoint:
        store = RWKV7MixerLayerStore(
            checkpoint.resolve(),
            checkpoint.resolve().parent / f".{checkpoint.name}-diagnostic-overlay",
        )
        mixer = store.load_mixer(args.layer, device=device, dtype=dtype).eval()
        with torch.no_grad():
            output = executor.forward_cached_layer_local(
                hidden_states,
                active_layer_index=args.layer,
                active_mixer=mixer,
                loaded_layer=loaded,
            )
        results.append(
            {
                "checkpoint": str(checkpoint.resolve()),
                "mixer_normalized_mse": float(
                    normalized_mse(
                        output.student_mixer_output, output.teacher_mixer_output
                    )
                ),
                "block_normalized_mse": float(
                    normalized_mse(
                        output.student_block_output, output.teacher_block_output
                    )
                ),
                "mixer_cosine": float(
                    torch.nn.functional.cosine_similarity(
                        output.student_mixer_output.float().flatten(),
                        output.teacher_mixer_output.float().flatten(),
                        dim=0,
                    )
                ),
                "block_cosine": float(
                    torch.nn.functional.cosine_similarity(
                        output.student_block_output.float().flatten(),
                        output.teacher_block_output.float().flatten(),
                        dim=0,
                    )
                ),
            }
        )
    print(json.dumps({"layer": args.layer, "rows": len(rows), "results": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
