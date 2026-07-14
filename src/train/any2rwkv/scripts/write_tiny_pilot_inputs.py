#!/usr/bin/env python3
"""Write deterministic local inputs for the 60-layer fixture GPU pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--tokens-per-row", type=int, default=96)
    parser.add_argument(
        "--execution-mode",
        choices=("resident", "streamed_layer_store"),
        default="resident",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rows < 128 or args.tokens_per_row < 32:
        raise SystemExit("tiny pilot requires at least 128 rows and 32 tokens per row")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data_path = output / "source.jsonl"
    with data_path.open("w", encoding="utf-8") as handle:
        for row in range(args.rows):
            tokens = [
                f"token_{3 + ((row * 17 + index * 13) % 61)}"
                for index in range(args.tokens_per_row)
            ]
            handle.write(
                json.dumps(
                    {
                        "sample_id": f"fixture-{row:08d}",
                        # Keep raw documents unique for dedup/split coverage;
                        # the fixture tokenizer intentionally maps this marker
                        # to <unk> so the token workload stays deterministic.
                        "text": f"row_{row:08d} " + " ".join(tokens),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    plan = {
        "schema_version": 1,
        "classification": "deterministic-60-layer-fixture-smoke-only",
        "seed": 20260714,
        "learning_rate": 0.0005,
        "burn_in_tokens": 16,
        "supervised_tokens": 16,
        "accumulation_steps": 1,
        "activation_checkpointing": True,
        "execution_mode": args.execution_mode,
        "stage_tokens_per_layer": {"signals": 16, "block": 16, "global": 16},
        "corrective_min_sweeps": 1,
        "corrective_max_sweeps": 1,
        "corrective_min_delta": 0.0,
    }
    plan_path = output / "distill-plan.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"data": str(data_path), "plan": str(plan_path), "rows": args.rows},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
