#!/usr/bin/env python3
"""Prepare deterministic Any2RWKV packed rows from local JSONL inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from transformers import AutoTokenizer

from any2rwkv.data import DataPreparationConfig, prepare_jsonl_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Any2RWKV data from already-downloaded local JSONL and tokenizer files."
    )
    parser.add_argument("--input", action="append", required=True, type=Path, help="Local JSONL input; repeatable")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-repository", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--tokenizer-path", required=True, type=Path)
    parser.add_argument("--tokenizer-repository", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--burn-in-tokens", required=True, type=int)
    parser.add_argument("--supervised-tokens", required=True, type=int)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--split-ratios-json", help="JSON object overriding all six split ratios")
    parser.add_argument("--exact-duplicate-policy", choices=("drop", "reject"), default="drop")
    parser.add_argument("--near-duplicate-policy", choices=("report", "reject"), default="report")
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.8)
    parser.add_argument("--id-field", default="sample_id")
    parser.add_argument("--text-field", default="text")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tokenizer_path = args.tokenizer_path.resolve()
    if not tokenizer_path.is_dir():
        raise SystemExit(f"tokenizer path must be an existing local directory: {tokenizer_path}")
    split_ratios = json.loads(args.split_ratios_json) if args.split_ratios_json else None
    config_kwargs = {
        "burn_in_tokens": args.burn_in_tokens,
        "supervised_tokens": args.supervised_tokens,
        "seed": args.seed,
        "exact_duplicate_policy": args.exact_duplicate_policy,
        "near_duplicate_policy": args.near_duplicate_policy,
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "id_field": args.id_field,
        "text_field": args.text_field,
    }
    if split_ratios is not None:
        config_kwargs["split_ratios"] = split_ratios
    config = DataPreparationConfig(**config_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    manifest = prepare_jsonl_dataset(
        args.input,
        output_dir=args.output_dir.resolve(),
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        dataset_repository=args.dataset_repository,
        dataset_revision=args.dataset_revision,
        tokenizer_repository=args.tokenizer_repository,
        tokenizer_revision=args.tokenizer_revision,
        config=config,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
