#!/usr/bin/env python3
"""Build the hash-bound Any2RWKV evaluator input from prepared split data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from any2rwkv.artifacts import file_sha256, write_json
from any2rwkv.evaluator_runner import combined_tokenizer_sha256


def _read_split(prepared: Path, payload: dict, name: str) -> list[dict]:
    splits = payload.get("splits")
    entry = splits.get(name) if isinstance(splits, dict) else None
    if not isinstance(entry, dict):
        raise SystemExit(f"prepared manifest has no {name} split")
    path = prepared.parent / str(entry.get("path", ""))
    if not path.is_file() or file_sha256(path) != entry.get("sha256"):
        raise SystemExit(f"prepared {name} split SHA-256 mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != int(entry.get("row_count", -1)):
        raise SystemExit(f"prepared {name} split row_count mismatch")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-manifest", required=True, type=Path)
    parser.add_argument("--tokenizer-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    prepared = args.prepared_manifest.resolve()
    payload = json.loads(prepared.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("status") != "prepared":
        raise SystemExit("prepared manifest must use schema_version=1 and status=prepared")
    validation_rows = _read_split(prepared, payload, "validation")
    smoke_rows = _read_split(prepared, payload, "smoke")
    if not validation_rows:
        raise SystemExit("evaluation validation split is empty")
    if len(smoke_rows) < 32:
        raise SystemExit("evaluation smoke split must contain at least 32 packed rows")

    tokenizer_path = args.tokenizer_path.resolve()
    tokenizer_sha = combined_tokenizer_sha256(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    validation_path = output / "validation.jsonl"
    with validation_path.open("w", encoding="utf-8") as handle:
        for row in validation_rows:
            tokens = row.get("input_ids")
            row_id = str(row.get("row_id", ""))
            if not row_id or not isinstance(tokens, list) or not all(type(value) is int for value in tokens):
                raise SystemExit(f"invalid packed validation row: {row_id}")
            handle.write(
                json.dumps(
                    {"sample_id": row_id, "input_ids": tokens}, sort_keys=True
                )
                + "\n"
            )
    smoke_path = output / "smoke-prompts.jsonl"
    with smoke_path.open("w", encoding="utf-8") as handle:
        for row in smoke_rows[:32]:
            tokens = row.get("input_ids")
            row_id = str(row.get("row_id", ""))
            if not row_id or not isinstance(tokens, list) or not all(type(value) is int for value in tokens):
                raise SystemExit(f"invalid packed smoke row: {row_id}")
            prompt = tokenizer.decode(tokens, skip_special_tokens=False)
            if not prompt:
                raise SystemExit(f"decoded smoke prompt is empty: {row_id}")
            handle.write(
                json.dumps(
                    {"sample_id": row_id, "prompt": prompt},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    packing = payload.get("packing", {})
    manifest = {
        "schema_version": 1,
        "split": "validation",
        "seed": int(payload["seed"]),
        "burn_in_tokens": int(packing["burn_in_tokens"]),
        "supervised_tokens": int(packing["supervised_tokens"]),
        "data_file": validation_path.name,
        "data_sha256": file_sha256(validation_path),
        "row_count": len(validation_rows),
        "smoke_file": smoke_path.name,
        "smoke_sha256": file_sha256(smoke_path),
        "smoke_row_count": 32,
        "smoke_new_tokens": 128,
        "bootstrap_samples": 10_000,
        "tokenizer_sha256": tokenizer_sha,
        "prepared_manifest": str(prepared),
        "prepared_manifest_sha256": file_sha256(prepared),
    }
    write_json(output / "evaluation-manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
