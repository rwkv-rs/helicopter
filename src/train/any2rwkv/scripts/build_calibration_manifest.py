#!/usr/bin/env python3
"""Decode the frozen calibration split into a ModelOpt calibration manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from any2rwkv.artifacts import file_sha256, write_json
from any2rwkv.evaluator_runner import combined_tokenizer_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-manifest", required=True, type=Path)
    parser.add_argument("--tokenizer-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.max_length <= 0 or args.batch_size <= 0:
        raise SystemExit("max-length and batch-size must be positive")

    prepared = args.prepared_manifest.resolve()
    payload = json.loads(prepared.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("status") != "prepared":
        raise SystemExit("prepared manifest must use schema_version=1 and status=prepared")
    splits = payload.get("splits")
    entry = splits.get("nvfp4_calibration") if isinstance(splits, dict) else None
    if not isinstance(entry, dict):
        raise SystemExit("prepared manifest has no nvfp4_calibration split")
    packed_path = prepared.parent / str(entry.get("path", ""))
    if not packed_path.is_file() or file_sha256(packed_path) != entry.get("sha256"):
        raise SystemExit("prepared nvfp4_calibration split SHA-256 mismatch")
    rows = [
        json.loads(line)
        for line in packed_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows or len(rows) != int(entry.get("row_count", -1)):
        raise SystemExit("prepared nvfp4_calibration row_count mismatch")

    tokenizer_path = args.tokenizer_path.resolve()
    tokenizer_sha = combined_tokenizer_sha256(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    decoded = []
    for row in rows:
        tokens = row.get("input_ids")
        row_id = str(row.get("row_id", ""))
        if not row_id or not isinstance(tokens, list) or not all(type(value) is int for value in tokens):
            raise SystemExit(f"invalid packed calibration row: {row_id}")
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        if not text:
            raise SystemExit(f"decoded calibration text is empty: {row_id}")
        decoded.append({"sample_id": row_id, "text": text})

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data_path = output / "calibration.jsonl"
    with data_path.open("w", encoding="utf-8") as handle:
        for row in decoded:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "split": "nvfp4_calibration",
        "data_file": data_path.name,
        "sha256": file_sha256(data_path),
        "row_count": len(decoded),
        "text_field": "text",
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "tokenizer_sha256": tokenizer_sha,
        "prepared_manifest": str(prepared),
        "prepared_manifest_sha256": file_sha256(prepared),
        "quality_gate_use_forbidden": True,
    }
    write_json(output / "calibration-manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
