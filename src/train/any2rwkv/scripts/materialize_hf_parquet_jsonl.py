#!/usr/bin/env python3
"""Materialize a deterministic, resumable JSONL slice from fixed HF Parquet files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from datasets import load_dataset
from huggingface_hub import hf_hub_url
from transformers import AutoTokenizer

from any2rwkv.artifacts import file_sha256
from any2rwkv.data import directory_sha256, normalize_text


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def _binding(args: argparse.Namespace, tokenizer_tree: str) -> dict[str, object]:
    return {
        "dataset_repository": args.dataset_repository,
        "dataset_revision": args.dataset_revision,
        "source_files": list(args.source_file),
        "tokenizer_path": str(args.tokenizer_path.resolve()),
        "tokenizer_tree_sha256": tokenizer_tree,
        "target_tokens": args.target_tokens,
        "text_field": args.text_field,
        "id_field": args.id_field,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repository", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--source-file", action="append", required=True)
    parser.add_argument("--tokenizer-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-tokens", required=True, type=int)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--checkpoint-rows", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.target_tokens <= 0 or args.checkpoint_rows <= 0:
        raise SystemExit("target-tokens and checkpoint-rows must be positive")
    tokenizer_path = args.tokenizer_path.resolve()
    if not tokenizer_path.is_dir():
        raise SystemExit(f"tokenizer path does not exist: {tokenizer_path}")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    progress_path = output.with_name(output.name + ".progress.json")
    manifest_path = output.with_name(output.name + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise SystemExit(f"refusing to overwrite completed materialization: {output}")

    tokenizer_tree = directory_sha256(tokenizer_path)
    binding = _binding(args, tokenizer_tree)
    rows_written = 0
    source_rows_consumed = 0
    token_count = 0
    if partial.exists() or progress_path.exists():
        if not partial.is_file() or not progress_path.is_file():
            raise SystemExit("partial output and progress must either both exist or both be absent")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("schema_version") != 1 or progress.get("binding") != binding:
            raise SystemExit("materialization resume binding mismatch")
        rows_written = int(progress["rows_written"])
        source_rows_consumed = int(progress["source_rows_consumed"])
        token_count = int(progress["token_count"])
        with partial.open("r", encoding="utf-8") as handle:
            actual_rows = sum(1 for _ in handle)
        if actual_rows != rows_written:
            raise SystemExit("materialization partial row count differs from progress")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    eos_id = tokenizer.eos_token_id
    urls = [
        hf_hub_url(
            args.dataset_repository,
            filename=filename,
            repo_type="dataset",
            revision=args.dataset_revision,
        )
        for filename in args.source_file
    ]
    stream = load_dataset(
        "parquet", data_files={"train": urls}, split="train", streaming=True
    )
    mode = "a" if rows_written else "w"
    with partial.open(mode, encoding="utf-8") as handle:
        for source_row, row in enumerate(stream):
            if source_row < source_rows_consumed:
                continue
            source_rows_consumed = source_row + 1
            if args.text_field not in row:
                raise SystemExit(f"source row {source_row} lacks {args.text_field!r}")
            text = normalize_text(row[args.text_field])
            if not text:
                continue
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            if eos_id is not None and (not input_ids or input_ids[-1] != eos_id):
                input_ids.append(eos_id)
            source_id = row.get(args.id_field)
            sample_id = str(source_id) if source_id not in (None, "") else f"row-{source_row:012d}"
            handle.write(
                json.dumps(
                    {"sample_id": f"fineweb-edu:{sample_id}", "text": text},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            rows_written += 1
            token_count += len(input_ids)
            if rows_written % args.checkpoint_rows == 0 or token_count >= args.target_tokens:
                handle.flush()
                os.fsync(handle.fileno())
                _write_json_atomic(
                    progress_path,
                    {
                        "schema_version": 1,
                        "status": "running",
                        "binding": binding,
                        "rows_written": rows_written,
                        "source_rows_consumed": source_rows_consumed,
                        "token_count": token_count,
                    },
                )
            if token_count >= args.target_tokens:
                break
        else:
            raise SystemExit(
                f"source exhausted at {token_count} tokens before target {args.target_tokens}"
            )

    partial.replace(output)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "binding": binding,
        "rows_written": rows_written,
        "source_rows_consumed": source_rows_consumed,
        "token_count": token_count,
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": file_sha256(output),
    }
    _write_json_atomic(manifest_path, manifest)
    progress_path.unlink(missing_ok=True)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
