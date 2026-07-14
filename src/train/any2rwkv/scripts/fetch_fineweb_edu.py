#!/usr/bin/env python3
"""Fetch a pinned FineWeb-Edu stream into an auditable local JSONL input.

This is the only network-facing data stage.  The subsequent ``prepare_data``
stage is deliberately offline and binds these bytes by SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from datasets import load_dataset
from transformers import AutoTokenizer


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sample_id(row: Mapping[str, Any]) -> str:
    for name in ("id", "sample_id"):
        value = row.get(name)
        if value is not None and str(value):
            return str(value)
    provenance = {
        name: row.get(name)
        for name in ("url", "dump", "file_path", "date")
        if row.get(name) is not None
    }
    provenance["text_sha256"] = hashlib.sha256(
        str(row.get("text", "")).encode("utf-8")
    ).hexdigest()
    return hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--subset", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer-path", required=True, type=Path)
    parser.add_argument("--target-tokens", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.target_tokens <= 0:
        raise SystemExit("--target-tokens must be positive")
    tokenizer_path = args.tokenizer_path.resolve()
    if not tokenizer_path.is_dir():
        raise SystemExit(f"local tokenizer path does not exist: {tokenizer_path}")
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing dataset input: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    stream = load_dataset(
        args.repository,
        name=args.subset,
        split=args.split,
        revision=args.revision,
        streaming=True,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent, text=True
    )
    temporary = Path(temporary_name)
    rows = 0
    tokens = 0
    seen_ids: set[str] = set()
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for source_row in stream:
                text = source_row.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                sample_id = source_sample_id(source_row)
                if sample_id in seen_ids:
                    continue
                seen_ids.add(sample_id)
                token_count = len(tokenizer.encode(text, add_special_tokens=False))
                if token_count == 0:
                    continue
                handle.write(
                    json.dumps(
                        {"sample_id": sample_id, "text": text},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                rows += 1
                tokens += token_count
                if tokens >= args.target_tokens:
                    break
            handle.flush()
            os.fsync(handle.fileno())
        if tokens < args.target_tokens:
            raise RuntimeError(
                f"stream exhausted at {tokens} tokenizer tokens; target={args.target_tokens}"
            )
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    result = {
        "schema_version": 1,
        "repository": args.repository,
        "revision": args.revision,
        "subset": args.subset,
        "split": args.split,
        "tokenizer_path": str(tokenizer_path),
        "target_tokens": args.target_tokens,
        "materialized_tokens": tokens,
        "row_count": rows,
        "output": str(output),
        "sha256": file_sha256(output),
    }
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    manifest.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
