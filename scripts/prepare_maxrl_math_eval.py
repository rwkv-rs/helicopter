#!/usr/bin/env python3
"""Prepare the fixed MaxRL math validation suite for Verl RL training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


@dataclass(frozen=True)
class Source:
    name: str
    sha256: str
    rows: int
    loader: str
    question_field: str
    answer_field: str
    output_dir: str
    data_source: str


SOURCES = {
    "aime24": Source(
        name="aime24",
        sha256="2602156eb2a0ab64393dc46617a3bfee3451ae3c985a7ea6ac2b55436e094a38",
        rows=30,
        loader="parquet",
        question_field="problem",
        answer_field="solution",
        output_dir="AIME24",
        data_source="aime24",
    ),
    "aime25": Source(
        name="aime25",
        sha256="b4e273c02d3e7fe1b74b59eae768fc8230bfb0f79539890cb56f4361caac0331",
        rows=30,
        loader="json",
        question_field="problem",
        answer_field="answer",
        output_dir="AIME25",
        data_source="aime25",
    ),
    "amc23": Source(
        name="amc23",
        sha256="b696e87ba47be4e879a60fd4ef1d4aa522ba78c5ae013c6aff5cc9788a397c5e",
        rows=40,
        loader="parquet",
        question_field="question",
        answer_field="answer",
        output_dir="AMC23",
        data_source="amc23",
    ),
    "math500": Source(
        name="math500",
        sha256="35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132",
        rows=500,
        loader="json",
        question_field="problem",
        answer_field="answer",
        output_dir="math500",
        data_source="HuggingFaceH4/MATH-500",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_record(record: dict[str, Any], index: int, source: Source) -> dict[str, Any]:
    question = str(record[source.question_field]).strip()
    ground_truth = str(record[source.answer_field]).strip()
    if not question or not ground_truth:
        raise RuntimeError(f"{source.name} row {index} has an empty question or answer")
    return {
        "data_source": source.data_source,
        "prompt": [{"role": "user", "content": f"{question} {INSTRUCTION}"}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": "test",
            "index": index,
            "question": question,
        },
    }


def prepare_source(raw_path: Path, output_root: Path, source: Source) -> Path:
    actual_sha256 = sha256_file(raw_path)
    if actual_sha256 != source.sha256:
        raise RuntimeError(
            f"{source.name} SHA-256 mismatch: expected {source.sha256}, found {actual_sha256}"
        )
    raw_dataset = load_dataset(source.loader, data_files=str(raw_path), split="train")
    if len(raw_dataset) != source.rows:
        raise RuntimeError(
            f"{source.name} expected {source.rows} rows, found {len(raw_dataset)}"
        )
    records = [convert_record(record, index, source) for index, record in enumerate(raw_dataset)]
    prepared = Dataset.from_list(records)
    output_dir = output_root / source.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test.parquet"
    temporary_path = output_dir / ".test.parquet.tmp"
    temporary_path.unlink(missing_ok=True)
    prepared.to_parquet(temporary_path)
    os.replace(temporary_path, output_path)
    return output_path


def write_manifest(path: Path, files: list[Path]) -> None:
    payload = {
        "schema_version": 1,
        "files": [
            {
                "path": str(file.resolve()),
                "size_bytes": file.stat().st_size,
                "sha256": sha256_file(file),
            }
            for file in files
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapo-train", type=Path, required=True)
    parser.add_argument("--aime24-raw", type=Path, required=True)
    parser.add_argument("--aime25-raw", type=Path, required=True)
    parser.add_argument("--amc23-raw", type=Path, required=True)
    parser.add_argument("--math500-raw", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_paths = {
        "aime24": args.aime24_raw,
        "aime25": args.aime25_raw,
        "amc23": args.amc23_raw,
        "math500": args.math500_raw,
    }
    outputs = [
        prepare_source(raw_paths[name], args.output_root, source)
        for name, source in SOURCES.items()
    ]
    if not args.dapo_train.is_file():
        raise RuntimeError(f"DAPO training file is missing: {args.dapo_train}")
    write_manifest(args.manifest, [args.dapo_train, *outputs])
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "outputs": [str(path) for path in outputs],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
