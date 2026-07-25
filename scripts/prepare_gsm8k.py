#!/usr/bin/env python3
"""Convert pinned raw GSM8K parquet files into Verl RL parquet files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from datasets import Dataset, load_dataset


DATA_SOURCE = "openai/gsm8k"
INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def extract_ground_truth(answer: str) -> str:
    match = re.search(r"#### (\-?[0-9\.\,]+)", answer)
    if match is None:
        raise ValueError("GSM8K answer does not contain a '####' ground truth")
    return match.group(1).replace(",", "")


def convert_split(raw_path: Path, output_path: Path, split: str) -> None:
    raw = load_dataset("parquet", data_files=str(raw_path), split="train")

    def convert(example: dict[str, str], index: int) -> dict[str, object]:
        question = example["question"]
        answer = example["answer"]
        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": f"{question} {INSTRUCTION}"}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": extract_ground_truth(answer)},
            "extra_info": {
                "split": split,
                "index": index,
                "answer": answer,
                "question": question,
            },
        }

    converted: Dataset = raw.map(convert, with_indices=True, remove_columns=raw.column_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted.to_parquet(str(output_path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(manifest_path: Path, outputs: list[Path], revision: str) -> None:
    payload = {
        "name": "gsm8k",
        "source": {"repo": DATA_SOURCE, "revision": revision},
        "files": [
            {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in outputs
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-train", type=Path, required=True)
    parser.add_argument("--raw-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    train_output = args.output_dir / "train.parquet"
    test_output = args.output_dir / "test.parquet"
    convert_split(args.raw_train, train_output, "train")
    convert_split(args.raw_test, test_output, "test")
    write_manifest(args.manifest, [train_output, test_output], args.revision)


if __name__ == "__main__":
    main()
