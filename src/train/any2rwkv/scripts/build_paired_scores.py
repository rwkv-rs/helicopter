#!/usr/bin/env python3
"""Bind task-native teacher/student scores into evaluator input artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_rows(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in rows:
                raise SystemExit(f"duplicate sample_id in {path}: {sample_id}")
            rows[sample_id] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, choices=("ruler", "downstream"))
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--student", required=True, type=Path)
    parser.add_argument("--teacher-sha256", required=True)
    parser.add_argument("--student-sha256", required=True)
    parser.add_argument("--runner-repository", required=True)
    parser.add_argument("--runner-revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for name in ("teacher_sha256", "student_sha256"):
        value = getattr(args, name)
        if len(value) != 64:
            raise SystemExit(f"{name} must be a SHA-256 digest")
    if len(args.runner_revision) != 40:
        raise SystemExit("runner revision must be a full commit SHA")
    teacher = read_rows(args.teacher)
    student = read_rows(args.student)
    if teacher.keys() != student.keys():
        raise SystemExit("teacher/student task-native sample ids differ")
    args.output.mkdir(parents=True, exist_ok=False)
    data = args.output / "paired-scores.jsonl"
    with data.open("w", encoding="utf-8") as handle:
        for sample_id in sorted(teacher):
            left, right = teacher[sample_id], student[sample_id]
            if str(left["group"]) != str(right["group"]):
                raise SystemExit(f"teacher/student group differs for {sample_id}")
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "group": str(left["group"]),
                        "teacher": float(left["score"]),
                        "student": float(right["score"]),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    manifest = {
        "schema_version": 1,
        "suite": args.suite,
        "runner_repository": args.runner_repository,
        "runner_revision": args.runner_revision,
        "teacher_sha256": args.teacher_sha256,
        "student_sha256": args.student_sha256,
        "data_file": data.name,
        "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "row_count": len(teacher),
        "pairing": "identical sample_id",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
