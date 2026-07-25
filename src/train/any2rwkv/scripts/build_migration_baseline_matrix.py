#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from any2rwkv.core.migration_baselines import build_migration_baseline_matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-manifest", action="append", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_migration_baseline_matrix(
        tuple(Path(value).resolve() for value in args.stage_manifest),
        student_checkpoint=Path(args.student_checkpoint).resolve(),
        output=Path(args.output).resolve(),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "student_sha256": result["student_sha256"],
                "stage_count": len(result["baselines"]),
                "output": str(Path(args.output).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
