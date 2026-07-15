#!/usr/bin/env python3
"""Compare equal-sample strict MaxRL global-batch quality artifacts."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


QUALITY_NONINFERIORITY_MARGIN = 0.03


def load_metadata(path: Path, *, expected_batch_size: int) -> dict[str, Any]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("status") != "done" or metadata.get("exit_code") != 0:
        raise RuntimeError(f"quality artifact is not successful: {path}")
    if metadata.get("run_phase") != "global-batch-quality":
        raise RuntimeError(f"quality artifact has the wrong run phase: {path}")
    if int(metadata.get("batch", {}).get("train_batch_size", -1)) != expected_batch_size:
        raise RuntimeError(
            f"quality artifact {path} must use train_batch_size={expected_batch_size}"
        )
    validation = metadata.get("validation")
    if not isinstance(validation, dict) or validation.get("final_cumulative_samples") != 6272:
        raise RuntimeError(f"quality artifact lacks the equal-sample validation contract: {path}")
    return metadata


def first_time_at_or_above(points: list[dict[str, Any]], metric: str, target: float) -> float | None:
    for point in points:
        value = float(point["metrics"][metric])
        wall_seconds = float(point["cumulative_wall_seconds"])
        if not math.isfinite(value) or not math.isfinite(wall_seconds):
            raise RuntimeError("quality comparison encountered a non-finite value")
        if value >= target:
            return wall_seconds
    return None


def compare_quality(
    baseline: dict[str, Any], candidate: dict[str, Any], *, metric: str | None = None
) -> dict[str, Any]:
    baseline_validation = baseline["validation"]
    candidate_validation = candidate["validation"]
    common_accuracy = sorted(
        set(baseline_validation["accuracy_metrics"])
        & set(candidate_validation["accuracy_metrics"])
    )
    if metric is None:
        if len(common_accuracy) != 1:
            raise RuntimeError(
                "quality comparison requires exactly one common accuracy metric or --metric; "
                f"found {common_accuracy}"
            )
        metric = common_accuracy[0]
    elif metric not in common_accuracy:
        raise RuntimeError(f"requested metric is not common to both artifacts: {metric}")

    baseline_points = baseline_validation["points"]
    candidate_points = candidate_validation["points"]
    baseline_final = float(baseline_points[-1]["metrics"][metric])
    candidate_final = float(candidate_points[-1]["metrics"][metric])
    target = baseline_final - QUALITY_NONINFERIORITY_MARGIN
    baseline_time = first_time_at_or_above(baseline_points, metric, target)
    candidate_time = first_time_at_or_above(candidate_points, metric, target)
    quality_passed = candidate_final >= target
    wall_clock_passed = (
        baseline_time is not None
        and candidate_time is not None
        and candidate_time <= baseline_time
    )
    accepted = quality_passed and wall_clock_passed
    return {
        "schema_version": 1,
        "decision": "accept-batch-112" if accepted else "retain-batch-56",
        "accepted": accepted,
        "metric": metric,
        "noninferiority_margin_absolute": QUALITY_NONINFERIORITY_MARGIN,
        "baseline": {
            "run_id": baseline["run_id"],
            "batch_size": 56,
            "final_accuracy": baseline_final,
            "time_to_target_seconds": baseline_time,
        },
        "candidate": {
            "run_id": candidate["run_id"],
            "batch_size": 112,
            "final_accuracy": candidate_final,
            "time_to_target_seconds": candidate_time,
        },
        "target_accuracy": target,
        "quality_gate_passed": quality_passed,
        "wall_clock_gate_passed": wall_clock_passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric")
    args = parser.parse_args()
    baseline = load_metadata(args.baseline, expected_batch_size=56)
    candidate = load_metadata(args.candidate, expected_batch_size=112)
    result = compare_quality(baseline, candidate, metric=args.metric)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["baseline_metadata"] = str(args.baseline.resolve())
    result["candidate_metadata"] = str(args.candidate.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
