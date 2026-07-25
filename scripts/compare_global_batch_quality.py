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
QUALITY_GEMM_POLICY = "fp32-state-fp16-io_no-fp16-accumulation"


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
    expected_config_name = f"strict-quality-b{expected_batch_size}.toml"
    config_path = Path(str(metadata.get("config", {}).get("path", "")))
    if config_path.name != expected_config_name:
        raise RuntimeError(
            f"quality artifact {path} must use {expected_config_name}, got {config_path.name!r}"
        )
    if metadata.get("precision") != "fp32io16" or metadata.get("wkv_mode") != "fp32io16":
        raise RuntimeError(f"quality artifact must use fp32io16 precision/WKV mode: {path}")
    if metadata.get("gemm_policy") != QUALITY_GEMM_POLICY:
        raise RuntimeError(f"quality artifact has the wrong GEMM policy: {path}")
    return metadata


def first_time_at_or_above(points: list[dict[str, Any]], metric: str, target: float) -> float | None:
    if not points:
        return None
    initial_wall_seconds = float(points[0]["cumulative_wall_seconds"])
    for point in points[1:]:
        if int(point["cumulative_samples"]) <= 0:
            continue
        value = float(point["metrics"][metric])
        wall_seconds = float(point["cumulative_wall_seconds"]) - initial_wall_seconds
        if not math.isfinite(value) or not math.isfinite(wall_seconds):
            raise RuntimeError("quality comparison encountered a non-finite value")
        if wall_seconds <= 0:
            raise RuntimeError("post-training quality wall time must be positive")
        if value >= target:
            return wall_seconds
    return None


def _required_path(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise RuntimeError(f"quality artifact is missing comparable field {path}")
        value = value[key]
    return value


def verify_comparable_contract(baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    equal_paths = (
        "source.commit",
        "source_revisions",
        "checkpoint.sha256",
        "dataset_manifest.sha256",
        "seed",
        "precision",
        "wkv_mode",
        "gemm_policy",
        "topology",
        "rollout_capacity",
        "measurement_contract",
        "environment",
    )
    for path in equal_paths:
        baseline_value = _required_path(baseline, path)
        candidate_value = _required_path(candidate, path)
        if baseline_value != candidate_value:
            raise RuntimeError(
                f"quality artifacts are not comparable for {path}: "
                f"{baseline_value!r} != {candidate_value!r}"
            )

    ignored_batch_fields = {"train_batch_size", "ppo_mini_batch_size"}
    baseline_batch = {
        key: value
        for key, value in _required_path(baseline, "batch").items()
        if key not in ignored_batch_fields
    }
    candidate_batch = {
        key: value
        for key, value in _required_path(candidate, "batch").items()
        if key not in ignored_batch_fields
    }
    if baseline_batch != candidate_batch:
        raise RuntimeError(
            "quality artifacts change non-global-batch fields: "
            f"{baseline_batch!r} != {candidate_batch!r}"
        )

    baseline_validation = _required_path(baseline, "validation")
    candidate_validation = _required_path(candidate, "validation")
    for field in ("accuracy_metrics", "common_metrics"):
        if baseline_validation.get(field) != candidate_validation.get(field):
            raise RuntimeError(f"quality artifacts do not share validation {field}")
    baseline_axis = [point.get("cumulative_samples") for point in baseline_validation["points"]]
    candidate_axis = [point.get("cumulative_samples") for point in candidate_validation["points"]]
    if baseline_axis != candidate_axis or baseline_axis != list(range(0, 6272 + 1, 896)):
        raise RuntimeError(
            "quality artifacts must share the fixed 0..6272 validation sample axis: "
            f"{baseline_axis!r} != {candidate_axis!r}"
        )


def compare_quality(
    baseline: dict[str, Any], candidate: dict[str, Any], *, metric: str | None = None
) -> dict[str, Any]:
    verify_comparable_contract(baseline, candidate)
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
        "wall_clock_basis": (
            "first post-training validation at or above target; elapsed from completion "
            "of initial validation"
        ),
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
