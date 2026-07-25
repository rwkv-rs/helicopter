from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..artifacts import file_sha256


def read_quality_dataset_evidence(
    reference: object, *, owner_path: Path, expected_sha256: str
) -> tuple[dict[str, Any], str, Path]:
    if not isinstance(reference, dict):
        raise ValueError("calibration protocol has no dataset manifest reference")
    manifest_path = Path(str(reference.get("artifact", "")))
    if not manifest_path.is_absolute():
        manifest_path = (owner_path.parent / manifest_path).resolve()
    claimed_sha = str(reference.get("artifact_sha256", ""))
    if (
        claimed_sha != expected_sha256
        or not manifest_path.is_file()
        or file_sha256(manifest_path) != claimed_sha
    ):
        raise ValueError("calibration dataset manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dedup = manifest.get("deduplication")
    split_assignment = manifest.get("split_assignment")
    splits = manifest.get("splits")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "prepared"
        or not isinstance(dedup, dict)
        or not isinstance(split_assignment, dict)
        or split_assignment.get("sample_ids_mutually_exclusive") is not True
        or not isinstance(splits, dict)
        or not {"distill_train", "validation", "ruler", "downstream", "smoke"}.issubset(splits)
    ):
        raise ValueError("calibration dataset manifest is incomplete")
    report_path = (manifest_path.parent / str(dedup.get("report_path", ""))).resolve()
    if not report_path.is_file() or file_sha256(report_path) != dedup.get("report_sha256"):
        raise ValueError("calibration deduplication report SHA-256 mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    near = report.get("near_duplicates")
    if not isinstance(near, dict) or near.get("candidate_search_complete") is not True:
        raise ValueError("near-duplicate candidate search is incomplete")
    policy = near.get("policy")
    pairs = near.get("pairs")
    if not isinstance(pairs, list) or policy not in {"report", "drop", "reject"}:
        raise ValueError("quality calibration has an invalid near-duplicate policy")
    if policy == "report" and pairs:
        raise ValueError("report-policy quality data still contains near duplicates")
    if policy == "reject" and pairs:
        raise ValueError("reject-policy quality data still contains near duplicates")
    if policy == "drop":
        dropped = set(near.get("dropped_sample_ids", []))
        if any(
            str(pair.get("left_sample_id", "")) not in dropped
            and str(pair.get("right_sample_id", "")) not in dropped
            for pair in pairs
            if isinstance(pair, dict)
        ):
            raise ValueError("drop-policy quality data retained a near-duplicate pair")
    return manifest, claimed_sha, manifest_path
