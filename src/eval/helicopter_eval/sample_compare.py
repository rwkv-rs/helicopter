from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def compare_sample_runs(
    *,
    baseline_manifest_path: str | Path,
    candidate_manifest_path: str | Path,
    baseline_results_path: str | Path | None = None,
    candidate_results_path: str | Path | None = None,
    max_examples: int = 20,
) -> dict[str, Any]:
    baseline_manifest = load_json_rows(baseline_manifest_path)
    candidate_manifest = load_json_rows(candidate_manifest_path)
    payload: dict[str, Any] = {
        "manifest": compare_manifests(
            baseline_manifest,
            candidate_manifest,
            max_examples=max_examples,
        )
    }
    if baseline_results_path or candidate_results_path:
        if not baseline_results_path or not candidate_results_path:
            raise ValueError("baseline and candidate result paths must be provided together")
        payload["results"] = compare_results(
            baseline_manifest,
            candidate_manifest,
            load_json_rows(baseline_results_path),
            load_json_rows(candidate_results_path),
            max_examples=max_examples,
        )
    return payload


def compare_manifests(
    baseline_manifest: Sequence[Mapping[str, Any]],
    candidate_manifest: Sequence[Mapping[str, Any]],
    *,
    max_examples: int = 20,
) -> dict[str, Any]:
    baseline_keys = [manifest_identity(row, index=index) for index, row in enumerate(baseline_manifest)]
    candidate_keys = [manifest_identity(row, index=index) for index, row in enumerate(candidate_manifest)]
    baseline_counts = Counter(baseline_keys)
    candidate_counts = Counter(candidate_keys)
    missing = list((baseline_counts - candidate_counts).elements())
    extra = list((candidate_counts - baseline_counts).elements())
    return {
        "baseline_total": len(baseline_manifest),
        "candidate_total": len(candidate_manifest),
        "same_order": baseline_keys == candidate_keys,
        "same_multiset": not missing and not extra,
        "matched_positions": sum(1 for left, right in zip(baseline_keys, candidate_keys) if left == right),
        "missing_in_candidate": missing[:max_examples],
        "extra_in_candidate": extra[:max_examples],
        "missing_count": len(missing),
        "extra_count": len(extra),
        "baseline_duplicates": _duplicates(baseline_counts, max_examples=max_examples),
        "candidate_duplicates": _duplicates(candidate_counts, max_examples=max_examples),
    }


def compare_results(
    baseline_manifest: Sequence[Mapping[str, Any]],
    candidate_manifest: Sequence[Mapping[str, Any]],
    baseline_results: Sequence[Mapping[str, Any]],
    candidate_results: Sequence[Mapping[str, Any]],
    *,
    max_examples: int = 20,
) -> dict[str, Any]:
    baseline_by_key = _results_by_manifest_key(baseline_results, baseline_manifest)
    candidate_by_key = _results_by_manifest_key(candidate_results, candidate_manifest)
    baseline_keys = set(baseline_by_key)
    candidate_keys = set(candidate_by_key)
    shared_keys = sorted(baseline_keys & candidate_keys)
    missing_in_candidate = sorted(baseline_keys - candidate_keys)
    extra_in_candidate = sorted(candidate_keys - baseline_keys)

    baseline_passed = {key for key, row in baseline_by_key.items() if _is_passed(row)}
    candidate_passed = {key for key, row in candidate_by_key.items() if _is_passed(row)}
    both_passed = sorted(baseline_passed & candidate_passed)
    baseline_only_passed = sorted(baseline_passed - candidate_passed)
    candidate_only_passed = sorted(candidate_passed - baseline_passed)
    pass_disagreements = sorted(
        key for key in shared_keys if _is_passed(baseline_by_key[key]) != _is_passed(candidate_by_key[key])
    )

    return {
        "baseline_total": len(baseline_results),
        "candidate_total": len(candidate_results),
        "matched": len(shared_keys),
        "missing_in_candidate_count": len(missing_in_candidate),
        "extra_in_candidate_count": len(extra_in_candidate),
        "baseline_passed": len(baseline_passed),
        "candidate_passed": len(candidate_passed),
        "both_passed": len(both_passed),
        "baseline_only_passed": len(baseline_only_passed),
        "candidate_only_passed": len(candidate_only_passed),
        "pass_disagreements": len(pass_disagreements),
        "missing_in_candidate": missing_in_candidate[:max_examples],
        "extra_in_candidate": extra_in_candidate[:max_examples],
        "baseline_only_passed_examples": _result_examples(
            baseline_only_passed,
            baseline_by_key,
            max_examples=max_examples,
        ),
        "candidate_only_passed_examples": _result_examples(
            candidate_only_passed,
            candidate_by_key,
            max_examples=max_examples,
        ),
        "pass_disagreement_examples": _disagreement_examples(
            pass_disagreements,
            baseline_by_key,
            candidate_by_key,
            max_examples=max_examples,
        ),
    }


def load_json_rows(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if target.suffix == ".json":
        payload = json.loads(stripped)
        if isinstance(payload, list):
            return [_as_row(item) for item in payload]
        if isinstance(payload, dict):
            for key in ("rows", "results", "evals", "samples", "manifest"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [_as_row(item) for item in value]
            return [_as_row(payload)]
    rows = []
    for line_index, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{target} line {line_index} is not a JSON object")
        rows.append(payload)
    return rows


def manifest_identity(row: Mapping[str, Any], *, index: int) -> str:
    for key in ("sample_sha256", "task_id", "source_id"):
        value = row.get(key)
        if value not in (None, ""):
            return f"{key}:{value}"
    source_sample_index = row.get("source_sample_index")
    if source_sample_index is None:
        source_sample_index = row.get("original_sample_index")
    if source_sample_index is None:
        source_sample_index = row.get("dataset_sample_index")
    if source_sample_index is not None:
        dataset = row.get("source_dataset") or row.get("dataset") or ""
        return f"source_sample_index:{dataset}:{source_sample_index}"
    sample_index = row.get("sample_index")
    if sample_index is None:
        sample_index = row.get("manifest_sample_index")
    if sample_index is not None:
        dataset = row.get("dataset") or ""
        return f"sample_index:{dataset}:{sample_index}"
    return f"manifest_order:{index}"


def result_identity(
    row: Mapping[str, Any],
    *,
    index: int,
    manifest_by_sample_index: Mapping[int, str],
    manifest_by_order: Mapping[int, str],
) -> str:
    sample_index = row.get("sample_index")
    if _int_like(sample_index) and int(sample_index) in manifest_by_sample_index:
        return manifest_by_sample_index[int(sample_index)]
    sample_order = row.get("sample_order")
    if _int_like(sample_order) and int(sample_order) in manifest_by_order:
        return manifest_by_order[int(sample_order)]
    for key in ("sample_sha256", "task_id", "source_id"):
        value = row.get(key)
        if value not in (None, ""):
            return f"{key}:{value}"
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("source_id", "task_id"):
            value = metadata.get(key)
            if value not in (None, ""):
                return f"{key}:{value}"
    return f"result_order:{index}"


def _results_by_manifest_key(
    results: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    manifest_by_sample_index = {
        int(row["sample_index"]): manifest_identity(row, index=index)
        for index, row in enumerate(manifest)
        if _int_like(row.get("sample_index"))
    }
    manifest_by_order = {
        int(row.get("sample_order", index)): manifest_identity(row, index=index)
        for index, row in enumerate(manifest)
        if _int_like(row.get("sample_order", index))
    }
    keyed: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(results):
        key = result_identity(
            row,
            index=index,
            manifest_by_sample_index=manifest_by_sample_index,
            manifest_by_order=manifest_by_order,
        )
        keyed[key] = row
    return keyed


def _is_passed(row: Mapping[str, Any]) -> bool:
    for key in ("is_passed", "passed", "correct"):
        if key in row:
            return _to_bool(row[key])
    return False


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "pass", "passed", "correct"}
    return False


def _result_examples(
    keys: Sequence[str],
    rows_by_key: Mapping[str, Mapping[str, Any]],
    *,
    max_examples: int,
) -> list[dict[str, Any]]:
    return [_result_example(key, rows_by_key[key]) for key in keys[:max_examples]]


def _disagreement_examples(
    keys: Sequence[str],
    baseline_by_key: Mapping[str, Mapping[str, Any]],
    candidate_by_key: Mapping[str, Mapping[str, Any]],
    *,
    max_examples: int,
) -> list[dict[str, Any]]:
    return [
        {
            "identity": key,
            "baseline": _result_example(key, baseline_by_key[key]),
            "candidate": _result_example(key, candidate_by_key[key]),
        }
        for key in keys[:max_examples]
    ]


def _result_example(key: str, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity": key,
        "sample_index": row.get("sample_index"),
        "task_id": row.get("task_id"),
        "is_passed": _is_passed(row),
        "answer": row.get("answer"),
        "reference_answer": row.get("reference_answer") or row.get("ref_answer"),
        "fail_reason": row.get("fail_reason"),
    }


def _duplicates(counts: Counter[str], *, max_examples: int) -> list[dict[str, Any]]:
    return [{"identity": key, "count": count} for key, count in counts.items() if count > 1][:max_examples]


def _int_like(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _as_row(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("JSON row is not an object")
    return value
