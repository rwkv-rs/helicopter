from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"sample output is empty: {path}")
    return rows


def _score(value: Any, *, location: str) -> float:
    if isinstance(value, bool):
        value = float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{location} is not a finite numeric score")
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{location} must be in [0, 1], got {score}")
    return score


def _sample_identity(task: str, row: Mapping[str, Any]) -> str:
    hashes = tuple(row.get(key) for key in ("doc_hash", "prompt_hash", "target_hash"))
    if all(isinstance(value, str) and value for value in hashes):
        return f"{task}:" + ":".join(hashes)  # type: ignore[arg-type]
    for key in ("sample_id", "id"):
        if key in row and str(row[key]):
            return f"{task}:{row[key]}"
    immutable = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "generation",
            "predicted_answer",
            "is_correct",
            "symbolic_correct",
            "resps",
            "filtered_resps",
        }
    }
    digest = hashlib.sha256(
        json.dumps(immutable, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return f"{task}:{digest}"


def normalize_lm_eval(
    *,
    results_json: Path | Sequence[Path],
    samples_dir: Path,
    task_metrics: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Normalize pinned lm-eval sample logs into paired-bootstrap input rows."""
    result_paths = (results_json,) if isinstance(results_json, Path) else tuple(results_json)
    if not result_paths:
        raise ValueError("at least one lm-eval results JSON is required")
    aggregated: dict[str, Any] = {}
    for result_path in result_paths:
        result_rows = _read_json(result_path).get("results")
        if not isinstance(result_rows, dict):
            raise ValueError(f"lm-eval results JSON has no results object: {result_path}")
        duplicate = set(aggregated) & set(result_rows)
        if duplicate:
            raise ValueError(f"lm-eval tasks appear in multiple result files: {sorted(duplicate)}")
        aggregated.update(result_rows)
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task, metric in task_metrics:
        task_result = aggregated.get(task)
        if not isinstance(task_result, dict):
            raise ValueError(f"lm-eval aggregate is missing frozen task {task}")
        aggregate_keys = (metric, f"{metric},none")
        if not any(key in task_result for key in aggregate_keys):
            raise ValueError(f"lm-eval task {task} has no frozen metric {metric}")
        candidates = sorted(samples_dir.glob(f"samples_{task}_*.jsonl"))
        if len(candidates) != 1:
            raise ValueError(f"expected exactly one sample log for {task}, found {len(candidates)}")
        for row_index, row in enumerate(_read_jsonl(candidates[0])):
            if metric not in row:
                raise ValueError(f"{candidates[0]} row {row_index} has no metric {metric}")
            sample_id = _sample_identity(task, row)
            if sample_id in seen:
                raise ValueError(f"duplicate normalized sample id: {sample_id}")
            seen.add(sample_id)
            output.append(
                {
                    "sample_id": sample_id,
                    "group": task,
                    "score": _score(row[metric], location=f"{task}[{row_index}].{metric}"),
                }
            )
    return sorted(output, key=lambda row: row["sample_id"])


def normalize_ruler2(
    *,
    inputs: Iterable[tuple[int, str, Path]],
    expected_tasks: Sequence[str],
    expected_lengths: Sequence[int],
    samples_per_bucket: int,
) -> list[dict[str, Any]]:
    """Normalize scored NeMo-Skills RULERv2 JSONL files without rescoring them."""
    paths = {(length, task): path for length, task, path in inputs}
    expected = {(length, task) for length in expected_lengths for task in expected_tasks}
    if set(paths) != expected:
        missing = sorted(expected - set(paths))
        extra = sorted(set(paths) - expected)
        raise ValueError(f"RULERv2 bucket mismatch; missing={missing}, extra={extra}")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for length, task in sorted(expected):
        rows = _read_jsonl(paths[(length, task)])
        if len(rows) != samples_per_bucket:
            raise ValueError(
                f"RULERv2 {task}@{length} has {len(rows)} samples; expected {samples_per_bucket}"
            )
        group = f"{task}@{length}"
        for row_index, row in enumerate(rows):
            available = [key for key in ("is_correct", "symbolic_correct") if key in row]
            if len(available) != 1:
                raise ValueError(f"{group} row {row_index} must contain exactly one correctness field")
            base_id = _sample_identity(task, row)
            sample_id = f"{length}:{base_id}"
            if sample_id in seen:
                raise ValueError(f"duplicate normalized sample id: {sample_id}")
            seen.add(sample_id)
            output.append(
                {
                    "sample_id": sample_id,
                    "group": group,
                    "score": _score(row[available[0]], location=f"{group}[{row_index}].{available[0]}"),
                }
            )
    return sorted(output, key=lambda row: row["sample_id"])


def write_score_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
