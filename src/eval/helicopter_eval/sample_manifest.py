from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


_IDENTITY_KEYS = (
    "task_id",
    "id",
    "instance_id",
    "question_id",
    "sample_id",
)
_METADATA_IDENTITY_KEYS = (
    "original_sample_index",
    "dataset_sample_index",
    "source_index",
    "source_id",
    "id",
    "task_id",
    "instance_id",
    "subject",
    "topic",
    "difficulty",
    "source",
    "dataset",
    "longbench_dataset",
    "category",
    "language",
)
_HASHED_CONTENT_KEYS = (
    "question",
    "prompt",
    "instruction",
    "context",
    "repo_text",
    "reference_answer",
    "answers",
    "choices",
    "messages",
    "turns",
    "tools",
    "task",
    "entry_point",
    "starter_code",
)
_PREVIEW_KEYS = ("question", "prompt", "instruction", "context")


def write_sample_manifest(
    samples: Sequence[Any],
    path: str | Path,
    *,
    config: Any,
    dataset: str,
    kind: str,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = [
        sample_to_manifest_row(sample, config=config, dataset=dataset, kind=kind, sample_order=index, source=source)
        for index, sample in enumerate(samples)
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return {
        "benchmark": str(getattr(config, "benchmark", "")),
        "dataset": dataset,
        "kind": kind,
        "manifest_path": str(target),
        "total": len(rows),
        "sample_size": getattr(config, "sample_size", None),
        "sample_seed": (
            getattr(config, "sample_seed", None)
            if getattr(config, "sample_size", None) is not None
            else None
        ),
        "sample_identity_sha256": _rows_identity_sha256(rows),
    }


def sample_to_manifest_row(
    sample: Any,
    *,
    config: Any,
    dataset: str,
    kind: str,
    sample_order: int,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _json_safe(sample)
    if not isinstance(payload, dict):
        payload = {"value": payload}
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    content_hashes = {
        key: _sha256_json(payload[key])
        for key in _HASHED_CONTENT_KEYS
        if key in payload and payload[key] not in (None, "", [], (), {})
    }
    metadata_hash = _sha256_json(metadata) if metadata else None
    row: dict[str, Any] = {
        "manifest_schema_version": 1,
        "benchmark": str(getattr(config, "benchmark", "")),
        "dataset": dataset,
        "kind": kind,
        "source_dataset": _source_value(source, config, "dataset_name"),
        "source_type": _source_value(source, config, "source_type"),
        "source_split": _source_value(source, config, "source_split", config_attr="split"),
        "source_url": _source_value(source, config, "source_url"),
        "source_urls": _source_value(source, config, "source_urls"),
        "source_path": _source_value(source, config, "source_path"),
        "source_root": getattr(config, "source_root", None),
        "row_adapter": _source_value(source, config, "row_adapter"),
        "sample_order": sample_order,
        "sample_index": payload.get("sample_index"),
        "source_sample_index": _source_sample_index(payload, metadata),
        "task_id": _task_id(payload, metadata),
        "metadata": _metadata_identity(metadata),
        "metadata_sha256": metadata_hash,
        "content_sha256": content_hashes,
        "sample_sha256": _sha256_json(payload),
        "preview": _preview(payload),
    }
    return {key: value for key, value in row.items() if value is not None}


def _source_value(source: Mapping[str, Any] | None, config: Any, key: str, *, config_attr: str | None = None) -> Any:
    if source and key in source:
        return source[key]
    return getattr(config, config_attr or key, None)


def _rows_identity_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    identity_rows = [
        {
            "sample_order": row.get("sample_order"),
            "sample_index": row.get("sample_index"),
            "source_sample_index": row.get("source_sample_index"),
            "task_id": row.get("task_id"),
            "sample_sha256": row.get("sample_sha256"),
        }
        for row in rows
    ]
    return _sha256_json(identity_rows)


def _source_sample_index(payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> Any:
    for key in ("original_sample_index", "dataset_sample_index", "source_index"):
        if key in metadata:
            return metadata[key]
    return payload.get("sample_index")


def _task_id(payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> str | None:
    for key in _IDENTITY_KEYS:
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    for key in ("source_id", "task_id", "id", "instance_id"):
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _metadata_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _json_safe(metadata[key])
        for key in _METADATA_IDENTITY_KEYS
        if key in metadata and metadata[key] is not None
    }


def _preview(payload: Mapping[str, Any]) -> dict[str, str]:
    previews: dict[str, str] = {}
    for key in _PREVIEW_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            previews[key] = _short_text(value)
    return previews


def _short_text(value: str, *, limit: int = 240) -> str:
    text = " ".join(value.strip().split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _sha256_json(value: Any) -> str:
    data = json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
