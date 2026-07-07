from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


INTERMEDIATE_FORMAT = "helicopter_agent_v1"

PATCH_KEY_CANDIDATES = (
    "model_patch",
    "patch",
    "output_patch",
    "prediction_patch",
)
TEXT_KEY_CANDIDATES = (
    "output",
    "completion",
    "text",
    "raw_output",
    "model_output",
    "assistant_output",
)
ID_KEY_CANDIDATES = (
    "instance_id",
    "sample_id",
    "task_id",
    "id",
)
NESTED_ID_CANDIDATES = (
    "metadata",
    "specific",
    "doc_specific",
    "sample",
)


@dataclass(frozen=True)
class ConversionError:
    row_index: int
    reason: str
    sample_id: str | None = None


def _coerce_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_coerce_text(item) for item in value if item is not None)
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def read_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] in "[{":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, Mapping)]
        if isinstance(payload, Mapping):
            if isinstance(payload.get("records"), list):
                return [dict(item) for item in payload["records"] if isinstance(item, Mapping)]
            if isinstance(payload.get("details"), str):
                details_path = Path(str(payload["details"]))
                if not details_path.is_absolute():
                    details_path = path.parent / details_path
                if details_path.exists():
                    return read_json_records(details_path)
            if isinstance(payload.get("details"), list):
                return [dict(item) for item in payload["details"] if isinstance(item, Mapping)]
            if isinstance(payload.get("samples"), list):
                return [dict(item) for item in payload["samples"] if isinstance(item, Mapping)]
            return [dict(payload)]
    rows = []
    for line in text.splitlines():
        value = line.strip()
        if not value:
            continue
        payload = json.loads(value)
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _nested_value(record: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    for parent_key in NESTED_ID_CANDIDATES:
        parent = _coerce_mapping(record.get(parent_key))
        for key in keys:
            value = parent.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


def record_sample_id(record: Mapping[str, Any]) -> str:
    return _nested_value(record, ID_KEY_CANDIDATES).strip()


def record_model_name(record: Mapping[str, Any], default: str | None = None) -> str:
    model = _nested_value(record, ("model_name_or_path", "model", "served_model_name")).strip()
    return model or str(default or "")


def response_text(record: Mapping[str, Any]) -> str:
    for key in PATCH_KEY_CANDIDATES:
        value = record.get(key)
        if value not in (None, ""):
            return _coerce_text(value)
    for key in TEXT_KEY_CANDIDATES:
        value = record.get(key)
        if value not in (None, ""):
            return _coerce_text(value)

    response_message = _coerce_mapping(record.get("response_message"))
    if response_message.get("content") not in (None, ""):
        return _coerce_text(response_message.get("content"))
    message = _coerce_mapping(record.get("message"))
    if message.get("content") not in (None, ""):
        return _coerce_text(message.get("content"))

    response = _coerce_mapping(record.get("response") or record.get("raw_response"))
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = _coerce_mapping(choices[0])
        choice_message = _coerce_mapping(first.get("message"))
        if choice_message.get("content") not in (None, ""):
            return _coerce_text(choice_message.get("content"))
        if first.get("text") not in (None, ""):
            return _coerce_text(first.get("text"))
    return ""


def _strip_patch_tags(text: str) -> str:
    match = re.search(r"<patch>\s*(.*?)\s*</patch>", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def _strip_fenced_block(text: str) -> str:
    matches = re.findall(r"```(?:diff|patch)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for value in matches:
        if _looks_like_patch(value):
            return value.strip()
    return text


def _looks_like_patch(text: str) -> bool:
    return bool(
        re.search(r"(?m)^diff --git ", text)
        or (re.search(r"(?m)^--- ", text) and re.search(r"(?m)^\+\+\+ ", text))
        or re.search(r"(?m)^@@ ", text)
    )


def extract_unified_diff(text: str) -> str:
    value = _strip_fenced_block(_strip_patch_tags(text.strip())).strip()
    if not value:
        return ""
    lines = value.splitlines()
    start_index = None
    for index, line in enumerate(lines):
        if line.startswith("diff --git ") or line.startswith("--- ") or line.startswith("@@ "):
            start_index = index
            break
    if start_index is not None:
        value = "\n".join(lines[start_index:]).strip()
    if not _looks_like_patch(value):
        return ""
    return value + "\n"


def canonical_intermediate_rows(
    records: list[Mapping[str, Any]],
    *,
    benchmark: str,
    model: str | None = None,
) -> tuple[list[dict[str, Any]], list[ConversionError]]:
    rows: list[dict[str, Any]] = []
    errors: list[ConversionError] = []
    for index, record in enumerate(records, start=1):
        sample_id = record_sample_id(record)
        if not sample_id:
            errors.append(ConversionError(index, "missing sample id"))
            continue
        content = response_text(record)
        rows.append(
            {
                "format": INTERMEDIATE_FORMAT,
                "benchmark": benchmark,
                "sample_id": sample_id,
                "model": record_model_name(record, model),
                "content": content,
                "tool_calls": record.get("tool_calls") or record.get("actual_calls") or [],
                "artifacts": {
                    "patch": extract_unified_diff(content),
                },
                "metadata": {
                    key: value
                    for key, value in record.items()
                    if key not in {"response", "raw_response", "message", "response_message"}
                },
            }
        )
    return rows, errors


def swebench_prediction_rows(
    records: list[Mapping[str, Any]],
    *,
    model: str,
    allow_empty_patch: bool = False,
) -> tuple[list[dict[str, str]], list[ConversionError]]:
    rows: list[dict[str, str]] = []
    errors: list[ConversionError] = []
    for index, record in enumerate(records, start=1):
        instance_id = record_sample_id(record)
        if not instance_id:
            errors.append(ConversionError(index, "missing instance_id/sample_id"))
            continue
        patch = extract_unified_diff(response_text(record))
        if not patch and not allow_empty_patch:
            errors.append(ConversionError(index, "missing unified diff patch", sample_id=instance_id))
            continue
        rows.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": record_model_name(record, model) or model,
                "model_patch": patch,
            }
        )
    return rows, errors


def conversion_errors_text(errors: list[ConversionError]) -> str:
    return "\n".join(
        f"row {error.row_index}: {error.reason}"
        + (f" ({error.sample_id})" if error.sample_id else "")
        for error in errors
    )
