from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping


_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class DatasetSource:
    repository: str
    revision: str
    source_file: str
    sha256: str

    def validate(self) -> None:
        if not _COMMIT_SHA.fullmatch(self.revision) and not re.fullmatch(
            r"bundled:[a-z0-9][a-z0-9._-]*", self.revision
        ):
            raise ValueError(
                "dataset revision must be an immutable commit SHA or bundled resource revision"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("dataset sha256 must be a 64-character lowercase digest")


@dataclass(frozen=True, slots=True)
class SnapshotRow:
    row_id: str
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RejectedRow:
    row_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    source: DatasetSource
    file_path: Path
    accepted_rows: tuple[SnapshotRow, ...]
    rejected_rows: tuple[RejectedRow, ...]

    @property
    def source_rows(self) -> int:
        return len(self.accepted_rows) + len(self.rejected_rows)


def verify_asset_manifest(
    manifest_path: Path,
    snapshot_path: Path,
    source: DatasetSource,
    *,
    asset_name: str,
) -> str:
    """Bind local bytes to the helicopter-dev immutable acquisition contract."""

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("dataset asset manifest is missing or invalid") from error
    expected_source = {
        "repo": source.repository,
        "revision": source.revision,
        "file": source.source_file,
    }
    if payload.get("name") != asset_name or payload.get("source") != expected_source:
        raise ValueError("dataset asset manifest source contract mismatch")
    files = payload.get("files")
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        raise ValueError("dataset asset manifest must describe exactly one file")
    entry = files[0]
    try:
        recorded_path = Path(entry["path"]).resolve(strict=True)
    except (KeyError, OSError, TypeError) as error:
        raise ValueError("dataset asset manifest path is invalid") from error
    if recorded_path != snapshot_path.resolve(strict=True):
        raise ValueError("dataset asset manifest points to a different snapshot")
    if entry.get("sha256") != source.sha256:
        raise ValueError("dataset asset manifest digest contract mismatch")
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return manifest_digest


def _stable_row_id(payload: dict[str, Any]) -> str:
    explicit = (
        payload.get("id") or payload.get("sample_id") or payload.get("instance_id")
    )
    if explicit is not None and str(explicit):
        return str(explicit)
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return f"row-{hashlib.sha256(canonical.encode()).hexdigest()}"


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def materialize_jsonl_snapshot(
    path: Path,
    source: DatasetSource,
    *,
    validate_row: Callable[[dict[str, Any]], str | None],
) -> DatasetSnapshot:
    source.validate()
    verified_bytes = path.read_bytes()
    actual_digest = hashlib.sha256(verified_bytes).hexdigest()
    if actual_digest != source.sha256:
        raise ValueError(
            f"dataset digest mismatch: expected {source.sha256}, found {actual_digest}"
        )

    accepted: list[SnapshotRow] = []
    rejected: list[RejectedRow] = []
    seen_row_ids: set[str] = set()
    try:
        verified_text = verified_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("dataset snapshot must be valid UTF-8") from error
    for line_number, line in enumerate(verified_text.splitlines(), 1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            rejected.append(
                RejectedRow(f"line-{line_number}", f"invalid_json:{error.msg}")
            )
            continue
        if not isinstance(payload, dict):
            rejected.append(RejectedRow(f"line-{line_number}", "row_must_be_object"))
            continue
        row_id = _stable_row_id(payload)
        if row_id in seen_row_ids:
            raise ValueError(f"duplicate stable dataset row identity: {row_id}")
        seen_row_ids.add(row_id)
        reason = validate_row(payload)
        if reason is None:
            accepted.append(SnapshotRow(row_id=row_id, payload=_freeze(payload)))
        else:
            rejected.append(RejectedRow(row_id=row_id, reason=reason))
    return DatasetSnapshot(
        source=source,
        file_path=path.resolve(),
        accepted_rows=tuple(accepted),
        rejected_rows=tuple(rejected),
    )


def materialize_parquet_snapshot(
    path: Path,
    source: DatasetSource,
    *,
    validate_row: Callable[[dict[str, Any]], str | None],
) -> DatasetSnapshot:
    """Verify once, then parse exactly those bytes without a check/use race."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    source.validate()
    verified_bytes = path.read_bytes()
    actual_digest = hashlib.sha256(verified_bytes).hexdigest()
    if actual_digest != source.sha256:
        raise ValueError(
            f"dataset digest mismatch: expected {source.sha256}, found {actual_digest}"
        )
    try:
        rows = pq.read_table(pa.BufferReader(verified_bytes)).to_pylist()
    except (pa.ArrowException, OSError, ValueError) as error:
        raise ValueError("dataset snapshot is not valid Parquet") from error
    accepted: list[SnapshotRow] = []
    rejected: list[RejectedRow] = []
    seen_row_ids: set[str] = set()
    for index, payload in enumerate(rows):
        if not isinstance(payload, dict):
            rejected.append(RejectedRow(f"row-{index}", "row_must_be_object"))
            continue
        row_id = _stable_row_id(payload)
        if row_id in seen_row_ids:
            raise ValueError(f"duplicate stable dataset row identity: {row_id}")
        seen_row_ids.add(row_id)
        reason = validate_row(payload)
        if reason is None:
            accepted.append(SnapshotRow(row_id=row_id, payload=_freeze(payload)))
        else:
            rejected.append(RejectedRow(row_id=row_id, reason=reason))
    return DatasetSnapshot(
        source=source,
        file_path=path.resolve(),
        accepted_rows=tuple(accepted),
        rejected_rows=tuple(rejected),
    )


def materialize_snapshot(
    path: Path,
    source: DatasetSource,
    *,
    validate_row: Callable[[dict[str, Any]], str | None],
) -> DatasetSnapshot:
    if path.suffix == ".jsonl":
        return materialize_jsonl_snapshot(path, source, validate_row=validate_row)
    if path.suffix == ".parquet":
        return materialize_parquet_snapshot(path, source, validate_row=validate_row)
    raise ValueError("dataset snapshot must be .jsonl or .parquet")


def select_snapshot_rows(
    snapshot: DatasetSnapshot, max_samples: int | None
) -> tuple[SnapshotRow, ...]:
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    ordered = tuple(sorted(snapshot.accepted_rows, key=lambda row: row.row_id))
    return ordered if max_samples is None else ordered[:max_samples]
