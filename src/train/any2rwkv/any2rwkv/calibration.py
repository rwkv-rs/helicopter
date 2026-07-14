from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .errors import ContractError


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CalibrationManifest:
    path: Path
    data_file: Path
    data_sha256: str
    row_count: int
    text_field: str
    max_length: int
    batch_size: int
    split: str

    def texts(self) -> tuple[str, ...]:
        rows: list[str] = []
        with self.data_file.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                    value = row[self.text_field]
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise ContractError(
                        f"invalid calibration row {line_number}: {error}"
                    ) from error
                if not isinstance(value, str) or not value.strip():
                    raise ContractError(
                        f"calibration row {line_number} has empty text"
                    )
                rows.append(value)
        if len(rows) != self.row_count:
            raise ContractError(
                f"calibration row count mismatch: expected {self.row_count}, "
                f"found {len(rows)}"
            )
        return tuple(rows)


def read_calibration_manifest(path: Path) -> CalibrationManifest:
    path = path.resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read calibration manifest {path}: {error}") from error
    if payload.get("schema_version") != 1:
        raise ContractError("calibration manifest schema_version must be 1")
    if payload.get("split") != "nvfp4_calibration":
        raise ContractError("calibration manifest must use split=nvfp4_calibration")
    data_file = Path(str(payload.get("data_file", "")))
    if not data_file.is_absolute():
        data_file = (path.parent / data_file).resolve()
    if not data_file.is_file():
        raise ContractError(f"calibration data file not found: {data_file}")
    expected_sha = str(payload.get("sha256", ""))
    actual_sha = file_sha256(data_file)
    if len(expected_sha) != 64 or actual_sha != expected_sha:
        raise ContractError(
            f"calibration SHA-256 mismatch: expected {expected_sha}, found {actual_sha}"
        )
    row_count = int(payload.get("row_count", 0))
    max_length = int(payload.get("max_length", 0))
    batch_size = int(payload.get("batch_size", 0))
    if row_count <= 0 or max_length <= 0 or batch_size <= 0:
        raise ContractError(
            "calibration row_count, max_length, and batch_size must be positive"
        )
    manifest = CalibrationManifest(
        path,
        data_file,
        actual_sha,
        row_count,
        str(payload.get("text_field", "text")),
        max_length,
        batch_size,
        "nvfp4_calibration",
    )
    manifest.texts()
    return manifest

