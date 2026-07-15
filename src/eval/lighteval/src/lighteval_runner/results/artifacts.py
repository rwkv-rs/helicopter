from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from ..execution import RunStatus, SampleAccounting


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    schema_version: int
    run_id: str
    status: RunStatus
    identity_digest: str
    identities: Mapping[str, Any]
    accounting: dict[str, int]
    artifacts: tuple[ArtifactEntry, ...]
    completed_at: str | None


class RunArtifacts:
    def __init__(self, output_root: Path, *, run_id: str | None = None) -> None:
        self.run_id = uuid4().hex if run_id is None else run_id
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.run_id):
            raise ValueError("run_id must be a normalized path-safe identifier")
        self.run_dir = output_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._entries: list[ArtifactEntry] = []
        self._relative_paths: set[str] = set()
        self._finalized = False

    @property
    def finalized(self) -> bool:
        return self._finalized

    def write_json(self, relative_path: str, payload: Any) -> ArtifactEntry:
        if self._finalized:
            raise RuntimeError("run artifacts are already finalized")
        if relative_path == "manifest.json":
            raise ValueError("manifest.json is reserved for finalization")
        if relative_path in self._relative_paths:
            raise FileExistsError(
                f"artifact already exists in this run: {relative_path}"
            )
        path = self._resolve_relative(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode()
        self._atomic_write(path, encoded)
        entry = ArtifactEntry(
            relative_path=relative_path,
            sha256=hashlib.sha256(encoded).hexdigest(),
            size_bytes=len(encoded),
        )
        self._entries.append(entry)
        self._relative_paths.add(relative_path)
        return entry

    def finalize(
        self,
        *,
        status: RunStatus,
        identity_digest: str,
        identities: Mapping[str, Any],
        accounting: SampleAccounting,
    ) -> Path:
        if self._finalized:
            raise RuntimeError("run artifacts are already finalized")
        if status not in {
            RunStatus.COMPLETED,
            RunStatus.PARTIAL,
            RunStatus.FAILED,
            RunStatus.INVALID,
            RunStatus.CANCELLED,
        }:
            raise ValueError("artifact manifest requires a terminal run status")
        if not re.fullmatch(r"[0-9a-f]{64}", identity_digest):
            raise ValueError("identity_digest must be a lowercase SHA-256 digest")
        if "run" not in identities:
            raise ValueError(
                "artifact identities must contain the canonical run identity"
            )
        canonical_identity = json.dumps(
            identities["run"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        if hashlib.sha256(canonical_identity).hexdigest() != identity_digest:
            raise ValueError(
                "identity_digest does not match the canonical run identity"
            )
        accounting.validate()
        manifest = ArtifactManifest(
            schema_version=2,
            run_id=self.run_id,
            status=status,
            identity_digest=identity_digest,
            identities=dict(identities),
            accounting=asdict(accounting),
            artifacts=tuple(self._entries),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        path = self.run_dir / "manifest.json"
        encoded = (
            json.dumps(asdict(manifest), sort_keys=True, ensure_ascii=False) + "\n"
        ).encode()
        self._atomic_write(path, encoded)
        self._finalized = True
        return path

    @staticmethod
    def _atomic_write(path: Path, encoded: bytes) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def _resolve_relative(self, relative_path: str) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or relative.name == "":
            raise ValueError("artifact path must be normalized and relative")
        return self.run_dir / relative


def verify_manifest(manifest_path: Path) -> ArtifactManifest:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        manifest = ArtifactManifest(
            schema_version=int(payload["schema_version"]),
            run_id=str(payload["run_id"]),
            status=RunStatus(payload["status"]),
            identity_digest=str(payload["identity_digest"]),
            identities=payload["identities"],
            accounting=payload["accounting"],
            artifacts=tuple(ArtifactEntry(**entry) for entry in payload["artifacts"]),
            completed_at=(
                str(payload["completed_at"])
                if payload.get("completed_at") is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("artifact manifest schema is invalid") from error
    if (
        manifest.schema_version not in {1, 2}
        or manifest_path.parent.name != manifest.run_id
    ):
        raise ValueError("artifact manifest run identity is invalid")
    if manifest.schema_version == 2:
        if manifest.completed_at is None:
            raise ValueError("artifact manifest completed_at is missing")
        try:
            completed_at = datetime.fromisoformat(manifest.completed_at)
        except ValueError as error:
            raise ValueError("artifact manifest completed_at is invalid") from error
        if completed_at.tzinfo is None:
            raise ValueError("artifact manifest completed_at must include timezone")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", manifest.run_id):
        raise ValueError("artifact manifest run id is unsafe")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest.identity_digest):
        raise ValueError("artifact manifest identity digest is invalid")
    try:
        accounting = SampleAccounting(**manifest.accounting)
        accounting.validate()
        run_identity = manifest.identities["run"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "artifact manifest accounting or identity is invalid"
        ) from error
    canonical_identity = json.dumps(
        run_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    if hashlib.sha256(canonical_identity).hexdigest() != manifest.identity_digest:
        raise ValueError(
            "artifact manifest identity digest does not match identity payload"
        )
    relative_paths = [entry.relative_path for entry in manifest.artifacts]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("artifact manifest contains duplicate artifact paths")
    for entry in manifest.artifacts:
        relative = Path(entry.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact manifest contains an unsafe path")
        artifact = manifest_path.parent / relative
        encoded = artifact.read_bytes()
        if (
            len(encoded) != entry.size_bytes
            or hashlib.sha256(encoded).hexdigest() != entry.sha256
        ):
            raise ValueError(f"artifact checksum mismatch: {entry.relative_path}")
    return manifest


def record_publication_attempt(manifest_path: Path, attempt: Mapping[str, Any]) -> Path:
    manifest = verify_manifest(manifest_path)
    retry_identity = attempt.get("retry_identity")
    if not isinstance(retry_identity, str) or not retry_identity:
        raise ValueError("publication attempt requires a retry identity")
    path = manifest_path.parent / "publication.json"
    history: list[dict[str, Any]] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("run_id") != manifest.run_id:
                raise ValueError("publication receipt belongs to a different run")
            history = existing["attempts"]
            if not isinstance(history, list):
                raise TypeError
            if history and history[-1].get("retry_identity") != retry_identity:
                raise ValueError("publication retry identity changed across attempts")
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
        ) as error:
            raise ValueError("publication receipt is invalid") from error
    history.append(dict(attempt))
    encoded = (
        json.dumps(
            {"run_id": manifest.run_id, "attempts": history},
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode()
    RunArtifacts._atomic_write(path, encoded)
    return path
