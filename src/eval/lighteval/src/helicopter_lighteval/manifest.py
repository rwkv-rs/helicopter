from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any
import uuid


MANIFEST_VERSION = 2
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(RuntimeError):
    pass


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def ensure_private_staging_root(staging_root: Path) -> Path:
    try:
        if staging_root.is_symlink():
            raise ManifestError("evaluation staging root must not be a symlink")
        created = False
        if not staging_root.exists():
            existing_parent = staging_root.parent
            while not existing_parent.exists():
                if existing_parent.parent == existing_parent:
                    raise ManifestError(
                        "evaluation staging root has no existing parent"
                    )
                existing_parent = existing_parent.parent
            parent_status = existing_parent.stat()
            if (
                not stat.S_ISDIR(parent_status.st_mode)
                or parent_status.st_uid != os.geteuid()
                or stat.S_IMODE(parent_status.st_mode) & 0o022
            ):
                raise ManifestError(
                    "evaluation staging root requires a current-user-owned "
                    "parent that is not group/world writable"
                )
            try:
                staging_root.mkdir(parents=True, exist_ok=False, mode=0o700)
                created = True
            except FileExistsError:
                pass
        staging_status = staging_root.lstat()
        if not stat.S_ISDIR(staging_status.st_mode) or stat.S_ISLNK(
            staging_status.st_mode
        ):
            raise ManifestError("evaluation staging root must be a directory")
        if created:
            staging_root.chmod(0o700)
            staging_status = staging_root.lstat()
        if (
            staging_status.st_uid != os.geteuid()
            or stat.S_IMODE(staging_status.st_mode) != 0o700
        ):
            raise ManifestError(
                "existing evaluation staging root must be owned by the "
                "current user and have mode 0700"
            )
        return staging_root.resolve()
    except OSError as error:
        raise ManifestError(
            f"cannot secure evaluation staging root: {error}"
        ) from error


def _reject_symlink_components(root: Path, candidate: Path) -> None:
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ManifestError("path is not a lexical child of staging root") from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ManifestError(f"staging child must not be a symlink: {current}")


def _write_json_atomic(path: Path, value: object) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = -1
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


@dataclass
class CampaignManifest:
    version: int
    resume_key: str
    campaign_id: str
    config_digest: str
    registry_digest: str
    eval_contract_digest: str
    weight_sha256: list[str]
    configured_selectors: list[str]
    resolved_selectors: list[str]
    skipped_selectors: list[str]
    registry_task_identities: list[str]
    shard_paths: dict[str, str] = field(default_factory=dict)
    attempted_shard_paths: dict[str, str] = field(default_factory=dict)
    failed_shard_paths: dict[str, list[str]] = field(default_factory=dict)
    failed_shard_errors: dict[str, str] = field(default_factory=dict)
    runtime_paths: dict[str, str] = field(default_factory=dict)
    model_executions: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_task_digests: dict[str, str] = field(default_factory=dict)
    acknowledged_task_digests: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "CampaignManifest":
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        if set(raw) != expected:
            raise ManifestError("campaign manifest fields do not match schema")
        manifest = cls(**raw)
        if isinstance(manifest.version, bool) or manifest.version != MANIFEST_VERSION:
            raise ManifestError("unsupported campaign manifest version")
        for name in (
            "resume_key",
            "config_digest",
            "registry_digest",
            "eval_contract_digest",
        ):
            value = getattr(manifest, name)
            if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
                raise ManifestError(f"campaign manifest has invalid {name}")
        if (
            not isinstance(manifest.weight_sha256, list)
            or not manifest.weight_sha256
            or any(
                not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None
                for digest in manifest.weight_sha256
            )
            or len(manifest.weight_sha256) != len(set(manifest.weight_sha256))
        ):
            raise ManifestError("campaign manifest has invalid weight_sha256")
        if (
            not isinstance(manifest.registry_task_identities, list)
            or not manifest.registry_task_identities
            or any(
                not isinstance(identity, str) or not identity
                for identity in manifest.registry_task_identities
            )
            or len(manifest.registry_task_identities)
            != len(set(manifest.registry_task_identities))
        ):
            raise ManifestError(
                "campaign manifest has invalid registry_task_identities"
            )
        for name in (
            "configured_selectors",
            "resolved_selectors",
            "skipped_selectors",
        ):
            values = getattr(manifest, name)
            if (
                not isinstance(values, list)
                or any(
                    not isinstance(value, str) or not value or value != value.strip()
                    for value in values
                )
                or len(values) != len(set(values))
            ):
                raise ManifestError(f"campaign manifest has invalid {name}")
        if not manifest.configured_selectors or not manifest.resolved_selectors:
            raise ManifestError("campaign manifest has no configured benchmarks")
        resolved = set(manifest.resolved_selectors)
        skipped = set(manifest.skipped_selectors)
        if not resolved.isdisjoint(skipped) or resolved | skipped != set(
            manifest.configured_selectors
        ):
            raise ManifestError(
                "campaign manifest selector status does not match configuration"
            )
        try:
            campaign_id = str(uuid.UUID(manifest.campaign_id))
        except (AttributeError, TypeError, ValueError) as error:
            raise ManifestError(
                "campaign manifest has an invalid campaign id"
            ) from error
        if campaign_id != manifest.campaign_id:
            raise ManifestError("campaign manifest campaign id is not canonical")
        _validate_path_map("shard_paths", manifest.shard_paths)
        _validate_path_map(
            "attempted_shard_paths",
            manifest.attempted_shard_paths,
        )
        _validate_path_map("runtime_paths", manifest.runtime_paths)
        if not isinstance(manifest.failed_shard_paths, dict):
            raise ManifestError(
                "campaign manifest failed_shard_paths must be an object"
            )
        for key, values in manifest.failed_shard_paths.items():
            if not isinstance(key, str) or not isinstance(values, list):
                raise ManifestError("campaign manifest failed_shard_paths is invalid")
            _validate_relative_paths("failed_shard_paths", values)
        if (
            not isinstance(manifest.failed_shard_errors, dict)
            or not all(
                isinstance(key, str)
                and key
                and isinstance(value, str)
                and value
                and value == value.strip()
                for key, value in manifest.failed_shard_errors.items()
            )
            or set(manifest.failed_shard_errors) != set(manifest.failed_shard_paths)
        ):
            raise ManifestError("campaign manifest failed_shard_errors is invalid")
        if not isinstance(manifest.model_executions, dict) or not all(
            isinstance(key, str) and isinstance(value, dict)
            for key, value in manifest.model_executions.items()
        ):
            raise ManifestError("campaign manifest model_executions is invalid")
        for name in (
            "pending_task_digests",
            "acknowledged_task_digests",
        ):
            value = getattr(manifest, name)
            if not isinstance(value, dict) or not all(
                isinstance(key, str)
                and isinstance(digest, str)
                and _DIGEST.fullmatch(digest) is not None
                for key, digest in value.items()
            ):
                raise ManifestError(f"campaign manifest {name} is invalid")
        if set(manifest.pending_task_digests) & set(manifest.acknowledged_task_digests):
            raise ManifestError("campaign manifest task digest states overlap")
        return manifest


def _validate_relative_paths(name: str, values: list[object]) -> None:
    for value in values:
        if not isinstance(value, str):
            raise ManifestError(f"campaign manifest {name} is invalid")
        path = Path(value)
        if (
            not path.parts
            or path.is_absolute()
            or "." in path.parts
            or ".." in path.parts
            or path.as_posix() != value
        ):
            raise ManifestError(f"campaign manifest {name} contains an unsafe path")


def _validate_path_map(name: str, value: object) -> None:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ManifestError(f"campaign manifest {name} must be an object")
    _validate_relative_paths(name, list(value.values()))


class ManifestStore:
    def __init__(self, staging_root: Path, resume_key: str) -> None:
        self.staging_root = staging_root.resolve()
        self.manifest_root = self.staging_root / "campaigns"
        self.path = self.manifest_root / f"{resume_key}.json"

    def load(self) -> CampaignManifest | None:
        _reject_symlink_components(self.staging_root, self.manifest_root)
        if not self.path.exists():
            return None
        if self.path.is_symlink():
            raise ManifestError("campaign manifest must not be a symlink")
        try:
            raw = json.loads(
                self.path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (OSError, ValueError) as error:
            raise ManifestError(f"cannot read campaign manifest: {error}") from error
        if not isinstance(raw, dict):
            raise ManifestError("campaign manifest must be an object")
        return CampaignManifest.from_json(raw)

    def save(self, manifest: CampaignManifest) -> None:
        try:
            _reject_symlink_components(self.staging_root, self.manifest_root)
            self.manifest_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            _reject_symlink_components(self.staging_root, self.path)
            _write_json_atomic(self.path, asdict(manifest))
        except (OSError, TypeError, ValueError) as error:
            raise ManifestError(f"cannot persist campaign manifest: {error}") from error

    def delete(self) -> None:
        _reject_symlink_components(self.staging_root, self.manifest_root)
        _reject_symlink_components(self.staging_root, self.path)
        if self.path.exists():
            if self.path.is_symlink():
                raise ManifestError("refusing to delete symlink manifest")
            self.path.unlink()
        if self.manifest_root.exists() and not any(self.manifest_root.iterdir()):
            self.manifest_root.rmdir()

    def quarantine(self) -> Path:
        _reject_symlink_components(self.staging_root, self.manifest_root)
        _reject_symlink_components(self.staging_root, self.path)
        if not self.path.is_file() or self.path.is_symlink():
            raise ManifestError("only a regular campaign manifest can be quarantined")
        try:
            quarantine_root = self.manifest_root / "quarantine"
            _reject_symlink_components(self.staging_root, quarantine_root)
            quarantine_root.mkdir(mode=0o700, exist_ok=True)
            quarantine_root.chmod(0o700)
            destination = quarantine_root / f"{self.path.stem}.{uuid.uuid4().hex}.json"
            _reject_symlink_components(self.staging_root, destination)
            os.replace(self.path, destination)
        except OSError as error:
            raise ManifestError(
                f"cannot quarantine campaign manifest: {error}"
            ) from error
        return destination


def campaign_directory(staging_root: Path, campaign_id: str) -> Path:
    root = staging_root.resolve()
    unresolved = root / "runs" / campaign_id
    _reject_symlink_components(root, unresolved)
    directory = unresolved.resolve()
    try:
        directory.relative_to(root)
    except ValueError as error:
        raise ManifestError("campaign directory escapes staging root") from error
    if directory in {root, root / "runs"}:
        raise ManifestError("campaign directory must be a child of staging root")
    return directory


def validate_campaign_child(
    campaign_dir: Path,
    child: Path,
) -> tuple[Path, Path]:
    campaign = campaign_dir.resolve()
    unresolved = child if child.is_absolute() else campaign / child
    try:
        lexical_relative = unresolved.relative_to(campaign)
    except ValueError as error:
        raise ManifestError("path is not a campaign child") from error
    if (
        not lexical_relative.parts
        or "." in lexical_relative.parts
        or ".." in lexical_relative.parts
    ):
        raise ManifestError("campaign child path is not normalized")
    _reject_symlink_components(campaign, unresolved)
    candidate = unresolved.resolve()
    try:
        resolved_relative = candidate.relative_to(campaign)
    except ValueError as error:
        raise ManifestError("campaign child path escapes campaign") from error
    if resolved_relative != lexical_relative:
        raise ManifestError("campaign child path changed during resolution")
    return candidate, resolved_relative


def remove_campaign_child_directory(
    campaign_dir: Path,
    child: Path,
) -> None:
    campaign = campaign_dir.resolve()
    candidate, relative = validate_campaign_child(campaign, child)
    if not relative.parts or candidate == campaign:
        raise ManifestError("refusing to delete campaign directory as a child")
    if candidate.exists():
        if candidate.is_symlink() or not candidate.is_dir():
            raise ManifestError("campaign child is not a safe directory")
        if not shutil.rmtree.avoids_symlink_attacks:
            raise ManifestError(
                "platform cannot safely remove campaign directory trees"
            )
        shutil.rmtree(candidate)
    parent = candidate.parent
    while parent != campaign and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent


def create_campaign_directory(staging_root: Path, campaign_id: str) -> Path:
    root = ensure_private_staging_root(staging_root)
    try:
        runs = root / "runs"
        _reject_symlink_components(root, runs)
        runs.mkdir(mode=0o700, exist_ok=True)
        runs.chmod(0o700)
        directory = campaign_directory(root, campaign_id)
        directory.mkdir(mode=0o700, exist_ok=True)
        directory.chmod(0o700)
        return directory
    except OSError as error:
        raise ManifestError(
            f"cannot create private campaign directory: {error}"
        ) from error


def remove_acknowledged_shard(
    *,
    staging_root: Path,
    campaign_id: str,
    shard_path: str,
) -> None:
    campaign = campaign_directory(staging_root, campaign_id)
    relative_path = Path(shard_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ManifestError("shard path must be a normalized relative child")
    remove_campaign_child_directory(campaign, relative_path)


def remove_empty_campaign(staging_root: Path, campaign_id: str) -> None:
    directory = campaign_directory(staging_root, campaign_id)
    if directory.exists():
        if directory.is_symlink() or not directory.is_dir():
            raise ManifestError("invalid campaign staging directory")
        if any(directory.iterdir()):
            raise ManifestError("campaign staging directory is not empty")
        directory.rmdir()
    runs = staging_root.resolve() / "runs"
    _reject_symlink_components(staging_root.resolve(), runs)
    if runs.exists() and not any(runs.iterdir()):
        runs.rmdir()


def write_control_metadata(
    *,
    staging_root: Path,
    campaign_id: str,
    metadata: dict[str, Any],
) -> Path:
    root = staging_root.resolve()
    unresolved_control_root = root / "control"
    _reject_symlink_components(root, unresolved_control_root)
    control_root = unresolved_control_root.resolve()
    control_root.relative_to(root)
    control_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    unresolved_path = control_root / f"{campaign_id}.json"
    _reject_symlink_components(root, unresolved_path)
    path = unresolved_path.resolve()
    try:
        path.relative_to(control_root)
    except ValueError as error:
        raise ManifestError("control metadata path escapes control root") from error
    if path.is_symlink():
        raise ManifestError("refusing to overwrite symlink control metadata")
    _write_json_atomic(path, metadata)
    return path
