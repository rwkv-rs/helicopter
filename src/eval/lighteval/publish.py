from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .config import (
    EvaluationConfigurationError,
    EvaluationEnvironment,
    EvaluationShard,
    EvaluationUnit,
    PROMPT_TEMPLATE_STOPS,
    RegistryTask,
    repository_root,
)
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


MAX_RESPONSE_BYTES = 8 * 1024 * 1024
LIGHTEVAL_VERSION = "0.13.0"


class ScoreboardError(RuntimeError):
    pass


class ScoreboardConflict(ScoreboardError):
    pass


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, new_url):
        return None


class ScoreboardClient:
    def __init__(self, environment: EvaluationEnvironment) -> None:
        self.base_url = environment.scoreboard_url.rstrip("/")
        self._token = environment.scoreboard_token
        self._opener = build_opener(_RejectRedirects())

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        try:
            body = None
            headers = {
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/json",
            }
            if payload is not None:
                raw = json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                body = gzip.compress(raw)
                headers.update(
                    {
                        "Content-Type": "application/json",
                        "Content-Encoding": "gzip",
                    }
                )
            if idempotency_key is not None:
                headers["Idempotency-Key"] = idempotency_key
            request = Request(
                f"{self.base_url}{path}",
                data=body,
                method=method,
                headers=headers,
            )
            with self._opener.open(request, timeout=60) as response:
                content_type = response.headers.get_content_type()
                if content_type != "application/json":
                    raise ScoreboardError(
                        "Scoreboard response Content-Type must be application/json"
                    )
                raw_response = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw_response) > MAX_RESPONSE_BYTES:
                    raise ScoreboardError("Scoreboard response exceeds size limit")
                decoded = json.loads(
                    raw_response,
                    parse_constant=_reject_json_constant,
                )
        except HTTPError as error:
            raw_detail = error.read(MAX_RESPONSE_BYTES + 1)
            if len(raw_detail) > MAX_RESPONSE_BYTES:
                raise ScoreboardError(
                    f"Scoreboard HTTP {error.code} response exceeds size limit"
                ) from error
            if error.code == 409:
                raise ScoreboardConflict(
                    f"Scoreboard HTTP {error.code} conflict"
                ) from error
            raise ScoreboardError(f"Scoreboard HTTP {error.code}") from error
        except ScoreboardError:
            raise
        except (
            OSError,
            TimeoutError,
            TypeError,
            UnicodeDecodeError,
            ValueError,
        ) as error:
            raise ScoreboardError(
                f"Scoreboard request failed: "
                f"{type(error).__module__}.{type(error).__qualname__}"
            ) from error
        if not isinstance(decoded, dict):
            raise ScoreboardError("Scoreboard response must be a JSON object")
        return decoded

    def create_campaign(
        self, payload: dict[str, Any], resume_key: str
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/evaluation-campaigns",
            payload=payload,
            idempotency_key=f"campaign:{resume_key}",
        )

    def preflight(self) -> dict[str, Any]:
        response = self._request(
            "GET",
            "/api/v1/evaluation-publication-preflight",
        )
        if (
            response.get("status") != "ready"
            or response.get("schema_version") != "lighteval-campaign-v2"
            or response.get("lighteval_version") != "0.13.0"
            or not isinstance(response.get("publisher_principal"), str)
            or not response["publisher_principal"]
        ):
            raise ScoreboardError("Scoreboard publication preflight is incompatible")
        return response

    def campaign_status(self, campaign_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/api/v1/evaluation-campaigns/{quote(campaign_id, safe='')}",
        )

    def publish_task(
        self,
        *,
        campaign_id: str,
        task_identity: str,
        payload: dict[str, Any],
        digest: str,
    ) -> dict[str, Any]:
        return self._request(
            "PUT",
            (
                f"/api/v1/evaluation-campaigns/{quote(campaign_id, safe='')}"
                f"/tasks/{quote(task_identity, safe='')}"
            ),
            payload=payload,
            idempotency_key=f"publish:{digest}",
        )

    def finalize(self, campaign_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            (f"/api/v1/evaluation-campaigns/{quote(campaign_id, safe='')}/finalize"),
            idempotency_key=f"finalize:{campaign_id}",
        )


def _direct_url(distribution: importlib.metadata.Distribution) -> dict[str, Any]:
    raw = distribution.read_text("direct_url.json")
    if raw is None:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise EvaluationConfigurationError(
            f"invalid direct_url.json for {distribution.metadata['Name']}"
        ) from error
    return value if isinstance(value, dict) else {}


def _dependency_source(name: str) -> dict[str, object]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise EvaluationConfigurationError(
            f"required eval dependency is not installed: {name}"
        ) from error
    return {
        "version": distribution.version,
        "direct_url": _direct_url(distribution),
    }


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise EvaluationConfigurationError(
                f"staging root has no existing parent: {path}"
            )
        candidate = candidate.parent
    return candidate


def run_preflight(environment: EvaluationEnvironment) -> dict[str, object]:
    lighteval = _dependency_source("lighteval")
    if lighteval["version"] != LIGHTEVAL_VERSION:
        raise EvaluationConfigurationError(
            f"lighteval must be exactly {LIGHTEVAL_VERSION}; "
            f"found {lighteval['version']}"
        )
    if lighteval["direct_url"]:
        raise EvaluationConfigurationError(
            "lighteval must be the locked registry release, not a direct "
            "URL or editable source"
        )

    vllm = _dependency_source("vllm")
    direct_url = vllm["direct_url"]
    dir_info = direct_url.get("dir_info") if isinstance(direct_url, dict) else None
    editable = isinstance(dir_info, dict) and dir_info.get("editable") is True
    expected = (repository_root() / "src" / "infer" / "vllm-rwkv").resolve()
    source_url = direct_url.get("url") if isinstance(direct_url, dict) else None
    if not isinstance(source_url, str) or not source_url.startswith("file://"):
        raise EvaluationConfigurationError(
            "vLLM must be installed editable from the repository submodule"
        )
    parsed_source = urlsplit(source_url)
    if parsed_source.scheme != "file" or parsed_source.netloc not in {"", "localhost"}:
        raise EvaluationConfigurationError(
            "vLLM editable source must be a local file URL"
        )
    actual = Path(unquote(parsed_source.path)).resolve()
    if not editable or actual != expected:
        raise EvaluationConfigurationError(
            f"vLLM editable source mismatch; expected {expected}, found {actual}"
        )

    existing_parent = _nearest_existing_parent(environment.staging_root)
    if not existing_parent.is_dir():
        raise EvaluationConfigurationError(
            f"staging root parent is not a directory: {existing_parent}"
        )
    parent_status = existing_parent.stat()
    if (
        parent_status.st_uid != os.geteuid()
        or stat.S_IMODE(parent_status.st_mode) & 0o022
    ):
        raise EvaluationConfigurationError(
            "staging root requires a current-user-owned existing parent "
            "that is not group/world writable"
        )
    if not os.access(existing_parent, os.W_OK | os.X_OK):
        raise EvaluationConfigurationError(
            f"staging root parent is not writable: {existing_parent}"
        )
    backend = ScoreboardClient(environment).preflight()
    return {
        "dependencies": {
            "lighteval": lighteval,
            "vllm": vllm,
        },
        "staging": {
            "root": str(environment.staging_root),
            "nearest_existing_parent": str(existing_parent),
            "ready": True,
        },
        "scoreboard": {
            "url": environment.scoreboard_url,
            "status": backend["status"],
            "schema_version": backend["schema_version"],
            "lighteval_version": backend["lighteval_version"],
            "publisher_principal": backend["publisher_principal"],
        },
    }


class ArtifactError(RuntimeError):
    pass


def canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    except (TypeError, ValueError) as error:
        raise ArtifactError(
            "standard artifacts contain non-canonical JSON data"
        ) from error


def content_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _validate_standard_file(shard_dir: Path, path: Path) -> Path:
    try:
        candidate, _ = validate_campaign_child(shard_dir, path)
    except ManifestError as error:
        raise ArtifactError(
            "standard artifact path is not a safe shard child"
        ) from error
    if path.is_symlink() or not candidate.is_file():
        raise ArtifactError("standard artifacts must be regular non-symlink files")
    return candidate


def _standard_artifacts(
    shard_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path, list[Path]]:
    import pyarrow.parquet as parquet

    result_files = sorted(shard_dir.glob("results/**/results_*.json"))
    if len(result_files) != 1:
        raise ArtifactError("expected exactly one standard results JSON per shard")
    result_file = _validate_standard_file(shard_dir, result_files[0])
    stamp = result_file.stem.removeprefix("results_")
    model_dir = result_file.parent.relative_to(shard_dir / "results")
    detail_files = sorted(
        (shard_dir / "details" / model_dir / stamp).glob(f"details_*_{stamp}.parquet")
    )
    if not detail_files:
        raise ArtifactError("expected standard details parquet files")
    detail_files = [_validate_standard_file(shard_dir, path) for path in detail_files]
    try:
        results = json.loads(
            result_file.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ArtifactError("standard results JSON is invalid") from error
    if not isinstance(results, dict):
        raise ArtifactError("standard results JSON must be an object")
    try:
        rows = [
            row for path in detail_files for row in parquet.read_table(path).to_pylist()
        ]
    except Exception as error:
        raise ArtifactError("standard details parquet is invalid") from error
    if any(not isinstance(row, dict) for row in rows):
        raise ArtifactError("standard detail rows must be objects")
    return results, rows, result_file, detail_files


def _completion_diagnostics(
    rows: list[dict[str, Any]],
    *,
    effective_limit: int,
    turn_boundary: str,
) -> dict[str, int | float]:
    completions = 0
    truncated = 0
    violations = 0
    for row in rows:
        response = row.get("model_response")
        if not isinstance(response, dict):
            raise ArtifactError("detail model_response must be an object")
        texts = response.get("text")
        tokens = response.get("output_tokens")
        if texts in (None, []):
            _validate_loglikelihood_response(response)
            continue
        if not isinstance(texts, list) or not isinstance(tokens, list):
            raise ArtifactError("completion text/output_tokens must be arrays")
        if len(texts) != len(tokens):
            raise ArtifactError("completion and output-token counts differ")
        _validate_optional_text_output(
            response,
            key="text_post_processed",
            expected_count=len(texts),
        )
        _validate_optional_text_output(
            response,
            key="reasonings",
            expected_count=len(texts),
            allow_none=True,
        )
        for text, token_ids in zip(texts, tokens, strict=True):
            if not isinstance(text, str):
                raise ArtifactError("completion text must be a string")
            _validate_token_group(token_ids)
            completions += 1
            truncated += int(len(token_ids) >= effective_limit)
            violations += int(turn_boundary in text)
    return {
        "samples": len(rows),
        "completions": completions,
        "truncated": truncated,
        "non_truncated": completions - truncated,
        "truncation_rate": truncated / completions if completions else 0.0,
        "turn_boundary_violations": violations,
        "turn_boundary_violation_rate": violations / completions
        if completions
        else 0.0,
    }


def _validate_token_group(value: object) -> None:
    if not isinstance(value, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in value
    ):
        raise ArtifactError("output tokens must be integer arrays")


def _validate_optional_text_output(
    response: dict[str, Any],
    *,
    key: str,
    expected_count: int,
    allow_none: bool = False,
) -> None:
    value = response.get(key)
    if value is None or (key == "reasonings" and value == []):
        return
    if (
        not isinstance(value, list)
        or len(value) != expected_count
        or any(
            not isinstance(item, str) and not (allow_none and item is None)
            for item in value
        )
    ):
        raise ArtifactError(f"{key} must align one-for-one with completion text")


def _validate_loglikelihood_response(response: dict[str, Any]) -> None:
    logprobs = response.get("logprobs")
    argmax = response.get("argmax_logits_eq_gold")
    if logprobs in (None, []) and argmax in (None, []):
        raise ArtifactError("empty completion lacks log-likelihood evidence")
    if logprobs not in (None, []):
        if not isinstance(logprobs, list) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in logprobs
        ):
            raise ArtifactError("logprobs must be a finite numeric array")
    if argmax not in (None, []):
        if not isinstance(argmax, list) or any(
            not isinstance(value, bool) for value in argmax
        ):
            raise ArtifactError("argmax evidence must be a boolean array")
    evidence_count = (
        len(logprobs) if isinstance(logprobs, list) and logprobs else len(argmax)
    )
    if (
        isinstance(logprobs, list)
        and logprobs
        and isinstance(argmax, list)
        and argmax
        and len(logprobs) != len(argmax)
    ):
        raise ArtifactError("log-likelihood evidence counts differ")
    output_tokens = response.get("output_tokens")
    if not isinstance(output_tokens, list) or not output_tokens:
        raise ArtifactError("log-likelihood output_tokens must be a non-empty array")
    for token_group in output_tokens:
        _validate_token_group(token_group)
        if not token_group:
            raise ArtifactError("log-likelihood output token groups must be non-empty")
    if len(output_tokens) != evidence_count:
        raise ArtifactError("log-likelihood evidence and output-token counts differ")
    _validate_optional_text_output(
        response,
        key="text_post_processed",
        expected_count=0,
    )
    _validate_optional_text_output(
        response,
        key="reasonings",
        expected_count=0,
        allow_none=True,
    )


def _sampling_config(
    results: dict[str, Any],
    *,
    prompt_template: object,
) -> dict[str, Any]:
    try:
        config = results["config_general"]["model_config"]
        generation = config["generation_parameters"]
    except (KeyError, TypeError) as error:
        raise ArtifactError("results lack model generation configuration") from error
    if not isinstance(config, dict) or not isinstance(generation, dict):
        raise ArtifactError("model generation configuration must be objects")
    sampling = {
        "temperature": generation.get("temperature"),
        "top_p": generation.get("top_p"),
        "top_k": generation.get("top_k"),
        "presence_penalty": generation.get("presence_penalty"),
        "frequency_penalty": generation.get("frequency_penalty"),
        "repetition_penalty": generation.get("repetition_penalty", 1.0),
        "penalty_decay": generation.get("penalty_decay"),
        "max_new_tokens": generation.get("max_new_tokens"),
        "stop": generation.get("stop_tokens"),
        "ignore_eos": False,
        "seed": config.get("seed"),
    }
    if (
        not isinstance(prompt_template, str)
        or prompt_template not in PROMPT_TEMPLATE_STOPS
    ):
        raise ArtifactError("model execution has an invalid prompt template")
    required = {
        "temperature": 0.96,
        "top_p": 0.76,
        "top_k": 32,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.1,
        "repetition_penalty": 1.0,
        "penalty_decay": 0.988,
        "max_new_tokens": 8192,
        "stop": [PROMPT_TEMPLATE_STOPS[prompt_template]],
        "ignore_eos": False,
    }
    mismatched = [
        key for key, expected in required.items() if sampling.get(key) != expected
    ]
    if mismatched:
        raise ArtifactError(
            "standard results violate the evaluation sampling contract: "
            + ", ".join(mismatched)
        )
    return sampling


def _validate_model_execution(
    model_execution: dict[str, object],
    unit: EvaluationUnit,
) -> None:
    expected_gemm_policy = (
        "fp16-accumulation" if unit.wkv_mode == "fp16" else "fp32-accumulation"
    )
    expected = {
        "weight_sha256": unit.weight.sha256,
        "weight_display_name": unit.weight.display_name,
        "wkv_mode": unit.wkv_mode,
        "prompt_template": unit.prompt_template,
        "gemm_policy": expected_gemm_policy,
    }
    mismatched = [
        key for key, value in expected.items() if model_execution.get(key) != value
    ]
    if mismatched:
        raise ArtifactError(
            "model execution does not match the planned unit: " + ", ".join(mismatched)
        )
    if unit.prompt_template not in PROMPT_TEMPLATE_STOPS:
        raise ArtifactError("planned unit has an invalid prompt template")
    for key in ("max_num_seqs", "max_num_batched_tokens"):
        value = model_execution.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ArtifactError(f"model execution {key} must be a positive integer")
    gpu = model_execution.get("gpu")
    if not isinstance(gpu, str) or not gpu:
        raise ArtifactError("model execution must identify the GPU")
    dependency_versions = model_execution.get("dependency_versions")
    if (
        not isinstance(dependency_versions, dict)
        or not {"lighteval", "vllm", "torch"}.issubset(dependency_versions)
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            for name, version in dependency_versions.items()
        )
    ):
        raise ArtifactError(
            "model execution must record lighteval, vllm, and torch versions"
        )


def _effective_limit(
    sampling: dict[str, Any],
    task_config: dict[str, Any],
) -> int:
    value = sampling.get("max_new_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        value = task_config.get("generation_size")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArtifactError("effective output limit is missing")
    return value


def _expected_task(
    unit: EvaluationUnit,
    task,
) -> dict[str, object]:
    return {
        "identity": f"{unit.weight.sha256}:{unit.wkv_mode}:{task.identity}",
        "weight_sha256": unit.weight.sha256,
        "weight_display_name": unit.weight.display_name,
        "wkv_mode": unit.wkv_mode,
        "selector": task.selector,
        "task_name": task.identity,
        "task_version": task.version,
        "module_family": task.module_family,
        "module": task.module,
        "dataset": task.dataset,
        "subset": task.subset,
        "evaluation_splits": list(task.evaluation_splits),
        "languages": list(task.languages),
        "upstream_tags": list(task.upstream_tags),
    }


def _is_uncertainty_aggregate(name: str) -> bool:
    return name == "stderr" or name.endswith("_stderr")


def _standard_task_sets(
    shard: EvaluationShard,
    registry_tasks: tuple[RegistryTask, ...],
) -> tuple[set[str], set[str]]:
    config_names = {task.identity for task in shard.tasks}
    aggregate_names = set(config_names)
    registry_names = {task.identity for task in registry_tasks}
    for task in shard.tasks:
        task_name, separator, few_shot = task.identity.rpartition("|")
        if not separator or ":" in task_name:
            continue
        # LightEval treats a root name with colon-qualified siblings as a
        # superset selector. Derive that exact expansion from the locked
        # registry instead of accepting arbitrary extra result tasks.
        members: set[str] = set()
        for identity in registry_names:
            registry_task_name = identity.rpartition("|")[0]
            if registry_task_name == task_name or registry_task_name.startswith(
                f"{task_name}:"
            ):
                members.add(f"{registry_task_name}|{few_shot}")
        if len(members) <= 1:
            continue
        config_names.update(members)
        aggregate_names.add(f"{task_name}:_average|{few_shot}")
    aggregate_names.update(config_names)
    return config_names, aggregate_names


def publications_from_shard(
    *,
    shard_dir: Path,
    campaign_id: str,
    unit: EvaluationUnit,
    shard: EvaluationShard,
    model_execution: dict[str, object],
    registry_tasks: tuple[RegistryTask, ...],
) -> list[tuple[str, dict[str, object], str]]:
    _validate_model_execution(model_execution, unit)
    prompt_template = unit.prompt_template
    results, rows, result_file, detail_files = _standard_artifacts(shard_dir)
    raw_task_results = results.get("results")
    raw_task_configs = results.get("config_tasks")
    general_config = results.get("config_general")
    if not isinstance(raw_task_results, dict) or not isinstance(raw_task_configs, dict):
        raise ArtifactError("results lack task aggregates/configs")
    if (
        not isinstance(general_config, dict)
        or general_config.get("max_samples") is not None
    ):
        raise ArtifactError("standard result does not prove max_samples=None")
    expected = {task.identity: task for task in shard.tasks}
    standard_config_names, standard_aggregate_names = _standard_task_sets(
        shard,
        registry_tasks,
    )
    result_names = {name for name in raw_task_results if name != "all"}
    if (
        result_names != standard_aggregate_names
        or set(raw_task_configs) != standard_config_names
    ):
        raise ArtifactError(
            "standard result task set does not match deterministic shard"
        )
    rows_by_task: dict[str, list[dict[str, Any]]] = {
        name: [] for name in standard_config_names
    }
    for row in rows:
        if (
            not isinstance(row.get("doc"), dict)
            or not isinstance(row.get("metric"), dict)
            or not isinstance(row.get("model_response"), dict)
        ):
            raise ArtifactError(
                "detail doc, metric, and model_response must be objects"
            )
        try:
            task_name = row["doc"]["task_name"]
        except (KeyError, TypeError) as error:
            raise ArtifactError("detail row lacks doc.task_name") from error
        if task_name not in rows_by_task:
            raise ArtifactError(f"unexpected detail task: {task_name}")
        rows_by_task[task_name].append(row)
    document_indices_by_task: dict[str, list[int]] = {}
    for task_name in sorted(standard_config_names):
        task_config = raw_task_configs[task_name]
        if not isinstance(task_config, dict):
            raise ArtifactError(f"invalid task config for {task_name}")
        original_docs = task_config.get("original_num_docs")
        effective_docs = task_config.get("effective_num_docs")
        skipped_multiselect_docs = task_config.get("skipped_multiselect_docs")
        document_indices: list[int] = []
        for row in rows_by_task[task_name]:
            try:
                document_index = row["doc"]["specific"]["helicopter_document_index"]
            except (KeyError, TypeError) as error:
                raise ArtifactError(
                    f"task detail lacks stable document index: {task_name}"
                ) from error
            if isinstance(document_index, bool) or not isinstance(document_index, int):
                raise ArtifactError(
                    f"task detail document index is invalid: {task_name}"
                )
            document_indices.append(document_index)
        if (
            isinstance(original_docs, bool)
            or not isinstance(original_docs, int)
            or isinstance(effective_docs, bool)
            or not isinstance(effective_docs, int)
            or isinstance(skipped_multiselect_docs, bool)
            or not isinstance(skipped_multiselect_docs, int)
            or original_docs <= 0
            or effective_docs <= 0
            or skipped_multiselect_docs < 0
            or original_docs != effective_docs + skipped_multiselect_docs
            or set(document_indices) != set(range(effective_docs))
        ):
            raise ArtifactError(
                f"task detail count does not account for the full evaluation "
                f"split: {task_name}"
            )
        document_indices_by_task[task_name] = document_indices
    sampling = _sampling_config(
        results,
        prompt_template=prompt_template,
    )
    dependency_versions = model_execution.get("dependency_versions")
    lighteval_version = (
        dependency_versions.get("lighteval")
        if isinstance(dependency_versions, dict)
        else None
    )
    if not isinstance(lighteval_version, str) or not lighteval_version:
        raise ArtifactError("model execution lacks the LightEval version")
    artifact = {
        "lighteval_version": lighteval_version,
        "results_path": str(result_file.relative_to(shard_dir)),
        "details_paths": [str(path.relative_to(shard_dir)) for path in detail_files],
    }
    publications: list[tuple[str, dict[str, object], str]] = []
    for task_name in sorted(expected):
        task_config = raw_task_configs[task_name]
        aggregates = raw_task_results[task_name]
        task_rows = rows_by_task[task_name]
        document_indices = document_indices_by_task[task_name]
        if not isinstance(task_config, dict) or not isinstance(aggregates, dict):
            raise ArtifactError(f"invalid task result/config for {task_name}")
        numeric: dict[str, float] = {}
        for key, value in aggregates.items():
            if (
                not isinstance(key, str)
                or not key
                or key != key.strip()
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise ArtifactError(f"task aggregate is invalid: {task_name}/{key}")
            if not math.isfinite(value):
                raise ArtifactError(f"task aggregate is not finite: {task_name}/{key}")
            numeric[key] = value
        primary_candidates = [
            key for key in numeric if not _is_uncertainty_aggregate(key)
        ]
        if not primary_candidates:
            raise ArtifactError(f"task has no native aggregate: {task_name}")
        primary_metric = primary_candidates[0]
        task = _expected_task(unit, expected[task_name])
        details = [
            {
                "sample_index": index,
                "document_index": document_indices[index],
                "doc": row["doc"],
                "metric": row["metric"],
                "model_response": row["model_response"],
            }
            for index, row in enumerate(task_rows)
        ]
        payload: dict[str, object] = {
            "schema_version": "lighteval-task-v2",
            "campaign_id": campaign_id,
            "task": task,
            "artifact": artifact,
            "task_config": task_config,
            "model": model_execution,
            "sampling_config": sampling,
            "primary_metric": primary_metric,
            "aggregates": numeric,
            "diagnostics": _completion_diagnostics(
                task_rows,
                effective_limit=_effective_limit(sampling, task_config),
                turn_boundary=PROMPT_TEMPLATE_STOPS[prompt_template],
            ),
            "details": details,
        }
        identity = str(task["identity"])
        publications.append((identity, payload, content_digest(payload)))
    return publications
