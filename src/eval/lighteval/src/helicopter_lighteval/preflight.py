from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import stat
from typing import Any
from urllib.parse import unquote, urlsplit

from .config import (
    EvaluationConfigurationError,
    EvaluationEnvironment,
    repository_root,
)
from .http_client import ScoreboardClient


LIGHTEVAL_VERSION = "0.13.0"


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
