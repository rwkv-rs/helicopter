from __future__ import annotations

import hashlib
import fnmatch
import json
import os
import re
import shutil
import stat
from pathlib import Path

from huggingface_hub import snapshot_download

from .artifacts import verify_scale_gate
from .checkpoint import read_checkpoint, sha256_file
from .errors import ContractError


SOURCE_PATTERNS = (
    "config.json",
    "generation_config.json",
    "tokenizer*",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "model*.safetensors",
    "model.safetensors.index.json",
)
_PINNED_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _source_identity(manifest: dict[str, object]) -> tuple[str, str, str]:
    classification = str(manifest.get("classification", ""))
    if classification not in {
        "real-proxy-model-not-60-layer-isomorphic",
        "final-scale-source-preflight-only",
    }:
        raise ContractError(f"unsupported frozen source classification: {classification}")
    repository = str(manifest.get("repository", "")).strip()
    revision = str(manifest.get("revision", ""))
    if not repository:
        raise ContractError("frozen source repository identity is missing")
    if _PINNED_REVISION_RE.fullmatch(revision) is None:
        raise ContractError(
            "frozen source revision must be a pinned 40-character commit SHA"
        )
    return classification, repository, revision


def _combined_sha256(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(name.encode())
        digest.update(value.encode())
    return digest.hexdigest()


def _expected_source_files(
    manifest: dict[str, object],
    destination: Path,
    *,
    classification: str,
    repository: str,
    revision: str,
) -> dict[str, str]:
    expected = manifest.get("files")
    if (
        expected is None
        and classification == "real-proxy-model-not-60-layer-isomorphic"
    ):
        expected = {str(manifest.get("weight_file", "")): manifest.get("weight_sha256")}
    if expected is None:
        download_manifest_path = destination / "source-download-manifest.json"
        if not download_manifest_path.is_file():
            raise ContractError(
                "scale source needs manifest files or source-download-manifest.json "
                "to prove immutable content identity"
            )
        download_manifest = json.loads(
            download_manifest_path.read_text(encoding="utf-8")
        )
        if (
            download_manifest.get("repository") != repository
            or download_manifest.get("revision") != revision
        ):
            raise ContractError(
                "source download manifest repository/revision identity differs"
            )
        expected = download_manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ContractError("frozen source manifest has no immutable file digests")
    normalized: dict[str, str] = {}
    for raw_name, raw_sha256 in expected.items():
        name = str(raw_name)
        sha256 = str(raw_sha256)
        if Path(name).name != name or _SHA256_RE.fullmatch(sha256) is None:
            raise ContractError("frozen source file digest binding is malformed")
        normalized[name] = sha256
    return normalized


def _copy_frozen_seed(seed: Path, temporary: Path) -> list[str]:
    if not seed.is_dir():
        raise ContractError(f"frozen source seed directory is missing: {seed}")
    temporary.mkdir(parents=True)
    copied: list[str] = []
    for source in sorted(seed.iterdir()):
        if not source.is_file() or not any(
            fnmatch.fnmatch(source.name, pattern) for pattern in SOURCE_PATTERNS
        ):
            continue
        shutil.copy2(source, temporary / source.name)
        copied.append(source.name)
    if not copied:
        raise ContractError(f"frozen source seed contains no checkpoint files: {seed}")
    return copied


def fetch_source(
    manifest_path: Path,
    destination: Path,
    *,
    scale_gate: Path | None = None,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    classification = manifest.get("classification")
    if classification not in {
        "real-proxy-model-not-60-layer-isomorphic",
        "final-scale-source-preflight-only",
    }:
        raise ContractError(
            f"unsupported frozen source classification: {classification}"
        )
    scale_evidence = None
    if classification == "final-scale-source-preflight-only":
        if scale_gate is None:
            raise ContractError("397B fetch-source requires --scale-gate pointing to the accepted proxy run")
        try:
            scale_evidence = verify_scale_gate(scale_gate.resolve())
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ContractError(f"397B scale gate rejected: {error}") from error
    repository = str(manifest["repository"])
    revision = str(manifest["revision"])
    expected_path = Path(str(manifest["remote_read_only_path"]))
    if destination.resolve() != expected_path:
        raise ContractError(f"source destination must match the frozen remote path: {expected_path}")
    if destination.exists():
        raise ContractError(f"source destination already exists; verify and reuse it instead: {destination}")
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise ContractError(f"partial source directory requires explicit repair before retry: {temporary}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    seed_value = manifest.get("remote_seed_path")
    if seed_value is not None:
        seed = Path(str(seed_value)).resolve()
        if seed == destination.resolve() or destination.resolve() in seed.parents:
            raise ContractError("frozen source seed must be independent from its destination")
        copied = _copy_frozen_seed(seed, temporary)
        acquisition = {
            "mode": "verified-local-seed",
            "seed_path": str(seed),
            "copied_files": copied,
        }
    else:
        snapshot_download(
            repo_id=repository,
            revision=revision,
            local_dir=temporary,
            allow_patterns=list(SOURCE_PATTERNS),
        )
        acquisition = {"mode": "huggingface-snapshot-download"}
    cache = temporary / ".cache"
    if cache.exists():
        shutil.rmtree(cache)
    if classification == "real-proxy-model-not-60-layer-isomorphic":
        weight = temporary / str(manifest["weight_file"])
        if not weight.is_file():
            raise ContractError(f"frozen source weight is missing after download: {weight.name}")
        actual_weight_sha = sha256_file(weight)
        if actual_weight_sha != manifest["weight_sha256"]:
            raise ContractError(
                f"source weight SHA-256 mismatch: expected {manifest['weight_sha256']} found {actual_weight_sha}"
            )
    else:
        checkpoint = read_checkpoint(temporary, require_final_layers=True)
        if checkpoint.contract.num_hidden_layers != 60:
            raise ContractError("397B source does not expose the frozen 60-layer text backbone")
    files = sorted(path for path in temporary.iterdir() if path.is_file())
    hashes = {path.name: sha256_file(path) for path in files}
    (temporary / "source-download-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repository": repository,
                "revision": revision,
                "acquisition": acquisition,
                "files": hashes,
                "scale_gate": scale_evidence,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.rename(destination)
    for path in destination.iterdir():
        if path.is_file():
            path.chmod(0o444)
    os.chmod(destination, 0o555)
    return {
        "path": str(destination),
        "repository": repository,
        "revision": revision,
        "acquisition": acquisition,
        "files": hashes,
        "scale_gate": scale_evidence,
    }


def verify_source(manifest_path: Path, destination: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ContractError("frozen source manifest must contain a JSON object")
    classification, repository, revision = _source_identity(manifest)
    destination = destination.resolve()
    checkpoint = read_checkpoint(
        destination,
        require_final_layers=classification == "final-scale-source-preflight-only",
    )
    expected_files = _expected_source_files(
        manifest,
        destination,
        classification=classification,
        repository=repository,
        revision=revision,
    )
    verified_files: dict[str, str] = {}
    for name, expected_sha256 in expected_files.items():
        source_file = destination / name
        if not source_file.is_file():
            raise ContractError(f"source identity file is missing: {source_file}")
        actual_sha256 = checkpoint.file_hashes.get(name) or sha256_file(source_file)
        if actual_sha256 != expected_sha256:
            raise ContractError(
                "source file SHA-256 mismatch: "
                f"{name}: expected {expected_sha256} found {actual_sha256}"
            )
        verified_files[name] = actual_sha256
    writable = sorted(
        name
        for name in verified_files
        if (destination / name).stat().st_mode
        & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    )
    expected_path_value = manifest.get("remote_read_only_path")
    expected_path = (
        Path(str(expected_path_value)).resolve()
        if expected_path_value is not None
        else None
    )
    return {
        "path": str(destination),
        "repository": repository,
        "revision": revision,
        "classification": classification,
        "layers": checkpoint.contract.num_hidden_layers,
        "selected_component": "text backbone",
        "files": verified_files,
        "combined_sha256": _combined_sha256(verified_files),
        "read_only": not writable,
        "writable_files": writable,
        "preferred_materialization_path": (
            str(expected_path) if expected_path is not None else None
        ),
        "equivalent_materialization": (
            expected_path is not None and destination != expected_path
        ),
    }
