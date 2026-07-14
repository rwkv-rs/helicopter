from __future__ import annotations

import hashlib
import json
import os
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
        raise ContractError(f"unsupported frozen source classification: {classification}")
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
    snapshot_download(
        repo_id=repository,
        revision=revision,
        local_dir=temporary,
        allow_patterns=list(SOURCE_PATTERNS),
    )
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
        "files": hashes,
        "scale_gate": scale_evidence,
    }


def verify_source(manifest_path: Path, destination: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("classification") == "final-scale-source-preflight-only":
        expected_path = Path(str(manifest["remote_read_only_path"]))
        if destination.resolve() != expected_path:
            raise ContractError(
                f"scale source path differs from the frozen manifest: {expected_path}"
            )
        revision = str(manifest.get("revision", ""))
        if len(revision) != 40 or destination.name != revision:
            raise ContractError("scale source directory must be named by its pinned 40-character revision")
        checkpoint = read_checkpoint(destination, require_final_layers=True)
        writable = [
            name
            for name in checkpoint.file_hashes
            if (destination / name).stat().st_mode
            & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        ]
        if writable:
            raise ContractError(f"scale source files are not read-only: {sorted(writable)}")
        digest = hashlib.sha256()
        for name, value in sorted(checkpoint.file_hashes.items()):
            digest.update(name.encode())
            digest.update(value.encode())
        return {
            "path": str(destination),
            "repository": manifest["repository"],
            "revision": revision,
            "layers": checkpoint.contract.num_hidden_layers,
            "selected_component": "text backbone",
            "files": checkpoint.file_hashes,
            "combined_sha256": digest.hexdigest(),
            "read_only": True,
        }
    weight = destination / str(manifest["weight_file"])
    expected_path = Path(str(manifest["remote_read_only_path"]))
    if destination.resolve() != expected_path:
        raise ContractError(
            f"proxy source path differs from the frozen manifest: {expected_path}"
        )
    revision = str(manifest.get("revision", ""))
    if len(revision) != 40 or destination.name != revision:
        raise ContractError(
            "proxy source directory must be named by its pinned 40-character revision"
        )
    if not weight.is_file():
        raise ContractError(f"source weight is missing: {weight}")
    actual = sha256_file(weight)
    if actual != manifest["weight_sha256"]:
        raise ContractError(f"source weight SHA-256 mismatch: expected {manifest['weight_sha256']} found {actual}")
    if os.access(weight, os.W_OK):
        raise ContractError("proxy source weight is writable; frozen source must be read-only")
    return {"path": str(destination), "weight": weight.name, "sha256": actual, "read_only": True}
