from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open

from .checkpoint import sha256_file
from .errors import ContractError


def validate_sharded_checkpoint(path: Path) -> dict[str, Any]:
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        single = path / "model.safetensors"
        if not single.is_file():
            raise ContractError(f"checkpoint index or single safetensors file is missing: {path}")
        with safe_open(single, framework="pt", device="cpu") as handle:
            names = sorted(handle.keys())
        if not names:
            raise ContractError("single-file checkpoint has no tensors")
        weight_map = {name: single.name for name in names}
        return {
            "tensor_count": len(names),
            "shard_count": 1,
            "weight_map_sha256": hashlib.sha256(
                json.dumps(weight_map, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "shard_sha256": {single.name: sha256_file(single)},
        }
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ContractError("checkpoint index has no weight_map")
    expected_by_shard: dict[str, set[str]] = {}
    for name, filename in weight_map.items():
        expected_by_shard.setdefault(str(filename), set()).add(str(name))
    actual_names: set[str] = set()
    file_hashes: dict[str, str] = {}
    for filename, expected in expected_by_shard.items():
        shard = path / filename
        if not shard.is_file():
            raise ContractError(f"checkpoint shard is missing: {filename}")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            actual = set(handle.keys())
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ContractError(
                f"checkpoint shard/index mismatch {filename}: missing={missing[:8]} unexpected={unexpected[:8]}"
            )
        overlap = actual_names & actual
        if overlap:
            raise ContractError(f"duplicate tensors across checkpoint shards: {sorted(overlap)[:8]}")
        actual_names.update(actual)
        file_hashes[filename] = sha256_file(shard)
    digest = hashlib.sha256(
        json.dumps(weight_map, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "tensor_count": len(actual_names),
        "shard_count": len(expected_by_shard),
        "weight_map_sha256": digest,
        "shard_sha256": file_hashes,
    }


def compare_fresh_process_manifests(left: dict[str, Any], right: dict[str, Any]) -> None:
    deterministic_keys = (
        "model_sha256",
        "backend",
        "transformers_version",
        "prompt_count",
        "new_tokens",
        "loading_info",
        "greedy_digest",
        "single_greedy_digest",
        "batch_greedy_digest",
        "logits_digest",
        "ppl",
        "shards",
    )
    for key in deterministic_keys:
        if key not in left or key not in right:
            raise ContractError(f"roundtrip process manifest is missing {key}")
    if left["loading_info"] != {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}:
        raise ContractError(f"first strict reload is not clean: {left['loading_info']}")
    if any(left[key] != right[key] for key in deterministic_keys):
        raise ContractError(
            "fresh-process Transformers version, checkpoint, loading, "
            "or deterministic outputs differ"
        )
    for manifest in (left, right):
        if (
            manifest.get("passed") is not True
            or manifest.get("backend") != "transformers"
            or manifest.get("strict_reload") is not True
            or manifest.get("single_batch_greedy_equal") is not True
            or manifest.get("batch_isolation", {}).get("passed") is not True
            or manifest.get("full_chunked_cache", {}).get("passed") is not True
            or manifest.get("state_reset", {}).get("passed") is not True
        ):
            raise ContractError(
                "fresh-process Transformers inference contract is incomplete"
            )
