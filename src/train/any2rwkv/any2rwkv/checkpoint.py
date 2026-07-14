from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from safetensors import safe_open

from .contract import SourceContract, validate_source_config
from .errors import ContractError


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointManifest:
    path: Path
    config: dict[str, object]
    contract: SourceContract
    shards: tuple[Path, ...]
    tokenizer_files: tuple[Path, ...]
    file_hashes: dict[str, str]

    def tensor_names(self) -> Iterator[str]:
        for shard in self.shards:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                yield from handle.keys()


def read_checkpoint(
    path: Path,
    *,
    require_final_layers: bool = True,
    text_backbone_only: bool = True,
) -> CheckpointManifest:
    path = path.resolve()
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ContractError(f"HF config.json not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    contract = validate_source_config(
        config,
        require_final_layers=require_final_layers,
        text_backbone_only=text_backbone_only,
    )
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ContractError(f"invalid safetensors index: {index_path}")
        shards = tuple(sorted({path / str(name) for name in weight_map.values()}))
    else:
        shards = tuple(sorted(path.glob("*.safetensors")))
    if not shards or any(not shard.is_file() for shard in shards):
        raise ContractError("checkpoint has no complete safetensors shard set")
    tokenizer_names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
    }
    tokenizer_files = tuple(sorted(path / name for name in tokenizer_names if (path / name).is_file()))
    present_tokenizer_names = {file.name for file in tokenizer_files}
    if "tokenizer_config.json" not in present_tokenizer_names:
        raise ContractError("tokenizer_config.json is missing")
    if not ({"tokenizer.json", "vocab.json"} & present_tokenizer_names):
        raise ContractError("tokenizer vocabulary/model data is missing")
    files = (config_path, *shards, *tokenizer_files, *((index_path,) if index_path.is_file() else ()))
    hashes = {file.name: sha256_file(file) for file in files}
    return CheckpointManifest(path, config, contract, shards, tokenizer_files, hashes)
