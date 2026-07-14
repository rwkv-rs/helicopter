from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open
from torch import Tensor

from .checkpoint import CheckpointManifest, sha256_file
from .errors import ContractError
from .target import canonical_text_name, is_vision_tensor


_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class LayerTensorIndex:
    layer_index: int
    tensor_names: tuple[str, ...]
    shard_paths: tuple[Path, ...]


class LayerTensorStore:
    """Read immutable HF safetensors one decoder layer at a time.

    The store keeps only names, paths and hashes resident. ``load_layer`` opens
    just the shards referenced by that layer and returns CPU tensors; callers
    own their lifetime and may stage them to one GPU at a time.
    """

    def __init__(self, checkpoint: CheckpointManifest) -> None:
        self.checkpoint = checkpoint
        raw_weight_map = _read_weight_map(checkpoint)
        layers: dict[int, list[tuple[str, str, Path]]] = {}
        all_entries: dict[str, tuple[str, Path]] = {}
        for source_name, shard_path in raw_weight_map.items():
            if is_vision_tensor(source_name):
                continue
            canonical_name = canonical_text_name(source_name)
            if canonical_name.startswith("mtp."):
                continue
            if canonical_name in all_entries:
                raise ContractError(
                    f"duplicate canonical text tensor: {canonical_name}"
                )
            all_entries[canonical_name] = (source_name, shard_path)
            match = _LAYER_INDEX.search(canonical_name)
            if match is None:
                continue
            layer_index = int(match.group(1))
            layers.setdefault(layer_index, []).append(
                (canonical_name, source_name, shard_path)
            )
        expected = set(range(checkpoint.contract.num_hidden_layers))
        if set(layers) != expected:
            raise ContractError(
                "layer tensor index does not cover every text decoder layer: "
                f"missing={sorted(expected - layers.keys())} "
                f"extra={sorted(layers.keys() - expected)}"
            )
        self._layers = {
            layer_index: tuple(sorted(entries))
            for layer_index, entries in layers.items()
        }
        self._all_entries = all_entries

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    def has_tensor(self, tensor_name: str) -> bool:
        return tensor_name in self._all_entries

    def index(self, layer_index: int) -> LayerTensorIndex:
        entries = self._entries(layer_index)
        return LayerTensorIndex(
            layer_index,
            tuple(canonical_name for canonical_name, _, _ in entries),
            tuple(sorted({shard_path for _, _, shard_path in entries})),
        )

    def verify_source_shards(self) -> dict[str, str]:
        verified: dict[str, str] = {}
        for shard_path in self.checkpoint.shards:
            actual = sha256_file(shard_path)
            expected = self.checkpoint.file_hashes.get(shard_path.name)
            if actual != expected:
                raise ContractError(f"source shard changed after preflight: {shard_path}")
            verified[shard_path.name] = actual
        return verified

    def load_layer(self, layer_index: int) -> dict[str, Tensor]:
        entries = self._entries(layer_index)
        by_shard: dict[Path, list[tuple[str, str]]] = {}
        for canonical_name, source_name, shard_path in entries:
            by_shard.setdefault(shard_path, []).append((canonical_name, source_name))
        tensors: dict[str, Tensor] = {}
        for shard_path, names in sorted(by_shard.items()):
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for canonical_name, source_name in names:
                    if canonical_name in tensors:
                        raise ContractError(
                            f"duplicate canonical tensor in layer {layer_index}: {canonical_name}"
                        )
                    tensors[canonical_name] = handle.get_tensor(source_name)
        return tensors

    def load_named_tensors(self, tensor_names: tuple[str, ...]) -> dict[str, Tensor]:
        missing = sorted(set(tensor_names) - self._all_entries.keys())
        if missing:
            raise ContractError(f"checkpoint lacks requested text tensors: {missing}")
        by_shard: dict[Path, list[tuple[str, str]]] = {}
        for canonical_name in tensor_names:
            source_name, shard_path = self._all_entries[canonical_name]
            by_shard.setdefault(shard_path, []).append((canonical_name, source_name))
        tensors: dict[str, Tensor] = {}
        for shard_path, names in sorted(by_shard.items()):
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for canonical_name, source_name in names:
                    tensors[canonical_name] = handle.get_tensor(source_name)
        return tensors

    def _entries(self, layer_index: int) -> tuple[tuple[str, str, Path], ...]:
        try:
            return self._layers[layer_index]
        except KeyError as error:
            raise ContractError(f"decoder layer index out of range: {layer_index}") from error


def _read_weight_map(checkpoint: CheckpointManifest) -> dict[str, Path]:
    index_path = checkpoint.path / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ContractError(f"invalid safetensors weight_map: {index_path}")
        return {
            str(tensor_name): checkpoint.path / str(shard_name)
            for tensor_name, shard_name in weight_map.items()
        }
    result: dict[str, Path] = {}
    for shard_path in checkpoint.shards:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for tensor_name in handle.keys():
                if tensor_name in result:
                    raise ContractError(f"duplicate tensor across shards: {tensor_name}")
                result[tensor_name] = shard_path
    return result
