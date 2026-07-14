from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .artifacts import file_sha256, write_json
from .configuration_any2rwkv import Any2RWKV7Config, Any2RWKVProxyConfig
from .errors import ContractError
from .mixer import ProjectionBoundaryRWKV7Attention


@dataclass(frozen=True)
class RWKV7MixerFactory:
    config: Any2RWKV7Config | Any2RWKVProxyConfig
    source_layer_types: tuple[str, ...]
    rotary_dim: int
    rope_theta: float

    @classmethod
    def from_checkpoint_config(cls, checkpoint_dir: Path) -> "RWKV7MixerFactory":
        payload = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
        model_type = payload.get("model_type")
        if model_type == Any2RWKV7Config.model_type:
            config = Any2RWKV7Config(**payload)
        elif model_type == Any2RWKVProxyConfig.model_type:
            config = Any2RWKVProxyConfig(**payload)
        else:
            raise ContractError(f"mixer store requires a final/proxy Any2RWKV checkpoint, got {model_type}")
        metadata = payload.get("any2rwkv")
        if not isinstance(metadata, dict):
            raise ContractError("Any2RWKV checkpoint lacks source metadata")
        source_layer_types = tuple(metadata.get("source_layer_types", ()))
        if len(source_layer_types) != config.num_hidden_layers:
            raise ContractError("source layer types do not cover every RWKV7 mixer")
        source_text = metadata.get("source_text_config")
        if not isinstance(source_text, dict):
            raise ContractError("Any2RWKV checkpoint lacks source text config")
        rope = source_text.get("rope_parameters", {})
        if not isinstance(rope, dict):
            rope = {}
        rotary_dim = int(
            source_text.get("head_dim", config.head_dim)
            * float(rope.get("partial_rotary_factor", source_text.get("partial_rotary_factor", 1.0)))
        )
        rotary_dim -= rotary_dim % 2
        return cls(
            config,
            source_layer_types,
            rotary_dim,
            float(rope.get("rope_theta", source_text.get("rope_theta", 10_000.0))),
        )

    def create(
        self,
        layer_index: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> ProjectionBoundaryRWKV7Attention:
        if not 0 <= layer_index < self.config.num_hidden_layers:
            raise ContractError(f"RWKV7 mixer layer out of range: {layer_index}")
        return ProjectionBoundaryRWKV7Attention(
            self.config,
            layer_index,
            source_used_rope=self.source_layer_types[layer_index] == "full_attention",
            rotary_dim=self.rotary_dim,
            rope_theta=self.rope_theta,
        ).to(device=device, dtype=dtype)


class RWKV7MixerLayerStore:
    """Load and atomically persist one RWKV7 mixer layer at a time."""

    def __init__(self, base_checkpoint_dir: Path, overlay_dir: Path) -> None:
        self.base_checkpoint_dir = base_checkpoint_dir.resolve()
        self.overlay_dir = overlay_dir.resolve()
        self.overlay_dir.mkdir(parents=True, exist_ok=True)
        self.factory = RWKV7MixerFactory.from_checkpoint_config(
            self.base_checkpoint_dir
        )
        index = json.loads(
            (self.base_checkpoint_dir / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ContractError("base mixer checkpoint has no weight_map")
        self.weight_map = {str(name): str(shard) for name, shard in weight_map.items()}

    def load_mixer(
        self,
        layer_index: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> ProjectionBoundaryRWKV7Attention:
        mixer = self.factory.create(layer_index, device=device, dtype=dtype)
        state = self._load_layer_state(layer_index)
        incompatible = mixer.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ContractError(
                f"mixer layer {layer_index} strict load failed: "
                f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
            )
        return mixer

    def save_mixer(
        self,
        layer_index: int,
        mixer: ProjectionBoundaryRWKV7Attention,
        *,
        cursor: dict[str, object],
    ) -> dict[str, object]:
        prefix = f"model.layers.{layer_index}.attn."
        path = self.overlay_dir / f"layer-{layer_index:03d}.safetensors"
        temporary = path.with_suffix(path.suffix + ".tmp")
        save_file(
            {
                prefix + name: tensor.detach().cpu().contiguous()
                for name, tensor in mixer.state_dict().items()
            },
            temporary,
        )
        temporary.replace(path)
        metadata = {
            "schema_version": 1,
            "layer_index": layer_index,
            "path": path.name,
            "sha256": file_sha256(path),
            "cursor": cursor,
        }
        write_json(self.overlay_dir / f"layer-{layer_index:03d}.json", metadata)
        return metadata

    def snapshot(self, destination: Path) -> Path:
        """Create an immutable, hash-verifiable all-layer sweep snapshot."""
        destination = destination.resolve()
        if destination.is_dir():
            self._validate_snapshot(destination)
            return destination
        temporary = destination.with_name(destination.name + ".tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        for layer_index in range(self.factory.config.num_hidden_layers):
            for suffix in (".safetensors", ".json"):
                source = self.overlay_dir / f"layer-{layer_index:03d}{suffix}"
                if not source.is_file():
                    raise ContractError(
                        f"cannot snapshot incomplete mixer overlay at layer {layer_index}"
                    )
                target = temporary / source.name
                if suffix == ".json":
                    shutil.copy2(source, target)
                else:
                    try:
                        os.link(source, target)
                    except OSError:
                        shutil.copy2(source, target)
        self._validate_snapshot(temporary)
        temporary.rename(destination)
        return destination

    def restore_snapshot(self, source: Path) -> None:
        """Atomically restore the selected all-layer sweep checkpoint."""
        source = source.resolve()
        self._validate_snapshot(source)
        for layer_index in range(self.factory.config.num_hidden_layers):
            for suffix in (".safetensors", ".json"):
                snapshot_path = source / f"layer-{layer_index:03d}{suffix}"
                target = self.overlay_dir / snapshot_path.name
                temporary = target.with_suffix(target.suffix + ".restore")
                temporary.unlink(missing_ok=True)
                if suffix == ".json":
                    shutil.copy2(snapshot_path, temporary)
                else:
                    try:
                        os.link(snapshot_path, temporary)
                    except OSError:
                        shutil.copy2(snapshot_path, temporary)
                temporary.replace(target)

    def _validate_snapshot(self, source: Path) -> None:
        for layer_index in range(self.factory.config.num_hidden_layers):
            tensor_path = source / f"layer-{layer_index:03d}.safetensors"
            metadata_path = source / f"layer-{layer_index:03d}.json"
            if not tensor_path.is_file() or not metadata_path.is_file():
                raise ContractError(
                    f"incomplete mixer sweep snapshot at layer {layer_index}: {source}"
                )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("layer_index") != layer_index
                or metadata.get("sha256") != file_sha256(tensor_path)
            ):
                raise ContractError(
                    f"mixer sweep snapshot hash mismatch at layer {layer_index}: {source}"
                )

    def _load_layer_state(self, layer_index: int) -> dict[str, torch.Tensor]:
        prefix = f"model.layers.{layer_index}.attn."
        overlay_path = self.overlay_dir / f"layer-{layer_index:03d}.safetensors"
        metadata_path = self.overlay_dir / f"layer-{layer_index:03d}.json"
        if overlay_path.is_file() or metadata_path.is_file():
            if not overlay_path.is_file() or not metadata_path.is_file():
                raise ContractError(f"incomplete mixer overlay for layer {layer_index}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("layer_index") != layer_index
                or metadata.get("sha256") != file_sha256(overlay_path)
            ):
                raise ContractError(f"mixer overlay hash/cursor mismatch at layer {layer_index}")
            with safe_open(overlay_path, framework="pt", device="cpu") as handle:
                return {
                    name.removeprefix(prefix): handle.get_tensor(name)
                    for name in handle.keys()
                    if name.startswith(prefix)
                }
        requested = {
            name: shard
            for name, shard in self.weight_map.items()
            if name.startswith(prefix)
        }
        if not requested:
            raise ContractError(f"base checkpoint lacks RWKV7 mixer layer {layer_index}")
        by_shard: dict[str, list[str]] = {}
        for name, shard in requested.items():
            by_shard.setdefault(shard, []).append(name)
        state: dict[str, torch.Tensor] = {}
        for shard, names in by_shard.items():
            with safe_open(
                self.base_checkpoint_dir / shard,
                framework="pt",
                device="cpu",
            ) as handle:
                for name in names:
                    state[name.removeprefix(prefix)] = handle.get_tensor(name)
        return state
