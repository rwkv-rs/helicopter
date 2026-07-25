from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..artifacts import file_sha256, write_json
from ..errors import ContractError


@dataclass(frozen=True)
class LayerInputBatch:
    row_indices: torch.Tensor
    hidden_states: torch.Tensor
    shared_states: torch.Tensor | None = None


@dataclass(frozen=True)
class LayerInputCacheEstimate:
    current_cache_bytes: int
    next_cache_bytes: int
    required_free_bytes: int


def estimate_layer_input_cache_bytes(
    *,
    row_count: int,
    sequence_length: int,
    hidden_size: int,
    dtype_bytes: int,
    current_has_shared_states: bool,
    next_has_shared_states: bool,
    reserve_ratio: float = 0.10,
) -> LayerInputCacheEstimate:
    if (
        row_count <= 0
        or sequence_length <= 0
        or hidden_size <= 0
        or dtype_bytes <= 0
        or reserve_ratio < 0
    ):
        raise ContractError("layer-input cache estimate arguments are invalid")
    base = row_count * sequence_length * hidden_size * dtype_bytes
    current = base * (2 if current_has_shared_states else 1)
    next_cache = base * (2 if next_has_shared_states else 1)
    required = math.ceil((current + next_cache) * (1 + reserve_ratio))
    return LayerInputCacheEstimate(current, next_cache, required)


def require_layer_input_cache_capacity(
    cache_root: Path,
    estimate: LayerInputCacheEstimate,
) -> dict[str, int]:
    existing = cache_root.resolve()
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    if usage.free < estimate.required_free_bytes:
        raise ContractError(
            "insufficient layer-input cache capacity: "
            f"required={estimate.required_free_bytes} free={usage.free} path={existing}"
        )
    return {
        "current_cache_bytes": estimate.current_cache_bytes,
        "next_cache_bytes": estimate.next_cache_bytes,
        "required_free_bytes": estimate.required_free_bytes,
        "available_free_bytes": usage.free,
    }


def write_layer_input_cache(
    destination: Path,
    *,
    layer_index: int,
    split: str,
    row_count: int,
    sequence_length: int,
    hidden_size: int,
    binding: Mapping[str, object],
    batches: Iterable[LayerInputBatch],
) -> Path:
    if layer_index < 0 or not split or row_count <= 0:
        raise ContractError("layer-input cache identity is invalid")
    destination = destination.resolve()
    temporary = destination.with_name(destination.name + ".tmp")
    if destination.exists():
        raise ContractError(f"layer-input cache already exists: {destination}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    shards: list[dict[str, object]] = []
    covered: set[int] = set()
    observed_shared_states: bool | None = None
    try:
        for shard_index, batch in enumerate(batches):
            row_indices = batch.row_indices.detach().cpu().to(torch.int64).contiguous()
            hidden_states = batch.hidden_states.detach().cpu().to(torch.bfloat16).contiguous()
            shared_states = (
                None
                if batch.shared_states is None
                else batch.shared_states.detach().cpu().to(torch.bfloat16).contiguous()
            )
            if row_indices.ndim != 1 or hidden_states.ndim != 3:
                raise ContractError("layer-input cache batch ranks are invalid")
            if (
                hidden_states.shape[0] != row_indices.numel()
                or hidden_states.shape[1] != sequence_length
                or hidden_states.shape[2] != hidden_size
            ):
                raise ContractError("layer-input cache hidden shape differs from contract")
            batch_has_shared_states = shared_states is not None
            if observed_shared_states is None:
                observed_shared_states = batch_has_shared_states
            elif observed_shared_states != batch_has_shared_states:
                raise ContractError("layer-input cache batches disagree on shared-state presence")
            if shared_states is not None and shared_states.shape != hidden_states.shape:
                raise ContractError("layer-input cache shared-state shape differs from hidden shape")
            values = [int(value) for value in row_indices.tolist()]
            if (
                not values
                or any(value < 0 or value >= row_count for value in values)
                or covered.intersection(values)
            ):
                raise ContractError("layer-input cache row coverage is invalid")
            covered.update(values)
            tensors = {"row_indices": row_indices, "hidden_states": hidden_states}
            if shared_states is not None:
                tensors["shared_states"] = shared_states
            path = temporary / f"shard-{shard_index:06d}.safetensors"
            save_file(tensors, path)
            shards.append(
                {
                    "path": path.name,
                    "sha256": file_sha256(path),
                    "row_indices": values,
                }
            )
        if covered != set(range(row_count)):
            missing = sorted(set(range(row_count)) - covered)
            raise ContractError(
                f"layer-input cache does not cover every row exactly once: missing={missing[:16]}"
            )
        manifest = {
            "schema_version": 1,
            "layer_index": layer_index,
            "split": split,
            "row_count": row_count,
            "sequence_length": sequence_length,
            "hidden_size": hidden_size,
            "dtype": "bfloat16",
            "has_shared_states": bool(observed_shared_states),
            "binding": dict(binding),
            "shards": shards,
        }
        write_json(temporary / "manifest.json", manifest)
        _fsync_tree(temporary)
        temporary.rename(destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def prepare_distributed_layer_input_cache(
    destination: Path,
    *,
    world_size: int,
    layer_index: int,
    split: str,
    row_count: int,
    sequence_length: int,
    hidden_size: int,
    has_shared_states: bool,
    binding: Mapping[str, object],
) -> dict[str, str]:
    """Inspect or create an immutable, resumable shared cache transaction."""
    identity = _distributed_cache_identity(
        world_size=world_size,
        layer_index=layer_index,
        split=split,
        row_count=row_count,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        has_shared_states=has_shared_states,
        binding=binding,
    )
    destination = destination.resolve()
    cache_id = _cache_identity_sha256(identity)
    staging = _distributed_staging_path(destination, cache_id)
    if destination.exists():
        _validate_published_distributed_cache(destination, identity)
        return {"state": "published", "cache_id": cache_id}
    conflicting = tuple(
        path
        for path in destination.parent.glob(destination.name + ".txn-*")
        if path != staging and ".rank-" not in path.name
    )
    if conflicting:
        raise ContractError(
            "distributed layer-input cache has a conflicting transaction: "
            f"{conflicting[0]}"
        )
    if staging.exists():
        _validate_transaction(staging, identity, cache_id)
        return {"state": "resume-staging", "cache_id": cache_id}
    staging.mkdir(parents=True)
    write_json(
        staging / "transaction.json",
        {
            "schema_version": 1,
            "cache_id": cache_id,
            "identity": identity,
        },
    )
    _fsync_tree(staging)
    return {"state": "created-staging", "cache_id": cache_id}


def write_distributed_layer_input_cache_partition(
    destination: Path,
    *,
    rank: int,
    world_size: int,
    layer_index: int,
    split: str,
    row_count: int,
    sequence_length: int,
    hidden_size: int,
    has_shared_states: bool,
    binding: Mapping[str, object],
    batches: Iterable[LayerInputBatch],
) -> dict[str, object]:
    """Atomically commit or reuse one rank's immutable cache partition."""
    if not 0 <= rank < world_size or world_size <= 1:
        raise ContractError("distributed layer-input cache rank identity is invalid")
    if layer_index < 0 or not split or row_count <= 0:
        raise ContractError("layer-input cache identity is invalid")
    identity = _distributed_cache_identity(
        world_size=world_size,
        layer_index=layer_index,
        split=split,
        row_count=row_count,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        has_shared_states=has_shared_states,
        binding=binding,
    )
    destination = destination.resolve()
    cache_id = _cache_identity_sha256(identity)
    staging = _distributed_staging_path(destination, cache_id)
    if not staging.is_dir():
        raise ContractError("distributed layer-input cache staging was not prepared")
    _validate_transaction(staging, identity, cache_id)
    partition = staging / f"rank-{rank:05d}"
    if partition.exists():
        _validate_partition(partition, identity, rank=rank, full_hash=True)
        return {"status": "reused", "rank": rank}
    partial = staging.with_name(staging.name + f".rank-{rank:05d}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    shards: list[dict[str, object]] = []
    covered: set[int] = set()
    observed_shared_states: bool | None = None
    try:
        for shard_index, batch in enumerate(batches):
            row_indices, tensors, batch_has_shared_states = _normalize_cache_batch(
                batch,
                row_count=row_count,
                sequence_length=sequence_length,
                hidden_size=hidden_size,
                covered=covered,
            )
            if observed_shared_states is None:
                observed_shared_states = batch_has_shared_states
            elif observed_shared_states != batch_has_shared_states:
                raise ContractError(
                    "layer-input cache batches disagree on shared-state presence"
                )
            if batch_has_shared_states != has_shared_states:
                raise ContractError(
                    "layer-input cache batch shared-state presence differs from contract"
                )
            path = partial / f"shard-{shard_index:06d}.safetensors"
            save_file(tensors, path)
            shards.append(
                {
                    "path": path.name,
                    "sha256": file_sha256(path),
                    "row_indices": row_indices,
                }
            )
        expected_rows = set(
            range(
                row_count * rank // world_size,
                row_count * (rank + 1) // world_size,
            )
        )
        if covered != expected_rows:
            raise ContractError(
                "distributed layer-input cache rank row partition is incomplete"
            )
        write_json(
            partial / "partition.json",
            {
                "schema_version": 1,
                "cache_id": cache_id,
                "rank": rank,
                "identity": identity,
                "shards": shards,
            },
        )
        _fsync_tree(partial)
        partial.rename(partition)
        _fsync_directory(staging)
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return {"status": "written", "rank": rank}


def publish_distributed_layer_input_cache(
    destination: Path,
    *,
    world_size: int,
    layer_index: int,
    split: str,
    row_count: int,
    sequence_length: int,
    hidden_size: int,
    has_shared_states: bool,
    binding: Mapping[str, object],
) -> Path:
    """Validate all immutable partitions and atomically rename the transaction."""
    if world_size <= 1:
        raise ContractError("distributed layer-input cache requires multiple ranks")
    identity = _distributed_cache_identity(
        world_size=world_size,
        layer_index=layer_index,
        split=split,
        row_count=row_count,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        has_shared_states=has_shared_states,
        binding=binding,
    )
    destination = destination.resolve()
    if destination.exists():
        _validate_published_distributed_cache(destination, identity)
        return destination
    cache_id = _cache_identity_sha256(identity)
    staging = _distributed_staging_path(destination, cache_id)
    _validate_transaction(staging, identity, cache_id)
    covered: set[int] = set()
    published_shards: list[dict[str, object]] = []
    for rank in range(world_size):
        partition = staging / f"rank-{rank:05d}"
        manifest = _validate_partition(
            partition, identity, rank=rank, full_hash=True
        )
        for shard in manifest.get("shards", []):
            source = partition / str(shard["path"])
            values = [int(value) for value in shard.get("row_indices", [])]
            if (
                not values
                or any(value < 0 or value >= row_count for value in values)
                or covered.intersection(values)
            ):
                raise ContractError(
                    "distributed layer-input cache shard coverage/hash is invalid"
                )
            covered.update(values)
            published_shards.append(
                {
                    "path": str(source.relative_to(staging)),
                    "sha256": str(shard["sha256"]),
                    "row_indices": values,
                }
            )
    if covered != set(range(row_count)):
        missing = sorted(set(range(row_count)) - covered)
        raise ContractError(
            "distributed layer-input cache does not cover every row exactly once: "
            f"missing={missing[:16]}"
        )
    write_json(
        staging / "manifest.json",
        {
            "schema_version": 1,
            "layer_index": layer_index,
            "split": split,
            "row_count": row_count,
            "sequence_length": sequence_length,
            "hidden_size": hidden_size,
            "dtype": "bfloat16",
            "has_shared_states": has_shared_states,
            "binding": dict(binding),
            "distributed_writer": {
                "world_size": world_size,
                "cache_id": cache_id,
                "row_partition": "balanced-contiguous-v1",
            },
            "shards": published_shards,
        },
    )
    _fsync_tree(staging)
    staging.rename(destination)
    _fsync_directory(destination.parent)
    return destination


def _normalize_cache_batch(
    batch: LayerInputBatch,
    *,
    row_count: int,
    sequence_length: int,
    hidden_size: int,
    covered: set[int],
) -> tuple[list[int], dict[str, torch.Tensor], bool]:
    row_indices = batch.row_indices.detach().cpu().to(torch.int64).contiguous()
    hidden_states = batch.hidden_states.detach().cpu().to(torch.bfloat16).contiguous()
    shared_states = (
        None
        if batch.shared_states is None
        else batch.shared_states.detach().cpu().to(torch.bfloat16).contiguous()
    )
    if row_indices.ndim != 1 or hidden_states.ndim != 3:
        raise ContractError("layer-input cache batch ranks are invalid")
    if (
        hidden_states.shape[0] != row_indices.numel()
        or hidden_states.shape[1] != sequence_length
        or hidden_states.shape[2] != hidden_size
    ):
        raise ContractError("layer-input cache hidden shape differs from contract")
    if shared_states is not None and shared_states.shape != hidden_states.shape:
        raise ContractError("layer-input cache shared-state shape differs from hidden shape")
    values = [int(value) for value in row_indices.tolist()]
    if (
        not values
        or any(value < 0 or value >= row_count for value in values)
        or covered.intersection(values)
    ):
        raise ContractError("layer-input cache row coverage is invalid")
    covered.update(values)
    tensors = {"row_indices": row_indices, "hidden_states": hidden_states}
    if shared_states is not None:
        tensors["shared_states"] = shared_states
    return values, tensors, shared_states is not None


def _distributed_cache_identity(
    *,
    world_size,
    layer_index,
    split,
    row_count,
    sequence_length,
    hidden_size,
    has_shared_states,
    binding,
):
    if world_size <= 1 or layer_index < 0 or not split or row_count <= 0:
        raise ContractError("distributed layer-input cache identity is invalid")
    return {
        "schema_version": 1,
        "world_size": world_size,
        "layer_index": layer_index,
        "split": split,
        "row_count": row_count,
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "dtype": "bfloat16",
        "has_shared_states": bool(has_shared_states),
        "row_partition": "balanced-contiguous-v1",
        "binding": dict(binding),
    }


def _cache_identity_sha256(identity: Mapping[str, object]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _distributed_staging_path(destination: Path, cache_id: str) -> Path:
    return destination.with_name(destination.name + f".txn-{cache_id}")


def _validate_transaction(staging: Path, identity, cache_id: str) -> None:
    path = staging / "transaction.json"
    if not path.is_file():
        raise ContractError("distributed layer-input cache transaction is incomplete")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("cache_id") != cache_id
        or payload.get("identity") != identity
    ):
        raise ContractError("distributed layer-input cache transaction identity differs")


def _validate_partition(partition: Path, identity, *, rank: int, full_hash: bool):
    path = partition / "partition.json"
    if not path.is_file():
        raise ContractError(f"distributed layer-input cache rank {rank} is incomplete")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("rank") != rank
        or payload.get("cache_id") != _cache_identity_sha256(identity)
        or payload.get("identity") != identity
    ):
        raise ContractError(f"distributed layer-input cache rank {rank} identity differs")
    covered: set[int] = set()
    for shard in payload.get("shards", []):
        shard_path = partition / str(shard.get("path", ""))
        values = [int(value) for value in shard.get("row_indices", [])]
        if not shard_path.is_file() or (
            full_hash and file_sha256(shard_path) != shard.get("sha256")
        ):
            raise ContractError(
                f"distributed layer-input cache rank {rank} shard hash differs"
            )
        if not values or covered.intersection(values):
            raise ContractError(
                f"distributed layer-input cache rank {rank} row coverage differs"
            )
        if full_hash:
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                stored_rows = [
                    int(value) for value in handle.get_tensor("row_indices").tolist()
                ]
            if stored_rows != values:
                raise ContractError(
                    f"distributed layer-input cache rank {rank} shard rows differ"
                )
        covered.update(values)
    expected_rows = set(
        range(
            int(identity["row_count"]) * rank // int(identity["world_size"]),
            int(identity["row_count"]) * (rank + 1) // int(identity["world_size"]),
        )
    )
    if covered != expected_rows:
        raise ContractError(
            f"distributed layer-input cache rank {rank} row coverage differs"
        )
    return payload


def _validate_published_distributed_cache(destination: Path, identity) -> None:
    reader = LayerInputCacheReader(
        destination,
        expected_binding=identity["binding"],
        verification="full",
    )
    manifest = reader.manifest
    writer = manifest.get("distributed_writer")
    if (
        manifest.get("layer_index") != identity["layer_index"]
        or manifest.get("split") != identity["split"]
        or manifest.get("row_count") != identity["row_count"]
        or manifest.get("sequence_length") != identity["sequence_length"]
        or manifest.get("hidden_size") != identity["hidden_size"]
        or manifest.get("has_shared_states") != identity["has_shared_states"]
        or not isinstance(writer, dict)
        or writer.get("world_size") != identity["world_size"]
        or writer.get("cache_id") != _cache_identity_sha256(identity)
    ):
        raise ContractError("published distributed layer-input cache identity differs")


class LayerInputCacheReader:
    def __init__(
        self,
        cache_dir: Path,
        *,
        expected_binding: Mapping[str, object] | None = None,
        verification: str = "full",
        max_cached_shards: int = 256,
        max_cached_bytes: int = 1024**3,
        pin_memory: bool | None = None,
    ) -> None:
        if verification not in {"full", "manifest"}:
            raise ContractError("layer-input cache verification mode is invalid")
        if max_cached_shards <= 0:
            raise ContractError("layer-input cache max_cached_shards must be positive")
        if max_cached_bytes <= 0:
            raise ContractError("layer-input cache max_cached_bytes must be positive")
        self.cache_dir = cache_dir.resolve()
        self.max_cached_shards = max_cached_shards
        self.max_cached_bytes = max_cached_bytes
        self._cached_bytes = 0
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
        self._shard_tensors: dict[
            Path, tuple[torch.Tensor, torch.Tensor | None]
        ] = {}
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ContractError(f"layer-input cache manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise ContractError("unsupported layer-input cache schema")
        if expected_binding is not None and self.manifest.get("binding") != dict(expected_binding):
            raise ContractError("layer-input cache binding differs from the requested prefix")
        self._row_locations: dict[int, tuple[Path, int]] = {}
        for shard in self.manifest.get("shards", []):
            path = self.cache_dir / str(shard.get("path", ""))
            if not path.is_file() or (
                verification == "full"
                and file_sha256(path) != shard.get("sha256")
            ):
                raise ContractError(f"layer-input cache shard hash mismatch: {path}")
            rows = shard.get("row_indices", [])
            if verification == "full":
                with safe_open(path, framework="pt", device="cpu") as handle:
                    stored_rows = [
                        int(value)
                        for value in handle.get_tensor("row_indices").tolist()
                    ]
                if stored_rows != [int(value) for value in rows]:
                    raise ContractError(
                        f"layer-input cache shard row mapping mismatch: {path}"
                    )
            for offset, row_index in enumerate(rows):
                row_index = int(row_index)
                if row_index in self._row_locations:
                    raise ContractError("layer-input cache contains duplicate row indices")
                self._row_locations[row_index] = (path, offset)
        if set(self._row_locations) != set(range(int(self.manifest.get("row_count", -1)))):
            raise ContractError("layer-input cache manifest row coverage is incomplete")

    @property
    def row_count(self) -> int:
        return int(self.manifest["row_count"])

    def read_rows(self, row_indices: Iterable[int]) -> LayerInputBatch:
        requested = tuple(int(value) for value in row_indices)
        if not requested:
            raise ContractError("layer-input cache read requires at least one row")
        by_path: dict[Path, list[tuple[int, int]]] = {}
        for output_offset, row_index in enumerate(requested):
            try:
                path, shard_offset = self._row_locations[row_index]
            except KeyError as error:
                raise ContractError(f"layer-input cache row is out of range: {row_index}") from error
            by_path.setdefault(path, []).append((output_offset, shard_offset))
        hidden_batch: torch.Tensor | None = None
        shared_batch: torch.Tensor | None = None
        for path, offsets in by_path.items():
            hidden, shared = self._load_shard(path)
            if hidden_batch is None:
                hidden_batch = torch.empty(
                    (len(requested), *hidden.shape[1:]),
                    dtype=hidden.dtype,
                    pin_memory=self.pin_memory,
                )
                if self.manifest.get("has_shared_states"):
                    if shared is None:
                        raise ContractError(
                            "layer-input cache shard lacks declared shared states"
                        )
                    shared_batch = torch.empty(
                        (len(requested), *shared.shape[1:]),
                        dtype=shared.dtype,
                        pin_memory=self.pin_memory,
                    )
            output_offsets = torch.tensor(
                [value[0] for value in offsets], dtype=torch.int64
            )
            shard_offsets = torch.tensor(
                [value[1] for value in offsets], dtype=torch.int64
            )
            hidden_batch.index_copy_(
                0, output_offsets, hidden.index_select(0, shard_offsets)
            )
            if shared_batch is not None:
                if shared is None:
                    raise ContractError(
                        "layer-input cache shard lacks declared shared states"
                    )
                shared_batch.index_copy_(
                    0, output_offsets, shared.index_select(0, shard_offsets)
                )
        if hidden_batch is None:
            raise ContractError("layer-input cache read failed to materialize hidden rows")
        return LayerInputBatch(
            torch.tensor(requested, dtype=torch.int64),
            hidden_batch,
            shared_batch,
        )

    def _load_shard(
        self, path: Path
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cached = self._shard_tensors.pop(path, None)
        if cached is not None:
            self._shard_tensors[path] = cached
            return cached
        with safe_open(path, framework="pt", device="cpu") as handle:
            hidden = handle.get_tensor("hidden_states")
            shared = (
                handle.get_tensor("shared_states")
                if self.manifest.get("has_shared_states")
                else None
            )
        value = (hidden, shared)
        value_bytes = hidden.numel() * hidden.element_size()
        if shared is not None:
            value_bytes += shared.numel() * shared.element_size()
        if value_bytes > self.max_cached_bytes:
            return value
        while self._shard_tensors and (
            len(self._shard_tensors) >= self.max_cached_shards
            or self._cached_bytes + value_bytes > self.max_cached_bytes
        ):
            oldest_path = next(iter(self._shard_tensors))
            evicted = self._shard_tensors.pop(oldest_path)
            self._cached_bytes -= sum(
                tensor.numel() * tensor.element_size()
                for tensor in evicted
                if tensor is not None
            )
        self._shard_tensors[path] = value
        self._cached_bytes += value_bytes
        return value


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
