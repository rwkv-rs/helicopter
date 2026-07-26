from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from any2rwkv.core import (
    LayerInputBatch,
    LayerInputCacheReader,
    estimate_layer_input_cache_bytes,
    prepare_distributed_layer_input_cache,
    publish_distributed_layer_input_cache,
    write_distributed_layer_input_cache_partition,
    write_layer_input_cache,
)
from any2rwkv.errors import ContractError


def test_cache_roundtrip_preserves_requested_row_order_and_shared_states(tmp_path: Path) -> None:
    hidden = torch.arange(5 * 3 * 4, dtype=torch.float32).reshape(5, 3, 4)
    shared = hidden + 100
    cache_dir = write_layer_input_cache(
        tmp_path / "layer-002-train",
        layer_index=2,
        split="distill_train",
        row_count=5,
        sequence_length=3,
        hidden_size=4,
        binding={"prefix_fingerprint": "a" * 64},
        batches=(
            LayerInputBatch(torch.tensor([0, 1]), hidden[:2], shared[:2]),
            LayerInputBatch(torch.tensor([2, 3, 4]), hidden[2:], shared[2:]),
        ),
    )
    reader = LayerInputCacheReader(
        cache_dir,
        expected_binding={"prefix_fingerprint": "a" * 64},
        pin_memory=False,
    )
    batch = reader.read_rows([4, 1, 3])
    torch.testing.assert_close(batch.hidden_states, hidden[[4, 1, 3]].bfloat16())
    torch.testing.assert_close(batch.shared_states, shared[[4, 1, 3]].bfloat16())
    assert batch.row_indices.tolist() == [4, 1, 3]
    cached_shards = len(reader._shard_tensors)
    repeated = reader.read_rows([1, 4])
    torch.testing.assert_close(repeated.hidden_states, hidden[[1, 4]].bfloat16())
    assert len(reader._shard_tensors) == cached_shards


def test_cache_rejects_incomplete_coverage_and_binding_or_hash_drift(tmp_path: Path) -> None:
    hidden = torch.zeros(1, 2, 3)
    with pytest.raises(ContractError, match="does not cover every row"):
        write_layer_input_cache(
            tmp_path / "incomplete",
            layer_index=0,
            split="validation",
            row_count=2,
            sequence_length=2,
            hidden_size=3,
            binding={"source": "x"},
            batches=(LayerInputBatch(torch.tensor([0]), hidden),),
        )
    cache_dir = write_layer_input_cache(
        tmp_path / "complete",
        layer_index=0,
        split="validation",
        row_count=1,
        sequence_length=2,
        hidden_size=3,
        binding={"source": "x"},
        batches=(LayerInputBatch(torch.tensor([0]), hidden),),
    )
    with pytest.raises(
        ContractError,
        match=r"binding differs from the request; keys=source",
    ):
        LayerInputCacheReader(cache_dir, expected_binding={"source": "y"})
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    shard = cache_dir / manifest["shards"][0]["path"]
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(ContractError, match="hash mismatch"):
        LayerInputCacheReader(cache_dir)


def test_cache_rejects_manifest_row_mapping_that_differs_from_shards(
    tmp_path: Path,
) -> None:
    hidden = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    cache_dir = write_layer_input_cache(
        tmp_path / "row-mapping",
        layer_index=0,
        split="distill_train",
        row_count=4,
        sequence_length=2,
        hidden_size=3,
        binding={"source": "x"},
        batches=(
            LayerInputBatch(torch.tensor([0, 1]), hidden[:2]),
            LayerInputBatch(torch.tensor([2, 3]), hidden[2:]),
        ),
    )
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_rows = manifest["shards"][0]["row_indices"]
    manifest["shards"][0]["row_indices"] = manifest["shards"][1]["row_indices"]
    manifest["shards"][1]["row_indices"] = first_rows
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ContractError, match="row mapping mismatch"):
        LayerInputCacheReader(cache_dir, expected_binding={"source": "x"})


def test_cache_shard_lru_is_bounded_by_bytes_per_rank(tmp_path: Path) -> None:
    hidden = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    cache_dir = write_layer_input_cache(
        tmp_path / "byte-bounded",
        layer_index=0,
        split="distill_train",
        row_count=4,
        sequence_length=2,
        hidden_size=3,
        binding={"source": "x"},
        batches=(
            LayerInputBatch(torch.tensor([0, 1]), hidden[:2]),
            LayerInputBatch(torch.tensor([2, 3]), hidden[2:]),
        ),
    )
    reader = LayerInputCacheReader(
        cache_dir,
        max_cached_shards=256,
        max_cached_bytes=24,
        pin_memory=False,
    )
    reader.read_rows([0, 2])
    assert reader._cached_bytes <= 24
    assert len(reader._shard_tensors) <= 1


def test_capacity_estimate_accounts_for_current_next_shared_state_double_buffer() -> None:
    estimate = estimate_layer_input_cache_bytes(
        row_count=10,
        sequence_length=20,
        hidden_size=30,
        dtype_bytes=2,
        current_has_shared_states=False,
        next_has_shared_states=True,
        reserve_ratio=0.10,
    )
    assert estimate.current_cache_bytes == 12_000
    assert estimate.next_cache_bytes == 24_000
    assert estimate.required_free_bytes == 39_600


def test_distributed_cache_partitions_publish_atomically(tmp_path: Path) -> None:
    destination = tmp_path / "distributed"
    binding = {"prefix_fingerprint": "b" * 64}
    hidden = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    transition = prepare_distributed_layer_input_cache(
        destination,
        world_size=2,
        layer_index=1,
        split="distill_train",
        row_count=4,
        sequence_length=2,
        hidden_size=3,
        has_shared_states=False,
        binding=binding,
    )
    assert transition["state"] == "created-staging"
    for rank, rows in ((0, [0, 1]), (1, [2, 3])):
        write_distributed_layer_input_cache_partition(
            destination,
            rank=rank,
            world_size=2,
            layer_index=1,
            split="distill_train",
            row_count=4,
            sequence_length=2,
            hidden_size=3,
            has_shared_states=False,
            binding=binding,
            batches=(LayerInputBatch(torch.tensor(rows), hidden[rows]),),
        )
        if rank == 0:
            resumed = prepare_distributed_layer_input_cache(
                destination,
                world_size=2,
                layer_index=1,
                split="distill_train",
                row_count=4,
                sequence_length=2,
                hidden_size=3,
                has_shared_states=False,
                binding=binding,
            )
            assert resumed["state"] == "resume-staging"
            reused = write_distributed_layer_input_cache_partition(
                destination,
                rank=0,
                world_size=2,
                layer_index=1,
                split="distill_train",
                row_count=4,
                sequence_length=2,
                hidden_size=3,
                has_shared_states=False,
                binding=binding,
                batches=(_ for _ in () if False),
            )
            assert reused == {"status": "reused", "rank": 0}
    assert not destination.exists()
    publish_distributed_layer_input_cache(
        destination,
        world_size=2,
        layer_index=1,
        split="distill_train",
        row_count=4,
        sequence_length=2,
        hidden_size=3,
        has_shared_states=False,
        binding=binding,
    )
    reader = LayerInputCacheReader(destination, expected_binding=binding)
    torch.testing.assert_close(
        reader.read_rows([3, 0, 2]).hidden_states,
        hidden[[3, 0, 2]].bfloat16(),
    )
    assert reader.manifest["distributed_writer"]["world_size"] == 2


def test_distributed_cache_rejects_partition_identity_drift(tmp_path: Path) -> None:
    destination = tmp_path / "identity-drift"
    binding = {"source": "fixture"}
    hidden = torch.zeros(1, 2, 3)
    prepare_distributed_layer_input_cache(
        destination,
        world_size=2,
        layer_index=0,
        split="validation",
        row_count=2,
        sequence_length=2,
        hidden_size=3,
        has_shared_states=False,
        binding=binding,
    )
    for rank, row in ((0, 0), (1, 1)):
        write_distributed_layer_input_cache_partition(
            destination,
            rank=rank,
            world_size=2,
            layer_index=0,
            split="validation",
            row_count=2,
            sequence_length=2,
            hidden_size=3,
            has_shared_states=False,
            binding=binding,
            batches=(LayerInputBatch(torch.tensor([row]), hidden),),
        )
        if rank == 1:
            staging = next(destination.parent.glob(destination.name + ".txn-*"))
            manifest_path = staging / f"rank-{rank:05d}" / "partition.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["identity"]["binding"] = {"source": "drifted"}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ContractError, match="identity differs"):
        publish_distributed_layer_input_cache(
            destination,
            world_size=2,
            layer_index=0,
            split="validation",
            row_count=2,
            sequence_length=2,
            hidden_size=3,
            has_shared_states=False,
            binding=binding,
        )
    assert not destination.exists()


def test_distributed_cache_rejects_cross_rank_duplicate_rows(tmp_path: Path) -> None:
    destination = tmp_path / "duplicate"
    binding = {"source": "fixture"}
    hidden = torch.zeros(1, 2, 3)
    prepare_distributed_layer_input_cache(
        destination,
        world_size=2,
        layer_index=0,
        split="validation",
        row_count=2,
        sequence_length=2,
        hidden_size=3,
        has_shared_states=False,
        binding=binding,
    )
    for rank in (0, 1):
        write_distributed_layer_input_cache_partition(
            destination,
            rank=rank,
            world_size=2,
            layer_index=0,
            split="validation",
            row_count=2,
            sequence_length=2,
            hidden_size=3,
            has_shared_states=False,
            binding=binding,
            batches=(LayerInputBatch(torch.tensor([rank]), hidden),),
        )
    staging = next(destination.parent.glob(destination.name + ".txn-*"))
    rank_one_manifest = staging / "rank-00001" / "partition.json"
    manifest = json.loads(rank_one_manifest.read_text(encoding="utf-8"))
    manifest["shards"][0]["row_indices"] = [0]
    rank_one_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ContractError, match="shard rows differ"):
        publish_distributed_layer_input_cache(
            destination,
            world_size=2,
            layer_index=0,
            split="validation",
            row_count=2,
            sequence_length=2,
            hidden_size=3,
            has_shared_states=False,
            binding=binding,
        )
    assert not destination.exists()


@pytest.mark.parametrize("row_count", [19, 3])
def test_distributed_cache_supports_eight_uneven_or_empty_rank_partitions(
    tmp_path: Path, row_count: int
) -> None:
    destination = tmp_path / f"eight-ranks-{row_count}"
    binding = {"source": "eight-rank-fixture"}
    hidden = torch.arange(row_count * 2 * 2, dtype=torch.float32).reshape(
        row_count, 2, 2
    )
    prepare_distributed_layer_input_cache(
        destination,
        world_size=8,
        layer_index=2,
        split="distill_train",
        row_count=row_count,
        sequence_length=2,
        hidden_size=2,
        has_shared_states=False,
        binding=binding,
    )
    for rank in range(8):
        start = row_count * rank // 8
        stop = row_count * (rank + 1) // 8
        batches = (
            (LayerInputBatch(torch.arange(start, stop), hidden[start:stop]),)
            if start < stop
            else ()
        )
        write_distributed_layer_input_cache_partition(
            destination,
            rank=rank,
            world_size=8,
            layer_index=2,
            split="distill_train",
            row_count=row_count,
            sequence_length=2,
            hidden_size=2,
            has_shared_states=False,
            binding=binding,
            batches=batches,
        )
    publish_distributed_layer_input_cache(
        destination,
        world_size=8,
        layer_index=2,
        split="distill_train",
        row_count=row_count,
        sequence_length=2,
        hidden_size=2,
        has_shared_states=False,
        binding=binding,
    )
    reader = LayerInputCacheReader(destination, expected_binding=binding)
    torch.testing.assert_close(
        reader.read_rows(reversed(range(row_count))).hidden_states,
        hidden.flip(0).bfloat16(),
    )


def test_distributed_cache_recovers_after_manifest_fsync_before_publish_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "publish-recovery"
    binding = {"source": "publish-recovery"}
    hidden = torch.zeros(2, 2, 2)
    prepare_distributed_layer_input_cache(
        destination,
        world_size=2,
        layer_index=1,
        split="validation",
        row_count=2,
        sequence_length=2,
        hidden_size=2,
        has_shared_states=False,
        binding=binding,
    )
    for rank in (0, 1):
        write_distributed_layer_input_cache_partition(
            destination,
            rank=rank,
            world_size=2,
            layer_index=1,
            split="validation",
            row_count=2,
            sequence_length=2,
            hidden_size=2,
            has_shared_states=False,
            binding=binding,
            batches=(LayerInputBatch(torch.tensor([rank]), hidden[rank : rank + 1]),),
        )
    staging = next(destination.parent.glob(destination.name + ".txn-*"))
    original_rename = Path.rename

    def fail_publish_rename(self: Path, target: Path):
        if self == staging and Path(target) == destination:
            raise OSError("injected publish rename failure")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_publish_rename)
    with pytest.raises(OSError, match="injected publish rename failure"):
        publish_distributed_layer_input_cache(
            destination,
            world_size=2,
            layer_index=1,
            split="validation",
            row_count=2,
            sequence_length=2,
            hidden_size=2,
            has_shared_states=False,
            binding=binding,
        )
    assert (staging / "manifest.json").is_file()
    assert not destination.exists()
    monkeypatch.setattr(Path, "rename", original_rename)
    resumed = prepare_distributed_layer_input_cache(
        destination,
        world_size=2,
        layer_index=1,
        split="validation",
        row_count=2,
        sequence_length=2,
        hidden_size=2,
        has_shared_states=False,
        binding=binding,
    )
    assert resumed["state"] == "resume-staging"
    publish_distributed_layer_input_cache(
        destination,
        world_size=2,
        layer_index=1,
        split="validation",
        row_count=2,
        sequence_length=2,
        hidden_size=2,
        has_shared_states=False,
        binding=binding,
    )
    LayerInputCacheReader(destination, expected_binding=binding)
