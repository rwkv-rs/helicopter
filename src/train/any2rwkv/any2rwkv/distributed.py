from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist
from torch import nn

from .errors import ContractError


GRADIENT_BUCKET_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int

    @classmethod
    def initialize(cls) -> "DistributedContext":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world_size == 1:
            return cls(rank=0, local_rank=0, world_size=1)
        if world_size != 8:
            raise ContractError(
                f"Any2RWKV distributed layer-major training requires world_size=8, found {world_size}"
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
            raise ContractError("8-GPU layer-major training requires eight visible CUDA devices")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                device_id=torch.device("cuda", local_rank),
            )
        return cls(rank=rank, local_rank=local_rank, world_size=world_size)

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.local_rank) if self.world_size > 1 else torch.device("cuda")

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier(device_ids=[self.local_rank])

    def synchronize_gradients(self, module: nn.Module) -> None:
        if self.world_size == 1:
            return
        gradients = []
        for parameter in module.parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            gradients.append(parameter.grad)
        for bucket in _gradient_buckets(gradients, GRADIENT_BUCKET_BYTES):
            flat = torch.cat([gradient.reshape(-1) for gradient in bucket])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(self.world_size)
            offset = 0
            for gradient in bucket:
                count = gradient.numel()
                gradient.copy_(flat[offset : offset + count].view_as(gradient))
                offset += count

    def synchronize_module_parameters(self, module: nn.Module) -> None:
        """Fail closed on structural drift, then copy rank-0 parameters to all ranks."""
        if self.world_size == 1:
            return
        signature = tuple(
            (name, tuple(parameter.shape), str(parameter.dtype))
            for name, parameter in module.named_parameters()
        )
        signatures = self.all_gather_objects(signature)
        if any(candidate != signatures[0] for candidate in signatures[1:]):
            raise ContractError("distributed module parameter signatures differ across ranks")
        for parameter in module.parameters():
            dist.broadcast(parameter.data, src=0)

    def validate_trainable_signature(self, module: nn.Module) -> None:
        """Ensure every rank will issue the same ordered gradient collectives."""
        if self.world_size == 1:
            return
        signature = tuple(
            (name, tuple(parameter.shape), str(parameter.dtype))
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        )
        signatures = self.all_gather_objects(signature)
        if any(candidate != signatures[0] for candidate in signatures[1:]):
            raise ContractError("distributed trainable parameter signatures differ across ranks")

    def aggregate_metrics(
        self, metrics: Mapping[str, float], sample_count: int
    ) -> dict[str, float]:
        if sample_count <= 0:
            raise ContractError("distributed metric shard must contain at least one sample")
        if self.world_size == 1:
            return dict(metrics)
        keys = tuple(sorted(metrics))
        values = torch.tensor(
            [sample_count, *(sample_count * float(metrics[key]) for key in keys)],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        total = float(values[0].item())
        return {key: float(values[index + 1].item() / total) for index, key in enumerate(keys)}

    def gather_scalar_metrics(
        self, metrics: Mapping[str, float]
    ) -> dict[str, tuple[float, ...]]:
        """Collect rank-local telemetry without changing training semantics."""
        keys = tuple(sorted(metrics))
        if self.world_size == 1:
            return {key: (float(metrics[key]),) for key in keys}
        local = torch.tensor(
            [float(metrics[key]) for key in keys],
            dtype=torch.float64,
            device=self.device,
        )
        gathered = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(gathered, local)
        return {
            key: tuple(float(row[index].item()) for row in gathered)
            for index, key in enumerate(keys)
        }

    def all_reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        """Sum an additive sufficient statistic across every training rank."""
        if self.world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def broadcast_tensor(self, value: torch.Tensor, *, source_rank: int = 0) -> torch.Tensor:
        """Broadcast an already allocated tensor without Python object serialization."""
        if self.world_size > 1:
            dist.broadcast(value, src=source_rank)
        return value

    def broadcast_object(self, value, *, source_rank: int = 0):
        values = [value if self.rank == source_rank else None]
        if self.world_size > 1:
            dist.broadcast_object_list(values, src=source_rank)
        return values[0]

    def all_gather_objects(self, value) -> tuple[object, ...]:
        if self.world_size == 1:
            return (value,)
        values: list[object | None] = [None] * self.world_size
        dist.all_gather_object(values, value)
        return tuple(values)

    def broadcast_path(self, path: Path | None) -> Path:
        values = [str(path) if self.is_primary and path is not None else None]
        if self.world_size > 1:
            dist.broadcast_object_list(values, src=0)
        if not values[0]:
            raise ContractError("primary rank did not publish a generation path")
        return Path(str(values[0]))

    def shard_rows(self, rows: tuple[int, ...]) -> tuple[int, ...]:
        shard = rows[self.rank :: self.world_size]
        if not shard:
            raise ContractError(
                "global distributed micro-batch must provide at least one row per rank"
            )
        return shard

    def close(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()


def _gradient_buckets(
    gradients: list[torch.Tensor], max_bytes: int
) -> tuple[tuple[torch.Tensor, ...], ...]:
    if max_bytes <= 0:
        raise ContractError("gradient bucket size must be positive")
    buckets: list[tuple[torch.Tensor, ...]] = []
    current: list[torch.Tensor] = []
    current_bytes = 0
    current_key: tuple[torch.device, torch.dtype] | None = None
    for gradient in gradients:
        key = (gradient.device, gradient.dtype)
        size = gradient.numel() * gradient.element_size()
        if current and (key != current_key or current_bytes + size > max_bytes):
            buckets.append(tuple(current))
            current = []
            current_bytes = 0
        current.append(gradient)
        current_bytes += size
        current_key = key
    if current:
        buckets.append(tuple(current))
    return tuple(buckets)
