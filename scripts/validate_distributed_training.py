#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json

import torch
import torch.distributed as dist
from torch import nn

from any2rwkv.distributed import DistributedContext
from any2rwkv.streaming_training import ActiveLayerOptimizer


def digest(module: nn.Module) -> str:
    value = hashlib.sha256()
    for parameter in module.parameters():
        value.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def main() -> None:
    context = DistributedContext.initialize()
    torch.manual_seed(20260714)
    active = nn.Linear(4, 4, bias=False, device=context.device)
    inactive = nn.Linear(4, 4, bias=False, device=context.device)
    inactive_before = digest(inactive)
    optimizer = ActiveLayerOptimizer(
        learning_rate=1e-3,
        gradient_sync=context.synchronize_gradients,
    )
    optimizer.activate(0, active)
    row_id = context.rank
    inputs = torch.full((1, 4), float(row_id + 1), device=context.device)
    targets = torch.full((1, 4), float((row_id + 1) * 2), device=context.device)
    loss = (active(inputs) - targets).square().mean()
    optimizer.backward(loss, accumulation_steps=1)

    active_digest = digest(active)
    inactive_after = digest(inactive)
    gathered: list[dict[str, object] | None] = [None] * context.world_size
    dist.all_gather_object(
        gathered,
        {
            "rank": context.rank,
            "row_id": row_id,
            "active_digest": active_digest,
            "inactive_before": inactive_before,
            "inactive_after": inactive_after,
        },
    )
    metric = context.aggregate_metrics({"row_value": float(row_id)}, 1)
    if context.is_primary:
        rows = [entry for entry in gathered if entry is not None]
        if sorted(int(entry["row_id"]) for entry in rows) != list(range(8)):
            raise SystemExit("distributed row shards do not cover ranks 0..7 exactly once")
        if len({str(entry["active_digest"]) for entry in rows}) != 1:
            raise SystemExit("all-reduced active layer parameters differ across ranks")
        if any(entry["inactive_before"] != entry["inactive_after"] for entry in rows):
            raise SystemExit("inactive layer changed during distributed active-layer step")
        if metric["row_value"] != 3.5:
            raise SystemExit(f"distributed metric reduction is wrong: {metric}")
        print(
            json.dumps(
                {
                    "status": "passed",
                    "world_size": context.world_size,
                    "row_coverage": list(range(8)),
                    "active_digest": rows[0]["active_digest"],
                    "inactive_unchanged": True,
                    "metric_mean": metric["row_value"],
                },
                sort_keys=True,
            )
        )
    context.barrier()
    context.close()


if __name__ == "__main__":
    main()
