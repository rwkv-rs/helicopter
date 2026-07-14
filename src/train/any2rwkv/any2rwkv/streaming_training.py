from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .errors import ContractError


@dataclass(frozen=True)
class ActiveLayerOptimizerSnapshot:
    layer_index: int
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    gradients: tuple[Tensor | None, ...]
    micro_step: int
    optimizer_step: int
    accumulation_step: int


class ActiveLayerOptimizer:
    """Own exactly one layer optimizer and release all state at layer switches."""

    def __init__(self, *, learning_rate: float) -> None:
        if learning_rate <= 0:
            raise ContractError("learning_rate must be positive")
        self.learning_rate = learning_rate
        self.layer_index: int | None = None
        self.module: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.micro_step = 0
        self.optimizer_step = 0
        self.accumulation_step = 0

    @property
    def is_active(self) -> bool:
        return self.module is not None

    def activate(
        self,
        layer_index: int,
        module: nn.Module,
        *,
        snapshot: ActiveLayerOptimizerSnapshot | None = None,
        trainable_names: set[str] | None = None,
    ) -> None:
        if self.is_active:
            raise ContractError("release the current active layer before activating another")
        if layer_index < 0:
            raise ContractError("layer_index must be non-negative")
        known_names = {name for name, _ in module.named_parameters()}
        if trainable_names is not None:
            unknown = trainable_names - known_names
            if unknown or not trainable_names:
                raise ContractError(
                    f"invalid streamed trainable parameter names: {sorted(unknown)}"
                )
        for name, parameter in module.named_parameters():
            parameter.requires_grad_(
                trainable_names is None or name in trainable_names
            )
        optimizer = torch.optim.AdamW(module.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        self.layer_index = layer_index
        self.module = module
        self.optimizer = optimizer
        self.scheduler = scheduler
        if snapshot is None:
            return
        if snapshot.layer_index != layer_index:
            raise ContractError("optimizer snapshot belongs to a different layer")
        optimizer.load_state_dict(snapshot.optimizer)
        scheduler.load_state_dict(snapshot.scheduler)
        self.micro_step = snapshot.micro_step
        self.optimizer_step = snapshot.optimizer_step
        self.accumulation_step = snapshot.accumulation_step
        for parameter, gradient in zip(
            module.parameters(), snapshot.gradients, strict=True
        ):
            parameter.grad = None if gradient is None else gradient.to(parameter.device)

    def backward(self, loss: Tensor, *, accumulation_steps: int) -> bool:
        if not self.is_active or self.optimizer is None or self.module is None:
            raise ContractError("no active layer optimizer")
        if accumulation_steps <= 0:
            raise ContractError("accumulation_steps must be positive")
        if self.accumulation_step == 0:
            self.optimizer.zero_grad(set_to_none=True)
        (loss / accumulation_steps).backward()
        if not any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad)
            for parameter in self.module.parameters()
        ):
            raise ContractError("active streamed layer received no nonzero gradient")
        self.accumulation_step += 1
        self.micro_step += 1
        if self.accumulation_step < accumulation_steps:
            return False
        self.optimizer.step()
        assert self.scheduler is not None
        self.scheduler.step()
        self.optimizer_step += 1
        self.accumulation_step = 0
        return True

    def release(self) -> ActiveLayerOptimizerSnapshot:
        if (
            self.layer_index is None
            or self.module is None
            or self.optimizer is None
            or self.scheduler is None
        ):
            raise ContractError("no active layer optimizer")
        snapshot = ActiveLayerOptimizerSnapshot(
            self.layer_index,
            _to_cpu(self.optimizer.state_dict()),
            _to_cpu(self.scheduler.state_dict()),
            tuple(
                None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                for parameter in self.module.parameters()
            ),
            self.micro_step,
            self.optimizer_step,
            self.accumulation_step,
        )
        self.module.requires_grad_(False)
        for parameter in self.module.parameters():
            parameter.grad = None
        self.layer_index = None
        self.module = None
        self.optimizer = None
        self.scheduler = None
        return snapshot


def _to_cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value
