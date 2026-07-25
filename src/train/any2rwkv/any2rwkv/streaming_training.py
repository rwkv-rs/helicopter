from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

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
    parameter_signature: tuple[tuple[str, tuple[int, ...], str], ...]
    accumulation_weight: float = 0.0
    master_parameters: tuple[Tensor, ...] = ()
    trainable_names: tuple[str, ...] = ()
    relative_update_exempt_names: tuple[str, ...] = ()
    optimizer_contract: tuple[tuple[str, object], ...] = ()


class ActiveLayerOptimizer:
    """Own exactly one layer optimizer and release all state at layer switches."""

    def __init__(
        self,
        *,
        learning_rate: float,
        optimizer_name: str = "adamw",
        final_learning_rate: float | None = None,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        detailed_telemetry_interval_steps: int = 1,
        learning_rate_schedule: str = "constant",
        warmup_steps: int = 0,
        total_steps: int | None = None,
        min_learning_rate_ratio: float = 1.0,
        gradient_clip_norm: float | None = None,
        max_parameter_update_relative_l2: float | None = None,
        gradient_sync: Callable[[nn.Module], None] | None = None,
    ) -> None:
        if optimizer_name != "adamw":
            raise ContractError(
                "optimizer_name must be adamw; Muon requires a separate calibrated A/B path"
            )
        if learning_rate <= 0:
            raise ContractError("learning_rate must be positive")
        if final_learning_rate is None:
            final_learning_rate = learning_rate * min_learning_rate_ratio
        if not 0 < final_learning_rate <= learning_rate:
            raise ContractError(
                "final_learning_rate must be positive and no larger than learning_rate"
            )
        if (
            len(adam_betas) != 2
            or not 0 <= adam_betas[0] < 1
            or not 0 <= adam_betas[1] < 1
            or not math.isfinite(adam_epsilon)
            or adam_epsilon <= 0
            or not math.isfinite(weight_decay)
            or weight_decay < 0
        ):
            raise ContractError("AdamW betas, epsilon, or weight_decay are invalid")
        if detailed_telemetry_interval_steps <= 0:
            raise ContractError("detailed_telemetry_interval_steps must be positive")
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.final_learning_rate = final_learning_rate
        self.adam_betas = tuple(float(value) for value in adam_betas)
        self.adam_epsilon = float(adam_epsilon)
        self.weight_decay = float(weight_decay)
        self.detailed_telemetry_interval_steps = detailed_telemetry_interval_steps
        if learning_rate_schedule not in {
            "constant",
            "warmup-constant",
            "warmup-cosine",
        }:
            raise ContractError("learning_rate_schedule is invalid")
        if (
            warmup_steps < 0
            or not 0 < min_learning_rate_ratio <= 1
            or (
                learning_rate_schedule in {"warmup-constant", "warmup-cosine"}
                and (total_steps is None or total_steps <= warmup_steps)
            )
        ):
            raise ContractError("learning-rate schedule bounds are invalid")
        if learning_rate_schedule in {
            "constant",
            "warmup-constant",
        } and not math.isclose(
            final_learning_rate,
            learning_rate,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ContractError(
                "constant and warmup-constant schedules require identical initial/final learning rates"
            )
        if learning_rate_schedule in {"constant", "warmup-constant"}:
            final_learning_rate = learning_rate
            self.final_learning_rate = learning_rate
        self.learning_rate_schedule = learning_rate_schedule
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_learning_rate_ratio = final_learning_rate / learning_rate
        if gradient_clip_norm is not None and (
            not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0
        ):
            raise ContractError("gradient_clip_norm must be finite and positive")
        if max_parameter_update_relative_l2 is not None and (
            not math.isfinite(max_parameter_update_relative_l2)
            or max_parameter_update_relative_l2 <= 0
        ):
            raise ContractError(
                "max_parameter_update_relative_l2 must be finite and positive"
            )
        self.gradient_clip_norm = gradient_clip_norm
        self.max_parameter_update_relative_l2_limit = max_parameter_update_relative_l2
        self.gradient_sync = gradient_sync
        self.layer_index: int | None = None
        self.module: nn.Module | None = None
        self.module_parameters: tuple[nn.Parameter, ...] = ()
        self.master_parameters: nn.ParameterList | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.micro_step = 0
        self.optimizer_step = 0
        self.accumulation_step = 0
        self.accumulation_weight = 0.0
        self.last_gradient_l2 = 0.0
        self.max_gradient_l2 = 0.0
        self.max_parameter_gradient_l2 = 0.0
        self.worst_gradient_parameter: str | None = None
        self.last_learning_rate = 0.0
        self.max_parameter_update_l2 = 0.0
        self.worst_absolute_update_parameter: str | None = None
        self.max_parameter_update_relative_l2 = 0.0
        self.worst_update_parameter: str | None = None
        self.last_applied_gradient_l2 = 0.0
        self.max_applied_gradient_l2 = 0.0
        self.gradient_clip_count = 0
        self.min_gradient_clip_scale = 1.0
        self.max_applied_parameter_update_relative_l2 = 0.0
        self.update_trust_region_count = 0
        self.min_update_trust_scale = 1.0
        self.relative_update_exempt_names: frozenset[str] = frozenset()
        self.max_relative_update_exempt_l2 = 0.0
        self.worst_relative_update_exempt_parameter: str | None = None

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
            raise ContractError(
                "release the current active layer before activating another"
            )
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
            parameter.requires_grad_(trainable_names is None or name in trainable_names)
        active_trainable_names = tuple(
            sorted(
                name
                for name, parameter in module.named_parameters()
                if parameter.requires_grad
            )
        )
        snapshot_trainable_names = tuple(
            getattr(snapshot, "trainable_names", ()) if snapshot is not None else ()
        )
        if (
            snapshot_trainable_names
            and snapshot_trainable_names != active_trainable_names
        ):
            raise ContractError(
                "optimizer snapshot trainable parameter set differs from the active layer"
            )
        module_parameters = tuple(module.parameters())
        master_parameters = nn.ParameterList(
            [
                nn.Parameter(
                    parameter.detach().float().clone(),
                    requires_grad=parameter.requires_grad,
                )
                for parameter in module_parameters
            ]
        )
        optimizer = torch.optim.AdamW(
            master_parameters.parameters(),
            lr=self.learning_rate,
            betas=self.adam_betas,
            eps=self.adam_epsilon,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, self._learning_rate_multiplier
        )
        self.layer_index = layer_index
        self.module = module
        self.module_parameters = module_parameters
        self.master_parameters = master_parameters
        self.optimizer = optimizer
        self.scheduler = scheduler
        zero_initialized_names = frozenset(
            name
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
            and not bool(torch.count_nonzero(parameter.detach()).item())
        )
        snapshot_exempt_names = frozenset(
            getattr(snapshot, "relative_update_exempt_names", ())
            if snapshot is not None
            else ()
        )
        unknown_exempt_names = snapshot_exempt_names - set(active_trainable_names)
        if unknown_exempt_names:
            raise ContractError(
                "optimizer snapshot relative-update exemptions differ from the active layer"
            )
        self.relative_update_exempt_names = (
            snapshot_exempt_names if snapshot_exempt_names else zero_initialized_names
        )
        if snapshot is None:
            return
        snapshot_contract = tuple(getattr(snapshot, "optimizer_contract", ()))
        if snapshot_contract and snapshot_contract != self.optimizer_contract:
            raise ContractError(
                "optimizer snapshot contract differs from the active optimizer"
            )
        if snapshot.layer_index != layer_index:
            raise ContractError("optimizer snapshot belongs to a different layer")
        parameter_signature = _parameter_signature(module)
        if snapshot.parameter_signature != parameter_signature:
            raise ContractError(
                "optimizer snapshot parameter signature differs from the active layer"
            )
        optimizer.load_state_dict(snapshot.optimizer)
        scheduler.load_state_dict(snapshot.scheduler)
        if snapshot.master_parameters:
            if len(snapshot.master_parameters) != len(master_parameters):
                raise ContractError(
                    "optimizer snapshot FP32 master parameter count differs from the active layer"
                )
            with torch.no_grad():
                for saved, master in zip(
                    snapshot.master_parameters, master_parameters, strict=True
                ):
                    if saved.shape != master.shape:
                        raise ContractError(
                            "optimizer snapshot FP32 master parameter shape differs from the active layer"
                        )
                    master.copy_(saved.to(device=master.device, dtype=torch.float32))
        self.micro_step = snapshot.micro_step
        self.optimizer_step = snapshot.optimizer_step
        self.accumulation_step = snapshot.accumulation_step
        self.accumulation_weight = float(
            getattr(snapshot, "accumulation_weight", snapshot.accumulation_step)
        )
        for parameter, gradient in zip(
            module.parameters(), snapshot.gradients, strict=True
        ):
            parameter.grad = None if gradient is None else gradient.to(parameter.device)

    def backward(
        self,
        loss: Tensor,
        *,
        accumulation_steps: int,
        sample_weight: float = 1.0,
    ) -> bool:
        if not self.is_active or self.optimizer is None or self.module is None:
            raise ContractError("no active layer optimizer")
        if accumulation_steps <= 0:
            raise ContractError("accumulation_steps must be positive")
        if sample_weight <= 0:
            raise ContractError("sample_weight must be positive")
        if self.accumulation_step == 0:
            self._zero_grad()
            self.accumulation_weight = 0.0
        (loss * sample_weight).backward()
        self.accumulation_step += 1
        self.accumulation_weight += float(sample_weight)
        self.micro_step += 1
        if self.accumulation_step < accumulation_steps:
            return False
        self._normalize_accumulated_gradients()
        self._copy_gradients_to_master()
        if self.gradient_sync is not None:
            assert self.master_parameters is not None
            self.gradient_sync(self.master_parameters)
        self._step_with_telemetry()
        self._copy_master_weights_to_module()
        assert self.scheduler is not None
        self.scheduler.step()
        self.optimizer_step += 1
        self.accumulation_step = 0
        self.accumulation_weight = 0.0
        self._zero_grad()
        return True

    def _learning_rate_multiplier(self, step: int) -> float:
        if self.learning_rate_schedule == "constant":
            return 1.0
        assert self.total_steps is not None
        if self.warmup_steps and step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        if self.learning_rate_schedule == "warmup-constant":
            return 1.0
        decay_updates = self.total_steps - self.warmup_steps
        if decay_updates == 1:
            return self.min_learning_rate_ratio
        progress = min(max(step - self.warmup_steps, 0) / (decay_updates - 1), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return (
            self.min_learning_rate_ratio + (1 - self.min_learning_rate_ratio) * cosine
        )

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
                None
                if parameter.grad is None
                else parameter.grad.detach().cpu().clone()
                for parameter in self.module.parameters()
            ),
            self.micro_step,
            self.optimizer_step,
            self.accumulation_step,
            _parameter_signature(self.module),
            self.accumulation_weight,
            tuple(
                parameter.detach().cpu().clone()
                for parameter in self.master_parameters or ()
            ),
            tuple(
                sorted(
                    name
                    for name, parameter in self.module.named_parameters()
                    if parameter.requires_grad
                )
            ),
            tuple(sorted(self.relative_update_exempt_names)),
            self.optimizer_contract,
        )
        self.module.requires_grad_(False)
        for parameter in self.module.parameters():
            parameter.grad = None
        self.layer_index = None
        self.module = None
        self.module_parameters = ()
        self.master_parameters = None
        self.optimizer = None
        self.scheduler = None
        self.relative_update_exempt_names = frozenset()
        return snapshot

    def snapshot(self) -> ActiveLayerOptimizerSnapshot:
        """Copy durable state without releasing the active layer."""
        if (
            self.layer_index is None
            or self.module is None
            or self.optimizer is None
            or self.scheduler is None
        ):
            raise ContractError("no active layer optimizer")
        return ActiveLayerOptimizerSnapshot(
            self.layer_index,
            _to_cpu(self.optimizer.state_dict()),
            _to_cpu(self.scheduler.state_dict()),
            tuple(
                None
                if parameter.grad is None
                else parameter.grad.detach().cpu().clone()
                for parameter in self.module.parameters()
            ),
            self.micro_step,
            self.optimizer_step,
            self.accumulation_step,
            _parameter_signature(self.module),
            self.accumulation_weight,
            tuple(
                parameter.detach().cpu().clone()
                for parameter in self.master_parameters or ()
            ),
            tuple(
                sorted(
                    name
                    for name, parameter in self.module.named_parameters()
                    if parameter.requires_grad
                )
            ),
            tuple(sorted(self.relative_update_exempt_names)),
            self.optimizer_contract,
        )

    @property
    def optimizer_contract(self) -> tuple[tuple[str, object], ...]:
        """Stable optimizer identity saved with every resumable generation."""
        return (
            ("name", self.optimizer_name),
            ("learning_rate", self.learning_rate),
            ("final_learning_rate", self.final_learning_rate),
            ("learning_rate_schedule", self.learning_rate_schedule),
            ("warmup_steps", self.warmup_steps),
            ("total_steps", self.total_steps),
            ("betas", self.adam_betas),
            ("epsilon", self.adam_epsilon),
            ("weight_decay", self.weight_decay),
            ("gradient_clip_norm", self.gradient_clip_norm),
            (
                "max_parameter_update_relative_l2",
                self.max_parameter_update_relative_l2_limit,
            ),
            ("relative_update_exempt_policy", "zero-initialized-trainable-v1"),
            (
                "detailed_telemetry_interval_steps",
                self.detailed_telemetry_interval_steps,
            ),
        )

    def flush(self, *, accumulation_steps: int) -> bool:
        """Commit a partial final accumulation without replaying epoch rows."""
        if not self.is_active or self.optimizer is None or self.module is None:
            raise ContractError("no active layer optimizer")
        if accumulation_steps <= 0:
            raise ContractError("accumulation_steps must be positive")
        if self.accumulation_step == 0:
            return False
        self._normalize_accumulated_gradients()
        self._copy_gradients_to_master()
        if self.gradient_sync is not None:
            assert self.master_parameters is not None
            self.gradient_sync(self.master_parameters)
        self._step_with_telemetry()
        self._copy_master_weights_to_module()
        assert self.scheduler is not None
        self.scheduler.step()
        self.optimizer_step += 1
        self.accumulation_step = 0
        self.accumulation_weight = 0.0
        self._zero_grad()
        return True

    @torch.no_grad()
    def _step_with_telemetry(self) -> None:
        if (
            self.optimizer is None
            or self.module is None
            or self.master_parameters is None
        ):
            raise ContractError("active optimizer is unavailable")
        active = [
            (name, master)
            for (name, _), master in zip(
                self.module.named_parameters(), self.master_parameters, strict=True
            )
            if master.grad is not None
        ]
        gradients = [master.grad for _, master in active]
        self.last_gradient_l2 = float(
            torch.nn.utils.clip_grad_norm_(
                self.master_parameters,
                max_norm=(
                    self.gradient_clip_norm
                    if self.gradient_clip_norm is not None
                    else float("inf")
                ),
                error_if_nonfinite=True,
                foreach=True,
            ).item()
        )
        if self.last_gradient_l2 == 0:
            raise ContractError("active streamed layer received no nonzero gradient")
        self.max_gradient_l2 = max(self.max_gradient_l2, self.last_gradient_l2)
        gradient_clip_scale = 1.0
        if (
            self.gradient_clip_norm is not None
            and self.last_gradient_l2 > self.gradient_clip_norm
        ):
            gradient_clip_scale = self.gradient_clip_norm / self.last_gradient_l2
            self.gradient_clip_count += 1
            self.min_gradient_clip_scale = min(
                self.min_gradient_clip_scale, gradient_clip_scale
            )
        self.last_applied_gradient_l2 = self.last_gradient_l2 * gradient_clip_scale
        self.max_applied_gradient_l2 = max(
            self.max_applied_gradient_l2, self.last_applied_gradient_l2
        )
        record_detailed = (
            self.max_parameter_update_relative_l2_limit is not None
            or self.optimizer_step % self.detailed_telemetry_interval_steps == 0
        )
        if record_detailed:
            gradient_norms = torch.stack(
                [gradient.detach().float().norm() for gradient in gradients]
            ).cpu()
            worst_index = int(torch.argmax(gradient_norms).item())
            worst_gradient_l2 = float(gradient_norms[worst_index])
            if worst_gradient_l2 > self.max_parameter_gradient_l2:
                self.max_parameter_gradient_l2 = worst_gradient_l2
                self.worst_gradient_parameter = active[worst_index][0]
        before_values = (
            [master.detach().clone() for _, master in active] if record_detailed else []
        )
        self.last_learning_rate = float(self.optimizer.param_groups[0]["lr"])
        self.optimizer.step()
        if not record_detailed:
            return
        raw_updates: list[tuple[str, nn.Parameter, Tensor, float, float]] = []
        max_raw_relative = 0.0
        for (name, master), before in zip(active, before_values, strict=True):
            update_l2 = float((master.detach() - before).float().norm().item())
            if update_l2 > self.max_parameter_update_l2:
                self.max_parameter_update_l2 = update_l2
                self.worst_absolute_update_parameter = name
            if name in self.relative_update_exempt_names:
                relative = 0.0
                if update_l2 > self.max_relative_update_exempt_l2:
                    self.max_relative_update_exempt_l2 = update_l2
                    self.worst_relative_update_exempt_parameter = name
            else:
                parameter_l2 = max(float(before.float().norm().item()), 1e-30)
                relative = update_l2 / parameter_l2
                if relative > self.max_parameter_update_relative_l2:
                    self.max_parameter_update_relative_l2 = relative
                    self.worst_update_parameter = name
                max_raw_relative = max(max_raw_relative, relative)
            raw_updates.append((name, master, before, update_l2, relative))
        update_trust_scale = 1.0
        if (
            self.max_parameter_update_relative_l2_limit is not None
            and max_raw_relative > self.max_parameter_update_relative_l2_limit
        ):
            update_trust_scale = (
                self.max_parameter_update_relative_l2_limit / max_raw_relative
            )
            for _, master, before, _, _ in raw_updates:
                master.copy_(before + (master.detach() - before) * update_trust_scale)
            self.update_trust_region_count += 1
            self.min_update_trust_scale = min(
                self.min_update_trust_scale, update_trust_scale
            )
        if raw_updates:
            self.max_applied_parameter_update_relative_l2 = max(
                self.max_applied_parameter_update_relative_l2,
                max(relative * update_trust_scale for *_, relative in raw_updates),
            )

    def telemetry(self) -> dict[str, float | int | str | None]:
        return {
            "optimizer_name": self.optimizer_name,
            "configured_learning_rate": self.learning_rate,
            "final_learning_rate": self.final_learning_rate,
            "adam_beta1": self.adam_betas[0],
            "adam_beta2": self.adam_betas[1],
            "adam_epsilon": self.adam_epsilon,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "detailed_telemetry_interval_steps": (
                self.detailed_telemetry_interval_steps
            ),
            "last_gradient_l2": self.last_gradient_l2,
            "max_gradient_l2": self.max_gradient_l2,
            "last_applied_gradient_l2": self.last_applied_gradient_l2,
            "max_applied_gradient_l2": self.max_applied_gradient_l2,
            "gradient_clip_norm": self.gradient_clip_norm,
            "gradient_clip_count": self.gradient_clip_count,
            "min_gradient_clip_scale": self.min_gradient_clip_scale,
            "max_parameter_gradient_l2": self.max_parameter_gradient_l2,
            "worst_gradient_parameter": self.worst_gradient_parameter,
            "last_learning_rate": self.last_learning_rate,
            "max_parameter_update_l2": self.max_parameter_update_l2,
            "worst_absolute_update_parameter": (self.worst_absolute_update_parameter),
            "max_parameter_update_relative_l2": (self.max_parameter_update_relative_l2),
            "max_parameter_update_relative_l2_limit": (
                self.max_parameter_update_relative_l2_limit
            ),
            "max_applied_parameter_update_relative_l2": (
                self.max_applied_parameter_update_relative_l2
            ),
            "update_trust_region_count": self.update_trust_region_count,
            "min_update_trust_scale": self.min_update_trust_scale,
            "worst_update_parameter": self.worst_update_parameter,
            "relative_update_exempt_parameter_count": len(
                self.relative_update_exempt_names
            ),
            "max_relative_update_exempt_l2": self.max_relative_update_exempt_l2,
            "worst_relative_update_exempt_parameter": (
                self.worst_relative_update_exempt_parameter
            ),
        }

    def _normalize_accumulated_gradients(self) -> None:
        if self.module is None or self.accumulation_weight <= 0:
            raise ContractError("active optimizer has no accumulated sample weight")
        scale = 1.0 / self.accumulation_weight
        for parameter in self.module.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)

    def _copy_gradients_to_master(self) -> None:
        if self.master_parameters is None:
            raise ContractError("active optimizer has no FP32 master parameters")
        for parameter, master in zip(
            self.module_parameters, self.master_parameters, strict=True
        ):
            master.grad = (
                None
                if parameter.grad is None
                else parameter.grad.detach().float().clone()
            )

    @torch.no_grad()
    def _copy_master_weights_to_module(self) -> None:
        if self.master_parameters is None:
            raise ContractError("active optimizer has no FP32 master parameters")
        for parameter, master in zip(
            self.module_parameters, self.master_parameters, strict=True
        ):
            parameter.copy_(master.to(dtype=parameter.dtype))

    def _zero_grad(self) -> None:
        if self.optimizer is None:
            raise ContractError("active optimizer is unavailable")
        self.optimizer.zero_grad(set_to_none=True)
        for parameter in self.module_parameters:
            parameter.grad = None


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


def _parameter_signature(
    module: nn.Module,
) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    return tuple(
        (name, tuple(parameter.shape), str(parameter.dtype))
        for name, parameter in module.named_parameters()
    )
