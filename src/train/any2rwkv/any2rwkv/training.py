from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from .distill import (
    ActiveLayerTrainer,
    BurnInWindow,
    LossBreakdown,
    LossWeights,
    normalized_mse,
)
from .errors import ContractError
from .hybrid import HybridModelPatcher


@dataclass(frozen=True)
class DistillationBatch:
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor
    teacher_mixer_output: Tensor
    teacher_block_output: Tensor
    teacher_logits: Tensor
    rollout_teacher: Tensor | None = None


class LayerwiseDistillationEngine:
    """Real Qwen teacher shell with one patched RWKV7 mixer receiving gradients."""

    def __init__(
        self,
        patcher: HybridModelPatcher,
        *,
        lr: float,
        trace_path: Path,
        trace_binding: Mapping[str, str] | None = None,
    ) -> None:
        self.patcher = patcher
        self.adapters = [record.adapter for record in patcher.records]
        self.trainer = ActiveLayerTrainer(self.adapters, lr=lr)
        self.trace_path = trace_path
        self.trace_binding = dict(trace_binding or {})
        self.active_layer: int | None = None
        self.converted_prefix = 0
        self.weights = LossWeights.for_stage("signals")
        self.window = BurnInWindow(0, 1, True, 20260714)

    def begin_layer(
        self,
        layer: int,
        *,
        converted_prefix: int,
        loss_stage: str,
        burn_in_tokens: int,
        supervised_tokens: int,
        seed: int,
        fully_recurrent: bool = False,
        resume_accumulation: bool = False,
        trainable_names: set[str] | None = None,
    ) -> None:
        if supervised_tokens < 2:
            raise ContractError("supervised window must contain at least two tokens for shifted CE")
        if fully_recurrent:
            self.patcher.configure(
                active_layer=layer,
                converted_layers=set(range(len(self.adapters))),
                reset_gradients=not resume_accumulation,
            )
        else:
            self.patcher.configure(
                active_layer=layer,
                converted_prefix=converted_prefix,
                reset_gradients=not resume_accumulation,
            )
        if resume_accumulation:
            if self.trainer.accumulation_step <= 0 or self.trainer.active_layer != layer:
                raise ContractError(
                    "resume_accumulation requires saved gradients for the same active layer"
                )
        else:
            self.trainer.activate(layer, trainable_names=trainable_names)
        self.active_layer = layer
        self.converted_prefix = converted_prefix
        self.weights = LossWeights.for_stage(loss_stage)
        self.window = BurnInWindow(burn_in_tokens, supervised_tokens, True, seed)

    def _supervised(self, value: Tensor) -> Tensor:
        start = self.window.burn_in_tokens
        stop = start + self.window.supervised_tokens
        if value.ndim < 2 or value.shape[1] < stop:
            raise ContractError(
                f"trace has {value.shape[1] if value.ndim >= 2 else 'no'} token axis; "
                f"burn-in contract requires {stop}"
            )
        return value[:, start:stop]

    def step(self, batch: DistillationBatch, *, accumulation_steps: int = 1) -> dict[str, object]:
        if self.active_layer is None:
            raise ContractError("begin_layer must be called before distillation step")
        adapter = self.adapters[self.active_layer]
        student_mixer_all, student_block_all, teacher_mixer_all, teacher_block_all = (
            self.patcher.forward_active_layer_local(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                position_ids=(batch.attention_mask.long().cumsum(-1) - 1).clamp_min(0),
                active_layer=self.active_layer,
            )
        )
        if adapter.last_state is None:
            raise ContractError("active RWKV7 trace was not captured")
        student_mixer = self._supervised(student_mixer_all)
        teacher_mixer = self._supervised(teacher_mixer_all.to(student_mixer.device))
        student_block = self._supervised(student_block_all)
        teacher_block = self._supervised(teacher_block_all.to(student_block.device))
        cosine = 1 - torch.nn.functional.cosine_similarity(
            student_block.flatten(0, -2), teacher_block.flatten(0, -2), dim=-1
        ).mean()
        shifted_ce = student_block.new_zeros(())
        rollout = student_block.new_zeros(())
        losses = LossBreakdown(
            intermediate_mse=normalized_mse(student_mixer, teacher_mixer),
            block_mse=normalized_mse(student_block, teacher_block),
            cosine=cosine,
            token_kl=student_block.new_zeros(()),
            shifted_ce=shifted_ce,
            rollout=rollout,
        )
        total = losses.weighted(self.weights)
        optimizer_stepped = self.trainer.backward(total, accumulation_steps=accumulation_steps)
        active_parameters = list(adapter.parameters())
        gradient_norm = torch.sqrt(
            sum(
                parameter.grad.detach().float().square().sum()
                for parameter in active_parameters
                if parameter.grad is not None
            )
        )
        row = {
            "active_layer": self.active_layer,
            "converted_prefix": self.converted_prefix,
            "micro_step": self.trainer.micro_step,
            "optimizer_step": self.trainer.optimizer_step,
            "accumulation_step": self.trainer.accumulation_step,
            "optimizer_stepped": optimizer_stepped,
            "burn_in_tokens": self.window.burn_in_tokens,
            "supervised_tokens": self.window.supervised_tokens,
            "loss_weights": asdict(self.weights),
            "losses": {name: float(value.detach()) for name, value in losses.items()},
            "total_loss": float(total.detach()),
            "active_gradient_norm": float(gradient_norm),
            "binding": self.trace_binding,
        }
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        return row
