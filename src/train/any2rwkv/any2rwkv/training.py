from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from .distill import (
    DEFAULT_VOCAB_CHUNK_SIZE,
    ActiveLayerTrainer,
    BurnInWindow,
    LossBreakdown,
    LossWeights,
    chunked_token_kl,
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
    teacher_state: Tensor | None = None
    rollout_teacher: Tensor | None = None


class LayerwiseDistillationEngine:
    """Real Qwen teacher shell with one patched RWKV7 mixer receiving gradients."""

    def __init__(
        self,
        patcher: HybridModelPatcher,
        *,
        lr: float,
        trace_path: Path,
        activation_checkpointing: bool = False,
        trace_binding: Mapping[str, str] | None = None,
    ) -> None:
        self.patcher = patcher
        self.adapters = [record.adapter for record in patcher.records]
        self.trainer = ActiveLayerTrainer(self.adapters, lr=lr)
        self.trace_path = trace_path
        self.activation_checkpointing = activation_checkpointing
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
                checkpoint_suffix=self.activation_checkpointing,
            )
        else:
            self.patcher.configure(
                active_layer=layer,
                converted_prefix=converted_prefix,
                reset_gradients=not resume_accumulation,
                checkpoint_suffix=self.activation_checkpointing,
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
        captured: dict[str, Tensor] = {}

        def block_hook(module, args, output):
            captured["block"] = output[0] if isinstance(output, tuple) else output

        handle = self.patcher.layers[self.active_layer].register_forward_hook(block_hook)
        try:
            output = self.patcher.teacher(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=None,
                use_cache=False,
            )
        finally:
            handle.remove()
        adapter = self.adapters[self.active_layer]
        if adapter.last_output is None or adapter.last_state is None or "block" not in captured:
            raise ContractError("active RWKV7 trace was not captured")
        student_mixer = self._supervised(adapter.last_output)
        teacher_mixer = self._supervised(batch.teacher_mixer_output.to(student_mixer.device))
        student_block = self._supervised(captured["block"])
        teacher_block = self._supervised(batch.teacher_block_output.to(student_block.device))
        student_logits = self._supervised(output.logits)
        teacher_logits = self._supervised(batch.teacher_logits.to(student_logits.device))
        labels = self._supervised(batch.labels.to(student_logits.device))
        cosine = 1 - torch.nn.functional.cosine_similarity(
            student_block.flatten(0, -2), teacher_block.flatten(0, -2), dim=-1
        ).mean()
        state_supervision = "not_available_for_source_mixer"
        teacher_state = batch.teacher_state
        if teacher_state is not None and tuple(teacher_state.shape) != tuple(adapter.last_state.shape):
            transposed = teacher_state.transpose(-1, -2)
            if tuple(transposed.shape) == tuple(adapter.last_state.shape):
                teacher_state = transposed
            else:
                teacher_state = None
                state_supervision = "source_target_state_geometry_mismatch"
        if teacher_state is None:
            state_mse = student_logits.new_zeros(())
        else:
            state_supervision = "aligned_final_state"
            state_mse = normalized_mse(
                adapter.last_state, teacher_state.to(adapter.last_state.device)
            )
        shifted_ce = torch.nn.functional.cross_entropy(
            student_logits[:, :-1].reshape(-1, student_logits.shape[-1]), labels[:, 1:].reshape(-1)
        )
        rollout = student_logits.new_zeros(())
        if batch.rollout_teacher is not None:
            rollout = normalized_mse(student_logits, self._supervised(batch.rollout_teacher.to(student_logits.device)))
        losses = LossBreakdown(
            intermediate_mse=normalized_mse(student_mixer, teacher_mixer),
            state_mse=state_mse,
            block_mse=normalized_mse(student_block, teacher_block),
            cosine=cosine,
            token_kl=chunked_token_kl(
                student_logits,
                teacher_logits,
                vocab_chunk_size=DEFAULT_VOCAB_CHUNK_SIZE,
            ),
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
            "state_supervision": state_supervision,
            "losses": {name: float(value.detach()) for name, value in losses.items()},
            "total_loss": float(total.detach()),
            "active_gradient_norm": float(gradient_norm),
            "binding": self.trace_binding,
        }
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        return row
