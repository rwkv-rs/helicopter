from __future__ import annotations

import hashlib
import io
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .errors import ContractError


DEFAULT_VOCAB_CHUNK_SIZE = 8192


@dataclass(frozen=True)
class LossBreakdown:
    intermediate_mse: Tensor
    state_mse: Tensor
    block_mse: Tensor
    cosine: Tensor
    token_kl: Tensor
    shifted_ce: Tensor
    rollout: Tensor

    @property
    def total(self) -> Tensor:
        return sum(asdict(self).values())

    def weighted(self, weights: "LossWeights") -> Tensor:
        return sum(getattr(self, name) * getattr(weights, name) for name in asdict(self))


@dataclass(frozen=True)
class LossWeights:
    intermediate_mse: float
    state_mse: float
    block_mse: float
    cosine: float
    token_kl: float
    shifted_ce: float
    rollout: float

    @classmethod
    def for_stage(cls, stage: str) -> "LossWeights":
        stages = {
            "signals": cls(1.0, 1.0, 0.25, 0.1, 0.0, 0.0, 0.0),
            "block": cls(0.25, 0.25, 1.0, 0.25, 0.1, 0.0, 0.0),
            "global": cls(0.1, 0.1, 0.5, 0.1, 1.0, 0.25, 0.0),
            "rollout": cls(0.0, 0.1, 0.25, 0.1, 0.5, 0.25, 1.0),
        }
        try:
            return stages[stage]
        except KeyError as error:
            raise ContractError(f"unknown distillation loss stage: {stage}") from error


def normalized_mse(student: Tensor, teacher: Tensor) -> Tensor:
    return torch.mean((student - teacher).square()) / torch.clamp(torch.mean(teacher.square()), min=1e-12)


def token_kl(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
    teacher = torch.softmax(teacher_logits.float(), dim=-1)
    return torch.nn.functional.kl_div(
        torch.log_softmax(student_logits.float(), dim=-1), teacher, reduction="batchmean"
    )


def chunked_token_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    vocab_chunk_size: int,
) -> Tensor:
    """Compute the exact teacher-to-student KL with bounded FP32 temporaries."""
    if student_logits.shape != teacher_logits.shape:
        raise ContractError("student and teacher logits must have identical shapes")
    if student_logits.ndim < 2 or vocab_chunk_size <= 0:
        raise ContractError("chunked KL requires logits and a positive vocab_chunk_size")
    vocab_size = student_logits.shape[-1]
    teacher_log_z: Tensor | None = None
    student_log_z: Tensor | None = None
    for start in range(0, vocab_size, vocab_chunk_size):
        stop = min(start + vocab_chunk_size, vocab_size)
        teacher_part = torch.logsumexp(teacher_logits[..., start:stop].float(), dim=-1)
        student_part = torch.logsumexp(student_logits[..., start:stop].float(), dim=-1)
        teacher_log_z = teacher_part if teacher_log_z is None else torch.logaddexp(teacher_log_z, teacher_part)
        student_log_z = student_part if student_log_z is None else torch.logaddexp(student_log_z, student_part)
    assert teacher_log_z is not None and student_log_z is not None
    total = teacher_log_z.new_zeros(())
    for start in range(0, vocab_size, vocab_chunk_size):
        stop = min(start + vocab_chunk_size, vocab_size)
        teacher_chunk = teacher_logits[..., start:stop].float()
        student_chunk = student_logits[..., start:stop].float()
        teacher_log_probability = teacher_chunk - teacher_log_z.unsqueeze(-1)
        student_log_probability = student_chunk - student_log_z.unsqueeze(-1)
        total = total + torch.sum(
            teacher_log_probability.exp()
            * (teacher_log_probability - student_log_probability)
        )
    return total / student_logits.shape[0]


def layerwise_losses(
    *,
    student_intermediate: Tensor,
    teacher_intermediate: Tensor,
    student_state: Tensor,
    teacher_state: Tensor,
    student_block: Tensor,
    teacher_block: Tensor,
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    rollout_student: Tensor | None = None,
    rollout_teacher: Tensor | None = None,
) -> LossBreakdown:
    cosine = 1 - torch.nn.functional.cosine_similarity(
        student_block.flatten(0, -2), teacher_block.flatten(0, -2), dim=-1
    ).mean()
    ce = torch.nn.functional.cross_entropy(student_logits[..., :-1, :].reshape(-1, student_logits.shape[-1]), labels[..., 1:].reshape(-1))
    rollout = student_logits.new_zeros(()) if rollout_student is None else normalized_mse(rollout_student, rollout_teacher)
    return LossBreakdown(
        normalized_mse(student_intermediate, teacher_intermediate),
        normalized_mse(student_state, teacher_state),
        normalized_mse(student_block, teacher_block),
        cosine,
        chunked_token_kl(
            student_logits,
            teacher_logits,
            vocab_chunk_size=DEFAULT_VOCAB_CHUNK_SIZE,
        ),
        ce,
        rollout,
    )


class ActiveLayerTrainer:
    """Owns one optimizer per mixer and exposes only the active mixer's params."""

    def __init__(self, layers: Sequence[nn.Module], *, lr: float = 1e-3) -> None:
        if not layers:
            raise ContractError("layerwise trainer requires at least one layer")
        self.layers = list(layers)
        self.optimizers = [torch.optim.AdamW(layer.parameters(), lr=lr) for layer in self.layers]
        self.schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0) for optimizer in self.optimizers]
        self.active_layer = 0
        self.micro_step = 0
        self.optimizer_step = 0
        self.accumulation_step = 0
        self.data_cursor = 0
        self.sweep_index = 0
        self.visit_cursor = 0
        self.validation_history: list[float] = []
        self.stop_counters = {"completed_sweeps": 0, "below_min_delta": 0}
        self.selected_checkpoint: str | None = None
        self.active_trainable_names: tuple[str, ...] | None = None
        self.scaler_state = {"enabled": False, "reason": "bf16-correctness-path"}

    def activate(
        self,
        index: int,
        *,
        trainable_names: set[str] | None = None,
    ) -> None:
        if not 0 <= index < len(self.layers):
            raise ContractError(f"active layer out of range: {index}")
        if self.accumulation_step:
            raise ContractError("cannot switch active layer with uncommitted accumulated gradients")
        self.active_layer = index
        known_names = {name for name, _ in self.layers[index].named_parameters()}
        if trainable_names is not None:
            unknown = trainable_names - known_names
            if unknown:
                raise ContractError(
                    f"active layer trainable-name set contains unknown parameters: {sorted(unknown)}"
                )
            if not trainable_names:
                raise ContractError("active layer trainable-name set is empty")
            self.active_trainable_names = tuple(sorted(trainable_names))
        else:
            self.active_trainable_names = None
        for layer_index, layer in enumerate(self.layers):
            for name, parameter in layer.named_parameters():
                parameter.requires_grad_(
                    layer_index == index
                    and (
                        trainable_names is None
                        or name in trainable_names
                    )
                )
                parameter.grad = None

    def backward(self, loss: Tensor, *, accumulation_steps: int = 1) -> bool:
        if accumulation_steps <= 0:
            raise ContractError("accumulation_steps must be positive")
        optimizer = self.optimizers[self.active_layer]
        if self.accumulation_step == 0:
            optimizer.zero_grad(set_to_none=True)
        (loss / accumulation_steps).backward()
        active = list(self.layers[self.active_layer].parameters())
        if not any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in active):
            raise ContractError("active layer received no nonzero gradient; global loss bridge is cut")
        for index, layer in enumerate(self.layers):
            if index != self.active_layer and any(parameter.grad is not None for parameter in layer.parameters()):
                raise ContractError(f"inactive layer {index} accumulated parameter gradients")
        self.accumulation_step += 1
        self.micro_step += 1
        self.data_cursor += 1
        if self.accumulation_step < accumulation_steps:
            return False
        optimizer.step()
        self.schedulers[self.active_layer].step()
        self.optimizer_step += 1
        self.accumulation_step = 0
        return True

    def step(self, loss: Tensor) -> None:
        self.backward(loss)

    def state_dict(self) -> dict[str, object]:
        return {
            "active_layer": self.active_layer,
            "micro_step": self.micro_step,
            "optimizer_step": self.optimizer_step,
            "accumulation_step": self.accumulation_step,
            "data_cursor": self.data_cursor,
            "sweep_index": self.sweep_index,
            "visit_cursor": self.visit_cursor,
            "validation_history": list(self.validation_history),
            "stop_counters": dict(self.stop_counters),
            "selected_checkpoint": self.selected_checkpoint,
            "active_trainable_names": self.active_trainable_names,
            "scaler": dict(self.scaler_state),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
            "gradients": [
                [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in layer.parameters()]
                for layer in self.layers
            ],
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.active_layer = int(state["active_layer"])
        self.micro_step = int(state["micro_step"])
        self.optimizer_step = int(state["optimizer_step"])
        self.accumulation_step = int(state["accumulation_step"])
        self.data_cursor = int(state["data_cursor"])
        self.sweep_index = int(state["sweep_index"])
        self.visit_cursor = int(state["visit_cursor"])
        self.validation_history = list(state["validation_history"])
        self.stop_counters = dict(state["stop_counters"])
        self.selected_checkpoint = state["selected_checkpoint"]
        saved_trainable_names = state.get("active_trainable_names")
        self.active_trainable_names = (
            None
            if saved_trainable_names is None
            else tuple(saved_trainable_names)
        )
        self.scaler_state = dict(state["scaler"])
        for optimizer, optimizer_state in zip(self.optimizers, state["optimizers"], strict=True):
            optimizer.load_state_dict(optimizer_state)
        for scheduler, scheduler_state in zip(self.schedulers, state["schedulers"], strict=True):
            scheduler.load_state_dict(scheduler_state)
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        accumulation_step = self.accumulation_step
        self.accumulation_step = 0
        self.activate(
            self.active_layer,
            trainable_names=(
                None
                if self.active_trainable_names is None
                else set(self.active_trainable_names)
            ),
        )
        self.accumulation_step = accumulation_step
        for layer, gradients in zip(self.layers, state["gradients"], strict=True):
            for parameter, gradient in zip(layer.parameters(), gradients, strict=True):
                parameter.grad = None if gradient is None else gradient.detach().clone()


def state_digest(layers: Iterable[nn.Module], trainer: ActiveLayerTrainer) -> str:
    buffer = io.BytesIO()
    torch.save({"layers": [layer.state_dict() for layer in layers], "trainer": trainer.state_dict()}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def save_training_checkpoint(
    path: Path,
    layers: Sequence[nn.Module],
    trainer: ActiveLayerTrainer,
    *,
    metadata: Mapping[str, object],
) -> str:
    payload = {
        "schema_version": 1,
        "layers": [layer.state_dict() for layer in layers],
        "trainer": trainer.state_dict(),
        "metadata": dict(metadata),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    content = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return hashlib.sha256(content).hexdigest()


def load_training_checkpoint(
    path: Path,
    layers: Sequence[nn.Module],
    trainer: ActiveLayerTrainer,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or len(payload.get("layers", ())) != len(layers):
        raise ContractError("training checkpoint schema or layer count mismatch")
    for layer, state in zip(layers, payload["layers"], strict=True):
        layer.load_state_dict(state, strict=True)
    trainer.load_state_dict(payload["trainer"])
    return dict(payload["metadata"])


def _atomic_torch_save(path: Path, payload: object) -> str:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    content = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return hashlib.sha256(content).hexdigest()


def save_sharded_training_checkpoint(
    root: Path,
    layers: Sequence[nn.Module],
    trainer: ActiveLayerTrainer,
    *,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Atomically persist only the mutable layer plus a small resume cursor.

    Older layer shards remain immutable until that layer becomes active again,
    avoiding an all-model rewrite on every optimizer or accumulation step.
    """
    active = trainer.active_layer
    old_pointer = None
    layer_files: dict[str, dict[str, str]] = {}
    pointer_path = root / "latest.json"
    if pointer_path.is_file():
        old_pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        old_progress = torch.load(
            root / str(old_pointer["progress"]), map_location="cpu", weights_only=False
        )
        layer_files = dict(old_progress.get("layer_files", {}))
    metadata_tag = hashlib.sha256(
        json.dumps(dict(metadata), sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    layer_path = (
        root
        / "layers"
        / f"layer-{active:03d}-micro-{trainer.micro_step:09d}-opt-{trainer.optimizer_step:09d}-{metadata_tag}.pt"
    )
    layer_payload = {
        "schema_version": 1,
        "layer": active,
        "model": layers[active].state_dict(),
        "optimizer": trainer.optimizers[active].state_dict(),
        "scheduler": trainer.schedulers[active].state_dict(),
        "gradients": [
            None if parameter.grad is None else parameter.grad.detach().cpu()
            for parameter in layers[active].parameters()
        ],
    }
    layer_sha = _atomic_torch_save(layer_path, layer_payload)
    previous_active_file = layer_files.get(str(active), {}).get("path")
    layer_files[str(active)] = {
        "path": str(layer_path.relative_to(root)),
        "sha256": layer_sha,
    }
    progress = {
        "schema_version": 1,
        "layer_count": len(layers),
        "active_layer": active,
        "micro_step": trainer.micro_step,
        "optimizer_step": trainer.optimizer_step,
        "accumulation_step": trainer.accumulation_step,
        "data_cursor": trainer.data_cursor,
        "sweep_index": trainer.sweep_index,
        "visit_cursor": trainer.visit_cursor,
        "validation_history": list(trainer.validation_history),
        "stop_counters": dict(trainer.stop_counters),
        "selected_checkpoint": trainer.selected_checkpoint,
        "scaler": dict(trainer.scaler_state),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "metadata": dict(metadata),
        "layer_files": layer_files,
    }
    progress_path = root / (
        f"progress-micro-{trainer.micro_step:09d}-opt-{trainer.optimizer_step:09d}-{metadata_tag}.pt"
    )
    progress_sha = _atomic_torch_save(progress_path, progress)
    pointer = {
        "schema_version": 1,
        "progress": str(progress_path.relative_to(root)),
        "progress_sha256": progress_sha,
        "active_layer": active,
        "optimizer_step": trainer.optimizer_step,
        "micro_step": trainer.micro_step,
    }
    temporary = pointer_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(pointer, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(pointer_path)
    if previous_active_file and previous_active_file != layer_files[str(active)]["path"]:
        (root / previous_active_file).unlink(missing_ok=True)
    if old_pointer and old_pointer.get("progress") != pointer["progress"]:
        (root / str(old_pointer["progress"])).unlink(missing_ok=True)
    return pointer


def load_sharded_training_checkpoint(
    root: Path,
    layers: Sequence[nn.Module],
    trainer: ActiveLayerTrainer,
) -> dict[str, object]:
    pointer = json.loads((root / "latest.json").read_text(encoding="utf-8"))
    progress_path = root / str(pointer["progress"])
    content = progress_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != pointer["progress_sha256"]:
        raise ContractError("training progress SHA-256 mismatch")
    progress = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
    if progress.get("schema_version") != 1 or int(progress["layer_count"]) != len(layers):
        raise ContractError("sharded training checkpoint schema or layer count mismatch")
    active = int(progress["active_layer"])
    active_payload = None
    for index_text, entry in progress.get("layer_files", {}).items():
        index = int(index_text)
        if not 0 <= index < len(layers):
            raise ContractError(f"training checkpoint has invalid layer index: {index}")
        layer = layers[index]
        layer_path = root / str(entry["path"])
        layer_content = layer_path.read_bytes()
        if hashlib.sha256(layer_content).hexdigest() != entry["sha256"]:
            raise ContractError(f"training layer shard SHA-256 mismatch: {layer_path}")
        payload = torch.load(layer_path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 1 or int(payload["layer"]) != index:
            raise ContractError(f"invalid layer training shard: {layer_path}")
        layer.load_state_dict(payload["model"], strict=True)
        trainer.optimizers[index].load_state_dict(payload["optimizer"])
        trainer.schedulers[index].load_state_dict(payload["scheduler"])
        if index == active:
            active_payload = payload
    trainer.micro_step = int(progress["micro_step"])
    trainer.optimizer_step = int(progress["optimizer_step"])
    trainer.accumulation_step = 0
    trainer.data_cursor = int(progress["data_cursor"])
    trainer.sweep_index = int(progress["sweep_index"])
    trainer.visit_cursor = int(progress["visit_cursor"])
    trainer.validation_history = list(progress["validation_history"])
    trainer.stop_counters = dict(progress["stop_counters"])
    trainer.selected_checkpoint = progress["selected_checkpoint"]
    trainer.scaler_state = dict(progress["scaler"])
    trainer.activate(active)
    trainer.accumulation_step = int(progress["accumulation_step"])
    if active_payload is None:
        raise ContractError("training checkpoint is missing the active layer shard")
    for parameter, gradient in zip(
        layers[active].parameters(), active_payload["gradients"], strict=True
    ):
        parameter.grad = None if gradient is None else gradient.to(parameter.device)
    torch.set_rng_state(progress["torch_rng"])
    random.setstate(progress["python_rng"])
    return dict(progress["metadata"])


@dataclass(frozen=True)
class BurnInWindow:
    burn_in_tokens: int
    supervised_tokens: int
    reset_at_document: bool
    seed: int

    def split(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        stop = self.burn_in_tokens + self.supervised_tokens
        if tokens.shape[-1] < stop:
            raise ContractError(f"sample has {tokens.shape[-1]} tokens but burn-in contract requires {stop}")
        return tokens[..., : self.burn_in_tokens], tokens[..., self.burn_in_tokens : stop]


def progressive_schedule(layer_count: int = 60) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tuple(range(layer_count)), tuple(reversed(range(layer_count)))


def _hidden(output: Tensor | tuple[Tensor, ...]) -> Tensor:
    return output[0] if isinstance(output, tuple) else output


class HybridReplacementRunner(nn.Module):
    """Teacher prefix / active student / teacher suffix with a live KL bridge."""

    def __init__(
        self,
        teacher_layers: Sequence[nn.Module],
        student_layers: Sequence[nn.Module],
        teacher_lm_head: nn.Module,
    ) -> None:
        super().__init__()
        if len(teacher_layers) != len(student_layers):
            raise ContractError("teacher/student layer counts differ")
        self.teacher_layers = nn.ModuleList(teacher_layers).eval().requires_grad_(False)
        self.student_layers = nn.ModuleList(student_layers)
        self.teacher_lm_head = teacher_lm_head.eval().requires_grad_(False)

    def isolated(self, hidden: Tensor, *, active_layer: int) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            for layer in self.teacher_layers[:active_layer]:
                hidden = _hidden(layer(hidden))
        active_output = _hidden(self.student_layers[active_layer](hidden.detach()))
        bridged = active_output
        # Teacher suffix parameters are frozen, but this deliberately is not
        # torch.no_grad(): logits loss must retain d(logits)/d(active_output).
        for layer in self.teacher_layers[active_layer + 1 :]:
            bridged = _hidden(layer(bridged))
        return active_output, self.teacher_lm_head(bridged)

    def progressive(self, hidden: Tensor, *, active_layer: int) -> tuple[Tensor, Tensor]:
        for index in range(active_layer):
            with torch.no_grad():
                hidden = _hidden(self.student_layers[index](hidden))
        active_output = _hidden(self.student_layers[active_layer](hidden.detach()))
        bridged = active_output
        for layer in self.teacher_layers[active_layer + 1 :]:
            bridged = _hidden(layer(bridged))
        return active_output, self.teacher_lm_head(bridged)

    def fully_recurrent(self, hidden: Tensor) -> Tensor:
        for layer in self.student_layers:
            hidden = _hidden(layer(hidden))
        return self.teacher_lm_head(hidden)


@dataclass(frozen=True)
class BaselineResult:
    name: str
    normalized_mse: float
    cosine: float
    token_kl: float
    ppl: float


def zero_step_baselines(
    candidates: Mapping[str, tuple[Tensor, Tensor]],
    *,
    teacher_output: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
) -> tuple[BaselineResult, ...]:
    required = {"random", "naive_copy", "gdn_algebraic", "kv_repeat", "kv_expand", "activation_fitted"}
    missing = required - candidates.keys()
    if missing:
        raise ContractError(f"zero-step baseline matrix is incomplete: {sorted(missing)}")
    results: list[BaselineResult] = []
    for name, (output, logits) in candidates.items():
        if logits.shape != teacher_logits.shape:
            raise ContractError(f"baseline {name} logits shape differs from teacher")
        mse = normalized_mse(output, teacher_output)
        cos = torch.nn.functional.cosine_similarity(output.flatten(), teacher_output.flatten(), dim=0)
        kl = chunked_token_kl(
            logits,
            teacher_logits,
            vocab_chunk_size=DEFAULT_VOCAB_CHUNK_SIZE,
        )
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        results.append(BaselineResult(name, float(mse), float(cos), float(kl), float(torch.exp(ce))))
    return tuple(results)


@dataclass
class SweepController:
    min_sweeps: int = 1
    max_sweeps: int = 3
    min_delta: float = 0.001
    history: list[dict[str, object]] | None = None

    def __post_init__(self) -> None:
        self.history = [] if self.history is None else self.history

    def complete(self, *, start_checkpoint: str, end_checkpoint: str, validation_kl: float, token_budget: int) -> dict[str, object]:
        index = len(self.history)
        previous = None if not self.history else float(self.history[-1]["validation_kl"])
        delta = None if previous is None else previous - validation_kl
        row = {
            "sweep_index": index,
            "order": "59..0",
            "start_checkpoint": start_checkpoint,
            "end_checkpoint": end_checkpoint,
            "token_budget": token_budget,
            "validation_kl": validation_kl,
            "delta": delta,
        }
        self.history.append(row)
        best = min(self.history, key=lambda item: float(item["validation_kl"]))
        completed = len(self.history)
        stop = completed >= self.max_sweeps or (
            completed >= self.min_sweeps and delta is not None and delta < self.min_delta
        )
        result = {
            **row,
            "stop": stop,
            "selected_checkpoint": best["end_checkpoint"],
            "rollback_checkpoint": best["end_checkpoint"],
        }
        self.history[-1] = result
        return result
