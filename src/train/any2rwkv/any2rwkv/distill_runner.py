from __future__ import annotations

import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from .artifacts import file_sha256, write_json
from .errors import ContractError
from .recipes import resolve_recipe
from .core import (
    DistillationExecutionRequest,
    ExperimentTracker,
    PerformanceProfileCacheRequest,
    write_experiment_report,
)
from .core.training_control_calibration import validate_training_control_artifact


@dataclass(frozen=True)
class LocalLossWeights:
    mixer_mse: float
    block_mse: float
    cosine: float


@dataclass(frozen=True)
class GlobalLossWeights:
    token_kl: float
    shifted_ce: float


@dataclass(frozen=True)
class DistillationPlan:
    classification: str
    evidence_tier: str
    seed: int
    optimizer_name: str
    learning_rate: float
    final_learning_rate: float
    optimizer_betas: tuple[float, float]
    optimizer_epsilon: float
    optimizer_weight_decay: float
    optimizer_telemetry_interval_steps: int
    learning_rate_warmup_steps: int | None
    local_learning_rate_by_mixer_kind: tuple[tuple[str, float], ...]
    learning_rate_schedule: str
    learning_rate_warmup_ratio: float
    min_learning_rate_ratio: float
    gradient_clip_norm: float | None
    max_parameter_update_relative_l2: float | None
    burn_in_tokens: int
    supervised_tokens: int
    accumulation_steps: int
    micro_batch_size: int
    gradient_checkpointing: bool
    distributed_world_size: int
    cache_shard_rows: int
    checkpoint_interval_micro_batches: int
    activation_fit_rows: int
    activation_fit_ridge: float
    activation_fit_functional_steps: int
    activation_fit_functional_learning_rate: float
    activation_fit_attention_time_mix_ablation: bool
    layer_min_epochs: int
    layer_max_epochs: int
    layer_learning_rate_schedule_epochs: int | None
    layer_fixed_epochs: int | None
    layer_min_delta: float
    layer_patience: int
    corrective_min_sweeps: int
    corrective_max_sweeps: int
    corrective_min_delta: float
    local_loss_weights: LocalLossWeights
    global_loss_weights: GlobalLossWeights
    training_control_evidence_status: str
    training_control_evidence_artifact: str | None
    training_control_evidence_sha256: str | None
    execution_mode: str
    max_estimated_weight_bytes_moved: int | None
    cache_teacher_layers: bool
    max_teacher_cache_bytes: int | None
    corrective_resident_model_max_bytes: int | None
    max_cuda_reserved_bytes: int | None
    max_layer_input_cache_bytes: int | None
    max_cached_layer_input_bytes_per_rank: int | None
    exploratory_layer_limit: int | None
    comparison_baseline_run: str | None
    comparison_changes: tuple[str, ...]
    wandb_mode: str
    wandb_project: str | None
    wandb_entity: str | None
    wandb_group: str | None
    wandb_tags: tuple[str, ...]
    wandb_job_types: tuple[tuple[str, str], ...]


def read_distillation_plan(path: Path) -> DistillationPlan:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, ValueError) as error:
        raise ContractError(f"invalid distillation plan JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ContractError("distillation plan must be a JSON object")
    if payload.get("schema_version") != 3:
        raise ContractError(
            "strict layer-major distillation plan schema_version must be 3"
        )
    optimizer_payload = payload.get("optimizer")
    if optimizer_payload is not None:
        required_optimizer_fields = {
            "name",
            "learning_rate",
            "final_learning_rate",
            "warmup_steps",
            "betas",
            "epsilon",
            "weight_decay",
        }
        if (
            not isinstance(optimizer_payload, dict)
            or set(optimizer_payload) != required_optimizer_fields
        ):
            raise ContractError(
                f"optimizer must contain exactly {sorted(required_optimizer_fields)}"
            )
        if "learning_rate" in payload and float(payload["learning_rate"]) != float(
            optimizer_payload["learning_rate"]
        ):
            raise ContractError(
                "legacy learning_rate conflicts with the explicit optimizer contract"
            )
        raw_betas = optimizer_payload["betas"]
        if (
            not isinstance(raw_betas, list)
            or len(raw_betas) != 2
            or any(isinstance(value, bool) for value in raw_betas)
        ):
            raise ContractError("optimizer betas must be a two-element JSON array")
        optimizer_name = str(optimizer_payload["name"])
        learning_rate = float(optimizer_payload["learning_rate"])
        final_learning_rate = float(optimizer_payload["final_learning_rate"])
        optimizer_betas = (float(raw_betas[0]), float(raw_betas[1]))
        optimizer_epsilon = float(optimizer_payload["epsilon"])
        optimizer_weight_decay = float(optimizer_payload["weight_decay"])
        if type(optimizer_payload["warmup_steps"]) is not int:
            raise ContractError("optimizer warmup_steps must be a JSON integer")
        learning_rate_warmup_steps = int(optimizer_payload["warmup_steps"])
    else:
        optimizer_name = "adamw"
        learning_rate = float(payload.get("learning_rate", 0))
        final_learning_rate = learning_rate * float(
            payload.get("min_learning_rate_ratio", 1.0)
        )
        optimizer_betas = (0.9, 0.999)
        optimizer_epsilon = 1e-8
        optimizer_weight_decay = 0.0
        learning_rate_warmup_steps = None
    raw_telemetry_interval = payload.get("optimizer_telemetry_interval_steps", 1)
    if type(raw_telemetry_interval) is not int:
        raise ContractError("optimizer_telemetry_interval_steps must be a JSON integer")
    optimizer_telemetry_interval_steps = raw_telemetry_interval
    tracking_payload = payload.get("tracking")
    if tracking_payload is None:
        wandb_mode = "disabled"
        wandb_project = None
        wandb_entity = None
        wandb_group = None
        wandb_tags = ()
        wandb_job_types = ()
    else:
        tracking_fields = {
            "mode",
            "project",
            "entity",
            "group",
            "tags",
            "job_types",
        }
        raw_job_types = (
            tracking_payload.get("job_types")
            if isinstance(tracking_payload, dict)
            else None
        )
        if (
            not isinstance(tracking_payload, dict)
            or set(tracking_payload) != tracking_fields
            or tracking_payload.get("mode") not in {"online", "offline", "disabled"}
            or not isinstance(tracking_payload.get("tags"), list)
            or any(
                not isinstance(tag, str) or not tag
                for tag in tracking_payload.get("tags", [])
            )
            or not isinstance(raw_job_types, dict)
            or set(raw_job_types) != {"profile", "distill", "corrective"}
            or any(
                not isinstance(value, str) or not value
                for value in raw_job_types.values()
            )
        ):
            raise ContractError(
                "tracking must contain mode/project/entity/group/tags/job_types "
                "with valid values"
            )
        wandb_mode = str(tracking_payload["mode"])
        wandb_project = tracking_payload["project"]
        wandb_entity = tracking_payload["entity"]
        wandb_group = tracking_payload["group"]
        wandb_tags = tuple(str(tag) for tag in tracking_payload["tags"])
        wandb_job_types = tuple(
            (name, str(raw_job_types[name]))
            for name in ("profile", "distill", "corrective")
        )
        for name, value in (
            ("project", wandb_project),
            ("entity", wandb_entity),
            ("group", wandb_group),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ContractError(
                    f"tracking {name} must be null or a nonempty string"
                )
        if wandb_mode != "disabled" and not wandb_project:
            raise ContractError("enabled W&B tracking requires a project")
    if payload.get("evidence_tier") != "fixture" and optimizer_payload is None:
        raise ContractError(
            "every non-fixture plan requires an explicit optimizer contract"
        )
    raw_gradient_checkpointing = payload.get("gradient_checkpointing", False)
    if type(raw_gradient_checkpointing) is not bool:
        raise ContractError("gradient_checkpointing must be a JSON boolean")
    comparison = payload.get("experiment_comparison")
    if comparison is None:
        comparison_baseline_run = None
        comparison_changes = ()
    elif (
        not isinstance(comparison, dict)
        or set(comparison) != {"baseline_run_id", "only_changes"}
        or not isinstance(comparison.get("baseline_run_id"), str)
        or not comparison["baseline_run_id"]
        or not isinstance(comparison.get("only_changes"), list)
        or not comparison["only_changes"]
        or any(
            not isinstance(value, str) or not value
            for value in comparison["only_changes"]
        )
    ):
        raise ContractError(
            "experiment_comparison requires baseline_run_id and nonempty only_changes"
        )
    else:
        comparison_baseline_run = comparison["baseline_run_id"]
        comparison_changes = tuple(comparison["only_changes"])
    if payload.get("evidence_tier") in {"p1", "scale"}:
        required_activation_fit_controls = {
            "activation_fit_rows",
            "activation_fit_ridge",
            "activation_fit_functional_steps",
            "activation_fit_functional_learning_rate",
        }
        missing = sorted(required_activation_fit_controls.difference(payload))
        if missing:
            raise ContractError(
                f"P1/397B plans require explicit activation-fit controls: {missing}"
            )
    forbidden = {
        "stage_tokens_per_layer",
        "activation_checkpointing",
        "checkpoint_suffix",
    }
    present = sorted(forbidden.intersection(payload))
    if present:
        raise ContractError(
            f"suffix/stage-major distillation settings are forbidden: {present}"
        )
    if (
        "exploratory_layer_limit" in payload
        and type(payload["exploratory_layer_limit"]) is not int
    ):
        raise ContractError("exploratory_layer_limit must be a JSON integer")
    cache_teacher_layers = payload.get("cache_teacher_layers", False)
    if type(cache_teacher_layers) is not bool:
        raise ContractError("cache_teacher_layers must be a JSON boolean")
    attention_time_mix_ablation = payload.get(
        "activation_fit_attention_time_mix_ablation", False
    )
    if type(attention_time_mix_ablation) is not bool:
        raise ContractError(
            "activation_fit_attention_time_mix_ablation must be a JSON boolean"
        )
    if (
        "max_cached_layer_input_bytes_per_rank" in payload
        and type(payload["max_cached_layer_input_bytes_per_rank"]) is not int
    ):
        raise ContractError(
            "max_cached_layer_input_bytes_per_rank must be a JSON integer"
        )
    local_loss = _loss_weights(
        payload.get("local_loss_weights"),
        LocalLossWeights,
        "local_loss_weights",
    )
    global_loss = _loss_weights(
        payload.get("global_loss_weights"),
        GlobalLossWeights,
        "global_loss_weights",
    )
    evidence = payload.get("training_control_evidence")
    if not isinstance(evidence, dict):
        raise ContractError("training_control_evidence must be an object")
    evidence_status = str(evidence.get("status", ""))
    evidence_artifact = evidence.get("artifact")
    evidence_sha = evidence.get("artifact_sha256")
    if evidence_status not in {
        "fixture-only",
        "exploratory-unvalidated",
        "calibrated",
    }:
        raise ContractError("training control evidence status is invalid")
    if evidence_status == "calibrated":
        if (
            not isinstance(evidence_artifact, str)
            or not evidence_artifact
            or not isinstance(evidence_sha, str)
            or len(evidence_sha) != 64
        ):
            raise ContractError(
                "calibrated training controls require an artifact and SHA-256"
            )
        artifact_path = Path(evidence_artifact)
        if not artifact_path.is_absolute():
            artifact_path = (path.parent / artifact_path).resolve()
        if not artifact_path.is_file() or file_sha256(artifact_path) != evidence_sha:
            raise ContractError(
                "training-control calibration artifact SHA-256 mismatch"
            )
        try:
            validate_training_control_artifact(artifact_path, expected_plan=payload)
        except ValueError as error:
            raise ContractError(
                "training-control calibration did not select this plan"
            ) from error
    elif evidence_artifact is not None or evidence_sha is not None:
        raise ContractError("uncalibrated training controls cannot claim an artifact")
    raw_learning_rate_profiles = payload.get("local_learning_rate_by_mixer_kind", {})
    if not isinstance(raw_learning_rate_profiles, dict) or any(
        not isinstance(kind, str)
        or not kind
        or isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or float(rate) <= 0
        for kind, rate in raw_learning_rate_profiles.items()
    ):
        raise ContractError(
            "local_learning_rate_by_mixer_kind must map nonempty names to positive rates"
        )
    legacy_functional_steps = payload.get("activation_fit_time_mix_steps")
    legacy_functional_learning_rate = payload.get(
        "activation_fit_time_mix_learning_rate"
    )
    if (
        legacy_functional_steps is not None
        and payload.get("activation_fit_functional_steps") is not None
        and legacy_functional_steps != payload["activation_fit_functional_steps"]
    ) or (
        legacy_functional_learning_rate is not None
        and payload.get("activation_fit_functional_learning_rate") is not None
        and legacy_functional_learning_rate
        != payload["activation_fit_functional_learning_rate"]
    ):
        raise ContractError(
            "legacy time-mix activation-fit aliases conflict with functional controls"
        )
    activation_fit_functional_steps = payload.get(
        "activation_fit_functional_steps", legacy_functional_steps
    )
    activation_fit_functional_learning_rate = payload.get(
        "activation_fit_functional_learning_rate",
        legacy_functional_learning_rate,
    )
    if (
        activation_fit_functional_steps is None
        or activation_fit_functional_learning_rate is None
    ):
        raise ContractError(
            "distillation plans require explicit functional activation-fit controls"
        )
    plan = DistillationPlan(
        classification=str(payload.get("classification", "")),
        evidence_tier=str(payload.get("evidence_tier", "")),
        seed=int(payload.get("seed", 0)),
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        final_learning_rate=final_learning_rate,
        optimizer_betas=optimizer_betas,
        optimizer_epsilon=optimizer_epsilon,
        optimizer_weight_decay=optimizer_weight_decay,
        optimizer_telemetry_interval_steps=optimizer_telemetry_interval_steps,
        learning_rate_warmup_steps=learning_rate_warmup_steps,
        local_learning_rate_by_mixer_kind=tuple(
            sorted(
                (str(kind), float(rate))
                for kind, rate in raw_learning_rate_profiles.items()
            )
        ),
        learning_rate_schedule=str(payload.get("learning_rate_schedule", "constant")),
        learning_rate_warmup_ratio=float(
            payload.get("learning_rate_warmup_ratio", 0.0)
        ),
        min_learning_rate_ratio=(
            final_learning_rate / learning_rate if learning_rate > 0 else 1.0
        ),
        gradient_clip_norm=(
            float(payload["gradient_clip_norm"])
            if payload.get("gradient_clip_norm") is not None
            else None
        ),
        max_parameter_update_relative_l2=(
            float(payload["max_parameter_update_relative_l2"])
            if payload.get("max_parameter_update_relative_l2") is not None
            else None
        ),
        burn_in_tokens=int(payload.get("burn_in_tokens", 0)),
        supervised_tokens=int(payload.get("supervised_tokens", 0)),
        accumulation_steps=int(payload.get("accumulation_steps", 0)),
        micro_batch_size=int(payload.get("micro_batch_size", 0)),
        gradient_checkpointing=raw_gradient_checkpointing,
        distributed_world_size=int(payload.get("distributed_world_size", 1)),
        cache_shard_rows=int(payload.get("cache_shard_rows", 0)),
        checkpoint_interval_micro_batches=int(
            payload.get("checkpoint_interval_micro_batches", 0)
        ),
        activation_fit_rows=int(payload.get("activation_fit_rows", 0)),
        activation_fit_ridge=float(payload.get("activation_fit_ridge", 1e-3)),
        activation_fit_functional_steps=int(activation_fit_functional_steps),
        activation_fit_functional_learning_rate=float(
            activation_fit_functional_learning_rate
        ),
        activation_fit_attention_time_mix_ablation=attention_time_mix_ablation,
        layer_min_epochs=int(payload.get("layer_min_epochs", 1)),
        layer_max_epochs=int(payload.get("layer_max_epochs", 2)),
        layer_learning_rate_schedule_epochs=(
            int(payload["layer_learning_rate_schedule_epochs"])
            if payload.get("layer_learning_rate_schedule_epochs") is not None
            else None
        ),
        layer_fixed_epochs=(
            int(payload["layer_fixed_epochs"])
            if payload.get("layer_fixed_epochs") is not None
            else None
        ),
        layer_min_delta=float(payload.get("layer_min_delta", 0.001)),
        layer_patience=int(payload.get("layer_patience", 1)),
        corrective_min_sweeps=int(payload.get("corrective_min_sweeps", 0)),
        corrective_max_sweeps=int(payload.get("corrective_max_sweeps", 0)),
        corrective_min_delta=float(payload.get("corrective_min_delta", -1)),
        local_loss_weights=local_loss,
        global_loss_weights=global_loss,
        training_control_evidence_status=evidence_status,
        training_control_evidence_artifact=evidence_artifact,
        training_control_evidence_sha256=evidence_sha,
        execution_mode=str(payload.get("execution_mode", "streamed_layer_store")),
        max_estimated_weight_bytes_moved=(
            int(payload["max_estimated_weight_bytes_moved"])
            if "max_estimated_weight_bytes_moved" in payload
            else None
        ),
        cache_teacher_layers=cache_teacher_layers,
        max_teacher_cache_bytes=(
            int(payload["max_teacher_cache_bytes"])
            if "max_teacher_cache_bytes" in payload
            else None
        ),
        corrective_resident_model_max_bytes=(
            int(payload["corrective_resident_model_max_bytes"])
            if "corrective_resident_model_max_bytes" in payload
            else None
        ),
        max_cuda_reserved_bytes=(
            int(payload["max_cuda_reserved_bytes"])
            if "max_cuda_reserved_bytes" in payload
            else None
        ),
        max_layer_input_cache_bytes=(
            int(payload["max_layer_input_cache_bytes"])
            if "max_layer_input_cache_bytes" in payload
            else None
        ),
        max_cached_layer_input_bytes_per_rank=(
            int(payload["max_cached_layer_input_bytes_per_rank"])
            if "max_cached_layer_input_bytes_per_rank" in payload
            else None
        ),
        exploratory_layer_limit=(
            int(payload["exploratory_layer_limit"])
            if "exploratory_layer_limit" in payload
            else None
        ),
        comparison_baseline_run=comparison_baseline_run,
        comparison_changes=comparison_changes,
        wandb_mode=wandb_mode,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
        wandb_tags=wandb_tags,
        wandb_job_types=wandb_job_types,
    )
    if plan.exploratory_layer_limit is not None and (
        plan.evidence_tier != "exploratory" or plan.exploratory_layer_limit <= 0
    ):
        raise ContractError(
            "exploratory_layer_limit must be positive and is only allowed for "
            "exploratory evidence"
        )
    if (
        not plan.classification
        or plan.evidence_tier not in {"fixture", "exploratory", "p1", "scale"}
        or plan.seed < 0
        or plan.learning_rate <= 0
        or plan.optimizer_name != "adamw"
        or not 0 < plan.final_learning_rate <= plan.learning_rate
        or len(plan.optimizer_betas) != 2
        or any(not 0 <= value < 1 for value in plan.optimizer_betas)
        or not math.isfinite(plan.optimizer_epsilon)
        or plan.optimizer_epsilon <= 0
        or not math.isfinite(plan.optimizer_weight_decay)
        or plan.optimizer_weight_decay < 0
        or plan.optimizer_telemetry_interval_steps <= 0
        or (
            plan.learning_rate_warmup_steps is not None
            and plan.learning_rate_warmup_steps < 0
        )
        or plan.learning_rate_schedule
        not in {
            "constant",
            "warmup-constant",
            "warmup-cosine",
        }
        or not 0 <= plan.learning_rate_warmup_ratio < 1
        or not 0 < plan.min_learning_rate_ratio <= 1
        or (
            plan.gradient_clip_norm is not None
            and (
                not math.isfinite(plan.gradient_clip_norm)
                or plan.gradient_clip_norm <= 0
            )
        )
        or (
            plan.max_parameter_update_relative_l2 is not None
            and (
                not math.isfinite(plan.max_parameter_update_relative_l2)
                or plan.max_parameter_update_relative_l2 <= 0
            )
        )
        or (
            plan.learning_rate_schedule == "constant"
            and (
                plan.learning_rate_warmup_ratio != 0
                or plan.min_learning_rate_ratio != 1
                or (plan.learning_rate_warmup_steps or 0) != 0
            )
        )
        or plan.burn_in_tokens < 0
        or plan.supervised_tokens < 2
        or plan.accumulation_steps <= 0
        or plan.micro_batch_size <= 0
        or plan.gradient_checkpointing
        or plan.distributed_world_size not in {1, 8}
        or (plan.evidence_tier != "fixture" and plan.distributed_world_size != 8)
        or plan.cache_shard_rows <= 0
        or plan.checkpoint_interval_micro_batches < 0
        or plan.activation_fit_rows < 0
        or plan.activation_fit_ridge <= 0
        or plan.activation_fit_functional_steps < 1
        or plan.activation_fit_functional_learning_rate <= 0
        or plan.layer_min_epochs < 1
        or plan.layer_max_epochs < plan.layer_min_epochs
        or (
            plan.layer_learning_rate_schedule_epochs is not None
            and (
                plan.layer_learning_rate_schedule_epochs < 1
                or plan.layer_learning_rate_schedule_epochs > plan.layer_max_epochs
            )
        )
        or (
            plan.layer_fixed_epochs is not None
            and (
                plan.layer_fixed_epochs < plan.layer_min_epochs
                or plan.layer_fixed_epochs > plan.layer_max_epochs
            )
        )
        or plan.layer_min_delta < 0
        or plan.layer_patience < 1
        or plan.layer_min_epochs + plan.layer_patience > plan.layer_max_epochs
        or plan.corrective_min_sweeps < 1
        or plan.corrective_max_sweeps < plan.corrective_min_sweeps
        or plan.corrective_min_delta < 0
        or sum(plan.local_loss_weights.__dict__.values()) <= 0
        or sum(plan.global_loss_weights.__dict__.values()) <= 0
        or plan.execution_mode != "streamed_layer_store"
        or (
            plan.max_estimated_weight_bytes_moved is not None
            and plan.max_estimated_weight_bytes_moved <= 0
        )
        or (
            plan.max_teacher_cache_bytes is not None
            and plan.max_teacher_cache_bytes <= 0
        )
        or (
            plan.corrective_resident_model_max_bytes is not None
            and plan.corrective_resident_model_max_bytes <= 0
        )
        or (
            plan.corrective_resident_model_max_bytes is not None
            and plan.max_cuda_reserved_bytes is not None
            and plan.corrective_resident_model_max_bytes >= plan.max_cuda_reserved_bytes
        )
        or plan.cache_teacher_layers
        or (
            plan.max_cuda_reserved_bytes is not None
            and plan.max_cuda_reserved_bytes <= 0
        )
        or (
            plan.max_layer_input_cache_bytes is not None
            and plan.max_layer_input_cache_bytes <= 0
        )
        or (
            plan.max_cached_layer_input_bytes_per_rank is not None
            and plan.max_cached_layer_input_bytes_per_rank <= 0
        )
        or (
            plan.evidence_tier != "fixture"
            and plan.max_cached_layer_input_bytes_per_rank is None
        )
        or (
            plan.learning_rate_schedule in {"constant", "warmup-constant"}
            and plan.final_learning_rate != plan.learning_rate
        )
    ):
        raise ContractError(
            "distillation plan violates the suffix-free layer-major contract"
        )
    return plan


def _loss_weights(value, cls, name):
    if not isinstance(value, dict) or set(value) != set(cls.__annotations__):
        raise ContractError(
            f"{name} must contain exactly {sorted(cls.__annotations__)}"
        )
    weights = cls(**{key: float(value[key]) for key in cls.__annotations__})
    if any(weight < 0 for weight in weights.__dict__.values()):
        raise ContractError(f"{name} cannot contain negative weights")
    return weights


def validate_training_control_evidence(plan: DistillationPlan) -> None:
    if (
        plan.evidence_tier in {"p1", "scale"}
        and plan.training_control_evidence_status != "calibrated"
    ):
        raise ContractError(
            "P1/397B distillation is blocked until training controls are calibrated"
        )
    if (
        plan.evidence_tier in {"p1", "scale"}
        and plan.activation_fit_rows < plan.distributed_world_size
    ):
        raise ContractError(
            "P1/397B distillation requires activation-fit rows covering every rank"
        )


def validate_distributed_row_capacity(
    plan: DistillationPlan,
    token_rows: tuple[tuple[int, ...], ...],
    validation_rows: tuple[tuple[int, ...], ...],
) -> None:
    """Reject plans that cannot keep every required data-parallel rank active."""
    if plan.distributed_world_size == 1:
        return
    if len(token_rows) < plan.distributed_world_size:
        raise ContractError("distill_train must provide at least one row per rank")
    if len(validation_rows) < plan.distributed_world_size:
        raise ContractError("validation must provide at least one row per rank")
    if plan.activation_fit_rows > len(token_rows):
        raise ContractError(
            "activation_fit_rows exceeds the immutable distill_train row count"
        )


def read_distillation_texts(
    path: Path,
    *,
    expected_split: str = "distill_train",
) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("split") != expected_split:
        raise ContractError(
            f"dataset manifest must use schema_version=1 and split={expected_split}"
        )
    data_file = Path(str(payload.get("data_file", "")))
    if not data_file.is_absolute():
        data_file = (path.parent / data_file).resolve()
    expected = str(payload.get("sha256", ""))
    actual = file_sha256(data_file)
    if actual != expected:
        raise ContractError(
            f"distillation data SHA-256 mismatch: expected {expected}, found {actual}"
        )
    text_field = str(payload.get("text_field", "text"))
    rows: list[str] = []
    with data_file.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            text = row.get(text_field)
            if not isinstance(text, str) or not text.strip():
                raise ContractError("distillation data contains an empty text row")
            rows.append(text)
    if len(rows) != int(payload.get("row_count", 0)) or not rows:
        raise ContractError("distillation data row_count does not match the file")
    return tuple(rows)


def read_packed_token_rows(
    path: Path,
    *,
    split: str,
    burn_in_tokens: int,
    supervised_tokens: int,
) -> tuple[tuple[int, ...], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("status") != "prepared":
        raise ContractError(
            "packed dataset manifest must use schema_version=1 and status=prepared"
        )
    splits = payload.get("splits")
    entry = splits.get(split) if isinstance(splits, dict) else None
    if not isinstance(entry, dict):
        raise ContractError(f"packed dataset manifest has no split: {split}")
    data_file = Path(str(entry.get("path", "")))
    if not data_file.is_absolute():
        data_file = (path.parent / data_file).resolve()
    expected_sha = str(entry.get("sha256", ""))
    actual_sha = file_sha256(data_file)
    if actual_sha != expected_sha:
        raise ContractError(
            f"packed {split} SHA-256 mismatch: expected {expected_sha}, found {actual_sha}"
        )
    expected_length = burn_in_tokens + supervised_tokens
    rows: list[tuple[int, ...]] = []
    row_ids: set[str] = set()
    with data_file.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row_id = str(row.get("row_id", ""))
            values = row.get("input_ids")
            if (
                row.get("split") != split
                or not row_id
                or row_id in row_ids
                or not isinstance(values, list)
                or len(values) != expected_length
                or not all(type(value) is int and value >= 0 for value in values)
                or row.get("burn_in_tokens") != burn_in_tokens
                or row.get("supervised_tokens") != supervised_tokens
            ):
                raise ContractError(
                    f"invalid packed token row in split {split}: {row_id}"
                )
            row_ids.add(row_id)
            rows.append(tuple(values))
    if len(rows) != int(entry.get("row_count", -1)) or not rows:
        raise ContractError(f"packed {split} row_count does not match the file")
    if sum(map(len, rows)) != int(entry.get("token_count", -1)):
        raise ContractError(f"packed {split} token_count does not match the file")
    return tuple(rows)


def _write_baseline_result(
    path: Path,
    *,
    name: str,
    metrics: dict[str, object],
    binding: dict[str, object],
    token_budget: int,
) -> None:
    required_binding = {
        "student_sha256",
        "tokenizer_sha256",
        "dataset_sha256",
        "split",
        "seed",
        "burn_in_tokens",
        "precision",
        "token_budget",
    }
    if not required_binding.issubset(binding):
        raise ContractError(
            "migration baseline binding lacks the shared evaluation protocol"
        )
    if int(binding["token_budget"]) != int(token_budget):
        raise ContractError(
            "migration baseline token budget differs from the shared binding"
        )
    if name == "activation_fitted" and (
        metrics.get("solver_invoked") is not True
        or not isinstance(metrics.get("fit_report_sha256"), str)
        or len(str(metrics["fit_report_sha256"])) != 64
        or not isinstance(metrics.get("materialization_sha256"), str)
        or len(str(metrics["materialization_sha256"])) != 64
    ):
        raise ContractError(
            "activation_fitted baseline requires solver and materialization evidence"
        )
    payload = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {
            "schema_version": 2,
            "student_sha256": binding["student_sha256"],
            "binding": binding,
            "baselines": {},
        }
    )
    if (
        payload.get("schema_version") != 2
        or payload.get("student_sha256") != binding["student_sha256"]
        or payload.get("binding") != binding
    ):
        raise ContractError("migration baseline binding changed within one run")
    payload["baselines"][name] = {**metrics, "token_budget": token_budget}
    write_json(path, payload)


def _finalize_experiment_tracking(
    tracker: ExperimentTracker,
    *,
    run_dir: Path,
    result: dict[str, object] | None,
    status: str,
    exit_code: int,
    reason: str | None = None,
) -> None:
    """Make rank-0 W&B/report completion a distributed all-or-nothing step."""
    error_message: str | None = None
    if tracker.is_primary:
        try:
            if result is not None:
                tracker.log_result(result)
            tracker.finish(exit_code=exit_code)
            write_experiment_report(run_dir, status=status, reason=reason)
        except BaseException as error:
            error_message = f"{type(error).__name__}: {error}"
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        status_payload = [error_message if tracker.is_primary else None]
        torch.distributed.broadcast_object_list(status_payload, src=0)
        error_message = status_payload[0]
    if error_message is not None:
        raise ContractError(
            "distributed W&B/report finalization failed: " + error_message
        )


def _abort_experiment_tracking(
    tracker: ExperimentTracker,
    *,
    run_dir: Path,
    status: str,
    reason: str,
) -> None:
    """Finalize a failed rank locally without entering a new collective."""
    if not tracker.is_primary:
        return
    try:
        tracker.abort(reason=reason)
    finally:
        write_experiment_report(run_dir, status=status, reason=reason)


def run_distillation(
    *,
    source: Path,
    run_dir: Path,
    dataset_manifest: Path,
    training_config: Path,
    recipe_id: str,
    allow_proxy_layers: bool,
    resume: Path | None = None,
) -> dict[str, object]:
    resolved = resolve_recipe(recipe_id)
    if not torch.cuda.is_available():
        raise ContractError("distillation requires CUDA")
    plan = read_distillation_plan(training_config)
    validate_training_control_evidence(plan)
    tracker = ExperimentTracker.from_plan(run_dir=run_dir, plan=plan, action="distill")
    token_rows = read_packed_token_rows(
        dataset_manifest,
        split="distill_train",
        burn_in_tokens=plan.burn_in_tokens,
        supervised_tokens=plan.supervised_tokens,
    )
    validation_rows = read_packed_token_rows(
        dataset_manifest,
        split="validation",
        burn_in_tokens=plan.burn_in_tokens,
        supervised_tokens=plan.supervised_tokens,
    )
    validate_distributed_row_capacity(plan, token_rows, validation_rows)
    torch.manual_seed(plan.seed)
    source_manifest = resolved.source.load_checkpoint(
        source, require_final_layout=not allow_proxy_layers
    )
    inspection = resolved.source.inspect_checkpoint(
        source, require_final_layout=not allow_proxy_layers
    )
    resolved.recipe.validate_source(inspection)
    expected_target = resolved.target.build_target_config(
        source_manifest, require_final_layout=not allow_proxy_layers
    )
    resolved.target.validate_training_environment(
        head_size=int(expected_target["head_size"])
    )
    zero_step = run_dir / "checkpoint-zero-step"
    if not zero_step.is_dir():
        raise ContractError("zero-step checkpoint is missing; run convert first")
    zero_step_config = json.loads(
        (zero_step / "config.json").read_text(encoding="utf-8")
    )
    if int(zero_step_config.get("head_size", 0)) != int(expected_target["head_size"]):
        raise ContractError(
            "zero-step recurrent head geometry differs from the source-compatible target"
        )
    _validate_initialized_run_binding(
        run_dir=run_dir,
        source_manifest=source_manifest,
        zero_step=zero_step,
    )
    tracking_error: str | None = None
    resume_wandb_run = (run_dir / "experiment-tracking.json").is_file()
    try:
        tracker.start(
            config={
                "schema_version": 1,
                "classification": plan.classification,
                "evidence_tier": plan.evidence_tier,
                "seed": plan.seed,
                "optimizer": {
                    "name": plan.optimizer_name,
                    "learning_rate": plan.learning_rate,
                    "final_learning_rate": plan.final_learning_rate,
                    "warmup_steps": plan.learning_rate_warmup_steps,
                    "betas": list(plan.optimizer_betas),
                    "epsilon": plan.optimizer_epsilon,
                    "weight_decay": plan.optimizer_weight_decay,
                    "gradient_clip_norm": plan.gradient_clip_norm,
                },
                "batch": {
                    "per_rank": plan.micro_batch_size,
                    "accumulation_steps": plan.accumulation_steps,
                    "world_size": plan.distributed_world_size,
                    "gradient_checkpointing": plan.gradient_checkpointing,
                },
                "comparison": {
                    "baseline_run_id": plan.comparison_baseline_run,
                    "only_changes": list(plan.comparison_changes),
                },
                "training_config": str(training_config.resolve()),
                "dataset_manifest": str(dataset_manifest.resolve()),
            },
            resume_existing=resume_wandb_run,
        )
    except BaseException as error:
        tracking_error = f"{type(error).__name__}: {error}"
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tracking_status = [tracking_error if tracker.is_primary else None]
        torch.distributed.broadcast_object_list(tracking_status, src=0)
        tracking_error = tracking_status[0]
    if tracking_error is not None:
        raise ContractError(
            "distributed W&B initialization failed before training: " + tracking_error
        )
    try:
        result = resolved.recipe.run_layerwise_distillation(
            DistillationExecutionRequest(
                source_checkpoint=source_manifest,
                run_dir=run_dir,
                zero_step_dir=zero_step,
                token_rows=token_rows,
                validation_rows=validation_rows,
                plan=plan,
                training_config=training_config,
                dataset_manifest=dataset_manifest,
                resume=resume,
                progress_callback=tracker.log_progress,
            )
        )
    except BaseException as error:
        try:
            _abort_experiment_tracking(
                tracker,
                run_dir=run_dir,
                status=(
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                ),
                reason=f"{type(error).__name__}: {error}",
            )
        except BaseException as finalization_error:
            error.add_note(str(finalization_error))
        raise
    _finalize_experiment_tracking(
        tracker,
        run_dir=run_dir,
        result=result,
        status=str(result.get("status", "complete")),
        exit_code=0,
    )
    return result


def prepare_performance_profile_caches(
    *,
    source: Path,
    run_dir: Path,
    dataset_manifest: Path,
    training_config: Path,
    recipe_id: str,
    allow_proxy_layers: bool,
) -> dict[str, object]:
    resolved = resolve_recipe(recipe_id)
    if not torch.cuda.is_available():
        raise ContractError("performance profile cache preparation requires CUDA")
    plan = read_distillation_plan(training_config)
    token_rows = read_packed_token_rows(
        dataset_manifest,
        split="distill_train",
        burn_in_tokens=plan.burn_in_tokens,
        supervised_tokens=plan.supervised_tokens,
    )
    validation_rows = read_packed_token_rows(
        dataset_manifest,
        split="validation",
        burn_in_tokens=plan.burn_in_tokens,
        supervised_tokens=plan.supervised_tokens,
    )
    validate_distributed_row_capacity(plan, token_rows, validation_rows)
    torch.manual_seed(plan.seed)
    source_manifest = resolved.source.load_checkpoint(
        source, require_final_layout=not allow_proxy_layers
    )
    inspection = resolved.source.inspect_checkpoint(
        source, require_final_layout=not allow_proxy_layers
    )
    resolved.recipe.validate_source(inspection)
    expected_target = resolved.target.build_target_config(
        source_manifest, require_final_layout=not allow_proxy_layers
    )
    resolved.target.validate_training_environment(
        head_size=int(expected_target["head_size"])
    )
    zero_step = run_dir / "checkpoint-zero-step"
    if not zero_step.is_dir():
        raise ContractError("zero-step checkpoint is missing; run convert first")
    _validate_initialized_run_binding(
        run_dir=run_dir,
        source_manifest=source_manifest,
        zero_step=zero_step,
    )
    return resolved.recipe.prepare_performance_profile_caches(
        PerformanceProfileCacheRequest(
            source_checkpoint=source_manifest,
            run_dir=run_dir,
            zero_step_dir=zero_step,
            token_rows=token_rows,
            validation_rows=validation_rows,
            plan=plan,
            training_config=training_config,
            dataset_manifest=dataset_manifest,
        )
    )


def run_corrective_continuation(args) -> int:
    """Fork a completed recurrent checkpoint into an independent corrective-only run."""
    from .distributed import DistributedContext

    resolved = resolve_recipe(args.recipe)
    if not torch.cuda.is_available():
        raise ContractError("corrective continuation requires CUDA")
    distributed: DistributedContext | None = None
    tracker: ExperimentTracker | None = None
    tracking_finalized = False
    try:
        plan = read_distillation_plan(Path(args.training_config))
        validate_training_control_evidence(plan)
        if plan.exploratory_layer_limit is not None:
            raise ContractError(
                "corrective continuation forbids exploratory_layer_limit"
            )
        source_manifest = resolved.source.load_checkpoint(
            Path(args.source), require_final_layout=not args.allow_proxy_layers
        )
        inspection = resolved.source.inspect_checkpoint(
            Path(args.source), require_final_layout=not args.allow_proxy_layers
        )
        resolved.recipe.validate_source(inspection)
        expected_target = resolved.target.build_target_config(
            source_manifest, require_final_layout=not args.allow_proxy_layers
        )
        resolved.target.validate_training_environment(
            head_size=int(expected_target["head_size"])
        )
        token_rows = read_packed_token_rows(
            Path(args.dataset_manifest),
            split="distill_train",
            burn_in_tokens=plan.burn_in_tokens,
            supervised_tokens=plan.supervised_tokens,
        )
        validation_rows = read_packed_token_rows(
            Path(args.dataset_manifest),
            split="validation",
            burn_in_tokens=plan.burn_in_tokens,
            supervised_tokens=plan.supervised_tokens,
        )
        validate_distributed_row_capacity(plan, token_rows, validation_rows)
        parent_run = Path(args.parent_run).resolve()
        output = Path(args.output).resolve()
        parent_checkpoint = _select_parent_recurrent_checkpoint(parent_run)
        parent_binding = _checkpoint_binding(parent_checkpoint)
        parent_binding_sha256 = _binding_sha256(parent_binding)
        if parent_binding_sha256 != args.parent_checkpoint_sha256.lower():
            raise ContractError(
                "corrective parent checkpoint SHA-256 mismatch: "
                f"expected={args.parent_checkpoint_sha256.lower()} "
                f"actual={parent_binding_sha256}"
            )
        _validate_parent_run(
            parent_run=parent_run,
            parent_checkpoint=parent_checkpoint,
            source_manifest=source_manifest,
            recipe=resolved,
            precision=args.precision,
        )
        if int(os.environ.get("RANK", "0")) == 0:
            try:
                _prepare_or_validate_corrective_output(
                    output=output,
                    parent_run=parent_run,
                    parent_checkpoint=parent_checkpoint,
                    parent_binding=parent_binding,
                    source_manifest=source_manifest,
                    recipe=resolved,
                    plan_path=Path(args.training_config),
                    dataset_manifest=Path(args.dataset_manifest),
                    precision=args.precision,
                    run_id=args.run_id or output.name,
                    rwkv_hf_sha=args.rwkv_hf_sha,
                    rwkv_lm_sha=args.rwkv_lm_sha,
                )
            except BaseException as error:
                raise ContractError(
                    "corrective continuation preparation failed: " + repr(error)
                ) from error
        tracker = ExperimentTracker.from_plan(
            run_dir=output, plan=plan, action="corrective"
        )
        resume_wandb_run = (output / "experiment-tracking.json").is_file()
        tracker.start(
            config={
                "schema_version": 1,
                "action": "corrective",
                "evidence_tier": plan.evidence_tier,
                "seed": plan.seed,
                "optimizer": {
                    "name": plan.optimizer_name,
                    "learning_rate": plan.learning_rate,
                    "final_learning_rate": plan.final_learning_rate,
                    "warmup_steps": plan.learning_rate_warmup_steps,
                    "betas": list(plan.optimizer_betas),
                    "epsilon": plan.optimizer_epsilon,
                    "weight_decay": plan.optimizer_weight_decay,
                    "gradient_clip_norm": plan.gradient_clip_norm,
                },
                "parent_checkpoint": str(parent_checkpoint),
                "gradient_checkpointing": plan.gradient_checkpointing,
                "comparison": {
                    "baseline_run_id": plan.comparison_baseline_run,
                    "only_changes": list(plan.comparison_changes),
                },
            },
            resume_existing=resume_wandb_run,
        )
        distributed = DistributedContext.initialize()
        distributed.barrier()
        torch.manual_seed(plan.seed)
        result = resolved.recipe.run_corrective_distillation(
            DistillationExecutionRequest(
                source_checkpoint=source_manifest,
                run_dir=output,
                zero_step_dir=parent_checkpoint,
                token_rows=token_rows,
                validation_rows=validation_rows,
                plan=plan,
                training_config=Path(args.training_config),
                dataset_manifest=Path(args.dataset_manifest),
                resume=None,
                progress_callback=tracker.log_progress,
            )
        )
        _finalize_experiment_tracking(
            tracker,
            run_dir=output,
            result=result,
            status=str(result.get("status", "complete")),
            exit_code=0,
        )
        tracking_finalized = True
        if distributed.is_primary:
            metadata = json.loads(
                (output / "metadata.json").read_text(encoding="utf-8")
            )
            metadata["distillation"] = result
            metadata["status"] = result["status"]
            write_json(output / "metadata.json", metadata)
            print(json.dumps(result, sort_keys=True))
        distributed.barrier()
        return 0
    except BaseException as error:
        if tracker is not None and not tracking_finalized:
            try:
                _abort_experiment_tracking(
                    tracker,
                    run_dir=tracker.run_dir,
                    status=(
                        "interrupted"
                        if isinstance(error, KeyboardInterrupt)
                        else "failed"
                    ),
                    reason=f"{type(error).__name__}: {error}",
                )
                tracking_finalized = True
            except BaseException as finalization_error:
                error.add_note(str(finalization_error))
        raise
    finally:
        if distributed is not None:
            distributed.close()


def _select_parent_recurrent_checkpoint(parent_run: Path) -> Path:
    for name in ("checkpoint-global-corrective", "checkpoint-layerwise-local"):
        checkpoint = parent_run / name
        config_path = checkpoint / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metadata = config.get("any2rwkv")
        if (
            isinstance(metadata, dict)
            and metadata.get("recurrence") == "native_rwkv7"
            and (
                metadata.get("final_recurrent") is True
                or metadata.get("fully_recurrent_proxy") is True
            )
        ):
            return checkpoint.resolve()
    raise ContractError("parent run has no completed fully recurrent checkpoint")


def _checkpoint_binding(checkpoint: Path) -> dict[str, object]:
    binding = _hf_checkpoint_files_binding(checkpoint)
    binding["mixer_fingerprint"] = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    )["any2rwkv"]["mixer_overlay_fingerprint"]
    return binding


def _zero_step_checkpoint_binding(checkpoint: Path) -> dict[str, object]:
    binding = _hf_checkpoint_files_binding(checkpoint)
    for name in ("mapping.json", "mapping-coverage.json"):
        path = checkpoint / name
        if not path.is_file():
            raise ContractError(f"zero-step checkpoint is missing {name}")
        binding["files"][name] = file_sha256(path)
    return binding


def _hf_checkpoint_files_binding(checkpoint: Path) -> dict[str, object]:
    index = checkpoint / "model.safetensors.index.json"
    if not index.is_file():
        raise ContractError("HF checkpoint has no shard index")
    payload = json.loads(index.read_text(encoding="utf-8"))
    shards = sorted(set(payload.get("weight_map", {}).values()))
    if not shards:
        raise ContractError("HF checkpoint shard index is empty")
    tokenizer_files = tuple(
        name
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
            "chat_template.jinja",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
        )
        if (checkpoint / name).is_file()
    )
    files = (
        "config.json",
        "model.safetensors.index.json",
        *shards,
        *tokenizer_files,
    )
    return {
        "files": {name: file_sha256(checkpoint / name) for name in files},
    }


def _binding_sha256(binding: dict[str, object]) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_initialized_run_binding(*, run_dir, source_manifest, zero_step) -> None:
    run_metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if run_metadata.get("source", {}).get("files") != source_manifest.file_hashes:
        raise ContractError(
            "distillation source files differ from the source bound at conversion"
        )
    zero_step_binding = _zero_step_checkpoint_binding(zero_step)
    zero_step_sha256 = _binding_sha256(zero_step_binding)
    if run_metadata.get("zero_step") != {
        "binding": zero_step_binding,
        "sha256": zero_step_sha256,
    }:
        raise ContractError(
            "zero-step checkpoint differs from the checkpoint bound at conversion"
        )
    warm_start_plan = run_dir / "warm-start-plan.json"
    if not warm_start_plan.is_file():
        raise ContractError("warm-start plan bound at conversion is missing")
    if run_metadata.get("warm_start_plan") != {
        "path": "warm-start-plan.json",
        "sha256": file_sha256(warm_start_plan),
    }:
        raise ContractError("warm-start plan differs from the plan bound at conversion")


def _validate_parent_run(
    *,
    parent_run: Path,
    parent_checkpoint: Path,
    source_manifest,
    recipe,
    precision: str,
) -> None:
    metadata_path = parent_run / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError("corrective parent run has invalid metadata") from error
    expected_source = {
        "path": str(source_manifest.path),
        "files": source_manifest.file_hashes,
        "layers": source_manifest.contract.num_hidden_layers,
    }
    expected_recipe = {
        "id": recipe.recipe.recipe_id,
        "source_adapter": recipe.source.adapter_id,
        "target_adapter": recipe.target.adapter_id,
    }
    parent_source = metadata.get("source", {})
    source_matches = all(
        parent_source.get(field) == value for field, value in expected_source.items()
    )
    if (
        not source_matches
        or metadata.get("recipe") != expected_recipe
        or metadata.get("precision") != precision
    ):
        raise ContractError(
            "corrective parent run source/recipe/precision binding mismatch"
        )
    config = json.loads((parent_checkpoint / "config.json").read_text(encoding="utf-8"))
    if (
        int(config.get("num_hidden_layers", -1))
        != source_manifest.contract.num_hidden_layers
    ):
        raise ContractError("corrective parent checkpoint layer count mismatch")
    expected = recipe.target.build_target_config(
        source_manifest,
        require_final_layout=source_manifest.contract.num_hidden_layers == 60,
    )
    if int(config.get("head_size", 0)) != int(expected["head_size"]) or int(
        config.get("num_heads", 0)
    ) != int(expected["num_heads"]):
        raise ContractError(
            "corrective parent checkpoint changed source-compatible recurrent head geometry"
        )


def _materialize_corrective_base(
    *, parent_checkpoint: Path, output: Path, layer_count: int
) -> None:
    from .mixer_store import RWKV7MixerLayerStore

    local_marker = output / "checkpoint-layerwise-local"
    local_marker.mkdir()
    shutil.copy2(parent_checkpoint / "config.json", local_marker / "config.json")
    store = RWKV7MixerLayerStore(parent_checkpoint, output / "mixer-overlays")
    checkpoint_dtype = _checkpoint_mixer_dtype(parent_checkpoint)
    cursor = {
        "schedule": "corrective-continuation-base-v1",
        "parent_checkpoint": str(parent_checkpoint),
    }
    for layer_index in range(layer_count):
        mixer = store.load_mixer(
            layer_index,
            device="cpu",
            dtype=checkpoint_dtype,
        )
        store.save_mixer(layer_index, mixer, cursor={**cursor, "layer": layer_index})


def _checkpoint_mixer_dtype(checkpoint: Path) -> torch.dtype:
    from safetensors import safe_open

    index = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index.get("weight_map", {})
    try:
        tensor_name = next(name for name in sorted(weight_map) if ".attn." in name)
    except StopIteration as error:
        raise ContractError(
            "parent recurrent checkpoint contains no mixer tensor"
        ) from error
    with safe_open(
        checkpoint / str(weight_map[tensor_name]), framework="pt", device="cpu"
    ) as handle:
        return handle.get_tensor(tensor_name).dtype


def _prepare_or_validate_corrective_output(
    *,
    output: Path,
    parent_run: Path,
    parent_checkpoint: Path,
    parent_binding: dict[str, object],
    source_manifest,
    recipe,
    plan_path: Path,
    dataset_manifest: Path,
    precision: str,
    run_id: str,
    rwkv_hf_sha: str,
    rwkv_lm_sha: str,
) -> None:
    """Atomically publish a derived base, or validate it for deterministic resume."""
    from .artifacts import initialize_run

    source = {
        "path": str(source_manifest.path),
        "files": source_manifest.file_hashes,
        "layers": source_manifest.contract.num_hidden_layers,
    }
    continuation = {
        "parent_run": str(parent_run),
        "parent_checkpoint": str(parent_checkpoint),
        "parent_binding": parent_binding,
        "training_config_sha256": file_sha256(plan_path),
        "dataset_manifest_sha256": file_sha256(dataset_manifest),
    }
    if output.is_dir() and any(output.iterdir()):
        metadata_path = output / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ContractError(
                "existing corrective continuation output has invalid metadata"
            ) from error
        expected_recipe = {
            "id": recipe.recipe.recipe_id,
            "source_adapter": recipe.source.adapter_id,
            "target_adapter": recipe.target.adapter_id,
        }
        if (
            metadata.get("run_id") != run_id
            or metadata.get("source") != source
            or metadata.get("precision") != precision
            or metadata.get("recipe") != expected_recipe
            or metadata.get("corrective_continuation") != continuation
            or metadata.get("submodules")
            != {"rwkv-hf": rwkv_hf_sha, "rwkv-lm": rwkv_lm_sha}
        ):
            raise ContractError(
                "existing corrective continuation output binding mismatch"
            )
        _verify_materialized_corrective_base(
            output=output,
            parent_checkpoint=parent_checkpoint,
            parent_binding=parent_binding,
            layer_count=source_manifest.contract.num_hidden_layers,
        )
        return
    if output.exists():
        if not output.is_dir():
            raise ContractError("corrective continuation output is not a directory")
        output.rmdir()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.corrective-base.tmp")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        metadata = initialize_run(
            staging,
            run_id=run_id,
            source=source,
            precision=precision,
            command=sys.argv,
            product_root=Path(__file__).resolve().parents[4],
            rwkv_hf_sha=rwkv_hf_sha,
            rwkv_lm_sha=rwkv_lm_sha,
        )
        metadata["recipe"] = {
            "id": recipe.recipe.recipe_id,
            "source_adapter": recipe.source.adapter_id,
            "target_adapter": recipe.target.adapter_id,
        }
        metadata["corrective_continuation"] = continuation
        _materialize_corrective_base(
            parent_checkpoint=parent_checkpoint,
            output=staging,
            layer_count=source_manifest.contract.num_hidden_layers,
        )
        _verify_materialized_corrective_base(
            output=staging,
            parent_checkpoint=parent_checkpoint,
            parent_binding=parent_binding,
            layer_count=source_manifest.contract.num_hidden_layers,
        )
        if _checkpoint_binding(parent_checkpoint) != parent_binding:
            raise ContractError(
                "parent recurrent checkpoint changed during materialization"
            )
        metadata["status"] = "corrective-base-prepared"
        write_json(staging / "metadata.json", metadata)
        _fsync_tree(staging)
        staging.rename(output)
        _fsync_directory(output.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _verify_materialized_corrective_base(
    *,
    output: Path,
    parent_checkpoint: Path,
    parent_binding: dict[str, object],
    layer_count: int,
) -> None:
    from .mixer_store import RWKV7MixerLayerStore

    marker = output / "checkpoint-layerwise-local" / "config.json"
    if not marker.is_file():
        raise ContractError("corrective continuation base marker is missing")
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    if (
        marker_payload.get("any2rwkv", {}).get("mixer_overlay_fingerprint")
        != parent_binding["mixer_fingerprint"]
    ):
        raise ContractError("corrective continuation base fingerprint mismatch")
    overlay = output / "mixer-overlays"
    for layer_index in range(layer_count):
        if (
            not (overlay / f"layer-{layer_index:03d}.safetensors").is_file()
            or not (overlay / f"layer-{layer_index:03d}.json").is_file()
        ):
            raise ContractError(
                f"corrective continuation base is missing layer {layer_index}"
            )
    store = RWKV7MixerLayerStore(parent_checkpoint, overlay)
    if store.fingerprint() != parent_binding["mixer_fingerprint"]:
        raise ContractError(
            "materialized corrective mixer fingerprint differs from parent"
        )


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
