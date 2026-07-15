from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .contracts import EvaluationRequest


DISABLED_CACHE_POLICY_REVISION = "cache-disabled-v1"


class Eligibility(StrEnum):
    OFFICIAL = "official"
    PROXY = "proxy"
    SANITY = "sanity"
    INVALID = "invalid"


class SampleStatus(StrEnum):
    SCORED = "scored"
    MODEL_INVALID = "model_invalid"
    PROVIDER_ERROR = "provider_error"
    CACHE_ERROR = "cache_error"
    SCORER_ERROR = "scorer_error"
    HARNESS_ERROR = "harness_error"
    CANCELLED = "cancelled"


class RunStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    INVALID = "invalid"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class TaskIdentity:
    suite: str
    task: str
    version: str
    dataset_digest: str
    split: str
    fewshot: int
    prompt_revision: str
    scorer_revision: str
    generation_contract: str
    cot_mode: str
    repair_strategy: str
    eligibility: Eligibility


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    served_name: str
    checkpoint_sha256: str
    tokenizer_revision: str
    chat_template_revision: str


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    server_revision: str
    wkv_mode: str
    precision: str
    gemm_policy: str
    launch_contract: str


@dataclass(frozen=True, slots=True)
class GenerationLimitResolution:
    task_default: int
    override: int | None
    override_source: str | None
    final: int

    def __post_init__(self) -> None:
        if self.task_default <= 0 or self.final <= 0:
            raise ValueError("generation limits must be positive")
        if self.override is None and self.override_source is not None:
            raise ValueError("override_source requires an override")
        if self.override is not None:
            if self.override <= 0:
                raise ValueError("generation limit override must be positive")
            if not self.override_source:
                raise ValueError("generation limit override requires provenance")
            if self.final != self.override:
                raise ValueError(
                    "final generation limit must equal the explicit override"
                )
        elif self.final != self.task_default:
            raise ValueError("final generation limit must equal the task default")


def identity_digest(*identities: object, config_digest: str) -> str:
    payload: list[Any] = []
    for identity in identities:
        payload.append(
            asdict(identity) if hasattr(identity, "__dataclass_fields__") else identity
        )
    canonical = json.dumps(
        {"identities": payload, "config_digest": config_digest},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class SampleAccounting:
    source_rows: int
    dataset_accepted: int
    dataset_rejected: int
    selected: int
    formatter_accepted: int
    formatter_rejected: int
    scored: int
    model_invalid: int = 0
    provider_error: int = 0
    cache_error: int = 0
    scorer_error: int = 0
    harness_error: int = 0
    cancelled: int = 0

    def validate(self) -> None:
        values = asdict(self)
        if any(value < 0 for value in values.values()):
            raise ValueError("sample accounting values must be non-negative")
        if self.source_rows != self.dataset_accepted + self.dataset_rejected:
            raise ValueError(
                "source_rows must equal dataset_accepted + dataset_rejected"
            )
        if self.selected != self.formatter_accepted + self.formatter_rejected:
            raise ValueError(
                "selected must equal formatter_accepted + formatter_rejected"
            )
        terminal = (
            self.scored
            + self.model_invalid
            + self.provider_error
            + self.cache_error
            + self.scorer_error
            + self.harness_error
            + self.cancelled
        )
        if self.formatter_accepted != terminal:
            raise ValueError("formatter_accepted must equal terminal sample partitions")


_RUN_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PLANNED: frozenset(
        {RunStatus.RUNNING, RunStatus.CANCELLED, RunStatus.INVALID}
    ),
    RunStatus.RUNNING: frozenset(
        {RunStatus.FINALIZING, RunStatus.PARTIAL, RunStatus.FAILED, RunStatus.CANCELLED}
    ),
    RunStatus.FINALIZING: frozenset(
        {RunStatus.COMPLETED, RunStatus.PARTIAL, RunStatus.FAILED, RunStatus.INVALID}
    ),
}


def validate_run_transition(current: RunStatus, target: RunStatus) -> None:
    if target not in _RUN_TRANSITIONS.get(current, frozenset()):
        raise ValueError(f"invalid run state transition: {current} -> {target}")


@dataclass(slots=True)
class RunLifecycle:
    current: RunStatus = RunStatus.PLANNED

    def advance(self, target: RunStatus) -> None:
        validate_run_transition(self.current, target)
        self.current = target


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    run_id: str
    task_identity: TaskIdentity
    model_identity: ModelIdentity
    provider_identity: ProviderIdentity
    config_digest: str
    asset_manifest_digest: str
    output_dir: Path
    max_samples: int | None
    generation_limit: GenerationLimitResolution

    @property
    def cache_namespace(self) -> str:
        return identity_digest(
            self.task_identity,
            self.model_identity,
            self.provider_identity,
            self.generation_limit,
            config_digest=self.config_digest,
        )


def create_execution_plan(
    request: EvaluationRequest,
    *,
    run_id: str,
    task_identity: TaskIdentity,
    model_identity: ModelIdentity,
    provider_identity: ProviderIdentity,
    config_digest: str,
    asset_manifest_digest: str,
    task_generation_limit: int,
) -> ExecutionPlan:
    override = request.generation_limit_override
    return ExecutionPlan(
        run_id=run_id,
        task_identity=task_identity,
        model_identity=model_identity,
        provider_identity=provider_identity,
        config_digest=config_digest,
        asset_manifest_digest=asset_manifest_digest,
        output_dir=Path(request.output_root) / run_id,
        max_samples=request.max_samples,
        generation_limit=GenerationLimitResolution(
            task_default=task_generation_limit,
            override=override,
            override_source=request.generation_limit_override_source,
            final=override or task_generation_limit,
        ),
    )
