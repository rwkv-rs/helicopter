from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MetricContract(StrictModel):
    name: str
    minimum: float
    maximum: float
    aggregation: Literal["mean", "sum"]
    binary_correctness: bool = False

    @model_validator(mode="after")
    def validate_range(self) -> "MetricContract":
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("metric bounds must be finite")
        if self.minimum >= self.maximum:
            raise ValueError("metric minimum must be less than maximum")
        return self


class TaskIdentity(StrictModel):
    suite: str
    task: str
    version: str
    split: str
    fewshot: int = Field(ge=0)
    prompt_revision: str
    scorer_revision: str
    generation_contract: str
    cot_mode: Literal["none", "cot"]
    repair_strategy: Literal["A", "B", "C", "not-applicable"]
    dataset_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_metric: str
    metrics: tuple[MetricContract, ...]

    @model_validator(mode="after")
    def validate_metrics(self) -> "TaskIdentity":
        names = [metric.name for metric in self.metrics]
        if not names or len(names) != len(set(names)):
            raise ValueError("task metric names must be non-empty and unique")
        if self.primary_metric not in names:
            raise ValueError("task primary metric must exist in metric contract")
        return self


class ModelIdentity(StrictModel):
    served_name: str
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_revision: str
    chat_template_revision: str


class ProviderIdentity(StrictModel):
    server_revision: str
    wkv_mode: str
    precision: str
    gemm_policy: str
    launch_contract: str
    attestation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    attestation_verified: bool
    attestation_present: bool
    attestation_mismatches: tuple[str, ...]


class RunIdentity(StrictModel):
    task: TaskIdentity
    model: ModelIdentity
    provider: ProviderIdentity
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    eligibility: Literal["official", "proxy", "sanity", "invalid"]
    comparable: bool

    @model_validator(mode="after")
    def validate_eligibility(self) -> "RunIdentity":
        if self.eligibility == "official" and (
            not self.comparable
            or not self.provider.attestation_verified
            or not self.provider.attestation_present
            or self.provider.attestation_mismatches
        ):
            raise ValueError(
                "official identity requires comparable verified provider attestation"
            )
        return self


class CreateRunRequest(StrictModel):
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    identity: RunIdentity
    expected_sample_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_sample_ids(self) -> "CreateRunRequest":
        if not self.expected_sample_ids or any(
            not sample_id for sample_id in self.expected_sample_ids
        ):
            raise ValueError("expected_sample_ids must contain non-empty identities")
        if len(set(self.expected_sample_ids)) != len(self.expected_sample_ids):
            raise ValueError("expected_sample_ids must be unique")
        return self

    @property
    def sample_set_digest(self) -> str:
        encoded = "\n".join(sorted(self.expected_sample_ids)).encode()
        return hashlib.sha256(encoded).hexdigest()


class ProviderUsageEvidence(StrictModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class GenerationEvidence(StrictModel):
    raw_completion: str
    output_token_ids: tuple[int, ...]
    output_token_count: int = Field(ge=0)
    finish_reason: str
    stop_reason: str | int | None
    terminal_reason: Literal["stop", "length"]
    truncated: bool
    generation_limit: int = Field(gt=0)
    prompt_text: str
    prompt_token_ids: tuple[int, ...]
    request_id: str
    usage: ProviderUsageEvidence

    @model_validator(mode="after")
    def validate_terminal_evidence(self) -> "GenerationEvidence":
        if self.output_token_count != len(self.output_token_ids):
            raise ValueError("output_token_count must equal output token id count")
        if self.terminal_reason == "stop" and self.truncated:
            raise ValueError("stop terminal reason cannot be truncated")
        if self.terminal_reason == "length" and not self.truncated:
            raise ValueError("length terminal reason must be truncated")
        if (
            self.terminal_reason == "length"
            and self.output_token_count != self.generation_limit
        ):
            raise ValueError("length termination must occur at the generation limit")
        if self.output_token_count > self.generation_limit:
            raise ValueError("output token count exceeds generation limit")
        if self.usage.prompt_tokens != len(self.prompt_token_ids):
            raise ValueError("prompt token usage must equal prompt token id count")
        if self.usage.completion_tokens != self.output_token_count:
            raise ValueError("completion token usage must equal output token count")
        if (
            self.usage.total_tokens
            != self.usage.prompt_tokens + self.usage.completion_tokens
        ):
            raise ValueError("total token usage is inconsistent")
        return self


class ScoringEvidence(StrictModel):
    raw_completion: str
    scored_completion: str
    scorer_revision: str
    repair_strategy: Literal["A", "B", "C", "not-applicable"]
    repair_action: Literal["none", "close-think-and-therefore", "append-therefore"]


SampleStatus = Literal[
    "scored",
    "model_invalid",
    "provider_error",
    "cache_error",
    "scorer_error",
    "harness_error",
    "cancelled",
]


class SampleEvidence(StrictModel):
    sample_id: str
    attempt: int = Field(gt=0)
    status: SampleStatus
    prompt: str
    raw_completion: str = ""
    scored_completion: str = ""
    generation: GenerationEvidence | None = None
    scoring: ScoringEvidence | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    provenance: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_evidence(self) -> "SampleEvidence":
        if any(not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("sample metrics must be finite")
        if self.status == "scored":
            if self.generation is None or self.scoring is None or not self.metrics:
                raise ValueError(
                    "scored sample requires generation, scoring, and metrics"
                )
            if self.raw_completion != self.generation.raw_completion:
                raise ValueError("sample raw completion must match generation evidence")
            if self.raw_completion != self.scoring.raw_completion:
                raise ValueError("scoring evidence must preserve raw completion")
            if self.scored_completion != self.scoring.scored_completion:
                raise ValueError("sample scored completion must match scoring evidence")
            if self.prompt != self.generation.prompt_text:
                raise ValueError("sample prompt must match signed generation prompt")
        elif self.metrics:
            raise ValueError("unscored sample must not contribute metrics")
        return self


class SampleAccounting(StrictModel):
    source_rows: int = Field(ge=0)
    dataset_accepted: int = Field(ge=0)
    dataset_rejected: int = Field(ge=0)
    selected: int = Field(ge=0)
    formatter_accepted: int = Field(ge=0)
    formatter_rejected: int = Field(ge=0)
    scored: int = Field(ge=0)
    model_invalid: int = Field(ge=0, default=0)
    provider_error: int = Field(ge=0, default=0)
    cache_error: int = Field(ge=0, default=0)
    scorer_error: int = Field(ge=0, default=0)
    harness_error: int = Field(ge=0, default=0)
    cancelled: int = Field(ge=0, default=0)

    @model_validator(mode="after")
    def close_partitions(self) -> "SampleAccounting":
        if self.source_rows != self.dataset_accepted + self.dataset_rejected:
            raise ValueError("source_rows partition does not close")
        if self.selected != self.formatter_accepted + self.formatter_rejected:
            raise ValueError("selected partition does not close")
        if self.formatter_accepted != sum(
            (
                self.scored,
                self.model_invalid,
                self.provider_error,
                self.cache_error,
                self.scorer_error,
                self.harness_error,
                self.cancelled,
            )
        ):
            raise ValueError("formatter_accepted terminal partition does not close")
        return self


class RejectionEvidence(StrictModel):
    stage: Literal["dataset", "formatter"]
    row_identity: str
    reason_code: str
    message: str


class ManifestEvidence(StrictModel):
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    accounting_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    terminal_status: Literal["completed", "partial", "failed", "invalid", "cancelled"]
    checksums_verified: Literal[True]


@dataclass(frozen=True, slots=True)
class IngestEvidenceSummary:
    latest_samples: dict[str, SampleEvidence]
    status_counts: dict[str, int]
    generated_samples: int
    truncated_samples: int
    dataset_rejections: int
    formatter_rejections: int


def summarize_ingest_evidence(
    samples: tuple[SampleEvidence, ...],
    rejections: tuple[RejectionEvidence, ...],
) -> IngestEvidenceSummary:
    keys = [(sample.sample_id, sample.attempt) for sample in samples]
    if len(set(keys)) != len(keys):
        raise ValueError("sample attempt identities must be unique")
    latest: dict[str, SampleEvidence] = {}
    for sample in sorted(samples, key=lambda item: item.attempt):
        latest[sample.sample_id] = sample
    status_counts = {
        name: 0
        for name in (
            "scored",
            "model_invalid",
            "provider_error",
            "cache_error",
            "scorer_error",
            "harness_error",
            "cancelled",
        )
    }
    for sample in latest.values():
        status_counts[sample.status] += 1
    return IngestEvidenceSummary(
        latest_samples=latest,
        status_counts=status_counts,
        generated_samples=sum(
            sample.generation is not None for sample in latest.values()
        ),
        truncated_samples=sum(
            sample.generation is not None and sample.generation.truncated
            for sample in latest.values()
        ),
        dataset_rejections=sum(item.stage == "dataset" for item in rejections),
        formatter_rejections=sum(item.stage == "formatter" for item in rejections),
    )


class IngestRunRequest(StrictModel):
    samples: tuple[SampleEvidence, ...]
    accounting: SampleAccounting
    rejections: tuple[RejectionEvidence, ...] = ()
    metrics: dict[str, float]
    primary_metric: str | None = None
    truncated_samples: int = Field(ge=0)
    generated_samples: int = Field(ge=0)
    manifest: ManifestEvidence
    terminal_status: Literal["completed", "partial", "failed", "invalid", "cancelled"]

    @model_validator(mode="after")
    def validate_completion(self) -> "IngestRunRequest":
        if any(not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("aggregate metrics must be finite")
        summary = self.evidence_summary()
        if self.generated_samples != summary.generated_samples:
            raise ValueError(
                "generated_samples must equal samples with generation evidence"
            )
        if self.truncated_samples != summary.truncated_samples:
            raise ValueError("truncated_samples does not match generation evidence")
        for name, count in summary.status_counts.items():
            if getattr(self.accounting, name) != count:
                raise ValueError(f"accounting {name} does not match sample evidence")
        if summary.dataset_rejections != self.accounting.dataset_rejected:
            raise ValueError("dataset rejection evidence does not close accounting")
        if summary.formatter_rejections != self.accounting.formatter_rejected:
            raise ValueError("formatter rejection evidence does not close accounting")
        if self.manifest.terminal_status != self.terminal_status:
            raise ValueError("manifest terminal status differs from ingest status")
        if self.terminal_status == "completed":
            self._validate_completed(summary)
        return self

    def evidence_summary(self) -> IngestEvidenceSummary:
        return summarize_ingest_evidence(self.samples, self.rejections)

    def _validate_completed(self, summary: IngestEvidenceSummary) -> None:
        if self.primary_metric is None or self.primary_metric not in self.metrics:
            raise ValueError("completed run requires a primary aggregate metric")
        if any(
            summary.status_counts[name]
            for name in summary.status_counts
            if name != "scored"
        ):
            raise ValueError("completed run cannot contain unscored samples")
        if self.accounting.dataset_rejected or self.accounting.formatter_rejected:
            raise ValueError("completed run cannot hide rejected samples")
        if not self.accounting.scored:
            raise ValueError("performance-only run cannot be completed")


class RunResponse(StrictModel):
    run_id: str
    status: str
    revision: int
    disposition: Literal["created", "unchanged"]


class IngestResponse(StrictModel):
    run_id: str
    status: str
    revision: int
    disposition: Literal["created", "unchanged"]


class MigrationResponse(StrictModel):
    disposition: Literal["applied", "unchanged"]
    schema_state: Literal["ready"]


class HealthResponse(StrictModel):
    status: Literal["ok", "degraded"]
    schema_state: Literal["missing", "outdated", "ahead", "drift", "ready"]


class MetadataResponse(StrictModel):
    run_count: int = Field(ge=0)
    model_count: int = Field(ge=0)
    models: tuple[str, ...]


class LeaderboardItem(StrictModel):
    run_id: str
    suite: str
    task_name: str
    task_version: str
    split_name: str
    fewshot: int
    model_name: str
    dataset_digest: str
    cot_mode: str
    repair_strategy: str
    completed_at: str
    metric_name: str
    metric_value: float


class LeaderboardPage(StrictModel):
    items: tuple[LeaderboardItem, ...]
    next_cursor: str | None


class HistoryItem(StrictModel):
    run_id: str
    suite: str
    task_name: str
    task_version: str
    model_name: str
    cot_mode: str
    repair_strategy: str
    eligibility: str
    comparable: bool
    status: str
    revision: int
    created_at: str
    updated_at: str
    completed_at: str | None


class HistoryPage(StrictModel):
    items: tuple[HistoryItem, ...]
    next_cursor: str | None


class AggregateEvidence(StrictModel):
    accounting: SampleAccounting
    metrics: dict[str, float]
    truncated_samples: int = Field(ge=0)
    generated_samples: int = Field(ge=0)
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class PerformanceEvidence(StrictModel):
    revision: int = Field(ge=1)
    metrics: dict[str, float]
    updated_at: str


class RunDetail(StrictModel):
    run_id: str
    identity: RunIdentity
    status: str
    revision: int = Field(ge=1)
    created_at: str
    updated_at: str
    completed_at: str | None
    aggregate: AggregateEvidence | None
    performance: PerformanceEvidence | None


class SamplePage(StrictModel):
    items: tuple[SampleEvidence, ...]
    next_cursor: str | None


class PerformancePatchRequest(StrictModel):
    metrics: dict[str, float]

    @model_validator(mode="after")
    def validate_metrics(self) -> "PerformancePatchRequest":
        if not self.metrics or any(
            not name or not math.isfinite(value) for name, value in self.metrics.items()
        ):
            raise ValueError(
                "performance metrics require non-empty names and finite values"
            )
        return self


class PerformancePatchResponse(StrictModel):
    run_id: str
    revision: int = Field(ge=1)
    disposition: Literal["created"]


class ErrorBody(StrictModel):
    code: str
    message: str
    request_id: str
    details: tuple[dict[str, JsonValue], ...] | None = None


class ErrorEnvelope(StrictModel):
    error: ErrorBody
