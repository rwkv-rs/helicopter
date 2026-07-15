from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class MetricIdentity(StrictModel):
    name: str = Field(min_length=1)
    minimum: float
    maximum: float
    aggregation: Literal["mean", "sum"]
    binary_correctness: bool

    @model_validator(mode="after")
    def validate_range(self) -> MetricIdentity:
        if self.minimum > self.maximum:
            raise ValueError("metric minimum exceeds maximum")
        return self


class TaskIdentity(StrictModel):
    suite: str = Field(min_length=1)
    task: str = Field(min_length=1)
    version: str = Field(min_length=1)
    split: str = Field(min_length=1)
    fewshot: int = Field(ge=0)
    prompt_revision: str = Field(min_length=1)
    scorer_revision: str = Field(min_length=1)
    generation_contract: str = Field(min_length=1)
    cot_mode: Literal["none", "cot"]
    repair_strategy: str = Field(min_length=1)
    dataset_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_metric: str = Field(min_length=1)
    metrics: list[MetricIdentity] = Field(min_length=1)


class ModelIdentity(StrictModel):
    served_name: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_revision: str = Field(min_length=1)
    chat_template_revision: str = Field(min_length=1)


class ProviderIdentity(StrictModel):
    server_revision: str = Field(min_length=1)
    wkv_mode: str = Field(min_length=1)
    precision: str = Field(min_length=1)
    gemm_policy: str = Field(min_length=1)
    launch_contract: str = Field(min_length=1)
    attestation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    attestation_verified: bool
    attestation_present: bool
    attestation_mismatches: list[str]


class EvaluatorIdentity(StrictModel):
    product_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty: bool


class PublicationIdentity(StrictModel):
    task: TaskIdentity
    model: ModelIdentity
    provider: ProviderIdentity
    evaluator: EvaluatorIdentity
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    eligibility: Literal["official", "proxy", "sanity"]
    comparable: bool

    @model_validator(mode="after")
    def validate_eligibility(self) -> PublicationIdentity:
        provider_verified = (
            self.provider.attestation_verified
            and self.provider.attestation_present
            and not self.provider.attestation_mismatches
        )
        if self.comparable != provider_verified:
            raise ValueError("comparability does not match provider attestation")
        if self.eligibility == "official" and (
            not self.comparable or self.evaluator.dirty
        ):
            raise ValueError(
                "official publication requires verified evidence from a clean evaluator"
            )
        return self


class SampleAccounting(StrictModel):
    source_rows: int = Field(ge=0)
    dataset_accepted: int = Field(ge=0)
    dataset_rejected: int = Field(ge=0)
    selected: int = Field(ge=0)
    formatter_accepted: int = Field(ge=0)
    formatter_rejected: int = Field(ge=0)
    scored: int = Field(ge=0)
    model_invalid: int = Field(ge=0)
    provider_error: int = Field(ge=0)
    cache_error: int = Field(ge=0)
    scorer_error: int = Field(ge=0)
    harness_error: int = Field(ge=0)
    cancelled: int = Field(ge=0)


class TokenUsage(StrictModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> TokenUsage:
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("token usage total does not close")
        return self


class GenerationEvidence(StrictModel):
    output_token_count: int = Field(ge=0)
    finish_reason: str
    stop_reason: str | int | None
    terminal_reason: Literal["stop", "length"]
    truncated: bool
    generation_limit: int = Field(gt=0)
    request_id: str | None
    usage: TokenUsage | None

    @model_validator(mode="after")
    def validate_generation_evidence(self) -> GenerationEvidence:
        if self.truncated != (self.terminal_reason == "length"):
            raise ValueError("truncation does not match terminal reason")
        if self.truncated and self.output_token_count != self.generation_limit:
            raise ValueError("truncated generation did not reach its token limit")
        return self


class ScoringEvidence(StrictModel):
    scorer_revision: str = Field(min_length=1)
    repair_strategy: str = Field(min_length=1)
    repair_action: str = Field(min_length=1)


class PublishedSample(StrictModel):
    sample_index: int = Field(ge=0)
    sample_id: str = Field(min_length=1)
    attempt: int = Field(gt=0)
    status: Literal["scored"]
    prompt: str
    raw_completion: str
    scored_completion: str
    generation: GenerationEvidence
    scoring: ScoringEvidence
    metrics: dict[str, float]
    error_code: None
    error_message: None
    reference_answer: str | None
    provenance: dict[str, Any]


class ManifestEvidence(StrictModel):
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    accounting_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    terminal_status: Literal["completed"]
    checksums_verified: Literal[True]
    completed_at: datetime

    @field_validator("completed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("manifest completion time must include a timezone")
        return value


class EvaluationPublicationRequest(StrictModel):
    identity: PublicationIdentity
    samples: list[PublishedSample] = Field(min_length=1)
    accounting: SampleAccounting
    rejections: list[dict[str, str]]
    metrics: dict[str, float]
    primary_metric: str = Field(min_length=1)
    truncated_samples: int = Field(ge=0)
    generated_samples: int = Field(gt=0)
    performance: dict[str, Any]
    manifest: ManifestEvidence
    terminal_status: Literal["completed"]

    @model_validator(mode="after")
    def validate_completed_projection(self) -> EvaluationPublicationRequest:
        accounting = self.accounting
        if (
            accounting.source_rows
            != accounting.dataset_accepted + accounting.dataset_rejected
        ):
            raise ValueError("source row accounting does not close")
        if (
            accounting.selected
            != accounting.formatter_accepted + accounting.formatter_rejected
        ):
            raise ValueError("selected sample accounting does not close")
        terminal = sum(
            (
                accounting.scored,
                accounting.model_invalid,
                accounting.provider_error,
                accounting.cache_error,
                accounting.scorer_error,
                accounting.harness_error,
                accounting.cancelled,
            )
        )
        if accounting.formatter_accepted != terminal:
            raise ValueError("terminal sample accounting does not close")
        if (
            accounting.dataset_rejected
            or accounting.formatter_rejected
            or terminal != accounting.scored
            or accounting.scored != len(self.samples)
            or accounting.selected != len(self.samples)
            or self.rejections
        ):
            raise ValueError("completed publication must contain only scored samples")
        if {sample.sample_index for sample in self.samples} != set(
            range(len(self.samples))
        ):
            raise ValueError("sample_index must be a unique contiguous ordinal")
        if self.generated_samples != len(self.samples):
            raise ValueError("generated sample count does not match samples")
        if self.truncated_samples > self.generated_samples:
            raise ValueError("truncated sample count exceeds generated samples")
        if self.truncated_samples != sum(
            sample.generation.truncated for sample in self.samples
        ):
            raise ValueError("truncated sample count does not match sample evidence")
        if self.primary_metric != self.identity.task.primary_metric:
            raise ValueError("primary metric does not match task identity")
        if set(self.metrics) != {self.primary_metric}:
            raise ValueError(
                "publication must contain exactly the primary aggregate metric"
            )
        metric_identity = [
            metric
            for metric in self.identity.task.metrics
            if metric.name == self.primary_metric
        ]
        if len(metric_identity) != 1:
            raise ValueError("primary metric identity is missing or ambiguous")
        for sample in self.samples:
            if self.primary_metric not in sample.metrics:
                raise ValueError("sample is missing the primary metric")
            if sample.scoring.scorer_revision != self.identity.task.scorer_revision:
                raise ValueError("sample scorer revision does not match task identity")
            if sample.scoring.repair_strategy != self.identity.task.repair_strategy:
                raise ValueError("sample repair strategy does not match task identity")
        return self


class EvaluationPublicationResponse(StrictModel):
    run_id: str
    task_id: int
    status: Literal["completed"]
    disposition: Literal["created", "unchanged"]
