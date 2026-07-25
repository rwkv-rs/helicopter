"""HTTP and persistence contracts for stored evaluation results."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


AnswerOutcome = Literal["correct", "incorrect", "unanswered", "undetermined"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ModelVariant(Contract):
    label: str = Field(min_length=1, max_length=200)
    architecture: str = Field(min_length=1, max_length=100)
    generation: str = Field(min_length=1, max_length=100)
    parameters: str = Field(min_length=1, max_length=100)


class ComparisonOption(Contract):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)
    label: str = Field(min_length=1, max_length=200)
    short_label: str = Field(min_length=1, max_length=100)
    a_label: str = Field(min_length=1, max_length=100)
    b_label: str = Field(min_length=1, max_length=100)
    contract: str = Field(min_length=1, max_length=2000)


class ParameterGroup(Contract):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$", max_length=100)
    label: str = Field(min_length=1, max_length=100)
    a_model: ModelVariant
    b_model: ModelVariant
    parameter_delta_percent: float = Field(ge=0)
    comparable: bool


class ComparisonCoordinate(Contract):
    comparison: ComparisonOption
    parameter_group: ParameterGroup
    arm: Literal["a", "b"]


class BenchmarkMetadata(Contract):
    label: str = Field(min_length=1, max_length=200)
    domain: str = Field(min_length=1, max_length=100)
    evaluation_method: str = Field(min_length=1, max_length=100)
    score_multiplier: float = Field(gt=0)


class EvaluationMetadata(Contract):
    prompt_profile: str = Field(min_length=1, max_length=100)
    prompt_template: str
    precision: str = Field(min_length=1, max_length=100)


class ArtifactMetadata(Contract):
    lighteval_version: str = Field(min_length=1, max_length=100)
    results_path: str = Field(min_length=1)
    details_paths: list[str] = Field(min_length=1)


class Diagnostics(Contract):
    samples: int = Field(ge=0)
    completions: int = Field(ge=0)
    truncated: int = Field(ge=0)
    non_truncated: int = Field(ge=0)
    truncation_rate: float = Field(ge=0, le=1)
    turn_boundary_violations: int = Field(ge=0)
    turn_boundary_violation_rate: float = Field(ge=0, le=1)


class StandardDetail(Contract):
    doc: dict[str, JsonValue]
    metric: dict[str, JsonValue]
    model_response: dict[str, JsonValue]


class EvaluationPublication(Contract):
    schema_version: Literal["lighteval-standard-v1"]
    source_run_id: str = Field(min_length=1, max_length=200)
    artifact: ArtifactMetadata
    task_name: str = Field(min_length=1, max_length=500)
    task_config: dict[str, JsonValue]
    model: ModelVariant
    benchmark: BenchmarkMetadata
    evaluation: EvaluationMetadata
    comparisons: list[ComparisonCoordinate] = Field(default_factory=list)
    sampling_config: dict[str, JsonValue]
    primary_metric: str = Field(min_length=1, max_length=300)
    aggregates: dict[str, float]
    diagnostics: Diagnostics
    details: list[StandardDetail]

    @model_validator(mode="after")
    def validate_result(self) -> "EvaluationPublication":
        if self.primary_metric not in self.aggregates:
            raise ValueError("primary_metric must exist in aggregates")
        if self.diagnostics.samples != len(self.details):
            raise ValueError("diagnostics.samples must equal details length")
        for index, detail in enumerate(self.details):
            detail_task = detail.doc.get("task_name")
            if detail_task is not None and detail_task != self.task_name:
                raise ValueError(
                    f"details[{index}].doc.task_name does not match task_name"
                )
        return self


class PublicationReceipt(Contract):
    evaluation_id: str
    publication_id: str
    content_digest: str
    disposition: Literal["created", "unchanged"]


class EvaluationSummary(Contract):
    evaluation_id: str
    publication_id: str
    source_run_id: str
    source: str
    visibility: Literal["non_official"]
    created_at: str
    task_name: str
    model: ModelVariant
    benchmark: BenchmarkMetadata
    evaluation: EvaluationMetadata
    comparisons: list[ComparisonCoordinate]
    sampling_config: dict[str, JsonValue]
    primary_metric: str
    aggregates: dict[str, float]
    diagnostics: Diagnostics


class EvaluationList(Contract):
    evaluations: list[EvaluationSummary]
    generated_at: str


class SampleDetail(Contract):
    id: str
    sample_index: int
    outcome: AnswerOutcome
    doc: dict[str, JsonValue]
    metric: dict[str, JsonValue]
    model_response: dict[str, JsonValue]


class SampleGroup(Contract):
    outcome: AnswerOutcome
    total: int
    items: list[SampleDetail]


class SampleGroups(Contract):
    evaluation_id: str
    primary_metric: str
    groups: dict[AnswerOutcome, SampleGroup]


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sample_outcome(detail: StandardDetail, primary_metric: str) -> AnswerOutcome:
    value = detail.metric.get(primary_metric)
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        if value == 1:
            return "correct"
        if value == 0:
            return "incorrect"

    response = detail.model_response
    texts = response.get("text")
    processed = response.get("text_post_processed")
    has_text = any(
        isinstance(item, str) and item.strip()
        for values in (texts, processed)
        if isinstance(values, list)
        for item in values
    )
    has_logprob_evidence = any(
        response.get(key) not in (None, [])
        for key in ("logprobs", "argmax_logits_eq_gold")
    )
    if not has_text and not has_logprob_evidence:
        return "unanswered"
    return "undetermined"
