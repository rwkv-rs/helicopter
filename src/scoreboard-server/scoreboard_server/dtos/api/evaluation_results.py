"""Strict campaign, task publication, and complete-result contracts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


AnswerOutcome = Literal["correct", "incorrect", "unanswered", "undetermined"]
WkvMode = Literal["fp16", "fp32io16"]
PromptTemplate = Literal["bot", "assistant", "function_calling", "none"]
PROMPT_TEMPLATE_STOPS: dict[str, str] = {
    "bot": "✿",
    "assistant": "\nUser:",
    "function_calling": "\n### User",
}


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class EvaluatorMetadata(Contract):
    name: Literal["lighteval", "lm-eval"]
    version: str = Field(min_length=1, max_length=100)


class ExpectedTask(Contract):
    identity: str = Field(min_length=1, max_length=1000)
    weight_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weight_display_name: str = Field(min_length=1, max_length=500)
    wkv_mode: WkvMode
    selector: str = Field(min_length=1, max_length=500)
    task_name: str = Field(min_length=1, max_length=500)
    task_version: str = Field(min_length=1, max_length=100)
    module_family: str = Field(min_length=1, max_length=300)
    module: str = Field(min_length=1, max_length=500)
    dataset: str = Field(min_length=1, max_length=500)
    subset: str = Field(max_length=500)
    evaluation_splits: list[str] = Field(min_length=1)
    languages: list[str]
    upstream_tags: list[str]

    @model_validator(mode="after")
    def identity_matches_dimensions(self) -> "ExpectedTask":
        for name in (
            "identity",
            "weight_display_name",
            "selector",
            "task_name",
            "task_version",
            "module_family",
            "module",
            "dataset",
        ):
            value = getattr(self, name)
            if not value.strip() or value != value.strip():
                raise ValueError(f"{name} must be a non-empty trimmed string")
        if self.subset != self.subset.strip():
            raise ValueError("subset must be a trimmed string")
        expected = f"{self.weight_sha256}:{self.wkv_mode}:{self.task_name}"
        if self.identity != expected:
            raise ValueError("task identity does not match weight, WKV mode, and task")
        for name in (
            "evaluation_splits",
            "languages",
            "upstream_tags",
        ):
            values = getattr(self, name)
            if any(
                not value.strip() or value != value.strip() for value in values
            ) or len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique non-empty strings")
        return self


class CampaignCreate(Contract):
    schema_version: Literal["lighteval-campaign-v3", "lm-eval-campaign-v1"]
    run_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    eval_contract_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    lighteval_version: Literal["0.13.0"] | None = None
    evaluator: EvaluatorMetadata | None = None
    configured_selectors: list[str] = Field(min_length=1)
    resolved_selectors: list[str] = Field(min_length=1)
    skipped_selectors: list[str]
    expected_tasks: list[ExpectedTask] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_expected_tasks(self) -> "CampaignCreate":
        if self.schema_version == "lighteval-campaign-v3":
            if self.lighteval_version != "0.13.0" or self.evaluator is not None:
                raise ValueError("LightEval campaign requires lighteval_version")
        elif (
            self.lighteval_version is not None
            or self.evaluator is None
            or self.evaluator.name != "lm-eval"
            or self.evaluator.version != "0.4.12"
        ):
            raise ValueError("lm-eval campaign requires evaluator version 0.4.12")
        for name in (
            "configured_selectors",
            "resolved_selectors",
            "skipped_selectors",
        ):
            values = getattr(self, name)
            if any(
                not value.strip() or value != value.strip() for value in values
            ) or len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique trimmed strings")
        if not set(self.resolved_selectors).isdisjoint(self.skipped_selectors):
            raise ValueError("resolved and skipped selectors must be disjoint")
        if set(self.resolved_selectors) | set(self.skipped_selectors) != set(
            self.configured_selectors
        ):
            raise ValueError("selector status must partition configured selectors")
        task_selectors = {task.selector for task in self.expected_tasks}
        if not task_selectors.issubset(self.resolved_selectors):
            raise ValueError("expected tasks must use resolved selectors")
        if set(self.resolved_selectors) != task_selectors:
            raise ValueError("every resolved selector must produce an expected task")
        identities = [task.identity for task in self.expected_tasks]
        if len(identities) != len(set(identities)):
            raise ValueError("expected task identities must be unique")
        by_weight: dict[str, dict[str, list[ExpectedTask]]] = {}
        for task in self.expected_tasks:
            by_weight.setdefault(task.weight_sha256, {}).setdefault(
                task.task_name, []
            ).append(task)
        task_sets = {tuple(sorted(tasks)) for tasks in by_weight.values()}
        if len(task_sets) != 1:
            raise ValueError("every weight must evaluate the same task set")
        metadata_by_task: dict[str, tuple[object, ...]] = {}
        for weight_sha256, tasks in by_weight.items():
            display_names = {
                row.weight_display_name for rows in tasks.values() for row in rows
            }
            if len(display_names) != 1:
                raise ValueError(
                    f"weight display name differs for digest: {weight_sha256}"
                )
            for task_name, rows in tasks.items():
                if {row.wkv_mode for row in rows} != {"fp16", "fp32io16"}:
                    raise ValueError(f"task {task_name} must include both WKV modes")
                for row in rows:
                    metadata = (
                        row.task_version,
                        row.module_family,
                        row.module,
                        row.dataset,
                        row.subset,
                        tuple(row.evaluation_splits),
                        tuple(row.languages),
                        tuple(row.upstream_tags),
                        row.selector,
                    )
                    known = metadata_by_task.setdefault(task_name, metadata)
                    if known != metadata:
                        raise ValueError(
                            f"task metadata differs across weights/modes: {task_name}"
                        )
        return self

    @property
    def evaluator_name(self) -> str:
        return "lighteval" if self.evaluator is None else self.evaluator.name

    @property
    def evaluator_version(self) -> str:
        return self.lighteval_version or self.evaluator.version  # type: ignore[union-attr]


class CampaignReceipt(Contract):
    campaign_id: str
    disposition: Literal["created", "unchanged"]
    status: Literal["incomplete", "complete"]
    expected_task_count: int = Field(ge=1)
    acknowledged_task_digests: dict[str, str]


class PublicationPreflight(Contract):
    status: Literal["ready"]
    publisher_principal: str
    schema_version: Literal["lighteval-campaign-v3"]
    lighteval_version: Literal["0.13.0"]
    supported_campaign_schemas: list[str]
    evaluator_versions: dict[str, str]


class CampaignStatus(Contract):
    campaign_id: str
    status: Literal["incomplete", "complete"]
    expected_task_count: int = Field(ge=1)
    acknowledged_task_digests: dict[str, str]
    missing_task_identities: list[str]


class ModelExecution(Contract):
    weight_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weight_display_name: str = Field(min_length=1, max_length=500)
    wkv_mode: WkvMode
    prompt_template: PromptTemplate
    gemm_policy: Literal["fp16-accumulation", "fp32-accumulation"]
    gpu: str = Field(min_length=1, max_length=500)
    max_num_seqs: int = Field(gt=0)
    max_num_batched_tokens: int = Field(gt=0)
    dependency_versions: dict[str, str]
    evaluator: Literal["lighteval", "lm-eval"] = "lighteval"

    @model_validator(mode="after")
    def required_dependency_versions(self) -> "ModelExecution":
        required = {self.evaluator, "vllm", "torch"}
        if not required.issubset(self.dependency_versions) or any(
            not name or not version
            for name, version in self.dependency_versions.items()
        ):
            raise ValueError(
                "model execution must record evaluator, vllm, and torch versions"
            )
        return self


class ArtifactMetadata(Contract):
    lighteval_version: Literal["0.13.0"] | None = None
    evaluator: EvaluatorMetadata | None = None
    results_path: str = Field(min_length=1)
    details_paths: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def relative_standard_paths(self) -> "ArtifactMetadata":
        if (self.lighteval_version is None) == (self.evaluator is None):
            raise ValueError("artifact requires exactly one evaluator version")
        for raw in (self.results_path, *self.details_paths):
            path = PurePosixPath(raw)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "." in path.parts
                or str(path) != raw
            ):
                raise ValueError("artifact paths must be normalized relative paths")
        if self.evaluator is None and not self.results_path.startswith("results/"):
            raise ValueError("results_path must name a standard results child")
        if self.evaluator is None and any(
            not path.startswith("details/") for path in self.details_paths
        ):
            raise ValueError("details_paths must name standard details children")
        if len(self.details_paths) != len(set(self.details_paths)):
            raise ValueError("details_paths must be unique")
        return self

    @property
    def evaluator_name(self) -> str:
        return "lighteval" if self.evaluator is None else self.evaluator.name

    @property
    def evaluator_version(self) -> str:
        return self.lighteval_version or self.evaluator.version  # type: ignore[union-attr]


class Diagnostics(Contract):
    samples: int = Field(ge=0)
    completions: int = Field(ge=0)
    truncated: int = Field(ge=0)
    non_truncated: int = Field(ge=0)
    truncation_rate: float = Field(ge=0, le=1)
    turn_boundary_violations: int = Field(ge=0)
    turn_boundary_violation_rate: float = Field(ge=0, le=1)


class StandardDetail(Contract):
    sample_index: int = Field(ge=0)
    document_index: int = Field(ge=0)
    doc: dict[str, JsonValue]
    metric: dict[str, JsonValue]
    model_response: dict[str, JsonValue]


class TaskPublication(Contract):
    schema_version: Literal["lighteval-task-v2", "lm-eval-task-v1"]
    campaign_id: str
    task: ExpectedTask
    artifact: ArtifactMetadata
    task_config: dict[str, JsonValue]
    model: ModelExecution
    sampling_config: dict[str, JsonValue]
    primary_metric: str = Field(min_length=1, max_length=300)
    aggregates: dict[str, float]
    diagnostics: Diagnostics
    details: list[StandardDetail]

    @model_validator(mode="after")
    def validate_result(self) -> "TaskPublication":
        is_lm_eval = self.schema_version == "lm-eval-task-v1"
        if is_lm_eval != (self.model.evaluator == "lm-eval"):
            raise ValueError("task schema and model evaluator differ")
        if is_lm_eval != (self.model.prompt_template == "none"):
            raise ValueError("task schema and prompt template differ")
        if self.artifact.evaluator_name != self.model.evaluator:
            raise ValueError("artifact and model evaluators differ")
        if not self.aggregates or any(
            not key.strip() or key != key.strip() for key in self.aggregates
        ):
            raise ValueError(
                "aggregates must use non-empty trimmed native metric names"
            )
        if self.primary_metric not in self.aggregates:
            raise ValueError("primary_metric must exist in aggregates")
        if any(not math.isfinite(value) for value in self.aggregates.values()):
            raise ValueError("aggregates must be finite")
        ordinals = [detail.sample_index for detail in self.details]
        if ordinals != list(range(len(self.details))):
            raise ValueError("detail sample_index values must be consecutive from zero")
        for index, detail in enumerate(self.details):
            detail_task = detail.doc.get("task_name")
            if detail_task != self.task.task_name:
                raise ValueError(
                    f"details[{index}].doc.task_name does not match task_name"
                )
            specific = detail.doc.get("specific")
            if (
                not isinstance(specific, dict)
                or specific.get("helicopter_document_index") != detail.document_index
            ):
                raise ValueError(
                    f"details[{index}] stable document index is inconsistent"
                )
        if self.model.weight_sha256 != self.task.weight_sha256:
            raise ValueError("model weight digest does not match expected task")
        if self.model.weight_display_name != self.task.weight_display_name:
            raise ValueError("model display name does not match expected task")
        if self.model.wkv_mode != self.task.wkv_mode:
            raise ValueError("model WKV mode does not match expected task")
        expected_gemm_policy = (
            "fp16-accumulation"
            if self.model.wkv_mode == "fp16"
            else "fp32-accumulation"
        )
        if self.model.gemm_policy != expected_gemm_policy:
            raise ValueError("model GEMM policy does not match WKV mode")
        if self.artifact.evaluator_version != self.model.dependency_versions.get(
            self.model.evaluator
        ):
            raise ValueError("artifact and model evaluator versions differ")
        original_docs = self.task_config.get("original_num_docs")
        effective_docs = self.task_config.get("effective_num_docs")
        skipped_multiselect_docs = self.task_config.get("skipped_multiselect_docs")
        document_indices = {detail.document_index for detail in self.details}
        if (
            isinstance(original_docs, bool)
            or not isinstance(original_docs, int)
            or isinstance(effective_docs, bool)
            or not isinstance(effective_docs, int)
            or isinstance(skipped_multiselect_docs, bool)
            or not isinstance(skipped_multiselect_docs, int)
            or original_docs <= 0
            or effective_docs <= 0
            or skipped_multiselect_docs < 0
            or original_docs != effective_docs + skipped_multiselect_docs
            or document_indices != set(range(effective_docs))
        ):
            raise ValueError(
                "task config and details do not account for the full evaluation split"
            )
        if is_lm_eval:
            if self.artifact.evaluator_version != "0.4.12":
                raise ValueError("lm-eval artifact version must be 0.4.12")
            if (
                self.diagnostics.samples != len(self.details)
                or self.diagnostics.truncated
                + self.diagnostics.non_truncated
                != self.diagnostics.completions
            ):
                raise ValueError("lm-eval diagnostics do not match task details")
            return self
        required_sampling: dict[str, JsonValue] = {
            "temperature": 0.96,
            "top_p": 0.76,
            "top_k": 32,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.1,
            "repetition_penalty": 1.0,
            "penalty_decay": 0.988,
            "max_new_tokens": 8192,
            "stop": [PROMPT_TEMPLATE_STOPS[self.model.prompt_template]],
            "ignore_eos": False,
        }
        mismatched = [
            key
            for key, expected in required_sampling.items()
            if self.sampling_config.get(key) != expected
        ]
        if mismatched:
            raise ValueError(
                "sampling config violates eval contract: " + ", ".join(mismatched)
            )
        computed = compute_diagnostics(self)
        if self.diagnostics != computed:
            raise ValueError(
                "diagnostics do not match raw completions and output tokens"
            )
        return self


class TaskReceipt(Contract):
    evaluation_id: str
    task_identity: str
    content_digest: str
    disposition: Literal["created", "unchanged"]


class FinalizeReceipt(Contract):
    campaign_id: str
    status: Literal["complete"]
    task_count: int = Field(ge=1)


class CampaignProvenance(Contract):
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    eval_contract_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    lighteval_version: Literal["0.13.0"] | None = None
    evaluator: EvaluatorMetadata | None = None
    configured_selectors: list[str]
    resolved_selectors: list[str]
    skipped_selectors: list[str]
    publisher_principal: str

    @model_validator(mode="after")
    def exactly_one_evaluator(self) -> "CampaignProvenance":
        if (self.lighteval_version is None) == (self.evaluator is None):
            raise ValueError("provenance requires exactly one evaluator version")
        return self


class EvaluationSummary(Contract):
    evaluation_id: str
    campaign_id: str
    task_identity: str
    created_at: str
    completed_at: str
    task: ExpectedTask
    artifact: ArtifactMetadata
    task_config: dict[str, JsonValue]
    model: ModelExecution
    sampling_config: dict[str, JsonValue]
    primary_metric: str
    aggregates: dict[str, float]
    diagnostics: Diagnostics
    provenance: CampaignProvenance


class EvaluationList(Contract):
    evaluations: list[EvaluationSummary]
    generated_at: str
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(gt=0)
    next_offset: int | None


class SampleDetail(Contract):
    id: str
    sample_index: int
    document_index: int
    outcome: AnswerOutcome
    doc: dict[str, JsonValue]
    metric: dict[str, JsonValue]
    model_response: dict[str, JsonValue]


class SamplePage(Contract):
    evaluation_id: str
    primary_metric: str
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(gt=0)
    next_offset: int | None
    items: list[SampleDetail]


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


def _positive_limit(publication: TaskPublication) -> int:
    configured = publication.sampling_config.get("max_new_tokens")
    if (
        isinstance(configured, bool)
        or not isinstance(configured, int)
        or configured <= 0
    ):
        configured = publication.task_config.get("generation_size")
    if (
        isinstance(configured, bool)
        or not isinstance(configured, int)
        or configured <= 0
    ):
        raise ValueError("effective output limit must be a positive integer")
    return configured


def compute_diagnostics(publication: TaskPublication) -> Diagnostics:
    limit = _positive_limit(publication)
    completions = 0
    truncated = 0
    violations = 0
    for detail in publication.details:
        response = detail.model_response
        texts = response.get("text")
        output_tokens = response.get("output_tokens")
        if texts in (None, []):
            _validate_loglikelihood_response(response)
            continue
        if not isinstance(texts, list) or not isinstance(output_tokens, list):
            raise ValueError("completion text and output_tokens must be arrays")
        if len(texts) != len(output_tokens):
            raise ValueError("completion and output-token counts differ")
        _validate_optional_text_output(
            response,
            key="text_post_processed",
            expected_count=len(texts),
        )
        _validate_optional_text_output(
            response,
            key="reasonings",
            expected_count=len(texts),
            allow_none=True,
        )
        for text, tokens in zip(texts, output_tokens, strict=True):
            if not isinstance(text, str):
                raise ValueError("completion text must be a string")
            _validate_token_group(tokens)
            completions += 1
            truncated += int(len(tokens) >= limit)
            violations += int(
                PROMPT_TEMPLATE_STOPS[publication.model.prompt_template] in text
            )
    return Diagnostics(
        samples=len(publication.details),
        completions=completions,
        truncated=truncated,
        non_truncated=completions - truncated,
        truncation_rate=truncated / completions if completions else 0.0,
        turn_boundary_violations=violations,
        turn_boundary_violation_rate=violations / completions if completions else 0.0,
    )


def _validate_token_group(value: object) -> None:
    if not isinstance(value, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in value
    ):
        raise ValueError("output tokens must be integer arrays")


def _validate_optional_text_output(
    response: dict[str, JsonValue],
    *,
    key: str,
    expected_count: int,
    allow_none: bool = False,
) -> None:
    value = response.get(key)
    if value is None or (key == "reasonings" and value == []):
        return
    if (
        not isinstance(value, list)
        or len(value) != expected_count
        or any(
            not isinstance(item, str) and not (allow_none and item is None)
            for item in value
        )
    ):
        raise ValueError(f"{key} must align one-for-one with completion text")


def _validate_loglikelihood_response(
    response: dict[str, JsonValue],
) -> None:
    logprobs = response.get("logprobs")
    argmax = response.get("argmax_logits_eq_gold")
    if logprobs in (None, []) and argmax in (None, []):
        raise ValueError("empty completion lacks log-likelihood evidence")
    if logprobs not in (None, []):
        if not isinstance(logprobs, list) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in logprobs
        ):
            raise ValueError("logprobs must be a finite numeric array")
    if argmax not in (None, []):
        if not isinstance(argmax, list) or any(
            not isinstance(value, bool) for value in argmax
        ):
            raise ValueError("argmax evidence must be a boolean array")
    evidence_count = (
        len(logprobs) if isinstance(logprobs, list) and logprobs else len(argmax)
    )
    if (
        isinstance(logprobs, list)
        and logprobs
        and isinstance(argmax, list)
        and argmax
        and len(logprobs) != len(argmax)
    ):
        raise ValueError("log-likelihood evidence counts differ")
    output_tokens = response.get("output_tokens")
    if not isinstance(output_tokens, list) or not output_tokens:
        raise ValueError("log-likelihood output_tokens must be a non-empty array")
    for token_group in output_tokens:
        _validate_token_group(token_group)
        if not token_group:
            raise ValueError("log-likelihood output token groups must be non-empty")
    if len(output_tokens) != evidence_count:
        raise ValueError("log-likelihood evidence and output-token counts differ")
    _validate_optional_text_output(
        response,
        key="text_post_processed",
        expected_count=0,
    )
    _validate_optional_text_output(
        response,
        key="reasonings",
        expected_count=0,
        allow_none=True,
    )


def sample_outcome(detail: StandardDetail, primary_metric: str) -> AnswerOutcome:
    value = detail.metric.get(primary_metric)
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        if value == 1:
            return "correct"
        if value == 0:
            return "incorrect"
    response = detail.model_response
    has_text = any(
        isinstance(item, str) and item.strip()
        for key in ("text", "text_post_processed")
        for item in (response.get(key) if isinstance(response.get(key), list) else [])
    )
    has_logprob = any(
        response.get(key) not in (None, [])
        for key in ("logprobs", "argmax_logits_eq_gold")
    )
    has_native_response = any(
        response.get(key) not in (None, [], {})
        for key in ("filtered_resps", "resps")
    )
    if not has_text and not has_logprob and not has_native_response:
        return "unanswered"
    return "undetermined"
