from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .context import ContextBudgetPolicy, apply_context_budget
from .execution import DISABLED_CACHE_POLICY_REVISION, SampleStatus
from .generation import GenerationOutcome
from .provider.endpoint import (
    EndpointGenerationRequest,
    OpenAIEndpoint,
    ProviderResponseSchemaError,
    ProviderTransportError,
)
from .provider.attestation import ProviderAttestation
from .registry import TaskDefinition
from .results.records import SampleResult, ScoringEvidence
from .task_runtime import HarnessFailure, ModelOutputRejected, TaskRuntime


@dataclass(frozen=True, slots=True)
class SampleEvaluation:
    evidence: dict[str, Any] | None
    formatter_rejection: dict[str, str] | None
    primary_metric: float | None
    generated: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class SampleExecutionContext:
    model: str
    generation_limit: int
    cot_mode: str
    math_repair_strategy: str
    cache_namespace: str
    config_digest: str
    asset_manifest_digest: str
    provider_attestation: ProviderAttestation | None

    @property
    def generation_prompt_mode(self) -> str:
        return "open_think" if self.cot_mode == "cot" else "fake_think"

    @property
    def provider_attestation_digest(self) -> str | None:
        return (
            _digest(asdict(self.provider_attestation))
            if self.provider_attestation is not None
            else None
        )


class SampleEvaluator:
    """Own the complete typed lifecycle of one selected dataset row."""

    def __init__(
        self,
        *,
        definition: TaskDefinition,
        runtime: TaskRuntime,
        endpoint: OpenAIEndpoint,
        context: SampleExecutionContext,
    ) -> None:
        self._definition = definition
        self._runtime = runtime
        self._endpoint = endpoint
        self._context = context

    def evaluate(self, row_id: str, payload: Mapping[str, Any]) -> SampleEvaluation:
        try:
            prepared = self._runtime.prepare(payload)
        except (KeyError, TypeError, ValueError) as error:
            return SampleEvaluation(
                evidence=None,
                formatter_rejection={
                    "stage": "formatter",
                    "row_identity": row_id,
                    "reason_code": "task_formatter_rejected",
                    "message": str(error),
                },
                primary_metric=None,
                generated=False,
                truncated=False,
            )

        prompt = ""
        try:
            initial_context = self._endpoint.tokenize_context(
                self._context.model,
                prepared.context.sections,
                self._context.generation_prompt_mode,
            )
            max_prompt_tokens = (
                initial_context.max_model_len - self._context.generation_limit
            )
            if max_prompt_tokens <= 0:
                raise ValueError(
                    "generation limit leaves no provider context budget: "
                    "model="
                    f"{initial_context.max_model_len}, generation={self._context.generation_limit}"
                )
            budgeted_context = apply_context_budget(
                prepared.context,
                ContextBudgetPolicy(max_prompt_tokens),
                lambda sections: self._endpoint.context_token_count(
                    self._context.model,
                    sections,
                    self._context.generation_prompt_mode,
                ),
            )
            generation = self._endpoint.generate(
                EndpointGenerationRequest(
                    model=self._context.model,
                    context=budgeted_context,
                    generation_limit=self._context.generation_limit,
                    generation_prompt_mode=self._context.generation_prompt_mode,
                )
            )
            if generation.prompt_text is None:
                raise ProviderResponseSchemaError(
                    "provider omitted signed prompt text evidence"
                )
            prompt = generation.prompt_text
        except (
            ProviderTransportError,
            ProviderResponseSchemaError,
            RuntimeError,
            ValueError,
        ) as error:
            return SampleEvaluation(
                evidence=_error_sample(
                    row_id,
                    prompt,
                    SampleStatus.PROVIDER_ERROR,
                    "provider_generation_failed",
                    error,
                ),
                formatter_rejection=None,
                primary_metric=None,
                generated=False,
                truncated=False,
            )

        raw_completion = generation.raw_completion
        transform = self._definition.scoring_transform(
            prompt,
            raw_completion,
            generation.truncated,
            self._context.math_repair_strategy,
        )
        scoring = ScoringEvidence(
            raw_completion=raw_completion,
            scored_completion=transform.completion,
            scorer_revision=self._definition.contract.scorer_revision,
            repair_strategy=transform.strategy,
            repair_action=transform.action,
        )
        try:
            metric_payload = self._runtime.score(
                prepared,
                prompt=prompt,
                completion=transform.completion,
                output_token_ids=generation.output_token_ids,
            )
            metrics = {name: float(value) for name, value in metric_payload.items()}
            self._definition.contract.metric.validate(metrics)
        except ModelOutputRejected as error:
            return self._scoring_error(
                row_id,
                prompt,
                generation,
                scoring,
                SampleStatus.MODEL_INVALID,
                "task_output_rejected",
                error,
            )
        except HarnessFailure as error:
            return self._scoring_error(
                row_id,
                prompt,
                generation,
                scoring,
                SampleStatus.HARNESS_ERROR,
                "task_harness_failed",
                error,
            )
        except Exception as error:
            return self._scoring_error(
                row_id,
                prompt,
                generation,
                scoring,
                SampleStatus.SCORER_ERROR,
                "task_native_scorer_failed",
                error,
            )

        evidence = SampleResult(
            sample_id=row_id,
            prompt=prompt,
            status=SampleStatus.SCORED,
            metrics=metrics,
            generation=generation,
            scoring=scoring,
            provenance={
                "cache": {
                    "status": "disabled",
                    "policy_revision": DISABLED_CACHE_POLICY_REVISION,
                    "reason": "no verified persistent cache implementation",
                    "key": self._context.cache_namespace,
                },
                "context": {
                    "dropped_sections": list(budgeted_context.dropped_sections),
                    "prompt_token_count": budgeted_context.prompt_token_count,
                    "prompt_token_ids_digest": _digest(
                        list(generation.prompt_token_ids)
                    ),
                },
                "config_digest": self._context.config_digest,
                "asset_manifest_digest": self._context.asset_manifest_digest,
                "provider_attestation": self._context.provider_attestation_digest,
            },
        ).to_evidence_payload()
        primary = metrics[self._definition.contract.metric.primary_metric]
        return SampleEvaluation(evidence, None, primary, True, generation.truncated)

    @staticmethod
    def cancelled(row_id: str) -> dict[str, Any]:
        return _error_sample(
            row_id,
            "",
            SampleStatus.CANCELLED,
            "run_cancelled",
            RuntimeError("evaluation was cancelled"),
        )

    def _scoring_error(
        self,
        row_id: str,
        prompt: str,
        generation: GenerationOutcome,
        scoring: ScoringEvidence,
        status: SampleStatus,
        error_code: str,
        error: Exception,
    ) -> SampleEvaluation:
        return SampleEvaluation(
            evidence=_error_sample(
                row_id,
                prompt,
                status,
                error_code,
                error,
                generation=generation,
                scoring=scoring,
            ),
            formatter_rejection=None,
            primary_metric=None,
            generated=True,
            truncated=generation.truncated,
        )


def _error_sample(
    sample_id: str,
    prompt: str,
    status: SampleStatus,
    error_code: str,
    error: Exception,
    *,
    generation: GenerationOutcome | None = None,
    scoring: ScoringEvidence | None = None,
) -> dict[str, Any]:
    return SampleResult(
        sample_id=sample_id,
        prompt=prompt,
        status=status,
        metrics={},
        generation=generation,
        scoring=scoring,
        error_code=error_code,
        error_message=str(error),
    ).to_evidence_payload()


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
