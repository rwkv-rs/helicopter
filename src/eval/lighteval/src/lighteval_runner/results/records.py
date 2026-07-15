from __future__ import annotations

from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from ..execution import Eligibility, RunStatus, SampleAccounting, SampleStatus
from ..generation import GenerationOutcome


@dataclass(frozen=True, slots=True)
class ScoringEvidence:
    raw_completion: str
    scored_completion: str
    scorer_revision: str
    repair_strategy: str
    repair_action: str


@dataclass(frozen=True, slots=True)
class SampleResult:
    sample_id: str
    prompt: str
    status: SampleStatus
    metrics: Mapping[str, float]
    generation: GenerationOutcome | None = None
    scoring: ScoringEvidence | None = None
    attempt: int = 1
    provenance: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        if self.attempt <= 0:
            raise ValueError("sample attempt must be positive")
        if self.status is SampleStatus.SCORED:
            if self.generation is None or self.scoring is None or not self.metrics:
                raise ValueError(
                    "scored samples require generation, scoring, and metrics"
                )
            if self.scoring.raw_completion != self.generation.raw_completion:
                raise ValueError("scoring evidence must preserve the raw generation")
        elif self.metrics:
            raise ValueError("unscored samples must not contribute metrics")

    def to_evidence_payload(self) -> dict[str, Any]:
        raw = self.generation.raw_completion if self.generation is not None else ""
        scored = self.scoring.scored_completion if self.scoring is not None else raw
        generation = None
        if self.generation is not None:
            generation = {
                "raw_completion": self.generation.raw_completion,
                "output_token_ids": list(self.generation.output_token_ids),
                "output_token_count": self.generation.output_token_count,
                "finish_reason": self.generation.provider_finish_reason,
                "stop_reason": self.generation.provider_stop_reason,
                "terminal_reason": self.generation.stop_reason.value,
                "truncated": self.generation.truncated,
                "generation_limit": self.generation.generation_limit,
                "prompt_text": self.generation.prompt_text,
                "prompt_token_ids": list(self.generation.prompt_token_ids),
                "request_id": self.generation.request_id,
                "usage": (
                    asdict(self.generation.usage)
                    if self.generation.usage is not None
                    else None
                ),
            }
        return {
            "sample_id": self.sample_id,
            "attempt": self.attempt,
            "status": self.status.value,
            "prompt": self.prompt,
            "raw_completion": raw,
            "scored_completion": scored,
            "generation": generation,
            "scoring": asdict(self.scoring) if self.scoring is not None else None,
            "metrics": dict(self.metrics),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    status: RunStatus
    eligibility: Eligibility
    accounting: SampleAccounting
    metrics: Mapping[str, float]
    truncated_samples: int
    generated_samples: int

    @property
    def truncation_rate(self) -> float | None:
        if self.generated_samples == 0:
            return None
        return self.truncated_samples / self.generated_samples

    def validate_completed(
        self, *, provider_valid: bool, aggregation_valid: bool, manifest_committed: bool
    ) -> None:
        self.accounting.validate()
        if self.status is not RunStatus.COMPLETED:
            raise ValueError("completion gate requires completed run status")
        if not provider_valid or not aggregation_valid or not manifest_committed:
            raise ValueError(
                "completed run is missing provider, aggregation, or manifest evidence"
            )
        if self.generated_samples != self.accounting.scored:
            raise ValueError(
                "completed run must have generation evidence for every scored sample"
            )
        if (
            self.truncated_samples < 0
            or self.truncated_samples > self.generated_samples
        ):
            raise ValueError("truncated_samples must be within generated sample bounds")
        if self.accounting.scored == 0:
            raise ValueError("performance-only runs cannot be completed evaluations")
