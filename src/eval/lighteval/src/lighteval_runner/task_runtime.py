from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .context import ContextDocument


class ModelOutputRejected(ValueError):
    """The provider returned text, but it is not a valid answer for this task."""


class HarnessFailure(RuntimeError):
    """The signed task harness could not execute a model answer."""


@dataclass(frozen=True, slots=True)
class PreparedSample:
    context: ContextDocument
    scoring_state: object


@dataclass(frozen=True, slots=True)
class ScoringTransform:
    completion: str
    strategy: str
    action: str


def unchanged_scoring_text(
    prompt: str, raw_completion: str, truncated: bool, requested_strategy: str
) -> ScoringTransform:
    del prompt, truncated, requested_strategy
    return ScoringTransform(raw_completion, "not-applicable", "none")


class TaskRuntime(Protocol):
    def prepare(self, row: Mapping[str, Any]) -> PreparedSample: ...

    def score(
        self,
        sample: PreparedSample,
        *,
        prompt: str,
        completion: str,
        output_token_ids: tuple[int, ...],
    ) -> Mapping[str, float]: ...
