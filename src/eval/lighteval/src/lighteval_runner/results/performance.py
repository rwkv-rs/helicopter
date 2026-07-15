from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PerformanceEvidence:
    token_usage_attribution: str
    server_metrics_attribution: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    request_ids: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def summarize_run_performance(samples: list[dict[str, Any]]) -> PerformanceEvidence:
    """Aggregate only signed per-request usage contained in the current run evidence."""

    usages: list[dict[str, int]] = []
    request_ids: list[str] = []
    generated = [sample for sample in samples if sample.get("generation") is not None]
    for sample in generated:
        generation = sample["generation"]
        usage = generation.get("usage")
        request_id = generation.get("request_id")
        if (
            not isinstance(usage, dict)
            or not isinstance(request_id, str)
            or not request_id
        ):
            return PerformanceEvidence(
                "not_attributable",
                "not_attributable",
                None,
                None,
                None,
                (),
            )
        usages.append(usage)
        request_ids.append(request_id)
    if not usages or len(set(request_ids)) != len(request_ids):
        return PerformanceEvidence(
            "not_attributable", "not_attributable", None, None, None, ()
        )
    return PerformanceEvidence(
        token_usage_attribution="per_request_usage",
        server_metrics_attribution="not_attributable",
        prompt_tokens=sum(item["prompt_tokens"] for item in usages),
        completion_tokens=sum(item["completion_tokens"] for item in usages),
        total_tokens=sum(item["total_tokens"] for item in usages),
        request_ids=tuple(request_ids),
    )
