"""Canonical aliases for generation-only knowledge tasks owned by LightEval."""

from __future__ import annotations


def upstream_task_candidates(benchmark: str) -> tuple[str, ...]:
    """Return fixed-LightEval task names without copying task definitions."""

    normalized = benchmark.strip()
    if not normalized:
        return ()

    aliases: dict[str, tuple[str, ...]] = {
        "mmlu-pro": ("mmlu_pro",),
        "mmlu_pro": ("mmlu_pro",),
        "gpqa-diamond": ("gpqa:diamond",),
        "gpqa_diamond": ("gpqa:diamond",),
        "gpqa-main": ("gpqa:main",),
        "gpqa_main": ("gpqa:main",),
        "gpqa-extended": ("gpqa:extended",),
        "gpqa_extended": ("gpqa:extended",),
    }
    candidates = aliases.get(normalized)
    if candidates is not None:
        return candidates

    if normalized.startswith("mmlu-"):
        subject = normalized.removeprefix("mmlu-").replace("-", "_")
        return (f"mmlu:{subject}",)
    if normalized.startswith("mmlu:"):
        return (normalized,)

    # Do not accept arbitrary registry names here.  The family boundary must
    # be mutually exclusive: a coding task such as ``lcb:codegeneration``
    # must never be smuggled in through the knowledge resolver.
    return ()
