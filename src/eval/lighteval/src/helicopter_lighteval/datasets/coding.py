"""LiveCodeBench identity aliases and the current coding safety boundary."""

from __future__ import annotations

import re


CODING_UNSUPPORTED_REASON = "LiveCodeBench requires an isolated execution harness"


def upstream_task_candidates(benchmark: str) -> tuple[str, ...]:
    """Return fixed-LightEval LiveCodeBench task names."""

    normalized = benchmark.strip()
    if not normalized:
        return ()
    if normalized.startswith("lcb:"):
        return (normalized,)
    if normalized == "livecodebench":
        return ("lcb:codegeneration",)

    match = re.fullmatch(
        r"livecodebench-(release-)?(latest|v[1-6](?:-v[1-6])*)", normalized
    )
    if match is None:
        return ()
    prefix = "lcb:codegeneration_release_" if match.group(1) else "lcb:codegeneration_"
    suffix = match.group(2).replace("-", "_")
    return (f"{prefix}{suffix}",)
