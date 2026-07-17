"""Canonical aliases for single-turn instruction-following tasks owned by LightEval."""

from __future__ import annotations


def upstream_task_candidates(benchmark: str) -> tuple[str, ...]:
    """Return fixed-LightEval task names without copying task definitions."""

    aliases = {
        "ifeval": ("ifeval",),
        "ifbench": ("ifbench_test",),
        "ifbench-test": ("ifbench_test",),
    }
    return aliases.get(benchmark.strip(), ())
