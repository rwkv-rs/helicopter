from __future__ import annotations

import re
from typing import Any, Mapping

from ..benchmark import Benchmark, BenchmarkInfo, BenchmarkName, Field


CODING_UNSUPPORTED_REASON = "LiveCodeBench requires an isolated execution harness"


class LiveCodeBench(Benchmark):
    def get_query(self, row: Mapping[str, Any], document: Any) -> str:
        raise RuntimeError(CODING_UNSUPPORTED_REASON)


def get_livecodebench_info(benchmark_name: str) -> BenchmarkInfo | None:
    if benchmark_name == "livecodebench":
        lighteval_task_name = "lcb:codegeneration"
    else:
        match = re.fullmatch(
            r"livecodebench-(release-)?(latest|v[1-6](?:-v[1-6])*)",
            benchmark_name,
        )
        if match is None:
            return None
        prefix = (
            "lcb:codegeneration_release_" if match.group(1) else "lcb:codegeneration_"
        )
        lighteval_task_name = f"{prefix}{match.group(2).replace('-', '_')}"

    return BenchmarkInfo(
        name=BenchmarkName(benchmark_name),
        field=Field.CODING,
        display_name="LiveCodeBench",
        lighteval_task_name=lighteval_task_name,
        create=LiveCodeBench,
    )
