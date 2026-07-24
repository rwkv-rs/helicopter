from __future__ import annotations

from typing import Any, Mapping

from ..benchmark import Benchmark, BenchmarkInfo, BenchmarkName, Field, get_string


class Ifeval(Benchmark):
    def get_query(self, row: Mapping[str, Any], document: Any) -> str:
        return get_string(row, "prompt")


IFEVAL_INFO = BenchmarkInfo(
    name=BenchmarkName("ifeval"),
    field=Field.INSTRUCTION_FOLLOWING,
    display_name="IFEval",
    lighteval_task_name="ifeval",
    create=Ifeval,
)
