from __future__ import annotations

from typing import Any, Mapping

from ..benchmark import Benchmark, BenchmarkInfo, BenchmarkName, Field, get_string


class Asdiv(Benchmark):
    def get_query(self, row: Mapping[str, Any], document: Any) -> str:
        return "\n".join((get_string(row, "body"), get_string(row, "question")))


ASDIV_INFO = BenchmarkInfo(
    name=BenchmarkName("asdiv"),
    field=Field.MATHS,
    display_name="ASDiv",
    lighteval_task_name="asdiv",
    create=Asdiv,
)
