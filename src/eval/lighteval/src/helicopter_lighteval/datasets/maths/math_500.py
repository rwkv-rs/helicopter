from __future__ import annotations

from typing import Any, Mapping

from ..benchmark import Benchmark, BenchmarkInfo, BenchmarkName, Field, get_string


class Math500(Benchmark):
    def get_query(self, row: Mapping[str, Any], document: Any) -> str:
        return get_string(row, "problem")


MATH_500_INFO = BenchmarkInfo(
    name=BenchmarkName("math-500"),
    field=Field.MATHS,
    display_name="MATH-500",
    lighteval_task_name="math_500",
    create=Math500,
)
