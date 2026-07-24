from __future__ import annotations

from typing import Any, Mapping

from ..benchmark import Benchmark, BenchmarkInfo, BenchmarkName, Field, get_string


class Aime25(Benchmark):
    def get_query(self, row: Mapping[str, Any], document: Any) -> str:
        return get_string(row, "problem")


AIME25_INFO = BenchmarkInfo(
    name=BenchmarkName("aime25"),
    field=Field.MATHS,
    display_name="AIME25",
    lighteval_task_name="aime25",
    create=Aime25,
)
