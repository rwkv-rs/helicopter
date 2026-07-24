from __future__ import annotations

from typing import Any, Mapping

from ..benchmark import Benchmark, BenchmarkInfo, BenchmarkName, Field, get_string


class Aime24(Benchmark):
    def get_query(self, row: Mapping[str, Any], document: Any) -> str:
        return get_string(row, "problem")


AIME24_INFO = BenchmarkInfo(
    name=BenchmarkName("aime24"),
    field=Field.MATHS,
    display_name="AIME24",
    lighteval_task_name="aime24",
    create=Aime24,
)
