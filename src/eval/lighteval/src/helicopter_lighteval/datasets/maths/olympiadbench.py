from __future__ import annotations

from typing import Any, Mapping

from ..benchmark import Benchmark, BenchmarkInfo, BenchmarkName, Field, get_string


class OlympiadBench(Benchmark):
    def get_query(self, row: Mapping[str, Any], document: Any) -> str:
        return get_string(row, "question")


OLYMPIADBENCH_INFO = BenchmarkInfo(
    name=BenchmarkName("olympiadbench"),
    field=Field.MATHS,
    display_name="OlympiadBench",
    lighteval_task_name="olympiad_bench:OE_TO_maths_en_COMP",
    create=OlympiadBench,
)
