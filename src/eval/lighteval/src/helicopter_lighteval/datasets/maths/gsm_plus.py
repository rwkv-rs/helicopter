from __future__ import annotations

from typing import Any, Mapping

from ..benchmark import Benchmark, BenchmarkInfo, BenchmarkName, Field, get_string


class GsmPlus(Benchmark):
    def get_query(self, row: Mapping[str, Any], document: Any) -> str:
        return get_string(row, "question")


GSM_PLUS_INFO = BenchmarkInfo(
    name=BenchmarkName("gsm-plus"),
    field=Field.MATHS,
    display_name="GSM-Plus",
    lighteval_task_name="gsm_plus",
    create=GsmPlus,
)
