from __future__ import annotations

from typing import Any, Mapping

from ..benchmark import Benchmark, BenchmarkInfo, BenchmarkName, Field, get_string


class Gsm8k(Benchmark):
    def get_query(self, row: Mapping[str, Any], document: Any) -> str:
        return get_string(row, "question")


GSM8K_INFO = BenchmarkInfo(
    name=BenchmarkName("gsm8k"),
    field=Field.MATHS,
    display_name="GSM8K",
    lighteval_task_name="gsm8k",
    create=Gsm8k,
)
