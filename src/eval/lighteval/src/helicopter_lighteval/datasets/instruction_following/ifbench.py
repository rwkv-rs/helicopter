from __future__ import annotations

from typing import Any, Mapping

from ..benchmark import Benchmark, BenchmarkInfo, BenchmarkName, Field, get_string


class Ifbench(Benchmark):
    def get_query(self, row: Mapping[str, Any], document: Any) -> str:
        return get_string(row, "prompt")


IFBENCH_INFO = BenchmarkInfo(
    name=BenchmarkName("ifbench-test"),
    field=Field.INSTRUCTION_FOLLOWING,
    display_name="IFBench Test",
    lighteval_task_name="ifbench_test",
    create=Ifbench,
)
