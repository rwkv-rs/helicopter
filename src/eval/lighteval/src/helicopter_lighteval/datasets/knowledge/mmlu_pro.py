from __future__ import annotations

from typing import Any, Mapping

from ..benchmark import (
    Benchmark,
    BenchmarkInfo,
    BenchmarkName,
    Field,
    get_string,
    get_strings,
    render_choices,
)


class MmluPro(Benchmark):
    def get_query(self, row: Mapping[str, Any], document: Any) -> str:
        return render_choices(get_string(row, "question"), get_strings(row, "options"))


MMLU_PRO_INFO = BenchmarkInfo(
    name=BenchmarkName("mmlu-pro"),
    field=Field.KNOWLEDGE,
    display_name="MMLU-Pro",
    lighteval_task_name="mmlu_pro",
    create=MmluPro,
)
