from __future__ import annotations

from typing import Any, Mapping

from ..benchmark import Benchmark, get_string, get_strings, render_choices


class Mmlu(Benchmark):
    def get_query(self, row: Mapping[str, Any], document: Any) -> str:
        return render_choices(get_string(row, "question"), get_strings(row, "choices"))
