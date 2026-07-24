from __future__ import annotations

from typing import Any, Mapping

from ...benchmark import Benchmark, get_string, render_choices


class Gpqa(Benchmark):
    def get_query(self, row: Mapping[str, Any], document: Any) -> str:
        choices = [
            get_string(row, "Incorrect Answer 1"),
            get_string(row, "Incorrect Answer 2"),
            get_string(row, "Incorrect Answer 3"),
        ]
        answer_index = document.gold_index
        if isinstance(answer_index, bool) or not isinstance(answer_index, int):
            raise ValueError("GPQA document gold_index must be an integer")
        if answer_index < 0 or answer_index > len(choices):
            raise ValueError("GPQA document gold_index is outside the choice range")
        choices.insert(answer_index, get_string(row, "Correct Answer"))
        return render_choices(get_string(row, "Question"), choices)
