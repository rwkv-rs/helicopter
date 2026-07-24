from types import SimpleNamespace

import pytest

from helicopter_lighteval.datasets.instruction_following.ifbench import Ifbench
from helicopter_lighteval.datasets.instruction_following.ifeval import Ifeval


@pytest.mark.parametrize(
    ("benchmark", "prompt"),
    [
        (Ifeval(), "Write exactly three lines."),
        (Ifbench(), "Do not use the letter e."),
    ],
)
def test_instruction_following_benchmarks_preserve_dataset_prompt(
    benchmark: object, prompt: str
) -> None:
    assert benchmark.get_query({"prompt": prompt}, SimpleNamespace()) == prompt
