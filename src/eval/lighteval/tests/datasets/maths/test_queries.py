from types import SimpleNamespace

import pytest

from helicopter_lighteval.datasets.maths.aime24 import Aime24
from helicopter_lighteval.datasets.maths.aime25 import Aime25
from helicopter_lighteval.datasets.maths.asdiv import Asdiv
from helicopter_lighteval.datasets.maths.gsm8k import Gsm8k
from helicopter_lighteval.datasets.maths.gsm_plus import GsmPlus
from helicopter_lighteval.datasets.maths.math_500 import Math500
from helicopter_lighteval.datasets.maths.olympiadbench import OlympiadBench


@pytest.mark.parametrize(
    ("benchmark", "row", "expected"),
    [
        (Gsm8k(), {"question": "How many?"}, "How many?"),
        (GsmPlus(), {"question": "How many more?"}, "How many more?"),
        (Aime24(), {"problem": "Find x."}, "Find x."),
        (Aime25(), {"problem": "Find y."}, "Find y."),
        (Math500(), {"problem": "Prove it."}, "Prove it."),
        (
            Asdiv(),
            {"body": "There are three birds.", "question": "How many birds?"},
            "There are three birds.\nHow many birds?",
        ),
        (
            OlympiadBench(),
            {"question": "Show that n is even."},
            "Show that n is even.",
        ),
    ],
)
def test_maths_benchmarks_return_raw_queries(
    benchmark: object, row: dict[str, str], expected: str
) -> None:
    query = benchmark.get_query(row, SimpleNamespace())

    assert query == expected
    assert "Question:" not in query
    assert "Answer:" not in query
