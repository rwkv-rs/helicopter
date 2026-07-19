import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_maxrl_math_eval.py"
SPEC = importlib.util.spec_from_file_location("prepare_maxrl_math_eval", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


@pytest.mark.parametrize(
    ("name", "record", "answer"),
    (
        ("aime24", {"problem": "P24", "solution": r"\\boxed{24}"}, r"\\boxed{24}"),
        ("aime25", {"problem": "P25", "answer": 25}, "25"),
        ("amc23", {"question": "AMC", "answer": 23.0}, "23.0"),
        ("math500", {"problem": "M500", "answer": r"\\frac{1}{2}"}, r"\\frac{1}{2}"),
    ),
)
def test_convert_record_builds_verl_reward_schema(name, record, answer):
    converted = prepare.convert_record(record, 7, prepare.SOURCES[name])

    assert converted["data_source"] == prepare.SOURCES[name].data_source
    assert converted["prompt"] == [
        {
            "role": "user",
            "content": f"{record[prepare.SOURCES[name].question_field]} {prepare.INSTRUCTION}",
        }
    ]
    assert converted["ability"] == "math"
    assert converted["reward_model"] == {"style": "rule", "ground_truth": answer}
    assert converted["extra_info"]["index"] == 7
    assert converted["extra_info"]["split"] == "test"


def test_convert_record_rejects_empty_questions_and_answers():
    source = prepare.SOURCES["aime25"]
    with pytest.raises(RuntimeError, match="empty question or answer"):
        prepare.convert_record({"problem": "", "answer": "1"}, 0, source)
    with pytest.raises(RuntimeError, match="empty question or answer"):
        prepare.convert_record({"problem": "question", "answer": ""}, 0, source)
