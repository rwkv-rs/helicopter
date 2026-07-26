from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from helicopter_lighteval import evaluate
from helicopter_lighteval.config import LightEvalConfig
from helicopter_lighteval.publish import PublicationError


class Sampling:
    GENERATIVE = "generative"
    LOGPROBS = "logprobs"


def _document(*, gold_index, choices=None, methods=None):
    return SimpleNamespace(
        query="Question?",
        choices=choices or ["one", "two", "three"],
        gold_index=gold_index,
        sampling_methods=methods or [Sampling.LOGPROBS],
        specific={},
    )


def test_multiselect_is_skipped_and_single_choice_is_converted() -> None:
    assert evaluate._is_multiselect(
        _document(gold_index=[0, 2]),
        Sampling,
    )

    document = _document(gold_index=[1])
    assert evaluate._is_single_choice(document, Sampling)
    evaluate._convert_choice(document, Sampling)

    assert document.sampling_methods == [Sampling.GENERATIVE]
    assert "A. one" in document.query
    assert "Answer: <letter>" in document.query
    assert document.specific["helicopter_choice"] is True


@pytest.mark.parametrize(
    ("raw", "tokens", "expected"),
    [
        ("<think>work</think>Answer: B", [1, 2], "two"),
        ("<think>work</think>\\boxed{C}", [1], "three"),
        ("<think>work</think>Answer: A\nAnswer: B", [1], ""),
        ("Answer: A", [1], ""),
        ("<think>work</think>Answer: A", [], ""),
        (
            "<think>work</think>Answer: A",
            list(range(evaluate.MAX_NEW_TOKENS)),
            "",
        ),
    ],
)
def test_choice_answer_requires_one_complete_supported_answer(
    raw: str,
    tokens: list[int],
    expected: str,
) -> None:
    assert evaluate._choice_answer(raw, tokens, ["one", "two", "three"]) == expected


def test_expected_tasks_are_weight_mode_task_product(tmp_path: Path) -> None:
    weights = (tmp_path / "a.pth", tmp_path / "b.pth")
    config = LightEvalConfig(
        prompt_template="bot",
        publish=True,
        result_path=None,
        weights=weights,
        weight_hashes=("a" * 64, "b" * 64),
        benchmarks=("gsm8k",),
        wkv_modes=("fp16", "fp32io16"),
        scoreboard_url="https://example.test",
        scoreboard_token="secret",
        staging_root=tmp_path / "staging",
    )
    task = {
        "selector": "gsm8k",
        "task_name": "gsm8k|0",
        "task_version": "0",
        "module_family": "gsm8k",
        "module": "lighteval.tasks.tasks.gsm8k",
        "dataset": "openai/gsm8k",
        "subset": "main",
        "evaluation_splits": ["test"],
        "languages": ["english"],
        "upstream_tags": ["math"],
    }

    expected = evaluate._expected_tasks(config, [task])

    assert len(expected) == 4
    assert {(row["weight_sha256"], row["wkv_mode"]) for row in expected} == {
        ("a" * 64, "fp16"),
        ("a" * 64, "fp32io16"),
        ("b" * 64, "fp16"),
        ("b" * 64, "fp32io16"),
    }


def test_completed_run_cleanup_only_accepts_direct_child(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    run = staging / "campaign"
    run.mkdir()
    (run / "result").write_text("evidence")

    evaluate._remove_completed_run(run, staging)
    assert not run.exists()

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(PublicationError, match="unsafe"):
        evaluate._remove_completed_run(outside, staging)


def test_process_environment_restores_existing_and_missing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HELICOPTER_EXISTING", "before")
    monkeypatch.delenv("HELICOPTER_NEW", raising=False)

    with evaluate._process_environment(
        {"HELICOPTER_EXISTING": "during", "HELICOPTER_NEW": "during"}
    ):
        assert evaluate.os.environ["HELICOPTER_EXISTING"] == "during"
        assert evaluate.os.environ["HELICOPTER_NEW"] == "during"

    assert evaluate.os.environ["HELICOPTER_EXISTING"] == "before"
    assert "HELICOPTER_NEW" not in evaluate.os.environ
