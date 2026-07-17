from pathlib import Path

import pytest

from helicopter_lighteval.evaluation import EvaluationRequest, run_evaluation


def test_coding_fails_closed_before_model_or_server_import(tmp_path: Path) -> None:
    request = EvaluationRequest(
        model="model",
        task="lighteval/coding/livecodebench@0",
        endpoint_url="http://127.0.0.1:8000/v1",
        output_root=tmp_path,
    )
    outcome = run_evaluation(request)
    assert outcome.run_status == "unsupported"
    assert outcome.manifest_path is None


@pytest.mark.parametrize("family", ["function-calling", "agent"])
def test_absent_task_family_fails_closed_before_provider_import(
    tmp_path: Path, family: str
) -> None:
    outcome = run_evaluation(
        EvaluationRequest(
            model="model",
            task=f"lighteval/{family}/default@0",
            endpoint_url="http://127.0.0.1:8000/v1",
            output_root=tmp_path,
        )
    )
    assert outcome.run_status == "unsupported"
    assert outcome.manifest_path is None
