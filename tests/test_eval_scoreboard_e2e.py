from pathlib import Path

from helicopter_lighteval.evaluation import EvaluationRequest, run_evaluation


def test_unsupported_coding_never_reaches_scoreboard_or_provider(
    tmp_path: Path,
) -> None:
    outcome = run_evaluation(
        EvaluationRequest(
            model="rwkv-test",
            task="lighteval/knowledge/livecodebench@0",
            endpoint_url="http://127.0.0.1:1/v1",
            output_root=tmp_path,
            scoreboard_url="http://scoreboard",
            scoreboard_token="token",
        )
    )
    assert outcome.run_status == "unsupported"
    assert outcome.publication_status == "not_requested"
    assert outcome.manifest_path is None
