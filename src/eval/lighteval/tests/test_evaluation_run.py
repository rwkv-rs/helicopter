import json
from types import SimpleNamespace

from helicopter_lighteval import evaluation
from helicopter_lighteval.scoreboard import PublicationResult


class _Rows(list):
    _fingerprint = "dataset-fingerprint"


def test_run_writes_one_manifest_and_publishes_native_scores(
    tmp_path, monkeypatch
) -> None:
    task = SimpleNamespace(
        full_name="gsm8k|0",
        config=SimpleNamespace(
            name="gsm8k",
            version=0,
            hf_repo="openai/gsm8k",
            hf_subset="main",
            hf_revision=None,
            evaluation_splits=("test",),
            num_fewshots=0,
            scorer=None,
            generation_size=None,
        ),
        metrics=[
            SimpleNamespace(
                metric_name="extractive_match",
                sample_level_fn=lambda doc, response: 1.0,
            )
        ],
        dataset={"test": _Rows(["row-1", "row-2"])},
        generation_size=None,
    )
    document = SimpleNamespace(
        id=0,
        query="What is 1 + 1?",
        choices=["2"],
        gold_index=0,
        generation_size=7,
    )
    response = SimpleNamespace(text=["2"])
    monkeypatch.setattr(
        evaluation,
        "_load_task",
        lambda **kwargs: (task, [document]),
    )

    async def generate(**kwargs):
        return [response], ["length"]

    monkeypatch.setattr(evaluation, "_generate", generate)
    monkeypatch.setattr(
        evaluation,
        "_score",
        lambda **kwargs: (
            [{"extractive_match": 1.0}],
            {"results": {task.full_name: {"extractive_match": 1.0}}},
        ),
    )
    published = []
    import helicopter_lighteval.scoreboard as scoreboard

    monkeypatch.setattr(
        scoreboard,
        "publish_manifest",
        lambda **kwargs: (
            published.append(kwargs["manifest_path"])
            or PublicationResult("published", "publish:test", task_id=17)
        ),
    )

    outcome = evaluation.run_evaluation(
        evaluation.EvaluationRequest(
            model="model",
            task="lighteval/math/gsm8k@0",
            endpoint_url="http://server/v1",
            output_root=tmp_path,
            checkpoint_sha256="c" * 64,
            tokenizer_revision="tokenizer",
            chat_template_revision="chat",
            server_revision="server",
            wkv_mode="fp32io16",
            precision="fp16",
            gemm_policy="fp32",
            launch_contract="launch",
            max_samples=1,
            generation_limit=7,
            scoreboard_url="http://scoreboard",
            scoreboard_token="secret",
            product_revision="0" * 40,
        )
    )

    assert outcome.run_status == "completed"
    assert outcome.publication_status == "published"
    assert outcome.publication_task_id == 17
    manifest = json.loads(published[0].read_text(encoding="utf-8"))
    assert manifest["identities"]["run"]["eligibility"] == "sanity"
    assert manifest["accounting"]["dataset_accepted"] == 2
    assert manifest["accounting"]["selected"] == 1
    evidence = json.loads(
        (outcome.manifest_path.parent / "terminal_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    sample = evidence["samples"][0]
    assert sample["prompt"] == "What is 1 + 1?"
    assert sample["raw_completion"] == "2"
    assert sample["generation"] == {
        "finish_reason": "length",
        "generation_limit": 7,
        "truncated": True,
    }
    assert evidence["truncation_rate"] == 1.0
