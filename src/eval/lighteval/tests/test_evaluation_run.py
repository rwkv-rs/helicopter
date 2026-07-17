import json
from types import SimpleNamespace

import pytest

from helicopter_lighteval import evaluation
from helicopter_lighteval.scoreboard import PublicationResult
from helicopter_lighteval.vllm_rwkv import (
    AttestationDecision,
    GenerationEvidence,
    ModelIdentity,
    ProviderAttestation,
    ProviderIdentity,
    TerminalReason,
    TokenUsage,
)


class _Rows(list):
    _fingerprint = "dataset-fingerprint"


@pytest.mark.parametrize(
    ("attestation_verified", "expected_eligibility"),
    [(True, "sanity"), (False, "proxy")],
)
def test_run_writes_one_manifest_and_publishes_nonofficial_evidence(
    tmp_path, monkeypatch, attestation_verified, expected_eligibility
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
            prompt_function=lambda doc: doc.query,
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
    doc = SimpleNamespace(
        id=0,
        choices=["answer"],
        gold_index=0,
        generation_size=None,
    )
    task._docs = [doc]
    evidence = GenerationEvidence(
        raw_completion="answer",
        scored_completion="answer",
        output_token_ids=(7, 0),
        prompt_token_ids=(1,),
        prompt_text="prompt",
        finish_reason="stop",
        stop_reason=0,
        terminal_reason=TerminalReason.STOP,
        truncated=False,
        generation_limit=7,
        request_id="request-1",
        usage=TokenUsage(1, 2, 3),
        repair_strategy="A",
        repair_action="none",
    )
    actual_model = ModelIdentity(
        served_name="model",
        checkpoint_sha256="c" * 64,
        tokenizer_revision="tokenizer",
        chat_template_revision="chat",
    )
    actual_provider = ProviderIdentity(
        server_revision="server",
        wkv_mode="fp32io16",
        precision="fp16",
        gemm_policy="fp32",
        launch_contract="launch",
    )
    attestation = ProviderAttestation(
        model=actual_model,
        provider=actual_provider,
        capabilities=(
            "openai-chat",
            "output-token-ids",
            "terminal-reason",
            "prompt-evidence",
        ),
    )

    class FakeModel:
        def __init__(self, **kwargs):
            self.model_identity = actual_model
            self.provider_identity = actual_provider
            self.attestation = None
            self.evidence = {(task.full_name, "0"): [evidence]}

        def cleanup(self):
            return None

    class FakePipeline:
        def evaluate(self):
            assert task.generation_size == 7
            assert task.config.generation_size == 7
            assert task._docs[0].generation_size == 7
            return None

        def save_and_push_results(self):
            return None

        def get_results(self):
            return {"results": {task.full_name: {"extractive_match": 1.0}}}

        def get_details(self):
            return {
                task.full_name: [
                    SimpleNamespace(doc=doc, metric={"extractive_match": 1.0})
                ]
            }

    import helicopter_lighteval.vllm_rwkv as vllm_rwkv

    monkeypatch.setattr(
        vllm_rwkv,
        "fetch_attestation",
        lambda base_url: attestation if attestation_verified else None,
    )
    monkeypatch.setattr(
        vllm_rwkv,
        "attest",
        lambda **kwargs: AttestationDecision(
            official=attestation_verified,
            mismatches=() if attestation_verified else ("missing_attestation",),
        ),
    )
    monkeypatch.setattr(vllm_rwkv, "VllmRwkvModel", FakeModel)
    monkeypatch.setattr(
        evaluation,
        "_build_pipeline",
        lambda **kwargs: (FakePipeline(), task),
    )
    published: list = []
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
            allow_non_comparable=not attestation_verified,
        )
    )

    assert outcome.run_status == "completed"
    assert outcome.publication_status == "published"
    assert outcome.publication_task_id == 17
    assert len(published) == 1
    manifest = json.loads(published[0].read_text(encoding="utf-8"))
    assert manifest["identities"]["run"]["eligibility"] == expected_eligibility
    assert manifest["accounting"]["dataset_accepted"] == 2
    assert manifest["accounting"]["dataset_rejected"] == 0
    assert manifest["accounting"]["selected"] == 1
    terminal_evidence = json.loads(
        (outcome.manifest_path.parent / "terminal_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    assert terminal_evidence["samples"][0]["generation"]["generation_limit"] == 7
