from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from lighteval_runner import application
from lighteval_runner.contracts import EvaluationRequest
from lighteval_runner.execution import ModelIdentity, ProviderIdentity
from lighteval_runner.generation import normalize_generation_outcome
from lighteval_runner.provider.attestation import ProviderAttestation
from lighteval_runner.provider.endpoint import TokenizedContext
from lighteval_runner.results.artifacts import verify_manifest
from lighteval_runner.harnesses.coding import SandboxResult


def _attestation() -> ProviderAttestation:
    return ProviderAttestation(
        model=ModelIdentity("rwkv-test", "b" * 64, "tok-v1", "chat-v1"),
        provider=ProviderIdentity(
            "server-v1",
            "fp32io16",
            "fp16-io-fp32-state",
            "fp32-accumulation",
            "launch-v1",
        ),
        capabilities=(
            "openai-chat",
            "output-token-ids",
            "terminal-reason",
            "prompt-evidence",
        ),
    )


class _Endpoint:
    def __init__(self, **_kwargs) -> None:
        pass

    def close(self) -> None:
        pass

    def tokenize_context(self, _model, sections, _generation_prompt_mode):
        return TokenizedContext((1, 2), 4096)

    def context_token_count(self, _model, _sections, _generation_prompt_mode):
        return 2

    def generate(self, request):
        return normalize_generation_outcome(
            completion="2",
            output_token_ids=(2, 0),
            provider_finish_reason="stop",
            provider_stop_reason=0,
            generation_limit=request.generation_limit,
            prompt_text="Question: What is 1+1?\nAnswer:",
            prompt_token_ids=(1, 2),
        )


class _ProxyEndpoint(_Endpoint):
    completion = ""

    def generate(self, request):
        return normalize_generation_outcome(
            completion=self.completion,
            output_token_ids=(7, 0),
            provider_finish_reason="stop",
            provider_stop_reason=0,
            generation_limit=request.generation_limit,
            prompt_text="signed proxy prompt",
            prompt_token_ids=(1, 2),
        )


def _trust_test_snapshot(snapshot: Path, monkeypatch) -> tuple[str, Path]:
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    definition = application.get_task_definition("lighteval/math/gsm8k@0")
    trusted = replace(
        definition,
        asset_name="test_gsm8k",
        source_file=snapshot.name,
        snapshot_sha256=digest,
        expected_rows=1,
    )
    monkeypatch.setattr(
        application, "get_task_definition", lambda *_args, **_kwargs: trusted
    )
    manifest = snapshot.with_name(f"{snapshot.name}.manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "name": trusted.asset_name,
                "source": {
                    "repo": trusted.dataset_repository,
                    "revision": trusted.dataset_revision,
                    "file": trusted.source_file,
                },
                "files": [
                    {
                        "path": str(snapshot.resolve()),
                        "sha256": digest,
                        "size_bytes": snapshot.stat().st_size,
                    }
                ],
            }
        )
    )
    return digest, manifest


def test_application_runs_task_native_prompt_score_and_immutable_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = tmp_path / "gsm8k.jsonl"
    snapshot.write_text(
        json.dumps({"question": "What is 1+1?", "answer": "#### 2"}) + "\n"
    )
    snapshot_digest, snapshot_manifest = _trust_test_snapshot(snapshot, monkeypatch)
    monkeypatch.setattr(application, "OpenAIEndpoint", _Endpoint)
    monkeypatch.setattr(
        application, "fetch_provider_attestation", lambda **_kwargs: _attestation()
    )

    outcome = application.run_evaluation(
        EvaluationRequest(
            model="rwkv-test",
            task="lighteval/math/gsm8k@0",
            output_root=tmp_path / "runs",
            snapshot_path=snapshot,
            snapshot_manifest_path=snapshot_manifest,
            snapshot_sha256=snapshot_digest,
            endpoint_url="http://127.0.0.1:8000/v1",
            checkpoint_sha256="b" * 64,
            tokenizer_revision="tok-v1",
            chat_template_revision="chat-v1",
            expected_server_revision="server-v1",
            wkv_mode="fp32io16",
            precision="fp16-io-fp32-state",
            gemm_policy="fp32-accumulation",
            launch_contract="launch-v1",
            product_revision="d" * 40,
            cot_mode="cot",
        )
    )

    assert outcome.is_success
    assert outcome.summary == {
        "extractive_match": 1.0,
        "generated_samples": 1,
        "truncated_samples": 0,
        "truncation_rate": 0.0,
    }
    manifest = verify_manifest(outcome.manifest_path)
    assert manifest.status.value == "completed"
    samples = json.loads((outcome.manifest_path.parent / "samples.json").read_text())
    assert samples[0]["raw_completion"] == "2"
    assert samples[0]["scored_completion"] == "2"
    assert samples[0]["prompt"].startswith("Question:")


def test_application_records_provider_failure_without_fabricating_score(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = tmp_path / "gsm8k.jsonl"
    snapshot.write_text(
        json.dumps({"question": "What is 1+1?", "answer": "#### 2"}) + "\n"
    )
    snapshot_digest, snapshot_manifest = _trust_test_snapshot(snapshot, monkeypatch)
    monkeypatch.setattr(
        application, "fetch_provider_attestation", lambda **_kwargs: _attestation()
    )

    class FailingEndpoint(_Endpoint):
        def generate(self, request):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(application, "OpenAIEndpoint", FailingEndpoint)
    outcome = application.run_evaluation(
        EvaluationRequest(
            model="rwkv-test",
            task="lighteval/math/gsm8k@0",
            output_root=tmp_path / "runs",
            snapshot_path=snapshot,
            snapshot_manifest_path=snapshot_manifest,
            snapshot_sha256=snapshot_digest,
            endpoint_url="http://127.0.0.1:8000/v1",
            checkpoint_sha256="b" * 64,
            tokenizer_revision="tok-v1",
            chat_template_revision="chat-v1",
            expected_server_revision="server-v1",
            wkv_mode="fp32io16",
            precision="fp16-io-fp32-state",
            gemm_policy="fp32-accumulation",
            launch_contract="launch-v1",
            product_revision="d" * 40,
        )
    )
    assert outcome.run_status == "failed"
    assert outcome.summary["generated_samples"] == 0
    samples = json.loads((outcome.manifest_path.parent / "samples.json").read_text())
    assert samples[0]["status"] == "provider_error"
    assert samples[0]["metrics"] == {}


def _proxy_request(tmp_path: Path, task: str) -> EvaluationRequest:
    return EvaluationRequest(
        model="rwkv-test",
        task=task,
        output_root=tmp_path / "runs",
        snapshot_path=None,
        snapshot_manifest_path=None,
        snapshot_sha256=None,
        endpoint_url="http://127.0.0.1:8000/v1",
        checkpoint_sha256="b" * 64,
        tokenizer_revision="tok-v1",
        chat_template_revision="chat-v1",
        expected_server_revision="server-v1",
        wkv_mode="fp32io16",
        precision="fp16-io-fp32-state",
        gemm_policy="fp32-accumulation",
        launch_contract="launch-v1",
        product_revision="d" * 40,
        allow_non_comparable=True,
    )


def test_function_calling_proxy_runs_through_registry_and_application(
    tmp_path: Path, monkeypatch
) -> None:
    class Endpoint(_ProxyEndpoint):
        completion = '{"name":"search","arguments":{"query":"rwkv"}}'

    monkeypatch.setattr(application, "OpenAIEndpoint", Endpoint)
    monkeypatch.setattr(
        application, "fetch_provider_attestation", lambda **_kwargs: _attestation()
    )
    outcome = application.run_evaluation(
        _proxy_request(tmp_path, "helicopter-proxy/function-calling/exact-json@1")
    )
    assert outcome.run_status == "completed"
    assert outcome.summary["exact_function_call"] == 1.0
    manifest = verify_manifest(outcome.manifest_path)
    assert manifest.identities["run"]["task"]["suite"] == "helicopter-proxy"
    assert manifest.identities["run"]["task"]["task"] == "function-calling/exact-json"
    assert manifest.identities["run"]["eligibility"] == "proxy"
    assert manifest.identities["run"]["comparable"] is False


def test_coding_proxy_uses_signed_sandbox_result(tmp_path: Path, monkeypatch) -> None:
    class Endpoint(_ProxyEndpoint):
        completion = "value = int(input())\nprint(value * 2)"

    monkeypatch.setattr(application, "OpenAIEndpoint", Endpoint)
    monkeypatch.setattr(
        application, "fetch_provider_attestation", lambda **_kwargs: _attestation()
    )
    monkeypatch.setattr(
        "lighteval_runner.tasks.coding.run_python_submission",
        lambda source, *, stdin, policy: SandboxResult(
            revision="landlock-seccomp-python-v3",
            return_code=0,
            stdout=str(int(stdin) * 2) + "\n",
            stderr="",
            timed_out=False,
            output_limit_exceeded=False,
        ),
    )
    outcome = application.run_evaluation(
        _proxy_request(tmp_path, "helicopter-proxy/coding/python-stdio@1")
    )
    assert outcome.run_status == "completed"
    assert outcome.summary["sandbox_pass_rate"] == 1.0


def test_preflight_failure_commits_invalid_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = tmp_path / "gsm8k.jsonl"
    snapshot.write_text(
        json.dumps({"question": "What is 1+1?", "answer": "#### 2"}) + "\n"
    )
    snapshot_digest, snapshot_manifest = _trust_test_snapshot(snapshot, monkeypatch)
    monkeypatch.setattr(
        application, "fetch_provider_attestation", lambda **_kwargs: None
    )
    request = replace(
        _proxy_request(tmp_path, "lighteval/math/gsm8k@0"),
        snapshot_path=snapshot,
        snapshot_manifest_path=snapshot_manifest,
        snapshot_sha256=snapshot_digest,
        allow_non_comparable=False,
    )
    outcome = application.run_evaluation(request)
    assert outcome.run_status == "invalid"
    manifest = verify_manifest(outcome.manifest_path)
    assert manifest.status.value == "invalid"
    assert (outcome.manifest_path.parent / "failure.json").is_file()


def test_official_dataset_rejection_is_invalid_not_a_smaller_denominator(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = tmp_path / "gsm8k.jsonl"
    snapshot.write_text(json.dumps({"question": "missing answer"}) + "\n")
    snapshot_digest, snapshot_manifest = _trust_test_snapshot(snapshot, monkeypatch)
    monkeypatch.setattr(application, "OpenAIEndpoint", _Endpoint)
    monkeypatch.setattr(
        application, "fetch_provider_attestation", lambda **_kwargs: _attestation()
    )
    request = replace(
        _proxy_request(tmp_path, "lighteval/math/gsm8k@0"),
        snapshot_path=snapshot,
        snapshot_manifest_path=snapshot_manifest,
        snapshot_sha256=snapshot_digest,
        allow_non_comparable=False,
    )
    outcome = application.run_evaluation(request)
    assert outcome.run_status == "invalid"
    manifest = verify_manifest(outcome.manifest_path)
    assert manifest.accounting["dataset_rejected"] == 1
    assert manifest.accounting["scored"] == 0


def test_keyboard_interrupt_commits_cancelled_samples_and_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = tmp_path / "gsm8k.jsonl"
    snapshot.write_text(
        json.dumps({"question": "What is 1+1?", "answer": "#### 2"}) + "\n"
    )
    snapshot_digest, snapshot_manifest = _trust_test_snapshot(snapshot, monkeypatch)

    class CancelledEndpoint(_Endpoint):
        def generate(self, request):
            raise KeyboardInterrupt

    monkeypatch.setattr(application, "OpenAIEndpoint", CancelledEndpoint)
    monkeypatch.setattr(
        application, "fetch_provider_attestation", lambda **_kwargs: _attestation()
    )
    request = replace(
        _proxy_request(tmp_path, "lighteval/math/gsm8k@0"),
        snapshot_path=snapshot,
        snapshot_manifest_path=snapshot_manifest,
        snapshot_sha256=snapshot_digest,
        allow_non_comparable=False,
    )
    outcome = application.run_evaluation(request)
    assert outcome.run_status == "cancelled"
    manifest = verify_manifest(outcome.manifest_path)
    assert manifest.accounting["cancelled"] == 1
