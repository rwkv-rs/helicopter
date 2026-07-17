import gzip
import hashlib
import json

import httpx

from helicopter_lighteval.scoreboard import (
    build_publication_payload,
    publish_manifest,
    verify_manifest,
)


def _identity() -> dict:
    return {
        "task": {
            "suite": "lighteval",
            "task": "gsm8k",
            "version": "0",
            "split": "test",
            "fewshot": 0,
            "prompt_revision": "p" * 64,
            "scorer_revision": "s" * 64,
            "generation_contract": "helicopter-lighteval-openai-v1",
            "cot_mode": "none",
            "repair_strategy": "A",
            "dataset_digest": "d" * 64,
            "primary_metric": "extractive_match",
            "metrics": [
                {
                    "name": "extractive_match",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "aggregation": "mean",
                    "binary_correctness": True,
                }
            ],
        },
        "model": {
            "served_name": "model",
            "checkpoint_sha256": "c" * 64,
            "tokenizer_revision": "tokenizer",
            "chat_template_revision": "template",
        },
        "provider": {
            "server_revision": "server",
            "wkv_mode": "fp32io16",
            "precision": "fp16-io-fp32-state",
            "gemm_policy": "fp32-accumulation",
            "launch_contract": "helicopter-eval-v1",
            "attestation_digest": "a" * 64,
            "attestation_verified": True,
            "attestation_present": True,
            "attestation_mismatches": [],
        },
        "evaluator": {"product_revision": "e" * 40, "dirty": False},
        "config_digest": "f" * 64,
        "eligibility": "official",
        "comparable": True,
    }


def _write_run(tmp_path):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    usage = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    sample = {
        "sample_index": 0,
        "sample_id": "gsm8k|0:0",
        "attempt": 1,
        "status": "scored",
        "prompt": "Question",
        "raw_completion": "answer",
        "scored_completion": "answer",
        "generation": {
            "output_token_count": 2,
            "output_token_ids": [4, 0],
            "prompt_token_ids": [1],
            "prompt_text": "Question",
            "finish_reason": "stop",
            "stop_reason": 0,
            "terminal_reason": "stop",
            "truncated": False,
            "generation_limit": 4,
            "request_id": "req",
            "usage": usage,
        },
        "scoring": {
            "scorer_revision": "s" * 64,
            "repair_strategy": "A",
            "repair_action": "none",
        },
        "metrics": {"extractive_match": 1.0},
        "error_code": None,
        "error_message": None,
        "reference_answer": "answer",
        "provenance": {"task": "lighteval/math/gsm8k@0"},
    }
    terminal = {
        "samples": [sample],
        "metrics": {"extractive_match": 1.0},
        "native_metrics": {
            "extractive_match": 1.0,
            "secondary_metric": 0.5,
        },
        "generated_samples": 1,
        "truncated_samples": 0,
        "truncation_rate": 0.0,
        "performance": {"status": "not_attributable"},
    }
    terminal_path = run_dir / "terminal_evidence.json"
    terminal_path.write_text(json.dumps(terminal) + "\n", encoding="utf-8")
    identity = _identity()
    accounting = {
        "source_rows": 1,
        "dataset_accepted": 1,
        "dataset_rejected": 0,
        "selected": 1,
        "formatter_accepted": 1,
        "formatter_rejected": 0,
        "scored": 1,
        "model_invalid": 0,
        "provider_error": 0,
        "cache_error": 0,
        "scorer_error": 0,
        "harness_error": 0,
        "cancelled": 0,
    }
    manifest = {
        "schema_version": 1,
        "run_id": "run-1",
        "status": "completed",
        "identity_digest": hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "identities": {"run": identity},
        "accounting": accounting,
        "artifacts": [
            {
                "relative_path": "terminal_evidence.json",
                "sha256": hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
                "size_bytes": terminal_path.stat().st_size,
            }
        ],
        "completed_at": "2026-07-16T12:00:00+00:00",
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def test_projection_contains_only_strict_manifest_evidence(tmp_path) -> None:
    manifest_path = _write_run(tmp_path)
    payload = build_publication_payload(manifest_path)
    assert set(payload["manifest"]) == {
        "digest",
        "identity_digest",
        "accounting_digest",
        "terminal_status",
        "checksums_verified",
        "completed_at",
    }
    assert payload["manifest"]["terminal_status"] == "completed"
    assert "run_id" not in payload["manifest"]
    assert payload["native_metrics"] == {
        "extractive_match": 1.0,
        "secondary_metric": 0.5,
    }


def test_http_publication_is_gzip_bearer_and_digest_idempotent(tmp_path) -> None:
    manifest_path = _write_run(tmp_path)
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        captured["payload"] = json.loads(gzip.decompress(request.content))
        return httpx.Response(
            201, json={"status": "completed", "run_id": "run-1", "task_id": 7}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = publish_manifest(
        manifest_path=manifest_path,
        scoreboard_url="http://scoreboard",
        bearer_token="secret",
        client=client,
    )
    assert result.status == "published"
    assert captured["headers"]["authorization"] == "Bearer secret"
    assert captured["headers"]["content-encoding"] == "gzip"
    assert (
        captured["headers"]["idempotency-key"]
        == f"publish:{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}"
    )
    assert captured["payload"]["manifest"]["checksums_verified"] is True


def test_retry_reuses_manifest_digest_idempotency_key(tmp_path) -> None:
    manifest_path = _write_run(tmp_path)
    keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        keys.append(request.headers["idempotency-key"])
        return httpx.Response(
            200, json={"status": "completed", "run_id": "run-1", "task_id": 7}
        )

    from helicopter_lighteval.scoreboard import retry_publication

    client = httpx.Client(transport=httpx.MockTransport(handler))
    first = publish_manifest(
        manifest_path=manifest_path,
        scoreboard_url="http://scoreboard",
        bearer_token="secret",
        client=client,
    )
    second = retry_publication(
        manifest_path=manifest_path,
        scoreboard_url="http://scoreboard",
        bearer_token="secret",
        client=client,
    )
    assert first.status == second.status == "published"
    assert len(keys) == 2
    assert keys[0] == keys[1]


def test_transport_failure_keeps_a_failed_receipt(tmp_path) -> None:
    manifest_path = _write_run(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = publish_manifest(
        manifest_path=manifest_path,
        scoreboard_url="http://scoreboard",
        bearer_token="secret",
        client=client,
    )
    assert result.status == "failed"
    receipt = json.loads((manifest_path.parent / "publication.json").read_text())
    assert receipt["status"] == "failed"


def test_manifest_checksum_is_required(tmp_path) -> None:
    manifest_path = _write_run(tmp_path)
    (manifest_path.parent / "terminal_evidence.json").write_text(
        "tampered\n", encoding="utf-8"
    )
    try:
        verify_manifest(manifest_path)
    except Exception as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("tampered evidence must not verify")
