from __future__ import annotations

import json
import stat
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from helicopter_lighteval import publish
from helicopter_lighteval.publish import PublicationError


CAMPAIGN_ID = "11111111-1111-1111-1111-111111111111"


def _task() -> dict[str, object]:
    return {
        "identity": f"{'a' * 64}:fp16:gsm8k|0",
        "weight_sha256": "a" * 64,
        "weight_display_name": "model.pth",
        "wkv_mode": "fp16",
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


def _model() -> dict[str, object]:
    return {
        "weight_sha256": "a" * 64,
        "weight_display_name": "model.pth",
        "wkv_mode": "fp16",
        "prompt_template": "assistant",
        "gemm_policy": "fp16-accumulation",
        "gpu": "fixture",
        "max_num_seqs": 1280,
        "max_num_batched_tokens": 8192,
        "dependency_versions": {
            "lighteval": "0.13.0",
            "vllm": "fixture",
            "torch": "fixture",
        },
    }


def _sampling() -> dict[str, object]:
    return {
        "temperature": 0.96,
        "top_p": 0.76,
        "top_k": 32,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.1,
        "repetition_penalty": 1.0,
        "penalty_decay": 0.988,
        "max_new_tokens": 8192,
        "stop": ["\nUser:"],
        "ignore_eos": False,
    }


def _standard() -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    return (
        {
            "results": {
                "gsm8k|0": {
                    "exact_match": 1.0,
                    "exact_match_stderr": 0.0,
                }
            },
            "config_tasks": {
                "gsm8k|0": {
                    "original_num_docs": 1,
                    "effective_num_docs": 1,
                    "skipped_multiselect_docs": 0,
                }
            },
        },
        [
            {
                "doc": {
                    "task_name": "gsm8k|0",
                    "query": "1+1?",
                    "specific": {"helicopter_document_index": 0},
                },
                "metric": {"exact_match": 1.0},
                "model_response": {
                    "input": "1+1?",
                    "input_tokens": [1, 2],
                    "text": ["<think>x</think>2", "bad\nUser:"],
                    "text_post_processed": ["2", "bad"],
                    "output_tokens": [[3, 4], [5]],
                },
            }
        ],
        {
            "lighteval_version": "0.13.0",
            "results_path": "results/model/results_stamp.json",
            "details_paths": ["details/model/stamp/details_gsm8k_stamp.parquet"],
        },
    )


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def publish_task(
        self,
        campaign_id: str,
        identity: str,
        payload: dict[str, object],
    ) -> None:
        self.calls.append((campaign_id, identity, payload))


def test_publish_results_preserves_native_data_and_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(publish, "_read_standard_results", lambda _path: _standard())
    client = RecordingClient()

    count = publish.publish_results(
        output_dir=tmp_path,
        campaign_id=CAMPAIGN_ID,
        expected_tasks=[_task()],
        model=_model(),
        sampling_config=_sampling(),
        client=client,
    )

    assert count == 1
    campaign_id, identity, payload = client.calls[0]
    assert campaign_id == CAMPAIGN_ID
    assert identity == _task()["identity"]
    assert payload["aggregates"] == {
        "exact_match": 1.0,
        "exact_match_stderr": 0.0,
    }
    assert payload["primary_metric"] == "exact_match"
    assert payload["diagnostics"]["completions"] == 2
    assert payload["diagnostics"]["turn_boundary_violations"] == 1
    assert len(payload["details"][0]["model_response"]["text"]) == 2


def test_publish_results_fails_closed_on_incomplete_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results, rows, artifact = _standard()
    results["results"] = {}
    monkeypatch.setattr(
        publish,
        "_read_standard_results",
        lambda _path: (results, rows, artifact),
    )

    with pytest.raises(PublicationError, match="missing an expected task"):
        publish.publish_results(
            output_dir=tmp_path,
            campaign_id=CAMPAIGN_ID,
            expected_tasks=[_task()],
            model=_model(),
            sampling_config=_sampling(),
            client=RecordingClient(),
        )


def test_standard_result_reader_uses_results_and_details_pair(tmp_path: Path) -> None:
    results, rows, _artifact = _standard()
    result_path = tmp_path / "results/model/results_stamp.json"
    detail_path = tmp_path / "details/model/stamp" / "details_gsm8k|0_stamp.parquet"
    result_path.parent.mkdir(parents=True)
    detail_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(results), encoding="utf-8")
    parquet.write_table(pa.Table.from_pylist(rows), detail_path)

    loaded_results, loaded_rows, artifact = publish._read_standard_results(tmp_path)

    assert loaded_results == results
    assert loaded_rows == rows
    assert artifact["results_path"] == "results/model/results_stamp.json"
    assert artifact["details_paths"] == [
        "details/model/stamp/details_gsm8k|0_stamp.parquet"
    ]


def test_sample_audit_retains_bounded_text_and_scorer_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results, rows, artifact = _standard()
    rows.append(
        {
            "doc": {
                "task_name": "gsm8k|0",
                "query": "2+2?",
                "choices": ["4"],
                "gold_index": 0,
                "specific": {"helicopter_document_index": 1},
            },
            "metric": {"exact_match": 0.0},
            "model_response": {
                "input": "User✿2+2?✿\nBot✿<think",
                "input_tokens": [7, 8],
                "text": ["<think>x</think>5"],
                "text_post_processed": ["5"],
                "output_tokens": [[9, 10]],
            },
        }
    )
    rows[0]["doc"].update(choices=["2"], gold_index=0)
    rows[0]["model_response"]["input"] = "User✿1+1?✿\nBot✿<think"
    monkeypatch.setattr(
        publish,
        "_read_standard_results",
        lambda _path: (results, list(reversed(rows)), artifact),
    )
    destination = tmp_path / "actor" / "lighteval_sample_audit.json"

    publish.write_sample_audit(
        output_dir=tmp_path,
        destination=destination,
        task_names=["gsm8k|0"],
        weight_sha256="a" * 64,
        wkv_mode="fp32io16",
        samples_per_task=1,
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert payload == {
        "schema_version": 1,
        "weight_sha256": "a" * 64,
        "wkv_mode": "fp32io16",
        "samples_per_task": 1,
        "tasks": {
            "gsm8k|0": [
                {
                    "document_index": 0,
                    "question": "1+1?",
                    "model_input_text": "User✿1+1?✿\nBot✿<think",
                    "model_output_text": [
                        "<think>x</think>2",
                        "bad\nUser:",
                    ],
                    "scorer_input": {
                        "golds": ["2"],
                        "predictions": ["2", "bad"],
                    },
                    "scorer_output": {"exact_match": 1.0},
                    "standard_answer": ["2"],
                }
            ]
        },
    }
    assert "input_tokens" not in destination.read_text(encoding="utf-8")
    assert "output_tokens" not in destination.read_text(encoding="utf-8")


def test_sample_audit_preserves_asdiv_string_gold_bug(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results, rows, artifact = _standard()
    rows[0]["doc"].update(
        task_name="asdiv|0",
        choices="12",
        gold_index=[0],
    )
    monkeypatch.setattr(
        publish,
        "_read_standard_results",
        lambda _path: (results, rows, artifact),
    )
    destination = tmp_path / "lighteval_sample_audit.json"

    publish.write_sample_audit(
        output_dir=tmp_path,
        destination=destination,
        task_names=["asdiv|0"],
        weight_sha256="a" * 64,
        wkv_mode="fp32io16",
        samples_per_task=1,
    )

    sample = json.loads(destination.read_text(encoding="utf-8"))["tasks"][
        "asdiv|0"
    ][0]
    assert sample["standard_answer"] == ["12"]
    assert sample["scorer_input"]["golds"] == ["1"]


def test_prepare_staging_creates_private_owned_directory(tmp_path: Path) -> None:
    staging = tmp_path / "private" / "eval"
    resolved = publish.prepare_staging(staging)

    assert resolved == staging.resolve()
    assert stat.S_IMODE(staging.stat().st_mode) == 0o700

    staging.chmod(0o755)
    with pytest.raises(PublicationError, match="0700"):
        publish.prepare_staging(staging)


def test_canonical_json_is_stable_and_rejects_nan() -> None:
    assert publish.canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    with pytest.raises(PublicationError, match="canonical JSON"):
        publish.canonical_json({"metric": float("nan")})
