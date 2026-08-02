from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from helicopter_lm_eval.publish import publish_unit, task_metadata


class RecordingClient:
    def __init__(self) -> None:
        self.calls = []

    def publish_task(self, campaign_id, identity, payload):
        self.calls.append((campaign_id, identity, payload))


def test_task_metadata_expands_groups_to_leaf_tasks_without_loading_datasets():
    from lm_eval.tasks import TaskManager

    manager = TaskManager()
    metadata = task_metadata(manager, ("mmlu",), ("mmlu",))

    names = {row["task_name"] for row in metadata}
    assert "mmlu" not in names
    assert "mmlu_abstract_algebra" in names
    assert all(row["selector"] == "mmlu" for row in metadata)


def test_task_metadata_accepts_python_task_leaves_without_loading_datasets():
    from lm_eval.tasks import TaskManager

    manager = TaskManager()
    metadata = task_metadata(manager, ("squadv2",), ("squadv2",))

    assert [row["task_name"] for row in metadata] == ["squadv2"]
    assert metadata[0]["selector"] == "squadv2"
    assert metadata[0]["module_family"] == "squadv2"


def test_publish_unit_preserves_native_metrics_and_samples(tmp_path, monkeypatch):
    task = {
        "identity": f"{'a' * 64}:fp16:gsm8k",
        "weight_sha256": "a" * 64,
        "weight_display_name": "model.pth",
        "wkv_mode": "fp16",
        "selector": "gsm8k",
        "task_name": "gsm8k",
        "task_version": "3.0",
        "module_family": "gsm8k",
        "module": "tasks/gsm8k.yaml",
        "dataset": "openai/gsm8k",
        "subset": "main",
        "evaluation_splits": ["test"],
        "languages": [],
        "upstream_tags": ["math_word_problems"],
    }
    (tmp_path / "results.json").write_text(
        json.dumps(
            {
                "results": {
                    "gsm8k": {
                        "exact_match,strict-match": 0.5,
                        "exact_match_stderr,strict-match": 0.1,
                    }
                },
                "configs": {"gsm8k": {"output_type": "generate_until"}},
                "n-samples": {"gsm8k": {"original": 1, "effective": 1}},
                "samples": {
                    "gsm8k": [
                        {
                            "doc_id": 0,
                            "doc": {"question": "1+1?"},
                            "filtered_resps": ["2"],
                            "metrics": {"exact_match": 1.0},
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "artifacts.json").write_text(
        json.dumps(
            {
                "sample_artifacts": [
                    {"task_name": "gsm8k", "path": "samples/0000.json"}
                ]
            }
        ),
        encoding="utf-8",
    )
    unit = SimpleNamespace(
        weight_sha256="a" * 64,
        weight=Path("model.pth"),
        wkv_mode="fp16",
        manifest=SimpleNamespace(
            total_capacity=8,
            max_model_len=8192,
            vllm_version="0.23.1.dev0",
        ),
    )
    monkeypatch.setattr(
        "helicopter_lm_eval.publish.importlib.metadata.version",
        lambda name: "2.11.0" if name == "torch" else "fixture",
    )
    client = RecordingClient()
    config = SimpleNamespace(batch_size=4, eot_token_id=0, max_gen_toks=256)

    assert publish_unit(
        output_dir=tmp_path,
        campaign_id="campaign",
        expected=[task],
        unit=unit,
        config=config,
        client=client,
    ) == 1

    payload = client.calls[0][2]
    assert payload["schema_version"] == "lm-eval-task-v1"
    assert payload["primary_metric"] == "exact_match,strict-match"
    assert payload["artifact"]["evaluator"] == {
        "name": "lm-eval",
        "version": "0.4.12",
    }
    assert payload["model"]["prompt_template"] == "none"
    assert payload["sampling_config"]["batch_size"] == 4
    assert payload["details"][0]["document_index"] == 0
    assert payload["details"][0]["model_response"]["filtered_resps"] == ["2"]


def test_publish_unit_preserves_filter_metrics_and_document_ids(
    tmp_path, monkeypatch
):
    task = {
        "identity": f"{'a' * 64}:fp16:gsm8k",
        "weight_sha256": "a" * 64,
        "weight_display_name": "model.pth",
        "wkv_mode": "fp16",
        "selector": "gsm8k",
        "task_name": "gsm8k",
        "task_version": "3.0",
        "module_family": "gsm8k",
        "module": "tasks/gsm8k/gsm8k.yaml",
        "dataset": "openai/gsm8k",
        "subset": "main",
        "evaluation_splits": ["test"],
        "languages": [],
        "upstream_tags": [],
    }
    rows = [
        {
            "doc_id": 0,
            "doc": {"question": "1+1?"},
            "filter": filter_name,
            "metrics": ["exact_match"],
            "exact_match": score,
            "filtered_resps": ["2"],
        }
        for filter_name, score in (
            ("flexible-extract", 1.0),
            ("strict-match", 1.0),
        )
    ]
    (tmp_path / "results.json").write_text(
        json.dumps(
            {
                "results": {
                    "gsm8k": {
                        "alias": "gsm8k",
                        "exact_match,flexible-extract": 1.0,
                        "exact_match,strict-match": 1.0,
                        "sample_len": 1,
                    }
                },
                "configs": {"gsm8k": {"output_type": "generate_until"}},
                "n-samples": {"gsm8k": {"original": 1, "effective": 1}},
                "samples": {"gsm8k": rows},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "artifacts.json").write_text(
        json.dumps(
            {
                "sample_artifacts": [
                    {"task_name": "gsm8k", "path": "samples/0000.json"}
                ]
            }
        ),
        encoding="utf-8",
    )
    unit = SimpleNamespace(
        weight_sha256="a" * 64,
        weight=Path("model.pth"),
        wkv_mode="fp16",
        manifest=SimpleNamespace(
            total_capacity=8,
            max_model_len=8192,
            vllm_version="0.23.1.dev0",
        ),
    )
    config = SimpleNamespace(batch_size=4, eot_token_id=0, max_gen_toks=256)
    monkeypatch.setattr(
        "helicopter_lm_eval.publish.importlib.metadata.version",
        lambda _name: "2.11.0",
    )
    client = RecordingClient()

    publish_unit(
        output_dir=tmp_path,
        campaign_id="campaign",
        expected=[task],
        unit=unit,
        config=config,
        client=client,
    )

    payload = client.calls[0][2]
    assert payload["aggregates"] == {
        "exact_match,flexible-extract": 1.0,
        "exact_match,strict-match": 1.0,
    }
    assert [row["document_index"] for row in payload["details"]] == [0, 0]
    assert payload["details"][0]["metric"] == {
        "exact_match,flexible-extract": 1.0
    }
    assert payload["task_config"]["effective_num_docs"] == 1
