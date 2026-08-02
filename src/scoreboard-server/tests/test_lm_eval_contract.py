from __future__ import annotations

import pytest
from pydantic import ValidationError

from scoreboard_server.dtos.api.evaluation_results import (
    CampaignCreate,
    TaskPublication,
    sample_outcome,
)


def _task(mode: str = "fp16") -> dict:
    return {
        "identity": f"{'a' * 64}:{mode}:wikitext",
        "weight_sha256": "a" * 64,
        "weight_display_name": "model.pth",
        "wkv_mode": mode,
        "selector": "wikitext",
        "task_name": "wikitext",
        "task_version": "2.0",
        "module_family": "wikitext",
        "module": "tasks/wikitext/wikitext.yaml",
        "dataset": "EleutherAI/wikitext_document_level",
        "subset": "wikitext-2-raw-v1",
        "evaluation_splits": ["test"],
        "languages": [],
        "upstream_tags": [],
    }


def test_lm_eval_campaign_requires_complete_wkv_matrix() -> None:
    payload = {
        "schema_version": "lm-eval-campaign-v1",
        "run_key": "1" * 64,
        "config_digest": "2" * 64,
        "registry_digest": "3" * 64,
        "eval_contract_digest": "4" * 64,
        "evaluator": {"name": "lm-eval", "version": "0.4.12"},
        "configured_selectors": ["wikitext"],
        "resolved_selectors": ["wikitext"],
        "skipped_selectors": [],
        "expected_tasks": [_task("fp16"), _task("fp32io16")],
    }

    campaign = CampaignCreate.model_validate(payload)

    assert campaign.evaluator_name == "lm-eval"
    assert campaign.evaluator_version == "0.4.12"
    with pytest.raises(ValidationError, match="both WKV modes"):
        CampaignCreate.model_validate({**payload, "expected_tasks": [_task()]})


def test_lm_eval_task_contract_accepts_native_sample_shape() -> None:
    task = _task()
    publication = TaskPublication.model_validate(
        {
            "schema_version": "lm-eval-task-v1",
            "campaign_id": "campaign",
            "task": task,
            "artifact": {
                "evaluator": {"name": "lm-eval", "version": "0.4.12"},
                "results_path": "results.json",
                "details_paths": ["samples/0000.json"],
            },
            "task_config": {
                "original_num_docs": 1,
                "effective_num_docs": 1,
                "skipped_multiselect_docs": 0,
            },
            "model": {
                "weight_sha256": "a" * 64,
                "weight_display_name": "model.pth",
                "wkv_mode": "fp16",
                "prompt_template": "none",
                "gemm_policy": "fp16-accumulation",
                "gpu": "remote-vllm-pool",
                "max_num_seqs": 8,
                "max_num_batched_tokens": 8192,
                "dependency_versions": {
                    "lm-eval": "0.4.12",
                    "vllm": "0.23.1.dev0",
                    "torch": "2.11.0",
                },
                "evaluator": "lm-eval",
            },
            "sampling_config": {"request_type": "loglikelihood_rolling"},
            "primary_metric": "word_perplexity,none",
            "aggregates": {"word_perplexity,none": 12.5},
            "diagnostics": {
                "samples": 1,
                "completions": 0,
                "truncated": 0,
                "non_truncated": 0,
                "truncation_rate": 0.0,
                "turn_boundary_violations": 0,
                "turn_boundary_violation_rate": 0.0,
            },
            "details": [
                {
                    "sample_index": 0,
                    "document_index": 0,
                    "doc": {
                        "task_name": "wikitext",
                        "specific": {"helicopter_document_index": 0},
                    },
                    "metric": {"word_perplexity,none": 12.5},
                    "model_response": {
                        "filtered_resps": [[-10.0, False]],
                        "resps": [[[-10.0, False]]],
                    },
                }
            ],
        }
    )

    assert publication.artifact.evaluator_name == "lm-eval"
    assert sample_outcome(
        publication.details[0], publication.primary_metric
    ) == "undetermined"
