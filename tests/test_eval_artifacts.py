import copy
from dataclasses import replace
import hashlib
import math
from pathlib import Path

import pytest

from helicopter_eval import artifacts
from helicopter_eval.config import WeightIdentity
from helicopter_eval.plan import EvaluationShard, EvaluationUnit
from helicopter_eval.registry import RegistryTask


def _task() -> RegistryTask:
    return RegistryTask(
        selector="gsm8k",
        identity="gsm8k|0",
        name="gsm8k",
        version="0",
        module_family="gsm8k",
        module="lighteval.tasks.tasks.gsm8k",
        dataset="openai/gsm8k",
        subset="main",
        evaluation_splits=("test",),
        languages=("english",),
        upstream_tags=("math",),
    )


def _unit(tmp_path: Path) -> tuple[EvaluationUnit, EvaluationShard]:
    weight = tmp_path / "weight.pth"
    weight.write_bytes(b"weight")
    task = _task()
    shard = EvaluationShard("gsm8k:001-of-001", "gsm8k", (task,))
    unit = EvaluationUnit(
        WeightIdentity(
            configured_path="weight.pth",
            path=weight,
            display_name="weight.pth",
            sha256=hashlib.sha256(b"weight").hexdigest(),
        ),
        "fp16",
        (shard,),
    )
    return unit, shard


def _standard(root: Path):
    result_path = root / "results/model/results_stamp.json"
    details_path = root / "details/model/stamp/details_gsm8k_stamp.parquet"
    results = {
        "config_general": {
            "max_samples": None,
            "model_config": {
                "seed": 1234,
                "generation_parameters": {
                    "temperature": 0.96,
                    "top_p": 0.76,
                    "top_k": 32,
                    "presence_penalty": 1.0,
                    "frequency_penalty": 0.1,
                    "penalty_decay": 0.988,
                    "max_new_tokens": 8192,
                    "stop_tokens": ["\nUser:"],
                },
            },
        },
        "results": {
            "gsm8k|0": {
                "extractive_match": 1.0,
                "extractive_match_stderr": 0.0,
            },
            "all": {"extractive_match": 1.0},
        },
        "config_tasks": {
            "gsm8k|0": {
                "generation_size": 8192,
                "original_num_docs": 1,
                "effective_num_docs": 1,
                "skipped_multiselect_docs": 0,
            }
        },
    }
    rows = [
        {
            "doc": {
                "id": "0",
                "task_name": "gsm8k|0",
                "query": "1+1?",
                "specific": {"helicopter_document_index": 0},
            },
            "metric": {"extractive_match": 1.0},
            "model_response": {
                "input": "1+1?",
                "input_tokens": [1, 2],
                "text": ["<think>x</think>2", "bad\nUser:"],
                "text_post_processed": ["2", "bad"],
                "output_tokens": [[3, 4], [5]],
            },
        }
    ]
    return results, rows, result_path, [details_path]


def _model(unit: EvaluationUnit) -> dict[str, object]:
    return {
        "weight_sha256": unit.weight.sha256,
        "weight_display_name": unit.weight.display_name,
        "wkv_mode": unit.wkv_mode,
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


def test_standard_parser_preserves_native_metrics_and_multi_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit, shard = _unit(tmp_path)
    monkeypatch.setattr(
        artifacts,
        "_standard_artifacts",
        lambda _path: _standard(tmp_path),
    )
    publications = artifacts.publications_from_shard(
        shard_dir=tmp_path,
        campaign_id="11111111-1111-1111-1111-111111111111",
        unit=unit,
        shard=shard,
        model_execution=_model(unit),
        registry_tasks=shard.tasks,
    )
    identity, payload, digest = publications[0]
    assert identity.endswith(":gsm8k|0")
    assert len(digest) == 64
    assert payload["aggregates"] == {
        "extractive_match": 1.0,
        "extractive_match_stderr": 0.0,
    }
    assert payload["diagnostics"]["completions"] == 2
    assert payload["diagnostics"]["turn_boundary_violations"] == 1
    assert len(payload["details"][0]["model_response"]["text"]) == 2


def test_standard_parser_accepts_multiple_native_rows_for_one_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit, shard = _unit(tmp_path)
    standard = _standard(tmp_path)
    second_sampling_category = copy.deepcopy(standard[1][0])
    second_sampling_category["metric"] = {"extractive_match": 0.0}
    standard[1].append(second_sampling_category)
    monkeypatch.setattr(artifacts, "_standard_artifacts", lambda _path: standard)

    publications = artifacts.publications_from_shard(
        shard_dir=tmp_path,
        campaign_id="11111111-1111-1111-1111-111111111111",
        unit=unit,
        shard=shard,
        model_execution=_model(unit),
        registry_tasks=shard.tasks,
    )

    details = publications[0][1]["details"]
    assert [detail["sample_index"] for detail in details] == [0, 1]
    assert [detail["document_index"] for detail in details] == [0, 0]


def test_standard_parser_accounts_for_skipped_multiselect_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit, shard = _unit(tmp_path)
    standard = _standard(tmp_path)
    task_config = standard[0]["config_tasks"]["gsm8k|0"]
    task_config["original_num_docs"] = 2
    task_config["effective_num_docs"] = 1
    task_config["skipped_multiselect_docs"] = 1
    monkeypatch.setattr(artifacts, "_standard_artifacts", lambda _path: standard)

    publications = artifacts.publications_from_shard(
        shard_dir=tmp_path,
        campaign_id="11111111-1111-1111-1111-111111111111",
        unit=unit,
        shard=shard,
        model_execution=_model(unit),
        registry_tasks=shard.tasks,
    )

    assert publications[0][1]["task_config"]["skipped_multiselect_docs"] == 1


def test_standard_parser_accepts_only_registry_proven_superset_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_unit, _ = _unit(tmp_path)
    root_task = replace(
        _task(),
        identity="bbq|0",
        name="bbq",
        selector="bbq",
        module_family="bbq",
        module="lighteval.tasks.tasks.bbq",
        dataset="lighteval/bbq_helm",
        subset="all",
    )
    child_task = replace(
        root_task,
        identity="bbq:Age|0",
        name="bbq:Age",
        subset="Age",
    )
    shard = EvaluationShard("bbq:002-of-002", "bbq", (root_task,))
    unit = EvaluationUnit(base_unit.weight, base_unit.wkv_mode, (shard,))
    results, rows, result_path, detail_paths = _standard(tmp_path)
    native_aggregate = results["results"].pop("gsm8k|0")
    task_config = results["config_tasks"].pop("gsm8k|0")
    results["results"].update(
        {
            "bbq|0": copy.deepcopy(native_aggregate),
            "bbq:Age|0": copy.deepcopy(native_aggregate),
            "bbq:_average|0": copy.deepcopy(native_aggregate),
        }
    )
    results["config_tasks"].update(
        {
            "bbq|0": copy.deepcopy(task_config),
            "bbq:Age|0": copy.deepcopy(task_config),
        }
    )
    rows[0]["doc"]["task_name"] = "bbq|0"
    child_row = copy.deepcopy(rows[0])
    child_row["doc"]["task_name"] = "bbq:Age|0"
    rows.append(child_row)
    standard = results, rows, result_path, detail_paths
    monkeypatch.setattr(artifacts, "_standard_artifacts", lambda _path: standard)

    publications = artifacts.publications_from_shard(
        shard_dir=tmp_path,
        campaign_id="11111111-1111-1111-1111-111111111111",
        unit=unit,
        shard=shard,
        model_execution=_model(unit),
        registry_tasks=(root_task, child_task),
    )

    assert len(publications) == 1
    assert publications[0][0].endswith(":bbq|0")
    assert {detail["doc"]["task_name"] for detail in publications[0][1]["details"]} == {
        "bbq|0"
    }

    with pytest.raises(
        artifacts.ArtifactError,
        match="task set does not match deterministic shard",
    ):
        artifacts.publications_from_shard(
            shard_dir=tmp_path,
            campaign_id="11111111-1111-1111-1111-111111111111",
            unit=unit,
            shard=shard,
            model_execution=_model(unit),
            registry_tasks=(root_task,),
        )


def test_standard_parser_never_selects_stderr_as_primary_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit, shard = _unit(tmp_path)
    standard = _standard(tmp_path)
    standard[0]["results"]["gsm8k|0"] = {
        "stderr": 0.01,
        "extractive_match": 1.0,
    }
    monkeypatch.setattr(artifacts, "_standard_artifacts", lambda _path: standard)

    publication = artifacts.publications_from_shard(
        shard_dir=tmp_path,
        campaign_id="11111111-1111-1111-1111-111111111111",
        unit=unit,
        shard=shard,
        model_execution=_model(unit),
        registry_tasks=shard.tasks,
    )[0][1]

    assert publication["primary_metric"] == "extractive_match"


def test_standard_parser_rejects_non_numeric_native_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit, shard = _unit(tmp_path)
    standard = _standard(tmp_path)
    standard[0]["results"]["gsm8k|0"]["unexpected"] = "not-a-number"
    monkeypatch.setattr(artifacts, "_standard_artifacts", lambda _path: standard)

    with pytest.raises(artifacts.ArtifactError, match="aggregate is invalid"):
        artifacts.publications_from_shard(
            shard_dir=tmp_path,
            campaign_id="11111111-1111-1111-1111-111111111111",
            unit=unit,
            shard=shard,
            model_execution=_model(unit),
            registry_tasks=shard.tasks,
        )


def test_standard_parser_rejects_symlink_result_file(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    result_dir = tmp_path / "shard" / "results" / "model"
    result_dir.mkdir(parents=True)
    (result_dir / "results_stamp.json").symlink_to(outside)

    with pytest.raises(
        artifacts.ArtifactError,
        match="safe shard child|regular non-symlink",
    ):
        artifacts._standard_artifacts(tmp_path / "shard")


def test_standard_parser_accepts_logprob_rows_with_output_token_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit, shard = _unit(tmp_path)
    standard = _standard(tmp_path)
    standard[1][0]["model_response"] = {
        "input": "1+1?",
        "input_tokens": [1, 2],
        "text": [],
        "text_post_processed": None,
        "output_tokens": [[3], [4, 5]],
        "logprobs": [-0.1, -0.2],
        "argmax_logits_eq_gold": [True, False],
    }
    monkeypatch.setattr(artifacts, "_standard_artifacts", lambda _path: standard)

    publication = artifacts.publications_from_shard(
        shard_dir=tmp_path,
        campaign_id="11111111-1111-1111-1111-111111111111",
        unit=unit,
        shard=shard,
        model_execution=_model(unit),
        registry_tasks=shard.tasks,
    )[0][1]

    assert publication["diagnostics"]["samples"] == 1
    assert publication["diagnostics"]["completions"] == 0
    assert publication["details"][0]["model_response"]["output_tokens"] == [
        [3],
        [4, 5],
    ]


def test_standard_parser_rejects_misaligned_postprocessed_completions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit, shard = _unit(tmp_path)
    standard = _standard(tmp_path)
    standard[1][0]["model_response"]["text_post_processed"] = ["2"]
    monkeypatch.setattr(artifacts, "_standard_artifacts", lambda _path: standard)

    with pytest.raises(
        artifacts.ArtifactError,
        match="text_post_processed must align",
    ):
        artifacts.publications_from_shard(
            shard_dir=tmp_path,
            campaign_id="11111111-1111-1111-1111-111111111111",
            unit=unit,
            shard=shard,
            model_execution=_model(unit),
            registry_tasks=shard.tasks,
        )


def test_standard_parser_rejects_misaligned_logprob_token_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit, shard = _unit(tmp_path)
    standard = _standard(tmp_path)
    standard[1][0]["model_response"] = {
        "input": "1+1?",
        "input_tokens": [1, 2],
        "text": [],
        "output_tokens": [[3]],
        "logprobs": [-0.1, -0.2],
    }
    monkeypatch.setattr(artifacts, "_standard_artifacts", lambda _path: standard)

    with pytest.raises(
        artifacts.ArtifactError,
        match="log-likelihood evidence and output-token counts differ",
    ):
        artifacts.publications_from_shard(
            shard_dir=tmp_path,
            campaign_id="11111111-1111-1111-1111-111111111111",
            unit=unit,
            shard=shard,
            model_execution=_model(unit),
            registry_tasks=shard.tasks,
        )

    standard = _standard(tmp_path)
    standard[1][0]["model_response"] = {
        "input": "1+1?",
        "input_tokens": [1, 2],
        "text": [],
        "logprobs": [-0.1, -0.2],
    }
    monkeypatch.setattr(artifacts, "_standard_artifacts", lambda _path: standard)
    with pytest.raises(
        artifacts.ArtifactError,
        match="output_tokens must be a non-empty array",
    ):
        artifacts.publications_from_shard(
            shard_dir=tmp_path,
            campaign_id="11111111-1111-1111-1111-111111111111",
            unit=unit,
            shard=shard,
            model_execution=_model(unit),
            registry_tasks=shard.tasks,
        )

    standard = _standard(tmp_path)
    standard[1][0]["model_response"] = {
        "input": "1+1?",
        "input_tokens": [1, 2],
        "text": [],
        "output_tokens": [[]],
        "logprobs": [-0.1],
    }
    monkeypatch.setattr(artifacts, "_standard_artifacts", lambda _path: standard)
    with pytest.raises(
        artifacts.ArtifactError,
        match="output token groups must be non-empty",
    ):
        artifacts.publications_from_shard(
            shard_dir=tmp_path,
            campaign_id="11111111-1111-1111-1111-111111111111",
            unit=unit,
            shard=shard,
            model_execution=_model(unit),
            registry_tasks=shard.tasks,
        )


def test_standard_parser_rejects_nonfinite_and_partial_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit, shard = _unit(tmp_path)
    result = _standard(tmp_path)
    result[0]["results"]["gsm8k|0"]["extractive_match"] = math.nan
    monkeypatch.setattr(artifacts, "_standard_artifacts", lambda _path: result)
    with pytest.raises(artifacts.ArtifactError, match="not finite"):
        artifacts.publications_from_shard(
            shard_dir=tmp_path,
            campaign_id="11111111-1111-1111-1111-111111111111",
            unit=unit,
            shard=shard,
            model_execution=_model(unit),
            registry_tasks=shard.tasks,
        )

    result = _standard(tmp_path)
    result[0]["config_tasks"]["gsm8k|0"]["effective_num_docs"] = 0
    monkeypatch.setattr(artifacts, "_standard_artifacts", lambda _path: result)
    with pytest.raises(artifacts.ArtifactError, match="full evaluation split"):
        artifacts.publications_from_shard(
            shard_dir=tmp_path,
            campaign_id="11111111-1111-1111-1111-111111111111",
            unit=unit,
            shard=shard,
            model_execution=_model(unit),
            registry_tasks=shard.tasks,
        )


def test_standard_parser_rejects_non_standard_results_json(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "results/model/results_stamp.json"
    detail_path = tmp_path / "details/model/stamp/details_gsm8k_stamp.parquet"
    result_path.parent.mkdir(parents=True)
    detail_path.parent.mkdir(parents=True)
    result_path.write_text('{"results": {"score": NaN}}', encoding="utf-8")
    detail_path.touch()

    with pytest.raises(
        artifacts.ArtifactError,
        match="standard results JSON is invalid",
    ):
        artifacts._standard_artifacts(tmp_path)


def test_standard_parser_validates_sampling_and_model_execution_locally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit, shard = _unit(tmp_path)
    standard = _standard(tmp_path)
    standard[0]["config_general"]["model_config"]["generation_parameters"][
        "temperature"
    ] = 0.0
    monkeypatch.setattr(artifacts, "_standard_artifacts", lambda _path: standard)

    with pytest.raises(
        artifacts.ArtifactError,
        match="sampling contract: temperature",
    ):
        artifacts.publications_from_shard(
            shard_dir=tmp_path,
            campaign_id="11111111-1111-1111-1111-111111111111",
            unit=unit,
            shard=shard,
            model_execution=_model(unit),
            registry_tasks=shard.tasks,
        )

    model_execution = _model(unit)
    model_execution["weight_sha256"] = "f" * 64
    with pytest.raises(
        artifacts.ArtifactError,
        match="planned unit: weight_sha256",
    ):
        artifacts.publications_from_shard(
            shard_dir=tmp_path,
            campaign_id="11111111-1111-1111-1111-111111111111",
            unit=unit,
            shard=shard,
            model_execution=model_execution,
            registry_tasks=shard.tasks,
        )
