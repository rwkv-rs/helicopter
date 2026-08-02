from __future__ import annotations

import hashlib
import json
import stat
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from helicopter_lm_eval import evaluate
from helicopter_lm_eval.config import ConfigError, LMEvalConfig


ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "pool.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "global_step": 10,
                "wkv_mode": "fp32io16",
                "vllm_version": "0.23.1.dev0",
                "max_model_len": 10240,
                "replicas": [
                    {
                        "base_url": "http://127.0.0.1:8000",
                        "max_concurrency": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_task_resolution_accepts_common_request_types_and_groups() -> None:
    manager, resolved = evaluate._resolve_tasks(
        ("wikitext", "hellaswag", "gsm8k", "mmlu")
    )

    assert resolved == ("wikitext", "hellaswag", "gsm8k", "mmlu")
    assert manager.task_index["wikitext"].cfg["output_type"] == (
        "loglikelihood_rolling"
    )
    assert manager.task_index["hellaswag"].cfg["output_type"] == "multiple_choice"
    assert manager.task_index["gsm8k"].cfg["output_type"] == "generate_until"
    assert manager.task_index["mmlu"].cfg["group"] == "mmlu"


def test_task_resolution_supports_globs_and_rejects_unknown_selectors() -> None:
    _manager, resolved = evaluate._resolve_tasks(("arc_easy", "gsm8k*"))

    assert resolved[0] == "arc_easy"
    assert "gsm8k" in resolved
    assert "gsm8k_cot" in resolved
    with pytest.raises(ConfigError, match="unknown lm-eval task, group, tag, or pattern"):
        evaluate._resolve_tasks(("does-not-exist",))


def test_qwen35_alignment_suite_uses_stable_unlimited_selectors() -> None:
    with (ROOT / "configs/eval/lm_eval_qwen35.toml").open("rb") as stream:
        config = tomllib.load(stream)

    expected = (
        "mmlu_pro",
        "mmlu_redux_generative",
        "ceval-valid",
        "gpqa_diamond_cot_zeroshot",
        "ifeval",
        "mmmlu",
    )
    manager, resolved = evaluate._resolve_tasks(tuple(config["tasks"]))

    assert resolved == expected
    assert "limit" not in config
    assert manager.task_index["mmlu_pro"].cfg["group"] == "mmlu_pro"
    assert manager.task_index["mmlu_redux_generative"].cfg["group"] == (
        "mmlu_redux_generative"
    )
    assert manager.task_index["ceval-valid"].cfg["group"] == "ceval-valid"
    assert manager.task_index["gpqa_diamond_cot_zeroshot"].cfg["output_type"] == (
        "generate_until"
    )
    assert manager.task_index["ifeval"].cfg["output_type"] == "generate_until"
    assert manager.task_index["mmmlu"].cfg["group"] == "mmmlu"


def test_capability_suite_is_unlimited_and_disjoint_from_lighteval() -> None:
    with (ROOT / "configs/eval/lm_eval_capabilities.toml").open("rb") as stream:
        config = tomllib.load(stream)
    with (ROOT / "configs/eval/lighteval.toml").open("rb") as stream:
        lighteval = tomllib.load(stream)

    expected = (
        "race",
        "wmt14-en-fr",
        "lambada_openai",
        "blimp",
        "longbench_passage_retrieval_en",
    )
    manager, resolved = evaluate._resolve_tasks(tuple(config["tasks"]))

    assert resolved == expected
    assert "limit" not in config
    assert set(expected).isdisjoint(lighteval["benchmarks"])
    assert manager.task_index["race"].cfg["output_type"] == "multiple_choice"
    assert manager.task_index["wmt14-en-fr"].cfg["output_type"] == "generate_until"
    assert manager.task_index["lambada_openai"].cfg["output_type"] == "loglikelihood"
    assert manager.task_index["blimp"].cfg["group"] == "blimp"
    assert "longbench_synthetic_tasks" in manager.task_index[
        "longbench_passage_retrieval_en"
    ].tags


def test_catalog_delta_suite_is_native_unlimited_and_exact() -> None:
    with (ROOT / "configs/eval/lm_eval_catalog_delta.toml").open("rb") as stream:
        config = tomllib.load(stream)
    with (ROOT / "configs/eval/lighteval.toml").open("rb") as stream:
        lighteval = tomllib.load(stream)

    expected = ("gpqa_extended_zeroshot", "cmmlu")
    manager, resolved = evaluate._resolve_tasks(tuple(config["tasks"]))

    assert resolved == expected
    assert "limit" not in config
    assert config["log_samples"] is True
    assert set(expected).isdisjoint(lighteval["benchmarks"])
    gpqa = manager.task_index["gpqa_extended_zeroshot"].cfg
    assert gpqa["dataset_name"] == "gpqa_extended"
    assert gpqa["num_fewshot"] == 0
    assert gpqa["output_type"] == "multiple_choice"
    cmmlu = manager.task_index["cmmlu"].cfg
    assert cmmlu["group"] == "cmmlu"
    assert len(cmmlu["task"]) == 67
    assert [metric["metric"] for metric in cmmlu["aggregate_metric_list"]] == [
        "acc",
        "acc_norm",
    ]
    assert all(
        metric["weight_by_size"] for metric in cmmlu["aggregate_metric_list"]
    )


def test_capability_result_manifest_preserves_comparison_contract() -> None:
    result_path = ROOT / "docs/evaluation/lm_eval_capability_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))

    expected_tasks = {
        "race": 1045,
        "wmt14-en-fr": 3003,
        "lambada_openai": 5153,
        "blimp": 67000,
        "longbench_passage_retrieval_en": 200,
    }
    protocol = result["protocol"]

    assert result["schema_version"] == 1
    assert protocol["lm_eval_version"] == "0.4.12"
    assert protocol["max_length"] == 16382
    assert protocol["chat_template"] is None
    assert protocol["limit"] is None
    assert protocol["log_samples"] is True
    assert {
        task["selector"]: task["sample_count"] for task in protocol["tasks"]
    } == expected_tasks
    assert result["metric_directions"]["wmt14-en-fr.ter"] == "lower"

    models = {model["comparison_role"]: model for model in result["models"]}
    assert set(models) == {"target", "lower_bound", "upper_bound"}
    assert {model["sample_record_count"] for model in models.values()} == {76401}
    assert models["target"]["identity"]["weight_sha256"] == (
        "22fe129988f6e98480b344075597259a13ae4201c1d8dedf987246772e613586"
    )
    assert models["lower_bound"]["identity"]["revision"] == (
        "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
    )
    assert models["upper_bound"]["identity"]["revision"] == (
        "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
    )
    for model in models.values():
        assert set(model["metrics"]) == expected_tasks.keys()


def test_catalog_delta_result_manifest_preserves_scope_and_comparison() -> None:
    result_path = ROOT / "docs/evaluation/lm_eval_catalog_delta_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))

    scope = result["scope"]
    protocol = result["protocol"]
    task = protocol["task"]

    assert result["schema_version"] == 1
    assert scope["native_selectors"] == ["gpqa_extended_zeroshot", "cmmlu"]
    assert scope["completed_selectors"] == ["cmmlu"]
    assert scope["blocked"] == [
        {
            "selector": "gpqa_extended_zeroshot",
            "dataset": "Idavidrein/gpqa",
            "dataset_name": "gpqa_extended",
            "status": "blocked",
            "reason": (
                "gated dataset requires accepted terms and an authorized "
                "read-only Hugging Face token"
            ),
        }
    ]
    assert set(scope["excluded_non_native"]) == {
        "AMC23",
        "SWE-bench Verified",
        "SWE-bench Multilingual",
        "SWE-bench Pro",
    }
    assert protocol["lm_eval_version"] == "0.4.12"
    assert protocol["datasets_version"] == "3.6.0"
    assert protocol["limit"] is None
    assert protocol["log_samples"] is True
    assert task == {
        "selector": "cmmlu",
        "version": 1,
        "leaf_task_count": 67,
        "sample_count": 11582,
        "choice_request_count": 46328,
        "metrics": ["acc", "acc_norm"],
    }

    models = {model["comparison_role"]: model for model in result["models"]}
    assert set(models) == {"target", "lower_bound", "upper_bound"}
    assert {model["sample_record_count"] for model in models.values()} == {11582}
    assert models["target"]["identity"]["weight_sha256"] == (
        "22fe129988f6e98480b344075597259a13ae4201c1d8dedf987246772e613586"
    )
    assert models["lower_bound"]["identity"]["revision"] == (
        "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
    )
    assert models["upper_bound"]["identity"]["revision"] == (
        "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
    )
    assert models["target"]["metrics"]["acc"]["value"] < models["lower_bound"][
        "metrics"
    ]["acc"]["value"]
    assert models["lower_bound"]["metrics"]["acc"]["value"] < models[
        "upper_bound"
    ]["metrics"]["acc"]["value"]


def test_dry_run_preflights_without_starting_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _write_manifest(tmp_path)
    config = tmp_path / "eval.toml"
    config.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'backend = "vllm_http"',
                'tasks = ["wikitext"]',
                f'output_dir = "{tmp_path / "results"}"',
            ]
        ),
        encoding="utf-8",
    )
    closed: list[bool] = []

    class Pool:
        def __init__(self, configured_manifest):
            self.manifest = configured_manifest

        def preflight(self):
            return "rwkv-current"

        def close(self):
            closed.append(True)

    monkeypatch.setattr(evaluate, "VLLMHttpPool", Pool)

    assert (
        evaluate.run(
            config_path=config,
            env={"HELICOPTER_VLLM_POOL_MANIFEST": str(manifest)},
            dry_run=True,
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "ready"
    assert output["model_id"] == "rwkv-current"
    assert output["effective_max_length"] == 10238
    assert output["resolved_tasks"] == ["wikitext"]
    assert closed == [True]


def test_run_dispatches_choice_and_generation_tasks_to_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lm_eval

    manifest = _write_manifest(tmp_path)
    output_dir = tmp_path / "results"
    config = tmp_path / "eval.toml"
    config.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'backend = "vllm_http"',
                'tasks = ["hellaswag", "gsm8k"]',
                f'output_dir = "{output_dir}"',
                "batch_size = 4",
                "max_gen_toks = 512",
                "limit = 1",
            ]
        ),
        encoding="utf-8",
    )
    closed: list[bool] = []
    calls: list[dict[str, object]] = []

    class Pool:
        total_capacity = 2

        def __init__(self, configured_manifest):
            self.manifest = configured_manifest
            self.model_id = "rwkv-current"

        def preflight(self):
            return self.model_id

        def close(self):
            closed.append(True)

    def simple_evaluate(**kwargs):
        calls.append(kwargs)
        return {
            "results": {
                "hellaswag": {"acc,none": 0.5},
                "gsm8k": {"exact_match,strict-match": 0.25},
            },
            "versions": {"hellaswag": 1, "gsm8k": 3},
        }

    monkeypatch.setattr(evaluate, "VLLMHttpPool", Pool)
    monkeypatch.setattr(lm_eval, "simple_evaluate", simple_evaluate)

    assert (
        evaluate.run(
            config_path=config,
            env={"HELICOPTER_VLLM_POOL_MANIFEST": str(manifest)},
            dry_run=False,
        )
        == 0
    )

    assert len(calls) == 1
    assert calls[0]["tasks"] == ["hellaswag", "gsm8k"]
    assert calls[0]["batch_size"] == 4
    assert calls[0]["limit"] == 1
    model = calls[0]["model"]
    assert model.max_gen_toks == 512
    assert json.loads((output_dir / "summary.json").read_text())["tasks"] == [
        "hellaswag",
        "gsm8k",
    ]
    assert closed == [True]


def test_result_writer_preserves_raw_results_and_normalizes_metrics(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "results"
    output_dir.mkdir(mode=0o755)
    output_dir.chmod(0o755)
    config = LMEvalConfig(
        tasks=("wikitext",),
        output_dir=output_dir,
        batch_size=1,
        eot_token_id=0,
        max_gen_toks=256,
        limit=None,
        log_samples=False,
        vllm_pool_manifest=tmp_path / "pool.json",
        manifest=SimpleNamespace(
            global_step=10,
            wkv_mode="fp32io16",
            max_model_len=10240,
        ),
    )
    raw = {
        "results": {"wikitext": {"word_perplexity,none": 12.5}},
        "versions": {"wikitext": 2},
    }

    evaluate._write_results(
        config=config,
        model_id="rwkv-current",
        version="0.4.12",
        results=raw,
    )

    assert json.loads((output_dir / "results.json").read_text()) == raw
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["metrics"] == raw["results"]
    assert summary["model_id"] == "rwkv-current"
    assert summary["effective_max_length"] == 10238
    assert stat.S_IMODE(output_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((output_dir / "results.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((output_dir / "summary.json").stat().st_mode) == 0o600


def test_published_run_completes_every_matrix_unit_before_finalize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lm_eval

    weight_root = tmp_path / "weights"
    weight_root.mkdir()
    weight = weight_root / "model.pth"
    weight.write_bytes(b"checkpoint")
    digest = hashlib.sha256(b"checkpoint").hexdigest()
    manifests = []
    for index, mode in enumerate(("fp16", "fp32io16")):
        manifest = tmp_path / f"{mode}.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "global_step": 10,
                    "wkv_mode": mode,
                    "vllm_version": "0.23.1.dev0",
                    "max_model_len": 10240,
                    "weight_sha256": digest,
                    "weight_display_name": weight.name,
                    "replicas": [
                        {
                            "base_url": f"http://127.0.0.1:{8000 + index}",
                            "max_concurrency": 2,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        manifests.append(manifest)
    config = tmp_path / "campaign.toml"
    config.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'backend = "vllm_http"',
                "publish = true",
                'tasks = ["wikitext"]',
                f'output_dir = "{tmp_path / "results"}"',
                'weights = ["model.pth"]',
                'wkv_modes = ["fp16", "fp32io16"]',
                "pool_manifests = [",
                *(f'  "{path}",' for path in manifests),
                "]",
                "batch_size = 2",
                "max_gen_toks = 256",
            ]
        ),
        encoding="utf-8",
    )
    events: list[str] = []

    class Pool:
        total_capacity = 2

        def __init__(self, manifest):
            self.manifest = manifest
            self.model_id = f"rwkv-{manifest.wkv_mode}"

        def preflight(self):
            events.append(f"pool:{self.manifest.wkv_mode}")
            return self.model_id

        def close(self):
            events.append(f"close:{self.manifest.wkv_mode}")

    class Client:
        def __init__(self, _url, _token):
            pass

        def preflight(self, evaluator, version):
            events.append(f"scoreboard:{evaluator}:{version}")
            return {"status": "ready"}

        def create_campaign(self, payload, run_key):
            assert payload["run_key"] == run_key
            events.append("create")
            return {
                "campaign_id": "11111111-1111-1111-1111-111111111111",
                "expected_task_count": 2,
            }

        def publish_task(self, _campaign_id, identity, _payload):
            events.append(f"publish:{identity.split(':')[1]}")

        def finalize(self, _campaign_id, expected_count):
            assert expected_count == 2
            events.append("finalize")

    def simple_evaluate(**kwargs):
        mode = kwargs["model"].pool.manifest.wkv_mode
        events.append(f"evaluate:{mode}")
        return {
            "results": {"wikitext": {"word_perplexity,none": 12.5}},
            "configs": {"wikitext": {"output_type": "loglikelihood_rolling"}},
            "n-samples": {"wikitext": {"original": 1, "effective": 1}},
            "samples": {
                "wikitext": [
                    {
                        "doc_id": 0,
                        "doc": {"page": "text"},
                        "filter": "none",
                        "metrics": ["word_perplexity"],
                        "word_perplexity": 12.5,
                        "filtered_resps": [],
                    }
                ]
            },
            "versions": {"wikitext": 2},
        }

    def package_version(name):
        return "0.4.12" if name == "lm-eval" else "2.11.0"

    monkeypatch.setattr(evaluate, "VLLMHttpPool", Pool)
    monkeypatch.setattr(lm_eval, "simple_evaluate", simple_evaluate)
    monkeypatch.setattr(evaluate.importlib.metadata, "version", package_version)
    monkeypatch.setattr("helicopter_lighteval.publish.ScoreboardClient", Client)

    assert evaluate.run(
        config_path=config,
        env={
            "WEIGHT_PATH": str(weight_root),
            "HELICOPTER_SCOREBOARD_URL": "https://scoreboard.example",
            "HELICOPTER_SCOREBOARD_TOKEN": "secret",
        },
        dry_run=False,
    ) == 0

    assert events[:4] == [
        "scoreboard:lm-eval:0.4.12",
        "pool:fp16",
        "pool:fp32io16",
        "create",
    ]
    assert events.index("finalize") > events.index("publish:fp16")
    assert events.index("finalize") > events.index("publish:fp32io16")
