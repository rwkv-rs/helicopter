from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from lighteval.models.model_output import ModelResponse

from helicopter_cli import __main__ as helicopter_main
from helicopter_cli import (
    agent_format,
    agent_harness,
    commands,
    config,
    env,
    eval_run,
    function_calling,
    lighteval_export,
    lighteval_rwkv_skills_tasks,
    lighteval_tasks,
    performance,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "configs/example.toml"


def load_example_config() -> dict[str, object]:
    loaded, _ = config.load_config(ROOT, str(EXAMPLE_CONFIG))
    return loaded


def infer_args(**overrides: object) -> Namespace:
    values = {
        "model": "g1g-1.5b",
        "dry_run": True,
        "wkv_mode": None,
        "emb_device": None,
        "host": None,
        "port": None,
        "served_model_name": None,
        "tensor_parallel_size": None,
        "gpu_memory_utilization": None,
        "max_model_len": None,
        "max_num_seqs": None,
        "max_num_batched_tokens": None,
        "enable_auto_tool_choice": None,
        "vllm_env": None,
    }
    values.update(overrides)
    return Namespace(**values)


def takeoff_args(**overrides: object) -> Namespace:
    values = {
        "algorithm": "grpo",
        "model": "g1g-1.5b",
        "dataset": "gsm8k",
        "dry_run": True,
        "wkv_mode": None,
        "emb_device": None,
        "num_nodes": None,
        "num_devices": None,
        "override": None,
    }
    values.update(overrides)
    return Namespace(**values)


def lighteval_args(**overrides: object) -> Namespace:
    values = {
        "backend": "endpoint-litellm",
        "model": "g1g-1.5b",
        "tasks": "gsm8k",
        "model_args": None,
        "lighteval_model_name": None,
        "base_url": None,
        "provider": None,
        "api_key": None,
        "concurrent_requests": None,
        "max_model_length": None,
        "max_new_tokens": None,
        "max_samples": None,
        "output_dir": None,
        "dataset_loading_processes": None,
        "num_fewshot_seeds": None,
        "custom_tasks": None,
        "load_tasks_multilingual": None,
        "save_details": None,
        "push_to_hub": None,
        "public_run": None,
        "results_org": None,
        "job_id": None,
        "extra": None,
        "performance_output": None,
        "metrics_url": None,
        "scoreboard_task_id": None,
    }
    values.update(overrides)
    return Namespace(**values)


def lighteval_tasks_args(**overrides: object) -> Namespace:
    values = {
        "task_action": "list",
        "tasks": None,
        "custom_tasks": None,
        "load_tasks_multilingual": None,
        "num_samples": None,
        "show_config": None,
        "output": None,
        "format": "text",
        "contains": None,
        "limit": None,
        "include_supersets": None,
        "source": None,
        "source_format": "auto",
        "candidate_limit": 5,
    }
    values.update(overrides)
    return Namespace(**values)


def lighteval_export_args(**overrides: object) -> Namespace:
    values = {
        "details": ["results/lighteval/details/run"],
        "output": None,
        "format": "jsonl",
    }
    values.update(overrides)
    return Namespace(**values)


def command_options(command: list[str]) -> dict[str, str | bool]:
    options: dict[str, str | bool] = {}
    index = 0
    while index < len(command):
        item = command[index]
        if not item.startswith("--"):
            index += 1
            continue
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            options[item] = command[index + 1]
            index += 2
        else:
            options[item] = True
            index += 1
    return options


def hydra_pairs(plan: commands.CommandPlan) -> list[tuple[str, str]]:
    pairs = []
    for item in plan.command[3:]:
        if "=" in item:
            key, value = item.split("=", 1)
            pairs.append((key, value))
    return pairs


def hydra_map(plan: commands.CommandPlan) -> dict[str, str]:
    return dict(hydra_pairs(plan))


def hydra_values(plan: commands.CommandPlan, key: str) -> list[str]:
    return [value for pair_key, value in hydra_pairs(plan) if pair_key == key]


def build_takeoff_plan(
    loaded_config: dict[str, object],
    *,
    args: Namespace | None = None,
    loaded_env: dict[str, str] | None = None,
    venv_python: Path | None = None,
) -> commands.CommandPlan:
    if loaded_env is None:
        loaded_env = {"WEIGHT_PATH": "/weights/RWKV", "DATASETS_PATH": "/datasets"}
    if args is None:
        args = takeoff_args()
    if venv_python is None:
        venv_python = ROOT / ".venv/bin/python"
    original_exists = Path.exists
    with mock.patch.object(Path, "exists", autospec=True) as exists:
        exists.side_effect = lambda path: True if path == venv_python else original_exists(path)
        return commands.build_takeoff_plan(args, root=ROOT, env=loaded_env, config=loaded_config)


class ParserTests(unittest.TestCase):
    def test_agent_harness_parser_accepts_plan_surface(self) -> None:
        parser = helicopter_main.build_parser()
        args = parser.parse_args(
            [
                "eval",
                "agent-harness",
                "plan",
                "swe_bench_verified",
                "--model",
                "g1d-0.4b",
                "--output-dir",
                "tmp/agent",
            ]
        )

        self.assertEqual(args.eval_command, "agent-harness")
        self.assertEqual(args.agent_action, "plan")
        self.assertEqual(args.benchmark, "swe_bench_verified")
        self.assertEqual(args.model, "g1d-0.4b")
        self.assertEqual(args.output_dir, "tmp/agent")

    def test_agent_harness_parser_accepts_convert_surface(self) -> None:
        parser = helicopter_main.build_parser()
        args = parser.parse_args(
            [
                "eval",
                "agent-harness",
                "convert",
                "swe_bench_verified",
                "--input",
                "tmp/raw.jsonl",
                "--output",
                "tmp/predictions.jsonl",
                "--model",
                "g1d-0.4b",
            ]
        )

        self.assertEqual(args.agent_action, "convert")
        self.assertEqual(args.benchmark, "swe_bench_verified")
        self.assertEqual(args.input, "tmp/raw.jsonl")
        self.assertEqual(args.output, "tmp/predictions.jsonl")
        self.assertEqual(args.target, "auto")

    def test_function_calling_parser_keeps_small_lighteval_like_surface(self) -> None:
        parser = helicopter_main.build_parser()
        args = parser.parse_args(
            [
                "eval",
                "function-calling",
                "g1d-0.4b",
                "bfcl_v3",
                "--max-samples",
                "2",
                "--output-dir",
                "tmp/fc",
                "--scoreboard",
            ]
        )

        self.assertEqual(args.eval_command, "function-calling")
        self.assertEqual(args.model, "g1d-0.4b")
        self.assertEqual(args.tasks, "bfcl_v3")
        self.assertEqual(args.max_samples, 2)
        self.assertEqual(args.output_dir, "tmp/fc")
        self.assertTrue(args.scoreboard)
        self.assertFalse(hasattr(args, "wkv_mode"))
        self.assertFalse(hasattr(args, "tensor_parallel_size"))

    def test_function_calling_parser_rejects_vllm_detail_flags(self) -> None:
        parser = helicopter_main.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["eval", "function-calling", "g1d-0.4b", "bfcl_v3", "--wkv-mode", "fp16"])


class FunctionCallingTests(unittest.TestCase):
    def test_openai_tool_calls_are_normalized_and_scored(self) -> None:
        sample = function_calling.FunctionCallingSample(
            task_name="bfcl_exec_parallel",
            sample_id="unit",
            kind="bfcl",
            messages=[],
            tools=[],
            specific={
                "expected_calls_json": json.dumps(
                    [
                        {"name": "calc", "arguments": {"x": 1}},
                        {"name": "lookup", "arguments": {"key": "a"}},
                    ]
                )
            },
        )
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{\"key\":\"a\"}"},
                            },
                            {
                                "type": "function",
                                "function": {"name": "calc", "arguments": "{\"x\":1}"},
                            },
                        ]
                    }
                }
            ]
        }

        calls = function_calling.tool_calls_from_response(response)

        self.assertEqual(
            calls,
            [
                {"name": "lookup", "arguments": {"key": "a"}},
                {"name": "calc", "arguments": {"x": 1}},
            ],
        )
        self.assertEqual(function_calling.score_calls(sample, calls), 1.0)

    def test_function_calling_dry_run_does_not_load_samples(self) -> None:
        args = Namespace(
            model="g1d-0.4b",
            tasks="bfcl_v3",
            dry_run=True,
            base_url=None,
            output_dir=None,
            max_samples=2,
            no_server=False,
            keep_server=False,
            scoreboard=False,
        )
        with mock.patch.object(function_calling, "load_samples") as load_samples:
            with mock.patch("sys.stdout", new=io.StringIO()):
                exit_code = function_calling.run_function_calling_eval(
                    args,
                    root=ROOT,
                    env={},
                    config=load_example_config(),
                )

        self.assertEqual(exit_code, 0)
        load_samples.assert_not_called()

    def test_compact_response_message_keeps_content_and_finish_reason(self) -> None:
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "plain fallback",
                        "tool_calls": [],
                    },
                }
            ]
        }

        self.assertEqual(
            function_calling._compact_response_message(response),
            {
                "role": "assistant",
                "content": "plain fallback",
                "tool_calls": [],
                "finish_reason": "stop",
            },
        )


class DotenvTests(unittest.TestCase):
    def test_load_dotenv_supports_simple_export_and_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "PLAIN=value",
                        "export EXPORTED=enabled",
                        "QUOTED='space value'",
                        "# ignored",
                        "not-an-assignment",
                    ]
                )
            )

            self.assertEqual(
                env.load_dotenv(env_file),
                {
                    "PLAIN": "value",
                    "EXPORTED": "enabled",
                    "QUOTED": "space value",
                },
            )

    def test_load_env_keeps_command_scoped_environment_over_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env.local").write_text("WEIGHT_PATH=/from-file\n")

            with mock.patch.dict(os.environ, {"WEIGHT_PATH": "/from-env"}, clear=False):
                loaded_env, path = env.load_env(root, ".env.local")

            self.assertEqual(path, root / ".env.local")
            self.assertEqual(loaded_env["WEIGHT_PATH"], "/from-env")


class ConfigResolutionTests(unittest.TestCase):
    def test_default_config_uses_newest_local_toml_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / "configs/local"
            local.mkdir(parents=True)
            (root / "configs/example.toml").write_text("")
            (local / "202401010000.toml").write_text("")
            newest = local / "202606290720.toml"
            newest.write_text("")

            self.assertEqual(config.default_config_path(root), newest)

    def test_resolve_model_path_uses_weight_path_directory(self) -> None:
        loaded_config = load_example_config()
        loaded_env = {"WEIGHT_PATH": "/weights/RWKV"}

        model_path, model = config.resolve_model_path(loaded_config, "g1g-1.5b", root=ROOT, env=loaded_env)

        self.assertEqual(model["served_model_name"], "g1g-1.5b")
        self.assertEqual(
            model_path,
            Path("/weights/RWKV/rwkv7-g1g-1.5b-20260526-ctx8192.pth"),
        )


class CommandPlanTests(unittest.TestCase):
    @staticmethod
    def _browsecomp_encrypt(text: str, canary: str) -> str:
        payload = text.encode("utf-8")
        digest = hashlib.sha256(canary.encode("utf-8")).digest()
        key = (digest * ((len(payload) // len(digest)) + 1))[: len(payload)]
        return base64.b64encode(bytes(lhs ^ rhs for lhs, rhs in zip(payload, key))).decode("utf-8")

    def test_infer_plan_uses_vllm_rwkv_contract(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_infer_plan(
            infer_args(),
            root=ROOT,
            env={"WEIGHT_PATH": "/weights/RWKV"},
            config=loaded_config,
        )

        self.assertEqual(
            plan.command[:3],
            ["vllm", "serve", "/weights/RWKV/rwkv7-g1g-1.5b-20260526-ctx8192.pth"],
        )
        self.assertEqual(
            command_options(plan.command),
            {
                "--host": "0.0.0.0",
                "--port": "8000",
                "--tokenizer-mode": "rwkv",
                "--load-format": "auto",
                "--served-model-name": "g1g-1.5b",
                "--max-model-len": "8192",
            },
        )
        self.assertEqual(plan.cwd, ROOT)
        self.assertEqual(plan.shown_env, {})
        self.assertEqual({key for key in plan.env if key.startswith("VLLM_")}, set())

    def test_infer_plan_allows_explicit_vllm_env(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_infer_plan(
            infer_args(vllm_env=["VLLM_WSL2_ENABLE_PIN_MEMORY=1"]),
            root=ROOT,
            env={
                "WEIGHT_PATH": "/weights/RWKV",
                "VLLM_USE_V2_MODEL_RUNNER": "0",
            },
            config=loaded_config,
        )

        self.assertEqual(plan.shown_env["VLLM_WSL2_ENABLE_PIN_MEMORY"], "1")
        self.assertEqual(plan.env["VLLM_WSL2_ENABLE_PIN_MEMORY"], "1")
        self.assertNotIn("VLLM_USE_V2_MODEL_RUNNER", plan.env)

    def test_infer_plan_accepts_configured_vllm_env(self) -> None:
        loaded_config = load_example_config()
        loaded_config["infer"] = {
            **loaded_config["infer"],
            "vllm_env": {"VLLM_WSL2_ENABLE_PIN_MEMORY": 1},
        }

        plan = commands.build_infer_plan(
            infer_args(),
            root=ROOT,
            env={"WEIGHT_PATH": "/weights/RWKV"},
            config=loaded_config,
        )

        self.assertEqual(plan.shown_env["VLLM_WSL2_ENABLE_PIN_MEMORY"], "1")
        self.assertEqual(plan.env["VLLM_WSL2_ENABLE_PIN_MEMORY"], "1")

    def test_infer_plan_uses_model_specific_0_4b_runtime(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_infer_plan(
            infer_args(model="g1d-0.4b"),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        options = command_options(plan.command)
        self.assertEqual(plan.command[0], "/home/chase/GitHub/vllm-rwkv/.venv/bin/vllm")
        self.assertEqual(options["--served-model-name"], "g1d-0.4b")
        self.assertEqual(options["--gpu-memory-utilization"], "0.45")
        self.assertEqual(options["--max-model-len"], "8192")
        self.assertEqual(options["--max-num-seqs"], "8")
        self.assertEqual(options["--max-num-batched-tokens"], "8192")
        self.assertEqual(
            plan.shown_env,
            {
                "VLLM_RWKV7_EMB_DEVICE": "gpu",
                "VLLM_RWKV7_WKV_MODE": "fp16",
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
                "VLLM_USE_RAPID_SAMPLER": "0",
                "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
            },
        )

    def test_lighteval_plan_uses_official_litellm_endpoint(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_lighteval_plan(
            lighteval_args(max_samples=3),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        self.assertEqual(plan.command[1:5], ["-m", "lighteval", "endpoint", "litellm"])
        self.assertEqual(plan.command[6], "gsm8k")
        self.assertIn("model_name=openai/g1g-1.5b", plan.command[5])
        self.assertIn("provider=openai", plan.command[5])
        self.assertIn("base_url=http://127.0.0.1:8000/v1", plan.command[5])
        self.assertIn("max_model_length=8192", plan.command[5])
        self.assertIn("generation_parameters={max_new_tokens:512}", plan.command[5])
        options = command_options(plan.command)
        self.assertEqual(options["--output-dir"], str(ROOT / "results/lighteval"))
        self.assertEqual(options["--max-samples"], "3")
        self.assertEqual(options["--dataset-loading-processes"], "1")
        self.assertEqual(
            options["--custom-tasks"],
            str(ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py"),
        )
        self.assertTrue(options["--load-tasks-multilingual"])
        self.assertTrue(options["--save-details"])
        self.assertEqual(plan.env["OPENAI_API_KEY"], "EMPTY")

    def test_lighteval_plan_cli_max_new_tokens_overrides_config(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_lighteval_plan(
            lighteval_args(max_new_tokens=128),
            root=ROOT,
            env={"HELICOPTER_EVAL_MAX_NEW_TOKENS": "256"},
            config=loaded_config,
        )

        self.assertIn("generation_parameters={max_new_tokens:128}", plan.command[5])

    def test_lighteval_plan_keeps_api_key_out_of_command(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_lighteval_plan(
            lighteval_args(base_url="https://example.test/v1", api_key="secret-token"),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        self.assertNotIn("secret-token", " ".join(plan.command))
        self.assertEqual(plan.env["OPENAI_API_KEY"], "secret-token")

    def test_lighteval_performance_metrics_url_uses_endpoint_root(self) -> None:
        self.assertEqual(
            performance.derive_metrics_url("http://127.0.0.1:8000/v1"),
            "http://127.0.0.1:8000/metrics",
        )
        self.assertEqual(
            performance.derive_metrics_url("http://127.0.0.1:8000/custom/v1"),
            "http://127.0.0.1:8000/custom/metrics",
        )

    def test_lighteval_performance_reads_equals_output_dir_option(self) -> None:
        self.assertEqual(
            performance.output_dir_from_command(["lighteval", "--output-dir=results/run"]),
            Path("results/run"),
        )

    def test_lighteval_performance_report_derives_run_rates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            result_dir = output_dir / "results/openai/model"
            result_dir.mkdir(parents=True)
            result_file = result_dir / "results_2026-07-03T00-00-00.json"
            result_file.write_text(
                json.dumps(
                    {
                        "config_general": {
                            "model_name": "openai/g1g-7.2b",
                            "max_samples": 2,
                            "total_evaluation_time_secondes": "10.0",
                        },
                        "results": {
                            "gsm8k|0": {"acc": 0.5},
                            "all": {"acc": 0.5},
                        },
                    }
                )
            )

            report = performance.build_performance_report(
                command=["python", "-m", "lighteval"],
                exit_code=0,
                output_dir=output_dir,
                started_at_epoch=1.0,
                ended_at_epoch=11.0,
                metrics_url="http://127.0.0.1:8000/metrics",
                metrics_before={
                    "vllm:prompt_tokens_total": 100,
                    "vllm:generation_tokens_total": 10,
                },
                metrics_after={
                    "vllm:prompt_tokens_total": 130,
                    "vllm:generation_tokens_total": 30,
                },
            )
            self.assertEqual(performance.extract_lighteval_score_metrics(report["source_files"]["results"]), {"acc": 0.5})

        self.assertEqual(report["samples_completed"], 2)
        self.assertEqual(report["jobs_completed"], 1)
        self.assertEqual(report["models_completed"], 1)
        self.assertEqual(report["benchmarks_completed"], 1)
        self.assertEqual(report["prompt_tokens"], 30)
        self.assertEqual(report["generation_tokens"], 20)
        self.assertEqual(report["total_tokens"], 50)
        self.assertEqual(report["samples_per_hour"], 720.0)
        self.assertEqual(report["jobs_per_hour"], 360.0)
        self.assertEqual(report["models_per_day"], 8640.0)
        self.assertEqual(report["benchmarks_per_day"], 8640.0)
        self.assertEqual(report["tokens_per_second"], 5.0)
        self.assertEqual(report["completed_runs/day"], 8640.0)

    def test_lighteval_performance_metrics_embeds_scoreboard_subset(self) -> None:
        report = {
            "elapsed_seconds": 5.0,
            "samples_completed": 1,
            "jobs_completed": 1,
            "models_completed": 1,
            "benchmarks_completed": 1,
            "completed_runs": 1,
            "tokens_per_second": None,
            "samples_per_hour": 720.0,
            "jobs_per_hour": 720.0,
            "models_per_day": 17280.0,
            "benchmarks_per_day": 17280.0,
            "completed_runs/day": 17280.0,
        }

        embedded = performance.performance_metrics_from_report(report)

        self.assertEqual(embedded["samples_per_hour"], 720.0)
        self.assertEqual(embedded["tokens_per_second"], None)
        self.assertNotIn("command", embedded)

    def test_lighteval_tasks_plan_lists_registry(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_lighteval_tasks_plan(
            lighteval_tasks_args(load_tasks_multilingual=True),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        self.assertEqual(plan.command[1:5], ["-m", "lighteval", "tasks", "list"])
        self.assertIn("--load-tasks-multilingual", plan.command)
        self.assertEqual(
            command_options(plan.command)["--custom-tasks"],
            str(ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py"),
        )

    def test_lighteval_tasks_show_config_uses_local_compat_wrapper(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_lighteval_tasks_plan(
            lighteval_tasks_args(task_action="inspect", tasks="gsm8k", show_config=True, num_samples=1),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        self.assertEqual(plan.command[1:4], ["-m", "helicopter_cli.lighteval_tasks", "inspect"])
        self.assertIn("--show-config", plan.command)
        self.assertEqual(command_options(plan.command)["--num-samples"], "1")

    def test_lighteval_tasks_export_uses_local_registry_wrapper(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_lighteval_tasks_plan(
            lighteval_tasks_args(
                task_action="export",
                load_tasks_multilingual=True,
                output="tmp/tasks.txt",
                contains=["gsm"],
                limit=2,
                include_supersets=True,
            ),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        self.assertEqual(plan.command[1:4], ["-m", "helicopter_cli.lighteval_tasks", "export"])
        self.assertIn("--load-multilingual", plan.command)
        self.assertIn("--include-supersets", plan.command)
        options = command_options(plan.command)
        self.assertEqual(options["--output"], "tmp/tasks.txt")
        self.assertEqual(options["--contains"], "gsm")
        self.assertEqual(options["--limit"], "2")

    def test_lighteval_tasks_export_filters_registry_rows(self) -> None:
        class FakeRegistry:
            _task_registry = {"gsm8k": object(), "mmlu": object(), "tiny:gsm8k": object()}
            _task_superset_dict = {"tiny": ("tiny:gsm8k",)}

        with mock.patch.object(lighteval_tasks, "load_registry", return_value=FakeRegistry()):
            rows = lighteval_tasks.selected_task_rows(
                Namespace(
                    custom_tasks=None,
                    load_multilingual=False,
                    contains=["gsm"],
                    limit=None,
                    include_supersets=True,
                )
            )

        self.assertEqual(rows, [("task", "gsm8k"), ("task", "tiny:gsm8k")])
        self.assertEqual(
            lighteval_tasks.format_export(rows, "jsonl"),
            '{"kind": "task", "task": "gsm8k"}\n{"kind": "task", "task": "tiny:gsm8k"}\n',
        )

    def test_lighteval_tasks_coverage_uses_local_registry_wrapper(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_lighteval_tasks_plan(
            lighteval_tasks_args(
                task_action="coverage",
                source="benchmarks.txt",
                source_format="text",
                output="tmp/coverage.jsonl",
                format="jsonl",
                candidate_limit=7,
            ),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        self.assertEqual(plan.command[1:4], ["-m", "helicopter_cli.lighteval_tasks", "coverage"])
        options = command_options(plan.command)
        self.assertEqual(options["--source"], str(ROOT / "benchmarks.txt"))
        self.assertEqual(options["--source-format"], "text")
        self.assertEqual(options["--output"], "tmp/coverage.jsonl")
        self.assertEqual(options["--format"], "jsonl")
        self.assertEqual(options["--candidate-limit"], "7")

    def test_lighteval_tasks_judges_uses_local_registry_wrapper(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_lighteval_tasks_plan(
            lighteval_tasks_args(
                task_action="judges",
                tasks="aime24",
                load_tasks_multilingual=True,
                output="tmp/judges.jsonl",
                format="jsonl",
                contains=["aime"],
                limit=1,
            ),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        self.assertEqual(plan.command[1:4], ["-m", "helicopter_cli.lighteval_tasks", "judges"])
        self.assertIn("--load-multilingual", plan.command)
        options = command_options(plan.command)
        self.assertEqual(options["--custom-tasks"], str(ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py"))
        self.assertEqual(options["--output"], "tmp/judges.jsonl")
        self.assertEqual(options["--format"], "jsonl")
        self.assertEqual(options["--contains"], "aime")
        self.assertEqual(options["--limit"], "1")
        self.assertIn("aime24", plan.command)

    def test_lighteval_tasks_coverage_rejects_filter_flags(self) -> None:
        loaded_config = load_example_config()
        for overrides in (
            {"contains": ["gsm"]},
            {"limit": 5},
            {"include_supersets": True},
        ):
            with self.assertRaises(SystemExit):
                commands.build_lighteval_tasks_plan(
                    lighteval_tasks_args(task_action="coverage", source="benchmarks.txt", **overrides),
                    root=ROOT,
                    env={},
                    config=loaded_config,
                )

    def test_lighteval_tasks_export_rejects_unsupported_format(self) -> None:
        loaded_config = load_example_config()
        for fmt in ("summary", "tasks"):
            with self.assertRaises(SystemExit):
                commands.build_lighteval_tasks_plan(
                    lighteval_tasks_args(task_action="export", format=fmt),
                    root=ROOT,
                    env={},
                    config=loaded_config,
                )

    def test_lighteval_tasks_judges_rejects_tasks_format(self) -> None:
        loaded_config = load_example_config()
        with self.assertRaises(SystemExit):
            commands.build_lighteval_tasks_plan(
                lighteval_tasks_args(task_action="judges", tasks="aime24", format="tasks"),
                root=ROOT,
                env={},
                config=loaded_config,
            )

    def test_lighteval_tasks_judges_rejects_include_supersets(self) -> None:
        loaded_config = load_example_config()
        with self.assertRaises(SystemExit):
            commands.build_lighteval_tasks_plan(
                lighteval_tasks_args(task_action="judges", tasks="aime24", include_supersets=True),
                root=ROOT,
                env={},
                config=loaded_config,
            )

    def test_lighteval_tasks_list_rejects_filter_and_output_flags(self) -> None:
        loaded_config = load_example_config()
        for overrides in ({"output": "tmp/tasks.txt"}, {"contains": ["gsm"]}, {"limit": 3}):
            with self.assertRaises(SystemExit):
                commands.build_lighteval_tasks_plan(
                    lighteval_tasks_args(task_action="list", **overrides),
                    root=ROOT,
                    env={},
                    config=loaded_config,
                )

    def test_normalize_openai_base_url_appends_v1_to_bare_host(self) -> None:
        self.assertEqual(
            commands.normalize_openai_base_url("http://host:8000"),
            "http://host:8000/v1",
        )
        # Existing path (or /v1) is preserved, only trailing slashes trimmed.
        self.assertEqual(
            commands.normalize_openai_base_url("http://host:8000/v1/"),
            "http://host:8000/v1",
        )
        self.assertEqual(
            commands.normalize_openai_base_url("http://host:8000/custom"),
            "http://host:8000/custom",
        )

    def test_lighteval_tasks_coverage_resolves_registry_rows(self) -> None:
        mmmlu_targets = lighteval_tasks.OFFICIAL_LIGHTEVAL_ALIASES["mmmlu"]

        class FakeRegistry:
            _task_registry = {
                "gpqa:diamond": object(),
                "gsm8k": object(),
                "ifbench_multiturn": object(),
                "ifbench_test": object(),
                "supergpqa": object(),
                "tiny:gsm8k": object(),
            }
            _task_superset_dict = {
                "lcb": ("lcb:codegeneration",),
                "mmlu": ("mmlu:abstract_algebra",),
                **{target: (f"{target}:abstract_algebra",) for target in mmmlu_targets},
            }

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "benchmarks.txt"
            source.write_text(
                "gsm8k,maths\n"
                "gpqa_diamond,knowledge\n"
                "mmlu,knowledge\n"
                "mmmlu,knowledge\n"
                "supergpqa,knowledge\n"
                "ifbench,instruction_following\n"
                "livecodebench,coding\n"
                "missing_one,maths\n"
            )
            with mock.patch.object(lighteval_tasks, "load_registry", return_value=FakeRegistry()):
                rows = lighteval_tasks.coverage_rows(
                    Namespace(
                        custom_tasks=None,
                        load_multilingual=False,
                        source=str(source),
                        source_format="text",
                        candidate_limit=3,
                    )
                )

        self.assertEqual(rows[0].status, "exact_task")
        self.assertEqual(rows[1].status, "normalized_task")
        self.assertEqual(rows[1].targets, ("gpqa:diamond",))
        self.assertEqual(rows[2].status, "exact_superset")
        self.assertEqual(rows[3].status, "alias_superset_list")
        self.assertEqual(rows[3].targets, mmmlu_targets)
        self.assertEqual(rows[4].status, "exact_task")
        self.assertEqual(rows[5].status, "alias_task_list")
        self.assertEqual(rows[5].targets, ("ifbench_test", "ifbench_multiturn"))
        self.assertEqual(rows[6].status, "alias_superset")
        self.assertEqual(rows[6].targets, ("lcb",))
        self.assertEqual(rows[7].status, "missing")
        self.assertIn("direct\t7\n", lighteval_tasks.format_coverage(rows, "summary"))
        self.assertIn("not_direct\t1\n", lighteval_tasks.format_coverage(rows, "summary"))
        self.assertEqual(
            lighteval_tasks.format_coverage(rows, "tasks"),
            "gsm8k\n"
            "gpqa:diamond\n"
            "mmlu\n"
            + "".join(f"{target}\n" for target in mmmlu_targets)
            + "supergpqa\n"
            "ifbench_test\n"
            "ifbench_multiturn\n"
            "lcb\n",
        )

    def test_lighteval_tasks_json_source_preserves_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "agent_benchmarks.json"
            source.write_text(
                json.dumps(
                    {
                        "benchmarks": [
                            {
                                "name": "swe_bench_verified",
                                "field": "agent",
                                "display_name": "SWE-bench Verified",
                                "run_status": "local_lighteval_proxy_available",
                                "priority": "primary",
                            }
                        ],
                        "excluded": [
                            {"name": "gpqa_diamond", "field": "knowledge"},
                        ],
                    }
                )
            )

            rows = lighteval_tasks.load_source_benchmarks(str(source), "auto")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].name, "swe_bench_verified")
        self.assertEqual(rows[0].field, "agent")
        self.assertEqual(rows[0].metadata["run_status"], "local_lighteval_proxy_available")
        self.assertEqual(rows[0].metadata["priority"], "primary")

        coverage = [
            lighteval_tasks.CoverageRow(
                source=rows[0].name,
                field=rows[0].field,
                status="missing",
                target_kind=None,
                targets=(),
                candidates=(),
                metadata=rows[0].metadata,
            )
        ]
        json_row = json.loads(lighteval_tasks.format_coverage(coverage, "jsonl"))
        self.assertEqual(json_row["metadata"]["display_name"], "SWE-bench Verified")
        self.assertIn("local_lighteval_proxy_available", lighteval_tasks.format_coverage(coverage, "text"))

    def test_agent_benchmark_source_excludes_non_agent_benchmarks(self) -> None:
        raw_source = json.loads((ROOT / "benchmarks/agent_benchmarks.json").read_text())
        rows = lighteval_tasks.load_source_benchmarks(
            str(ROOT / "benchmarks/agent_benchmarks.json"),
            "auto",
        )
        names = {row.name for row in rows}
        pipelines = {
            pipeline["name"]: set(pipeline["benchmarks"])
            for pipeline in raw_source["pipelines"]
        }
        row_pipeline_names = {row.metadata["pipeline"] for row in rows}
        benchmark_names = {benchmark["name"] for benchmark in raw_source["benchmarks"]}
        excluded_names = {benchmark["name"] for benchmark in raw_source["excluded"]}

        self.assertIn("swe_bench_verified", names)
        self.assertIn("terminal_bench_2_1", names)
        self.assertIn("mcp_atlas", names)
        self.assertIn("hle_with_tools", names)
        self.assertIn("hy_euler_pro", names)
        self.assertNotIn("gpqa_diamond", names)
        self.assertNotIn("hle_no_tools", names)
        self.assertNotIn("cl_bench", names)
        self.assertNotIn("hy_math", names)
        self.assertTrue(all(row.field == "agent" for row in rows))
        self.assertEqual(
            set(pipelines),
            {
                "coding_agent",
                "search_agent",
                "tool_mcp_agent",
                "office_enterprise_workflow_agent",
                "stem_tool_agent",
            },
        )
        self.assertTrue(all(row.metadata["pipeline"] in pipelines for row in rows))
        self.assertEqual(row_pipeline_names, set(pipelines))
        self.assertEqual(set().union(*pipelines.values()), benchmark_names)
        self.assertTrue(benchmark_names.isdisjoint(excluded_names))
        self.assertEqual(
            pipelines["coding_agent"],
            {
                "swe_bench_verified",
                "swe_bench_multilingual",
                "swe_bench_pro",
                "terminal_bench_2_1",
                "nl2repo",
                "deepswe",
                "hy_backend_2_0",
                "hy_swe_max",
            },
        )
        self.assertEqual(pipelines["search_agent"], {"browsecomp", "wide_search", "deepsearchqa"})
        self.assertEqual(
            pipelines["tool_mcp_agent"],
            {"mcp_atlas", "toolathlon", "skillsbench", "hy_skillsworld"},
        )
        self.assertEqual(
            pipelines["office_enterprise_workflow_agent"],
            {
                "apex_agents",
                "claweval",
                "wildclawbench",
                "hy_companybench",
                "prodbench",
                "hy_finmodelbench",
                "e_bench",
            },
        )
        self.assertEqual(pipelines["stem_tool_agent"], {"hle_with_tools", "hy_euler_pro"})

    def test_agent_harness_source_binds_every_benchmark_to_profile(self) -> None:
        source = agent_harness.load_agent_harness_source(ROOT, None)
        names = {row.name for row in source.benchmarks}

        self.assertIn("swe_bench_verified", names)
        self.assertIn("terminal_bench_2_1", names)
        self.assertIn("mcp_atlas", names)
        self.assertIn("hle_with_tools", names)
        self.assertIn("swebench_official_docker", source.profiles)
        self.assertIn("terminal_bench_official_docker", source.profiles)
        self.assertTrue(all(row.harness_profile in source.profiles for row in source.benchmarks))
        self.assertEqual(
            source.profiles["swebench_official_docker"].entrypoint,
            "python -m swebench.harness.run_evaluation",
        )
        self.assertEqual(source.profiles["terminal_bench_official_docker"].sandbox, "docker")

    def test_agent_harness_preflight_reports_missing_required_tool(self) -> None:
        source = agent_harness.load_agent_harness_source(ROOT, None)

        with mock.patch.object(agent_harness.shutil, "which", return_value=None):
            rows = agent_harness.preflight_rows(
                source,
                pipeline=None,
                benchmark="swe_bench_verified",
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, "blocked")
        self.assertEqual(rows[0].missing_tools, ("docker",))

    def test_agent_harness_preflight_distinguishes_local_proxy_from_official_harness(self) -> None:
        source = agent_harness.load_agent_harness_source(ROOT, None)

        with mock.patch.object(agent_harness.shutil, "which", return_value="/usr/bin/docker"):
            rows = agent_harness.preflight_rows(
                source,
                pipeline=None,
                benchmark="swe_bench_verified",
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, "local_proxy_available_official_harness_required")

    def test_agent_harness_swebench_plan_uses_official_docker_harness(self) -> None:
        source = agent_harness.load_agent_harness_source(ROOT, None)
        args = Namespace(
            benchmark="swe_bench_verified",
            model="g1d-0.4b",
            base_url="http://127.0.0.1:8000/v1",
            output_dir="tmp/agent",
            n_concurrent=2,
            run_id="unit",
        )

        plan = agent_harness.plan_for_benchmark(
            source,
            root=ROOT,
            env={},
            config=load_example_config(),
            args=args,
        )

        self.assertEqual(plan["benchmark"]["name"], "swe_bench_verified")
        self.assertEqual(plan["harness_profile"]["kind"], "swebench")
        self.assertEqual(plan["steps"][0]["schema"]["model_patch"], "unified diff patch generated by the coding agent")
        self.assertEqual(
            plan["steps"][1]["command"][:3],
            ["python", "-m", "swebench.harness.run_evaluation"],
        )
        self.assertIn("SWE-bench/SWE-bench_Verified", plan["steps"][1]["command"])
        self.assertIn("--predictions_path", plan["steps"][1]["command"])
        self.assertIn("--max_workers", plan["steps"][1]["command"])
        self.assertIn("2", plan["steps"][1]["command"])

    def test_agent_format_extracts_swebench_predictions_from_rwkv_response(self) -> None:
        patch = "--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-old\n+new\n"
        records = [
            {
                "sample_id": "repo__project-1",
                "response_message": {
                    "role": "assistant",
                    "content": f"Here is the fix:\n```diff\n{patch}```",
                    "finish_reason": "stop",
                },
            }
        ]

        rows, errors = agent_format.swebench_prediction_rows(
            records,
            model="g1d-0.4b",
        )

        self.assertEqual(errors, [])
        self.assertEqual(
            rows,
            [
                {
                    "instance_id": "repo__project-1",
                    "model_name_or_path": "g1d-0.4b",
                    "model_patch": patch,
                }
            ],
        )

    def test_agent_format_reports_missing_swebench_patch(self) -> None:
        rows, errors = agent_format.swebench_prediction_rows(
            [{"instance_id": "repo__project-1", "output": "I cannot fix this."}],
            model="g1d-0.4b",
        )

        self.assertEqual(rows, [])
        self.assertEqual(errors[0].reason, "missing unified diff patch")
        self.assertIn("repo__project-1", agent_format.conversion_errors_text(errors))

    def test_agent_harness_convert_writes_swebench_predictions_jsonl(self) -> None:
        source = agent_harness.load_agent_harness_source(ROOT, None)
        patch = "--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-old\n+new\n"
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "rwkv.jsonl"
            output_path = Path(tmp) / "predictions.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "instance_id": "repo__project-1",
                        "output": f"<patch>\n{patch}</patch>",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = Namespace(
                benchmark="swe_bench_verified",
                input=str(input_path),
                output=str(output_path),
                model="g1d-0.4b",
                target="auto",
                allow_empty_patch=False,
                allow_invalid=False,
            )

            result = agent_harness.convert_agent_outputs(source, root=ROOT, args=args)

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(result["target"], "swebench-predictions")
        self.assertEqual(result["written"], 1)
        self.assertEqual(rows[0]["instance_id"], "repo__project-1")
        self.assertEqual(rows[0]["model_patch"], patch)

    def test_agent_format_reads_multirow_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"instance_id": "repo__project-1", "output": "first"}),
                        json.dumps({"instance_id": "repo__project-2", "output": "second"}),
                    ]
                ),
                encoding="utf-8",
            )

            records = agent_format.read_json_records(path)

        self.assertEqual([record["instance_id"] for record in records], ["repo__project-1", "repo__project-2"])

    def test_agent_format_intermediate_rows_keep_patch_artifact_and_metadata(self) -> None:
        patch = "diff --git a/example.py b/example.py\n--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-old\n+new\n"

        rows, errors = agent_format.canonical_intermediate_rows(
            [
                {
                    "task_id": "repo__project-1",
                    "model": "rwkv",
                    "output": patch,
                    "score": 0.0,
                }
            ],
            benchmark="swe_bench_verified",
            model="g1d-0.4b",
        )

        self.assertEqual(errors, [])
        self.assertEqual(rows[0]["format"], agent_format.INTERMEDIATE_FORMAT)
        self.assertEqual(rows[0]["sample_id"], "repo__project-1")
        self.assertEqual(rows[0]["artifacts"]["patch"], patch)
        self.assertEqual(rows[0]["metadata"]["score"], 0.0)

    def test_scoreboard_domain_classifies_agent_scope_narrowly(self) -> None:
        scoreboard_path = ROOT / "src/scoreboard-server"
        if str(scoreboard_path) not in sys.path:
            sys.path.insert(0, str(scoreboard_path))
        from scoreboard_server.cores.normalize import domain_for

        self.assertEqual(domain_for("swe_bench_verified"), "agent")
        self.assertEqual(domain_for("browsecomp"), "agent")
        self.assertEqual(domain_for("mcp_bench_multi_2server"), "agent")
        self.assertEqual(domain_for("tau3_bench_mock"), "agent")
        self.assertEqual(domain_for("bfcl_v3"), "function_call")
        self.assertEqual(domain_for("toolalpaca_eval_real"), "function_call")
        self.assertEqual(domain_for("gpqa_diamond"), "knowledge")
        self.assertEqual(domain_for("cl_bench"), "knowledge")
        self.assertEqual(domain_for("matharena_apex"), "math")

    def test_lighteval_tasks_judges_classifies_builtin_and_custom_metrics(self) -> None:
        class UpstreamAvgAtN:
            pass

        UpstreamAvgAtN.__module__ = "lighteval.metrics.metrics_sample"

        class FakeRegistry:
            _task_registry = {
                "aime24": SimpleNamespace(
                    metrics=(
                        SimpleNamespace(metric_name="pass@k:k=1", sample_level_fn=UpstreamAvgAtN()),
                        SimpleNamespace(metric_name="avg@n:n=1", sample_level_fn=UpstreamAvgAtN()),
                    )
                ),
                "tau3_bench_mock": SimpleNamespace(
                    metrics=(
                        lighteval_rwkv_skills_tasks.rwkv_tau_bench_static_plan_f1,
                        lighteval_rwkv_skills_tasks.rwkv_tau_bench_response_nonempty,
                    )
                ),
            }
            _task_superset_dict = {"demo_family": ("aime24", "tau3_bench_mock")}

        with mock.patch.object(lighteval_tasks, "load_registry", return_value=FakeRegistry()):
            rows = lighteval_tasks.judge_rows(
                Namespace(
                    custom_tasks=None,
                    load_multilingual=False,
                    tasks="demo_family",
                    contains=None,
                    limit=None,
                )
            )

        by_metric = {row.metric: row for row in rows}
        self.assertEqual(by_metric["avg@n:n=1"].source, "lighteval_builtin")
        self.assertEqual(by_metric["avg@n:n=1"].status, "ready")
        self.assertEqual(by_metric["avg@n:n=1"].judge_type, "avg_at_n")
        self.assertEqual(by_metric["tau_bench_static_plan_f1"].source, "helicopter_custom")
        self.assertEqual(by_metric["tau_bench_static_plan_f1"].status, "proxy")
        self.assertEqual(by_metric["tau_bench_response_nonempty"].status, "sanity")
        summary = lighteval_tasks.format_judges(rows, "summary")
        self.assertIn("tasks\t2\n", summary)
        self.assertIn("status\tready\t2\n", summary)
        self.assertIn("status\tproxy\t1\n", summary)
        self.assertIn("status\tsanity\t1\n", summary)

    def test_supergpqa_custom_task_prompt_keeps_all_options(self) -> None:
        doc = lighteval_rwkv_skills_tasks.supergpqa_prompt(
            {
                "question": "Pick the third option.",
                "options": ["zero", "one", "two", "three", "four"],
                "answer_letter": "C",
            },
            "supergpqa",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.gold_index, 2)
        self.assertEqual(doc.choices, [" A", " B", " C", " D", " E"])
        self.assertIn("E. four", doc.query)

    def test_rwkv_skills_custom_tasks_include_direct_math_and_code_ids(self) -> None:
        registered_tasks = {task.name for task in lighteval_rwkv_skills_tasks.TASKS_TABLE}
        self.assertTrue(
            {
                "algebra222",
                "amc23",
                "answer_judge",
                "arena_hard_v2",
                "agentbench_db",
                "agentbench_kg",
                "beyond_aime",
                "brumo25",
                "browsecomp",
                "browsecomp_plus",
                "browsecomp_zh",
                "college_math",
                "comp_math_24_25",
                "gaokao2023en",
                "hendrycks_math",
                "hmmt_feb25",
                "human_eval",
                "human_eval_cn",
                "human_eval_fix",
                "human_eval_plus",
                "longbench",
                "longbench_qa",
                "longcodeqa",
                "mcp_bench",
                "mcp_bench_multi_2server",
                "mcp_bench_multi_3server",
                "mcp_bench_single",
                "math_odyssey",
                "mawps",
                "mbpp_plus",
                "minerva_math",
                "omni_math",
                "polymath",
                "svamp",
                "swe_bench",
                "swe_bench_lite",
                "swe_bench_verified",
                "swe_bench_lite_oracle",
                "swe_bench_lite_bm25_13k",
                "supergpqa",
                "tau_bench_retail",
                "tau_bench_airline",
                "tau_bench_telecom",
                "tau2_bench_retail",
                "tau2_bench_airline",
                "tau2_bench_telecom",
                "tau3_bench_retail",
                "tau3_bench_airline",
                "tau3_bench_telecom",
                "tau3_bench_banking_knowledge",
                "tau3_bench_mock",
                "tau3_bench_mock_long_context",
                "wmt24pp",
            }.issubset(registered_tasks)
        )
        self.assertFalse(
            {
                "apibank_l1",
                "apibank_l2",
                "apibank_level1",
                "apibank_level2",
                "bfcl_multiple",
                "bfcl_exec_multiple",
                "bfcl_exec_multiple_ast",
                "bfcl_exec_parallel",
                "bfcl_exec_parallel_multiple",
                "bfcl_simple_python",
                "bfcl_exec_simple",
                "bfcl_exec_simple_ast",
                "bfcl_v3",
                "complexfuncbench_official",
                "complexfuncbench_subset",
                "longbench_qa_balanced",
                "mbpp",
                "toolalpaca_eval_real",
                "toolalpaca_eval_simulated",
            }
            & registered_tasks
        )

    def test_polymath_task_aggregates_all_languages_and_levels(self) -> None:
        self.assertEqual(len(lighteval_rwkv_skills_tasks.POLYMATH_LANGUAGES), 18)
        self.assertEqual(len(lighteval_rwkv_skills_tasks.POLYMATH_LEVELS), 4)
        self.assertEqual(len(lighteval_rwkv_skills_tasks.POLYMATH_URLS), 72)
        self.assertIn("zh/low.parquet", lighteval_rwkv_skills_tasks.POLYMATH_URLS[-1])

    def test_comp_math_static_data_file_is_packaged(self) -> None:
        self.assertTrue(Path(lighteval_rwkv_skills_tasks.COMP_MATH_24_25_PATH).is_file())

    def test_mcpbench_static_data_files_are_packaged(self) -> None:
        expected_counts = {
            "mcp_bench": 104,
            "mcp_bench_single": 56,
            "mcp_bench_multi_2server": 30,
            "mcp_bench_multi_3server": 18,
        }
        for name, expected_count in expected_counts.items():
            path = Path(lighteval_rwkv_skills_tasks.MCP_BENCH_PATHS[name])
            self.assertTrue(path.is_file(), name)
            with path.open(encoding="utf-8") as fh:
                self.assertEqual(sum(1 for _line in fh), expected_count, name)

    def test_agentbench_static_data_files_are_packaged(self) -> None:
        expected_counts = {"agentbench_db": 300, "agentbench_kg": 150}
        for name, expected_count in expected_counts.items():
            path = Path(lighteval_rwkv_skills_tasks.AGENTBENCH_PATHS[name])
            self.assertTrue(path.is_file(), name)
            with path.open(encoding="utf-8") as fh:
                self.assertEqual(sum(1 for _line in fh), expected_count, name)

    def test_taubench_static_data_files_are_packaged(self) -> None:
        expected_counts = {
            "tau_bench_retail": 40,
            "tau_bench_airline": 20,
            "tau_bench_telecom": 40,
            "tau2_bench_retail": 114,
            "tau2_bench_airline": 50,
            "tau2_bench_telecom": 114,
            "tau3_bench_retail": 114,
            "tau3_bench_airline": 50,
            "tau3_bench_telecom": 114,
            "tau3_bench_banking_knowledge": 97,
            "tau3_bench_mock": 3,
            "tau3_bench_mock_long_context": 2,
        }
        self.assertEqual(set(lighteval_rwkv_skills_tasks.TAU_BENCH_PATHS), set(expected_counts))
        for name, expected_count in expected_counts.items():
            path = Path(lighteval_rwkv_skills_tasks.TAU_BENCH_PATHS[name])
            self.assertTrue(path.is_file(), name)
            with path.open(encoding="utf-8") as fh:
                self.assertEqual(sum(1 for _line in fh), expected_count, name)

    def test_taubench_mock_update_row_exposes_update_tool(self) -> None:
        path = Path(lighteval_rwkv_skills_tasks.TAU_BENCH_PATHS["tau3_bench_mock"])
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        update_row = next(row for row in rows if row["task_id"] == "update_task_with_history_and_env_assertions")
        self.assertIn("update_task_status", update_row["available_action_names"])
        self.assertEqual(update_row["reference_action_names"], ["update_task_status"])
        self.assertIn("update_task_status", update_row["reference_plan"])

    def test_wmt24pp_task_uses_default_target_languages(self) -> None:
        self.assertEqual(lighteval_rwkv_skills_tasks.WMT24PP_TARGET_LANGUAGES, ("de_DE", "es_MX", "fr_FR", "it_IT", "ja_JP"))
        self.assertEqual(len(lighteval_rwkv_skills_tasks.WMT24PP_URLS), 5)
        self.assertIn("en-ja_JP.jsonl", lighteval_rwkv_skills_tasks.WMT24PP_URLS[-1])

    def test_wmt24pp_prompt_builds_translation_doc(self) -> None:
        doc = lighteval_rwkv_skills_tasks.wmt24pp_prompt(
            {
                "lp": "en-de_DE",
                "source": "Good morning.",
                "target": "Guten Morgen.",
            },
            "wmt24pp",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.choices, ["Guten Morgen."])
        self.assertIn("from English to German", doc.query)
        self.assertTrue(doc.query.endswith("German:"))

    def test_longbench_prompt_builds_references_and_truncates_context(self) -> None:
        doc = lighteval_rwkv_skills_tasks.longbench_prompt(
            {
                "dataset": "triviaqa",
                "input": "What is the answer?",
                "context": "A" * (lighteval_rwkv_skills_tasks.LONG_CONTEXT_PROMPT_MAX_CHARS + 1000),
                "answers": ["forty two", "42"],
            },
            "longbench_qa",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.specific["references"], ["forty two", "42"])
        self.assertIn("middle truncated", doc.query)
        self.assertLess(len(doc.query), lighteval_rwkv_skills_tasks.LONG_CONTEXT_PROMPT_MAX_CHARS + 1000)

    def test_longbench_metric_scores_json_answer(self) -> None:
        doc = lighteval_rwkv_skills_tasks.longbench_prompt(
            {
                "input": "Name it.",
                "context": "The answer is Ozalj.",
                "answers": ["Ozalj"],
            },
            "longbench_qa",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        exact = lighteval_rwkv_skills_tasks.LongBenchExactMatch()
        f1 = lighteval_rwkv_skills_tasks.LongBenchF1()
        response = ModelResponse(text=['{"answer":"Ozalj"}'])
        self.assertEqual(exact.compute(response, doc), 1.0)
        self.assertEqual(f1.compute(response, doc), 1.0)

    def test_longcodeqa_prompt_and_metric_accept_option_letter(self) -> None:
        doc = lighteval_rwkv_skills_tasks.longcodeqa_prompt(
            {
                "prompt": "Repository: Repository:\nexample\nQuestion:\nA) no\nB) yes\n",
                "question": "Question:\nA) no\nB) yes\n",
                "correct_letter": "B",
            },
            "longcodeqa",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("Repository:\nexample", doc.query)
        self.assertNotIn("Repository: Repository:", doc.query)
        self.assertEqual(doc.specific["correct_letter"], "B")
        self.assertEqual(doc.specific["allowed_letters"], ["A", "B"])
        metric = lighteval_rwkv_skills_tasks.LongCodeQAAccuracy()
        self.assertEqual(metric.compute(ModelResponse(text=['{"arguments":{"answer":"B"}}']), doc), 1.0)
        self.assertEqual(metric.compute(ModelResponse(text=["Answer: A"]), doc), 0.0)

    def test_browsecomp_prompt_decrypts_openai_csv_row(self) -> None:
        canary = "unit-canary"
        question = "Which city hosted the example event?"
        answer = "Ozalj"
        doc = lighteval_rwkv_skills_tasks.browsecomp_prompt(
            {
                "problem": self._browsecomp_encrypt(question, canary),
                "answer": self._browsecomp_encrypt(answer, canary),
                "problem_topic": "Geography",
                "canary": canary,
            },
            "browsecomp",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn(question, doc.query)
        self.assertEqual(doc.specific["references"], [answer])
        self.assertEqual(doc.specific["locale"], "en")
        metric = lighteval_rwkv_skills_tasks.BrowseCompExactMatch()
        self.assertEqual(metric.compute(ModelResponse(text=["Explanation: short\nExact Answer: Ozalj\nConfidence: 90%"]), doc), 1.0)

    def test_browsecomp_zh_prompt_decrypts_hf_parquet_row(self) -> None:
        canary = "BrowseComp-ZH"
        question = "这个示例问题的答案是什么？"
        answer = "示例答案"
        topic = "示例"
        doc = lighteval_rwkv_skills_tasks.browsecomp_prompt(
            {
                "Question": self._browsecomp_encrypt(question, canary),
                "Answer": self._browsecomp_encrypt(answer, canary),
                "Topic": self._browsecomp_encrypt(topic, canary),
                "canary": canary,
            },
            "browsecomp_zh",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn(question, doc.query)
        self.assertEqual(doc.specific["references"], [answer])
        self.assertEqual(doc.specific["locale"], "zh")
        self.assertEqual(doc.specific["topic"], topic)
        metric = lighteval_rwkv_skills_tasks.BrowseCompF1()
        self.assertEqual(metric.compute(ModelResponse(text=["最终答案: 示例答案"]), doc), 1.0)

    def test_browsecomp_plus_prompt_decrypts_hf_row(self) -> None:
        canary = lighteval_rwkv_skills_tasks.BROWSECOMP_PLUS_CANARY
        query = "Which town hosts the example festival?"
        answer = "Ozalj"
        evidence = "The example festival is hosted in Ozalj every summer."
        doc = lighteval_rwkv_skills_tasks.browsecomp_plus_prompt(
            {
                "query_id": "unit-1",
                "query": self._browsecomp_encrypt(query, canary),
                "answer": self._browsecomp_encrypt(answer, canary),
                "gold_docs": [
                    {
                        "docid": self._browsecomp_encrypt("doc-1", canary),
                        "text": self._browsecomp_encrypt(evidence, canary),
                        "url": self._browsecomp_encrypt("https://example.test/doc", canary),
                    }
                ],
                "evidence_docs": [],
                "negative_docs": [],
            },
            "browsecomp_plus",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn(query, doc.query)
        self.assertIn(evidence, doc.query)
        self.assertEqual(doc.specific["references"], [answer])
        self.assertEqual(doc.specific["query_id"], "unit-1")
        self.assertEqual(doc.specific["mode"], "oracle_context")
        metric = lighteval_rwkv_skills_tasks.BrowseCompExactMatch()
        self.assertEqual(metric.compute(ModelResponse(text=["Exact Answer: Ozalj"]), doc), 1.0)

    def test_bfcl_prompt_scores_json_tool_call_against_ast_ground_truth(self) -> None:
        doc = lighteval_rwkv_skills_tasks.bfcl_prompt(
            {
                "id": "exec_multiple_0",
                "question": [[{"role": "user", "content": "Compute a binomial probability."}]],
                "function": [
                    {
                        "name": "calc_binomial_probability",
                        "description": "Calculate binomial probability.",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
                "execution_result_type": ["exact_match"],
                "ground_truth": ["calc_binomial_probability(n=20, k=5, p=1/6)"],
            },
            "bfcl_exec_multiple",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("Available functions:", doc.query)
        self.assertEqual(doc.specific["sample_id"], "exec_multiple_0")
        metric = lighteval_rwkv_skills_tasks.BFCLAccuracy()
        self.assertEqual(
            metric.compute(
                ModelResponse(
                    text=['{"name":"calc_binomial_probability","arguments":{"n":20,"k":5,"p":0.1666666667}}']
                ),
                doc,
            ),
            1.0,
        )

    def test_bfcl_metric_matches_parallel_calls_orderlessly(self) -> None:
        doc = lighteval_rwkv_skills_tasks.bfcl_prompt(
            {
                "id": "exec_parallel_0",
                "question": [[{"role": "user", "content": "Run three probability calculations."}]],
                "function": [{"name": "calc_binomial_probability", "parameters": {"type": "object"}}],
                "ground_truth": [
                    "calc_binomial_probability(n=10, k=3, p=0.3)",
                    "calc_binomial_probability(n=15, k=5, p=0.3)",
                ],
            },
            "bfcl_exec_parallel",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        metric = lighteval_rwkv_skills_tasks.BFCLAccuracy()
        response = ModelResponse(
            text=[
                json.dumps(
                    [
                        {"name": "calc_binomial_probability", "arguments": {"n": 15, "k": 5, "p": 0.3}},
                        {"name": "calc_binomial_probability", "arguments": {"n": 10, "k": 3, "p": 0.3}},
                    ]
                )
            ]
        )
        self.assertEqual(metric.compute(response, doc), 1.0)

    def test_bfcl_metric_extracts_multiple_tool_call_blocks(self) -> None:
        doc = lighteval_rwkv_skills_tasks.bfcl_prompt(
            {
                "id": "exec_parallel_asin",
                "question": [[{"role": "user", "content": "Get ratings for two products."}]],
                "function": [{"name": "get_rating_by_amazon_ASIN", "parameters": {"type": "object"}}],
                "ground_truth": [
                    "get_rating_by_amazon_ASIN(ASIN='B08PPDJWC8')",
                    "get_rating_by_amazon_ASIN(ASIN='B07ZPKBL9V')",
                ],
            },
            "bfcl_exec_parallel",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        metric = lighteval_rwkv_skills_tasks.BFCLAccuracy()
        response = ModelResponse(
            text=[
                "<tool_call>\n"
                '{"name":"get_rating_by_amazon_ASIN","arguments":"{\\"ASIN\\":\\"B08PPDJWC8\\"}"}'
                "\n</tool_call>\n"
                "<tool_call>\n"
                '{"name":"get_rating_by_amazon_ASIN","arguments":"{\\"ASIN\\":\\"B07ZPKBL9V\\"}"}'
                "\n</tool_call>"
            ]
        )
        self.assertEqual(metric.compute(response, doc), 1.0)

    def test_bfcl_prompt_joins_possible_answer_by_id(self) -> None:
        with mock.patch.object(
            lighteval_rwkv_skills_tasks,
            "_bfcl_possible_answers",
            return_value={
                "simple_0": [
                    {
                        "calculate_triangle_area": {
                            "base": [10],
                            "height": [5],
                            "unit": ["units", ""],
                        }
                    }
                ]
            },
        ):
            doc = lighteval_rwkv_skills_tasks.bfcl_prompt(
                {
                    "id": "simple_0",
                    "question": [[{"role": "user", "content": "Find the triangle area."}]],
                    "function": [{"name": "calculate_triangle_area", "parameters": {"type": "object"}}],
                },
                "bfcl_simple_python",
            )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.specific["sample_id"], "simple_0")
        self.assertNotIn("expected_calls", doc.specific)
        self.assertIn("calculate_triangle_area", doc.specific["expected_calls_json"])
        metric = lighteval_rwkv_skills_tasks.BFCLAccuracy()
        self.assertEqual(
            metric.compute(
                ModelResponse(text=['{"name":"calculate_triangle_area","arguments":{"base":10,"height":5}}']),
                doc,
            ),
            1.0,
        )

    def test_apibank_prompt_scores_against_sandbox_result(self) -> None:
        expected_result = {"input": {"city": "Paris"}, "output": {"weather": "sunny"}, "exception": None}
        doc = lighteval_rwkv_skills_tasks.apibank_prompt(
            {
                "task_id": "apibank_level1__weather_001",
                "instruction": "User: What is the weather in Paris?",
                "tools_json": json.dumps(
                    [
                        {
                            "name": "GetWeather",
                            "description": "Get weather.",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        }
                    ]
                ),
                "expected_call_json": json.dumps({"name": "GetWeather", "arguments": {"city": "Paris"}}),
                "expected_result_json": json.dumps(expected_result),
            },
            "apibank_level1",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("API-Bank", doc.query)
        self.assertEqual(doc.specific["sample_id"], "apibank_level1__weather_001")
        self.assertNotIn("expected_tool_calls", doc.specific)

        case = self

        class FakeSandbox:
            def replay_history(self, source_path, turn_index):
                case.assertEqual(source_path, "")
                case.assertEqual(turn_index, 1)

            def api_call(self, name, arguments):
                case.assertEqual(name, "GetWeather")
                case.assertEqual(arguments, {"city": "Paris"})
                return lighteval_rwkv_skills_tasks.ApiBankCallResult(True, expected_result)

            def check_api_call_correctness(self, name, actual, expected):
                case.assertEqual(name, "GetWeather")
                return actual == expected

            def _api_info(self, name):
                case.assertEqual(name, "GetWeather")
                return {"input_parameters": {"city": {"type": "str"}}}

            @staticmethod
            def _coerce_arg(value, arg_type):
                return value

        with mock.patch.object(lighteval_rwkv_skills_tasks, "ApiBankSandbox", return_value=FakeSandbox()):
            metric = lighteval_rwkv_skills_tasks.APIBankAccuracy()
            self.assertEqual(
                metric.compute(
                    ModelResponse(text=['{"name":"GetWeather","arguments":{"city":"Paris"}}']),
                    doc,
                ),
                1.0,
            )

    def test_apibank_metric_accepts_gold_arguments_when_official_execution_fails(self) -> None:
        expected_call = {
            "name": "CancelTimedSwitch",
            "arguments": {"device_id": "10000025", "time": "2023-03-19 09:30:00"},
        }
        doc = lighteval_rwkv_skills_tasks.apibank_prompt(
            {
                "task_id": "apibank_level1__CancelTimedSwitch-level-1-1_002",
                "instruction": "User: Cancel the timed switch.",
                "tools_json": json.dumps(
                    [
                        {
                            "name": "CancelTimedSwitch",
                            "description": "Cancels a timed switch.",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ]
                ),
                "expected_call_json": json.dumps(expected_call),
                "expected_result_json": json.dumps(
                    {
                        "api_name": "CancelTimedSwitch",
                        "input": {"device_id": "10000025", "time": "2023-03-19 09:30:00"},
                        "output": "success",
                        "exception": None,
                    }
                ),
                "source_path": "CancelTimedSwitch-level-1-1.jsonl",
                "turn_index": 2,
            },
            "apibank_level1",
        )
        self.assertIsNotNone(doc)
        assert doc is not None

        case = self

        class FakeSandbox:
            def replay_history(self, source_path, turn_index):
                case.assertEqual(source_path, "CancelTimedSwitch-level-1-1.jsonl")
                case.assertEqual(turn_index, 2)

            def api_call(self, name, arguments):
                case.assertEqual(name, "CancelTimedSwitch")
                case.assertEqual(arguments, {"name": "10000025", "time": "2023-03-19 09:30:00"})
                return lighteval_rwkv_skills_tasks.ApiBankCallResult(False, error="device name does not exist.")

            def _api_info(self, name):
                case.assertEqual(name, "CancelTimedSwitch")
                return {"input_parameters": {"name": {"type": "str"}, "time": {"type": "str"}}}

            @staticmethod
            def _coerce_arg(value, arg_type):
                return value

        with mock.patch.object(lighteval_rwkv_skills_tasks, "ApiBankSandbox", return_value=FakeSandbox()):
            metric = lighteval_rwkv_skills_tasks.APIBankAccuracy()
            self.assertEqual(
                metric.compute(
                    ModelResponse(
                        text=[
                            '{"name":"CancelTimedSwitch","arguments":'
                            '{"name":"10000025","time":"2023-03-19 09:30:00"}}'
                        ]
                    ),
                    doc,
                ),
                1.0,
            )

    def test_apibank_checker_falls_back_when_official_checker_raises(self) -> None:
        class RaisingTool:
            def check_api_call_correctness(self, actual, expected):
                raise KeyError("time")

        class FakeSandbox(lighteval_rwkv_skills_tasks.ApiBankSandbox):
            def __init__(self):
                pass

            def init_tool(self, api_name):
                return RaisingTool()

        actual = {
            "api_name": "DeleteMeeting",
            "input": {
                "attendees": ["David Wang", "Amy Chen"],
                "end_time": "2023-03-27 11:00:00",
                "location": "Training Room",
                "meeting_topic": "New Employee Orientation",
                "start_time": "2023-03-27 09:00:00",
                "token": "token",
            },
            "output": "success",
            "exception": None,
        }
        self.assertTrue(FakeSandbox().check_api_call_correctness("DeleteMeeting", actual, copy.deepcopy(actual)))
        wrong = copy.deepcopy(actual)
        wrong["input"]["location"] = "Other Room"
        self.assertFalse(FakeSandbox().check_api_call_correctness("DeleteMeeting", actual, wrong))

    def test_toolalpaca_prompt_scores_matching_request(self) -> None:
        doc = lighteval_rwkv_skills_tasks.toolalpaca_prompt(
            {
                "task_id": "toolalpaca_eval_simulated__weather_000",
                "instruction": "Find the current weather in Paris.",
                "tools": [
                    {
                        "name": "getWeather",
                        "description": "Get weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                        "metadata": {
                            "path": "/weather/{city}",
                            "method": "get",
                            "operation": {
                                "parameters": [
                                    {
                                        "name": "city",
                                        "in": "path",
                                        "required": True,
                                        "schema": {"type": "string"},
                                    }
                                ]
                            },
                        },
                    }
                ],
                "expected_tool_calls": [
                    {
                        "name": "getWeather",
                        "arguments": {"city": "Paris"},
                        "argument_options": {"city": ["Paris"]},
                    }
                ],
            },
            "toolalpaca_eval_simulated",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("ToolAlpaca", doc.query)
        self.assertNotIn('"metadata"', doc.query)
        self.assertEqual(doc.specific["sample_id"], "toolalpaca_eval_simulated__weather_000")
        metric = lighteval_rwkv_skills_tasks.ToolAlpacaAccuracy()
        self.assertEqual(
            metric.compute(
                ModelResponse(text=['{"name":"getWeather","arguments":{"city":"Paris"}}']),
                doc,
            ),
            1.0,
        )

    def test_complexfuncbench_prompt_scores_matching_parallel_call_turn(self) -> None:
        doc = lighteval_rwkv_skills_tasks.complexfuncbench_prompt(
            {
                "id": "complex-case-1",
                "functions": [
                    {
                        "name": "SearchHotel",
                        "description": "Search hotels.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                    {
                        "name": "BookHotel",
                        "description": "Book hotels.",
                        "parameters": {
                            "type": "object",
                            "properties": {"hotel_id": {"type": "string"}},
                            "required": ["hotel_id"],
                        },
                    },
                ],
                "conversations": [
                    {"role": "user", "content": "Find and book h1 in Paris."},
                    {
                        "role": "assistant",
                        "function_call": [
                            {"name": "SearchHotel", "arguments": {"city": "Paris"}},
                            {"name": "BookHotel", "arguments": {"hotel_id": "h1"}},
                        ],
                    },
                    {"role": "observation", "content": [{"hotel_id": "h1"}, {"status": "booked"}]},
                ],
            },
            "complexfuncbench_official",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("ComplexFuncBench", doc.query)
        self.assertIn("SearchHotel", doc.query)
        self.assertEqual(doc.specific["sample_id"], "complex-case-1__turn_1")
        metric = lighteval_rwkv_skills_tasks.ComplexFuncBenchCallAccuracy()
        self.assertEqual(
            metric.compute(
                ModelResponse(
                    text=[
                        '[{"name":"SearchHotel","arguments":{"city":"Paris"}},'
                        '{"name":"BookHotel","arguments":{"hotel_id":"h1"}}]'
                    ]
                ),
                doc,
            ),
            1.0,
        )
        self.assertEqual(
            metric.compute(
                ModelResponse(
                    text=[
                        '[{"name":"BookHotel","arguments":{"hotel_id":"h1"}},'
                        '{"name":"SearchHotel","arguments":{"city":"Paris"}}]'
                    ]
                ),
                doc,
            ),
            0.0,
        )

    def test_arena_hard_prompt_scores_against_baseline_answer(self) -> None:
        doc = lighteval_rwkv_skills_tasks.arena_hard_prompt(
            {
                "uid": "arena-1",
                "prompt": "Explain why the sky appears blue.",
                "baseline_answer": "The sky appears blue because air molecules scatter shorter blue wavelengths.",
            },
            "arena_hard_v2",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("Arena-Hard", doc.query)
        self.assertEqual(doc.specific["sample_id"], "arena-1")
        metric = lighteval_rwkv_skills_tasks.ArenaHardBaselineF1()
        self.assertEqual(
            metric.compute(
                ModelResponse(text=["The sky appears blue because air molecules scatter shorter blue wavelengths."]),
                doc,
            ),
            1.0,
        )

    def test_swebench_prompt_scores_gold_patch_and_strips_context_patch(self) -> None:
        patch = "--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-old\n+new\n"
        doc = lighteval_rwkv_skills_tasks.swebench_prompt(
            {
                "instance_id": "repo__project-1",
                "repo": "repo/project",
                "base_commit": "abc123",
                "problem_statement": "Fix the example bug.",
                "hints_text": "Look at example.py.",
                "text": f"Relevant context before patch.\n<patch>\n{patch}</patch>\nDo not leak this.",
                "patch": f"<patch>\n{patch}</patch>",
            },
            "swe_bench_lite_oracle",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("SWE-bench", doc.query)
        self.assertIn("Relevant context before patch.", doc.query)
        self.assertNotIn("<patch>", doc.query)
        self.assertNotIn("-old", doc.query)
        self.assertEqual(doc.specific["sample_id"], "repo__project-1")
        self.assertEqual(doc.specific["harness_dataset_name"], "princeton-nlp/SWE-bench_Lite")

        f1 = lighteval_rwkv_skills_tasks.SweBenchPatchF1()
        nonempty = lighteval_rwkv_skills_tasks.SweBenchPatchNonEmpty()
        response = ModelResponse(text=[f"```diff\n{patch}```"])
        self.assertEqual(f1.compute(response, doc), 1.0)
        self.assertEqual(nonempty.compute(response, doc), 1.0)
        self.assertEqual(nonempty.compute(ModelResponse(text=["I cannot produce a patch."]), doc), 0.0)

    def test_mcpbench_prompt_scores_static_plan(self) -> None:
        doc = lighteval_rwkv_skills_tasks.mcpbench_prompt(
            {
                "task_id": "weather_data_000",
                "instruction": "Find the hourly forecast for Seattle using only Weather Data.",
                "task_file": "mcpbench_tasks_single_runner_format.json",
                "server_name": "Weather Data",
                "servers": ["Weather Data"],
                "combination_name": "Single Server: Weather Data",
                "combination_type": "single_server",
                "official_source": "Accenture/mcp-bench",
                "official_source_revision": "revision",
                "official_source_path": "tasks/mcpbench_tasks_single_runner_format.json",
                "task": {
                    "task_id": "weather_data_000",
                    "task_description": "Use Weather Data:getForecast with latitude and longitude for Seattle.",
                    "dependency_analysis": "Call geocoding first, then use the coordinates in Weather Data:getForecast.",
                },
            },
            "mcp_bench_single",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("MCP-Bench", doc.query)
        self.assertIn("Weather Data", doc.query)
        self.assertEqual(doc.specific["sample_id"], "weather_data_000")
        self.assertEqual(doc.specific["servers"], ["Weather Data"])

        f1 = lighteval_rwkv_skills_tasks.McpBenchStaticPlanF1()
        nonempty = lighteval_rwkv_skills_tasks.McpBenchResponseNonEmpty()
        response = ModelResponse(text=[doc.specific["reference_plans"][0]])
        self.assertEqual(f1.compute(response, doc), 1.0)
        self.assertEqual(nonempty.compute(response, doc), 1.0)
        self.assertEqual(nonempty.compute(ModelResponse(text=[""]), doc), 0.0)

    def test_agentbench_db_prompt_scores_final_answer(self) -> None:
        doc = lighteval_rwkv_skills_tasks.agentbench_db_prompt(
            {
                "task_id": "agentbench_db__00000",
                "task_name": "dbbench-std",
                "index": 0,
                "domain": "dbbench",
                "question": "What are the Notes when the Method is decision?",
                "additional_description": "The table is Jiu-Jitsu Championships Results.",
                "operation_type": "other",
                "tables": [
                    {
                        "table_name": "Jiu-Jitsu Championships Results",
                        "columns": ["Method", "Notes"],
                        "rows": [["Decision", "Women +60kg Bronze"]],
                    }
                ],
                "reference_answers": ["Women +60kg Bronze"],
                "reference_sql": "SELECT Notes FROM table WHERE Method = 'Decision';",
                "reference_plan": "SQL: SELECT Notes FROM table WHERE Method = 'Decision';\nFinal answer: [\"Women +60kg Bronze\"]",
            },
            "agentbench_db",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("AgentBench DBBench", doc.query)
        self.assertIn("execute", doc.query.lower())
        self.assertEqual(doc.specific["sample_id"], "agentbench_db__00000")

        f1 = lighteval_rwkv_skills_tasks.AgentBenchDbAnswerF1()
        nonempty = lighteval_rwkv_skills_tasks.AgentBenchResponseNonEmpty()
        self.assertEqual(f1.compute(ModelResponse(text=['{"final_answer":"Women +60kg Bronze"}']), doc), 1.0)
        self.assertEqual(nonempty.compute(ModelResponse(text=['{"final_answer":"Women +60kg Bronze"}']), doc), 1.0)
        self.assertEqual(nonempty.compute(ModelResponse(text=[""]), doc), 0.0)

    def test_agentbench_kg_prompt_scores_reference_plan(self) -> None:
        doc = lighteval_rwkv_skills_tasks.agentbench_kg_prompt(
            {
                "task_id": "agentbench_kg__00000",
                "task_name": "kg-std",
                "index": 0,
                "domain": "knowledgegraph",
                "question": "what is the attitude of the first dog and the german shepherds?",
                "entities": {"first dog": "m.05t073s", "german shepherds": "m.0km5c"},
                "reference_actions": ["get_relations(m.05t073s)", "intersection(#1,#2)"],
                "reference_answers": ["Obedient", "Intelligent"],
                "reference_plan": "Actions:\nget_relations(m.05t073s)\nintersection(#1,#2)\nFinal answer names: [\"Obedient\", \"Intelligent\"]",
            },
            "agentbench_kg",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("AgentBench KnowledgeGraph", doc.query)
        self.assertIn("get_relations", doc.query)
        self.assertEqual(doc.specific["sample_id"], "agentbench_kg__00000")

        f1 = lighteval_rwkv_skills_tasks.AgentBenchKgPlanF1()
        response = ModelResponse(text=[doc.specific["reference_plans"][0]])
        self.assertEqual(f1.compute(response, doc), 1.0)

    def test_taubench_prompt_scores_static_plan_without_leaking_criteria(self) -> None:
        reference_plan = json.dumps(
            {
                "actions": [{"name": "create_task", "arguments": {"user_id": "user_1", "title": "Important Meeting"}}],
                "env_assertions": [{"func_name": "assert_task_status"}],
                "nl_assertions": [],
                "reward_basis": ["DB", "ENV_ASSERTION"],
            },
            sort_keys=True,
        )
        doc = lighteval_rwkv_skills_tasks.taubench_prompt(
            {
                "sample_id": "tau3_bench_mock__create_task_1",
                "task_id": "create_task_1",
                "domain": "mock",
                "split": "base",
                "benchmark_version": "tau_v3_light",
                "instruction": "Create a task named Important Meeting for user_1.",
                "available_action_names": ["create_task", "update_task_status"],
                "reference_action_names": ["create_task"],
                "reference_actions": [{"name": "create_task", "arguments": {"user_id": "user_1", "title": "Important Meeting"}}],
                "reference_plan": reference_plan,
                "task": {
                    "id": "create_task_1",
                    "ticket": "Create a task named Important Meeting for user_1.",
                    "evaluation_criteria": {"actions": [{"name": "create_task"}]},
                },
                "official_source": "https://github.com/sierra-research/tau2-bench",
                "official_source_revision": "revision",
                "official_source_path": "data/tau2/domains/mock",
            },
            "tau3_bench_mock",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("TAU benchmark", doc.query)
        self.assertIn("create_task", doc.query)
        self.assertNotIn("evaluation_criteria", doc.query)
        self.assertEqual(doc.specific["sample_id"], "tau3_bench_mock__create_task_1")
        self.assertEqual(doc.specific["reference_action_names"], ["create_task"])

        f1 = lighteval_rwkv_skills_tasks.TauBenchStaticPlanF1()
        nonempty = lighteval_rwkv_skills_tasks.TauBenchResponseNonEmpty()
        response = ModelResponse(text=[doc.specific["reference_plans"][0]])
        self.assertEqual(f1.compute(response, doc), 1.0)
        self.assertEqual(nonempty.compute(response, doc), 1.0)
        self.assertEqual(nonempty.compute(ModelResponse(text=[""]), doc), 0.0)

    def test_lighteval_export_prefers_specific_sample_id(self) -> None:
        row = lighteval_export.export_rows_from_frame(
            [
                {
                    "doc": {
                        "id": "doc-id",
                        "task_name": "swe_bench_lite|0",
                        "query": "Patch:",
                        "choices": [""],
                        "gold_index": 0,
                        "specific": {"sample_id": "repo__project-1"},
                    },
                    "metric": {"swebench_patch_nonempty": 1.0},
                    "model_response": {"text": ["--- a/x\n+++ b/x\n"]},
                }
            ]
        )[0]

        self.assertEqual(row["sample_id"], "repo__project-1")
        self.assertTrue(row["is_correct"])

    def test_is_correct_ignores_graded_f1_metrics(self) -> None:
        # A small nonzero proxy token-F1 is not a correct answer.
        self.assertIsNone(lighteval_export.is_correct({"swebench_patch_f1": 0.05}))
        self.assertIsNone(lighteval_export.is_correct({"longbench_f1": 0.4}))
        self.assertIsNone(lighteval_export.is_correct({"f1": 0.9}))

    def test_is_correct_uses_binary_signal_over_graded(self) -> None:
        # extractiveness (graded) must not override the binary extractive_match=0.0.
        self.assertFalse(
            lighteval_export.is_correct({"extractive_match": 0.0, "extractiveness": 0.83})
        )
        self.assertTrue(
            lighteval_export.is_correct({"extractive_match": 1.0, "extractiveness": 0.1})
        )

    def test_is_correct_handles_non_dict_metric(self) -> None:
        self.assertIsNone(lighteval_export.is_correct(None))
        self.assertIsNone(lighteval_export.is_correct([1.0]))

    def test_export_rows_tolerate_non_dict_columns(self) -> None:
        # doc/metric/model_response arriving as non-dict cells must not crash.
        rows = lighteval_export.export_rows_from_frame(
            [
                {
                    "doc": ["not", "a", "dict"],
                    "metric": None,
                    "model_response": "plain string",
                }
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["is_correct"])
        self.assertIsNone(rows[0]["task_name"])

    def test_free_answer_prompt_normalizes_numeric_answers(self) -> None:
        doc = lighteval_rwkv_skills_tasks.free_answer_prompt(
            {
                "question": "What is 40 + 3?",
                "final_answer": 43.0,
            },
            "algebra222",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.choices, ["43"])
        self.assertIn("Question: What is 40 + 3?", doc.query)

    def test_free_answer_prompt_builds_svamp_problem(self) -> None:
        doc = lighteval_rwkv_skills_tasks.free_answer_prompt(
            {
                "Body": "Each pack costs 76 dollars.",
                "Question": "How much after a 25 dollar discount?",
                "Answer": 51.0,
            },
            "svamp",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("Each pack costs 76 dollars. How much", doc.query)
        self.assertEqual(doc.choices, ["51"])

    def test_free_answer_prompt_extracts_math_odyssey_sparse_row(self) -> None:
        doc = lighteval_rwkv_skills_tasks.free_answer_prompt(
            {
                "Problem_1": {
                    "question": "\\begin{problem}Compute $1+1$.\\end{problem}",
                    "answer": "$2$.",
                },
                "Problem_2": None,
            },
            "math_odyssey",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("Compute $1+1$.", doc.query)
        self.assertEqual(doc.choices, ["2"])

    def test_free_answer_prompt_keeps_empty_gold_rows(self) -> None:
        doc = lighteval_rwkv_skills_tasks.free_answer_prompt(
            {
                "problem": "This source row has no gold answer.",
                "answer": "",
            },
            "omni_math",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.choices, [""])

    def test_free_answer_prompt_extracts_boxed_solution_when_answer_missing(self) -> None:
        doc = lighteval_rwkv_skills_tasks.free_answer_prompt(
            {
                "problem": "Find x.",
                "solution": "Solving gives \\\\boxed{1.6}.",
            },
            "minerva_math",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.choices, ["1.6"])

    def test_free_answer_prompt_extracts_nested_boxed_solution(self) -> None:
        doc = lighteval_rwkv_skills_tasks.free_answer_prompt(
            {
                "problem": "Find x.",
                "solution": "Solving gives \\\\boxed{\\\\frac{1}{2}}.",
            },
            "minerva_math",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.choices, ["\\\\frac{1}{2}"])

    def test_answer_judge_prompt_uses_mean_annotation_score(self) -> None:
        doc = lighteval_rwkv_skills_tasks.answer_judge_prompt(
            {
                "question": "What is 2 + 2?",
                "gt_answer": "4",
                "gen_answer": "four",
                "annotations": [{"score": "1"}, {"score": "0"}, {"score": "1"}],
            },
            "answer_judge",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.choices, ["Judgement: Yes"])
        self.assertIn("Return exactly `Judgement: Yes` or `Judgement: No`.", doc.query)

    def test_human_eval_prompt_preserves_execution_specifics(self) -> None:
        doc = lighteval_rwkv_skills_tasks.code_generation_prompt(
            {
                "prompt": "def add_one(x):\n    ",
                "entry_point": "add_one",
                "test": "def check(candidate):\n    assert candidate(1) == 2",
            },
            "human_eval",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.specific["code_kind"], "human_eval")
        self.assertEqual(doc.specific["entry_point"], "add_one")
        self.assertIn("Return only executable Python code", doc.query)

    def test_mbpp_prompt_uses_base_test_list_for_base_task(self) -> None:
        doc = lighteval_rwkv_skills_tasks.code_generation_prompt(
            {
                "prompt": "Write a function to add one.",
                "code": "def add_one(x):\n    return x + 1\n",
                "test_imports": "[]",
                "test_list": "['assert add_one(1) == 2']",
                "test": "raise AssertionError('plus test should not be used')",
            },
            "mbpp",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.specific["code_kind"], "mbpp")
        self.assertEqual(doc.specific["entry_point"], "add_one")
        self.assertIn("assert add_one(1) == 2", doc.specific["test"])
        self.assertNotIn("plus test should not be used", doc.specific["test"])

    def test_code_metric_runs_human_eval_style_check(self) -> None:
        doc = lighteval_rwkv_skills_tasks.code_generation_prompt(
            {
                "prompt": "def add_one(x):\n    ",
                "entry_point": "add_one",
                "test": "def check(candidate):\n    assert candidate(1) == 2",
            },
            "human_eval",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        metric = lighteval_rwkv_skills_tasks.CodePassAtOne()
        passed = metric.compute(ModelResponse(text=["return x + 1"]), doc)
        failed = metric.compute(ModelResponse(text=["return x + 2"]), doc)
        self.assertEqual(passed, 1.0)
        self.assertEqual(failed, 0.0)

    def test_code_metric_runs_mbpp_style_assertions(self) -> None:
        doc = lighteval_rwkv_skills_tasks.code_generation_prompt(
            {
                "prompt": "Write a function to add one.",
                "code": "def add_one(x):\n    return x + 1\n",
                "test_imports": "[]",
                "test_list": "['assert add_one(1) == 2']",
            },
            "mbpp",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        metric = lighteval_rwkv_skills_tasks.CodePassAtOne()
        passed = metric.compute(ModelResponse(text=["def add_one(x):\n    return x + 1"]), doc)
        failed = metric.compute(ModelResponse(text=["def add_one(x):\n    return x + 2"]), doc)
        self.assertEqual(passed, 1.0)
        self.assertEqual(failed, 0.0)

    def test_lighteval_tasks_loads_rwkv_skills_registry_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "benchmark_registry.py"
            source.write_text(
                "\n".join(
                    [
                        "_EXPLICIT_METADATA: dict[str, object] = {",
                        '    canonical_slug("gsm8k"): _math("gsm8k"),',
                        '    canonical_slug("mmlu"): _knowledge("mmlu"),',
                        '    canonical_slug("bfcl_v3"): _function_calling("bfcl_v3", scheduler_jobs=()),',
                        "}",
                    ]
                )
            )

            rows = lighteval_tasks.load_source_benchmarks(str(source), "auto")

        self.assertEqual(
            rows,
            [
                lighteval_tasks.SourceBenchmark(name="gsm8k", field="maths"),
                lighteval_tasks.SourceBenchmark(name="mmlu", field="knowledge"),
                lighteval_tasks.SourceBenchmark(name="bfcl_v3", field="function_calling"),
            ],
        )

    def test_lighteval_export_plan_exports_details(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_lighteval_export_plan(
            lighteval_export_args(output="tmp/gsm8k.jsonl"),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        self.assertEqual(plan.command[1:3], ["-m", "helicopter_cli.lighteval_export"])
        self.assertIn(str(ROOT / "results/lighteval/details/run"), plan.command)
        options = command_options(plan.command)
        self.assertEqual(options["--output"], "tmp/gsm8k.jsonl")
        self.assertEqual(options["--format"], "jsonl")

    def test_takeoff_plan_uses_verl_module_entrypoint_and_default_overrides(self) -> None:
        loaded_config = load_example_config()
        venv_python = ROOT / ".venv/bin/python"

        plan = build_takeoff_plan(loaded_config, venv_python=venv_python)
        overrides = hydra_map(plan)
        optional_rollout_keys = {
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            "actor_rollout_ref.rollout.max_num_seqs",
            "actor_rollout_ref.rollout.max_num_batched_tokens",
        }

        self.assertEqual(plan.cwd, ROOT / "src/train/verl-rwkv")
        self.assertEqual(
            plan.command[:3],
            [
                str(venv_python),
                "-m",
                "verl.experimental.one_step_off_policy.main_ppo",
            ],
        )
        self.assertEqual(
            plan.shown_env,
            {
                "PYTHON": str(venv_python),
                "PYTHONPATH": str(ROOT / "src/infer/vllm-rwkv"),
                "RWKV_LM_PATH": str(ROOT / "src/train/rwkv-lm"),
                "RWKV_MODEL_PATH": "/weights/RWKV/rwkv7-g1g-1.5b-20260526-ctx8192.pth",
                "VLLM_RWKV7_EMB_DEVICE": "gpu",
                "VLLM_RWKV7_WKV_MODE": "fp32io16",
            },
        )
        self.assertEqual(
            {
                key: overrides[key]
                for key in (
                    "actor_rollout_ref.actor.use_dynamic_bsz",
                    "actor_rollout_ref.model.path",
                    "actor_rollout_ref.rollout.name",
                    "actor_rollout_ref.hybrid_engine",
                    "trainer.total_epochs",
                )
            },
            {
                "actor_rollout_ref.actor.use_dynamic_bsz": "False",
                "actor_rollout_ref.model.path": "/weights/RWKV/rwkv7-g1g-1.5b-20260526-ctx8192.pth",
                "actor_rollout_ref.rollout.name": "vllm",
                "actor_rollout_ref.hybrid_engine": "False",
                "trainer.total_epochs": "2",
            },
        )
        self.assertEqual(optional_rollout_keys & overrides.keys(), set())

    def test_takeoff_runtime_env_strips_dotenv_vllm_knobs(self) -> None:
        loaded_config = load_example_config()
        plan = build_takeoff_plan(
            loaded_config,
            loaded_env={
                "WEIGHT_PATH": "/weights/RWKV",
                "DATASETS_PATH": "/datasets",
                "HELICOPTER_VLLM_RWKV_PATH": "src/infer/vllm-rwkv",
                "VLLM_GPU_MEMORY_UTILIZATION": "0.85",
                "VLLM_MAX_NUM_SEQS": "2048",
                "VLLM_MAX_NUM_BATCHED_TOKENS": "65536",
                "VLLM_RWKV_PATH": "legacy/path",
                "VLLM_RWKV7_EMB_DEVICE": "cpu",
                "VLLM_USE_V2_MODEL_RUNNER": "1",
            },
        )
        overrides = hydra_map(plan)
        forbidden_env_keys = {
            "VLLM_GPU_MEMORY_UTILIZATION",
            "VLLM_MAX_NUM_SEQS",
            "VLLM_MAX_NUM_BATCHED_TOKENS",
            "VLLM_RWKV_PATH",
            "VLLM_USE_V2_MODEL_RUNNER",
        }
        forbidden_override_keys = {
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            "actor_rollout_ref.rollout.max_num_seqs",
            "actor_rollout_ref.rollout.max_num_batched_tokens",
        }

        self.assertEqual(plan.env["VLLM_RWKV7_WKV_MODE"], "fp32io16")
        self.assertEqual(plan.env["VLLM_RWKV7_EMB_DEVICE"], "gpu")
        self.assertEqual(plan.env["PYTHONPATH"], str(ROOT / "src/infer/vllm-rwkv"))
        self.assertEqual(forbidden_env_keys & plan.env.keys(), set())
        self.assertEqual(forbidden_override_keys & overrides.keys(), set())

    def test_infer_runtime_env_strips_dotenv_vllm_knobs(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_infer_plan(
            infer_args(),
            root=ROOT,
            env={
                "WEIGHT_PATH": "/weights/RWKV",
                "VLLM_RWKV7_WKV_MODE": "fp32io16",
                "VLLM_GPU_MEMORY_UTILIZATION": "0.85",
                "VLLM_MAX_NUM_SEQS": "2048",
            },
            config=loaded_config,
        )
        options = command_options(plan.command)
        forbidden_env_keys = {"VLLM_GPU_MEMORY_UTILIZATION", "VLLM_MAX_NUM_SEQS"}
        forbidden_option_keys = {"--gpu-memory-utilization", "--max-num-seqs"}

        self.assertEqual(plan.env["VLLM_RWKV7_WKV_MODE"], "fp32io16")
        self.assertEqual(forbidden_env_keys & plan.env.keys(), set())
        self.assertEqual(forbidden_option_keys & options.keys(), set())

    def test_takeoff_config_adv_estimator_becomes_hydra_overrides(self) -> None:
        loaded_config = load_example_config()
        takeoff = loaded_config["takeoff"]
        takeoff["grpo"] = {**takeoff["grpo"], "adv_estimator": "maxrl", "reward_manager": "dapo"}

        overrides = hydra_map(build_takeoff_plan(loaded_config))

        self.assertEqual(
            {
                "algorithm.adv_estimator": overrides["algorithm.adv_estimator"],
                "reward.reward_manager.name": overrides["reward.reward_manager.name"],
            },
            {
                "algorithm.adv_estimator": "maxrl",
                "reward.reward_manager.name": "dapo",
            },
        )

    def test_takeoff_rollout_gpu_count_becomes_top_level_and_actor_rollout_overrides(self) -> None:
        loaded_config = load_example_config()
        takeoff = loaded_config["takeoff"]
        takeoff["grpo"] = {
            **takeoff["grpo"],
            "trainer_n_gpus_per_node": 7,
            "rollout_n_gpus_per_node": 1,
            "rollout_data_parallel_size": 1,
            "rollout_pipeline_parallel_size": 1,
        }

        overrides = hydra_map(build_takeoff_plan(loaded_config))

        self.assertEqual(
            {
                key: overrides[key]
                for key in (
                    "trainer.n_gpus_per_node",
                    "rollout.n_gpus_per_node",
                    "actor_rollout_ref.rollout.n_gpus_per_node",
                    "actor_rollout_ref.rollout.data_parallel_size",
                    "actor_rollout_ref.rollout.pipeline_model_parallel_size",
                )
            },
            {
                "trainer.n_gpus_per_node": "7",
                "rollout.n_gpus_per_node": "1",
                "actor_rollout_ref.rollout.n_gpus_per_node": "1",
                "actor_rollout_ref.rollout.data_parallel_size": "1",
                "actor_rollout_ref.rollout.pipeline_model_parallel_size": "1",
            },
        )

    def test_takeoff_dataset_files_become_verl_file_lists(self) -> None:
        loaded_config = load_example_config()
        datasets = loaded_config["datasets"]
        datasets["dapo_math_17k"] = {
            "train_files": ["${DATASETS_PATH}/DAPO/dapo-math-17k.parquet"],
            "val_files": [
                "${DATASETS_PATH}/AIME24/test.parquet",
                "${DATASETS_PATH}/AIME25/test.parquet",
            ],
        }

        plan = build_takeoff_plan(loaded_config, args=takeoff_args(dataset="dapo_math_17k"))
        overrides = hydra_map(plan)

        self.assertEqual(
            {
                "data.train_files": overrides["data.train_files"],
                "data.val_files": overrides["data.val_files"],
            },
            {
                "data.train_files": "['/datasets/DAPO/dapo-math-17k.parquet']",
                "data.val_files": "['/datasets/AIME24/test.parquet','/datasets/AIME25/test.parquet']",
            },
        )
        self.assertEqual(
            set(plan.shown_env),
            {
                "PYTHON",
                "PYTHONPATH",
                "RWKV_LM_PATH",
                "RWKV_MODEL_PATH",
                "VLLM_RWKV7_EMB_DEVICE",
                "VLLM_RWKV7_WKV_MODE",
            },
        )

    def test_takeoff_defaults_keep_actor_kl_loss_disabled(self) -> None:
        loaded_config = load_example_config()
        overrides = hydra_map(build_takeoff_plan(loaded_config))

        self.assertEqual(
            {
                "actor_rollout_ref.actor.use_kl_loss": overrides["actor_rollout_ref.actor.use_kl_loss"],
                "actor_rollout_ref.actor.kl_loss_coef": overrides["actor_rollout_ref.actor.kl_loss_coef"],
            },
            {
                "actor_rollout_ref.actor.use_kl_loss": "False",
                "actor_rollout_ref.actor.kl_loss_coef": "0.0",
            },
        )

    def test_takeoff_explicit_dataset_files_do_not_require_dataset_root(self) -> None:
        loaded_config = load_example_config()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_path = tmp_path / "model.pth"
            model_path.write_bytes(b"")
            missing_dataset_root = tmp_path / "missing-datasets"
            loaded_config["models"]["local-test"] = {
                "path": str(model_path),
                "served_model_name": "local-test",
                "max_model_len": 8192,
            }
            loaded_config["datasets"]["dapo_math_17k"] = {
                "train_files": ["${DATASETS_PATH}/DAPO/dapo-math-17k.parquet"],
                "val_files": ["${DATASETS_PATH}/AIME24/test.parquet"],
            }
            args = Namespace(
                algorithm="grpo",
                model="local-test",
                dataset="dapo_math_17k",
                dry_run=False,
                wkv_mode=None,
                emb_device=None,
                num_nodes=None,
                num_devices=None,
                override=None,
            )

            plan = commands.build_takeoff_plan(
                args,
                root=ROOT,
                env={
                    "DATASETS_PATH": str(missing_dataset_root),
                    "HELICOPTER_PYTHON": "/usr/bin/python3",
                },
                config=loaded_config,
            )

        self.assertEqual(
            hydra_map(plan)["data.train_files"],
            f"['{missing_dataset_root}/DAPO/dapo-math-17k.parquet']",
        )

    def test_takeoff_partial_explicit_dataset_files_require_dataset_root(self) -> None:
        loaded_config = load_example_config()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_path = tmp_path / "model.pth"
            model_path.write_bytes(b"")
            missing_dataset_root = tmp_path / "missing-datasets"
            loaded_config["models"]["local-test"] = {
                "path": str(model_path),
                "served_model_name": "local-test",
                "max_model_len": 8192,
            }
            loaded_config["datasets"]["partial"] = {
                "train_files": ["${DATASETS_PATH}/partial/train.parquet"],
            }
            args = Namespace(
                algorithm="grpo",
                model="local-test",
                dataset="partial",
                dry_run=False,
                wkv_mode=None,
                emb_device=None,
                num_nodes=None,
                num_devices=None,
                override=None,
            )

            with self.assertRaises(SystemExit) as raised:
                commands.build_takeoff_plan(
                    args,
                    root=ROOT,
                    env={
                        "DATASETS_PATH": str(missing_dataset_root),
                        "HELICOPTER_PYTHON": "/usr/bin/python3",
                    },
                    config=loaded_config,
                )

        self.assertEqual(
            str(raised.exception),
            f"dataset root not found: {missing_dataset_root / 'partial'}",
        )

    def test_takeoff_user_overrides_are_appended_after_generated_overrides(self) -> None:
        loaded_config = load_example_config()
        plan = build_takeoff_plan(
            loaded_config,
            args=takeoff_args(override=["trainer.total_epochs=1", "trainer.save_freq=10"]),
        )

        self.assertEqual(hydra_values(plan, "trainer.total_epochs"), ["2", "1"])
        self.assertEqual(hydra_values(plan, "trainer.save_freq"), ["20", "10"])
        self.assertEqual(plan.command[-2:], ["trainer.total_epochs=1", "trainer.save_freq=10"])

    def test_takeoff_rejects_missing_default_venv_python(self) -> None:
        config = load_example_config()
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"HELICOPTER_PYTHON", "PYTHON", "HELICOPTER_VENV", "VENV", "REMOTE_VENV"}
        }
        venv_python = ROOT / ".venv/bin/python"
        original_exists = Path.exists

        with mock.patch.object(Path, "exists", autospec=True) as exists:
            exists.side_effect = lambda path: False if path == venv_python else original_exists(path)
            with self.assertRaises(SystemExit) as raised:
                commands.python_executable(config, root=ROOT, env=env, require_configured=True)

        self.assertEqual(
            str(raised.exception),
            f"Python executable not found: {venv_python}; run scripts/install_local.sh "
            "or set HELICOPTER_PYTHON / paths.python",
        )


def eval_run_args(**overrides: object) -> Namespace:
    values = {
        **vars(lighteval_args()),
        "tasks": None,
        "dry_run": True,
        "wkv_mode": None,
        "emb_device": None,
        "tensor_parallel_size": None,
        "gpu_memory_utilization": None,
        "max_num_seqs": None,
        "max_num_batched_tokens": None,
        "vllm_env": None,
        "no_server": False,
        "keep_server": False,
        "server_timeout": eval_run.DEFAULT_SERVER_TIMEOUT_S,
        "scoreboard": False,
    }
    values.update(overrides)
    return Namespace(**values)


class EvalRunTests(unittest.TestCase):
    def test_resolve_run_tasks_prefers_cli_then_config(self) -> None:
        loaded = load_example_config()
        loaded["lighteval"] = {**loaded.get("lighteval", {}), "tasks": ["gsm8k|0", "mmlu|0"]}
        self.assertEqual(
            eval_run.resolve_run_tasks(eval_run_args(tasks="aime24|0"), loaded),
            "aime24|0",
        )
        self.assertEqual(
            eval_run.resolve_run_tasks(eval_run_args(), loaded),
            "gsm8k|0,mmlu|0",
        )

    def test_resolve_run_tasks_requires_tasks(self) -> None:
        loaded = load_example_config()
        loaded["lighteval"] = {key: value for key, value in loaded.get("lighteval", {}).items() if key != "tasks"}
        with self.assertRaises(SystemExit):
            eval_run.resolve_run_tasks(eval_run_args(), loaded)

    def test_port_and_health_url_helpers(self) -> None:
        self.assertEqual(eval_run.port_from_base_url("http://127.0.0.1:8000/v1"), "8000")
        self.assertIsNone(eval_run.port_from_base_url("http://example.com/v1"))
        self.assertEqual(
            eval_run.health_url_for("http://127.0.0.1:8000/v1"),
            "http://127.0.0.1:8000/v1/models",
        )

    def test_dry_run_prints_infer_and_lighteval_plans(self) -> None:
        loaded = load_example_config()
        loaded["lighteval"] = {**loaded.get("lighteval", {}), "tasks": "gsm8k|0"}
        args = eval_run_args()
        env = {"WEIGHT_PATH": "/weights/RWKV"}
        stdout = io.StringIO()
        with mock.patch("sys.stdout", stdout):
            exit_code = eval_run.run_eval(args, root=ROOT, env=env, config=loaded)
        self.assertEqual(exit_code, 0)
        lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 2)
        self.assertIn("vllm serve", lines[0])
        self.assertIn("--port 8000", lines[0])
        self.assertIn("-m lighteval endpoint litellm", lines[1])
        self.assertIn("gsm8k|0", lines[1])

    def test_dry_run_with_remote_base_url_skips_server_plan(self) -> None:
        loaded = load_example_config()
        loaded["lighteval"] = {
            **loaded.get("lighteval", {}),
            "tasks": "gsm8k|0",
            "base_url": "http://eval-farm.internal:9000/v1",
        }
        stdout = io.StringIO()
        with mock.patch("sys.stdout", stdout):
            exit_code = eval_run.run_eval(
                eval_run_args(), root=ROOT, env={"WEIGHT_PATH": "/weights/RWKV"}, config=loaded
            )
        self.assertEqual(exit_code, 0)
        lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertIn("-m lighteval endpoint litellm", lines[0])
        self.assertNotIn("vllm serve", lines[0])

    def test_no_server_flag_skips_server_plan(self) -> None:
        loaded = load_example_config()
        loaded["lighteval"] = {**loaded.get("lighteval", {}), "tasks": "gsm8k|0"}
        stdout = io.StringIO()
        with mock.patch("sys.stdout", stdout):
            exit_code = eval_run.run_eval(
                eval_run_args(no_server=True),
                root=ROOT,
                env={"WEIGHT_PATH": "/weights/RWKV"},
                config=loaded,
            )
        self.assertEqual(exit_code, 0)
        lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertNotIn("vllm serve", lines[0])

    def test_infer_args_namespace_forwards_server_overrides(self) -> None:
        args = eval_run_args(
            wkv_mode="fp16",
            gpu_memory_utilization=0.5,
            vllm_env=["VLLM_WSL2_ENABLE_PIN_MEMORY=1"],
        )
        namespace = eval_run.infer_args_namespace(args, port="8123")
        self.assertEqual(namespace.model, "g1g-1.5b")
        self.assertEqual(namespace.port, "8123")
        self.assertEqual(namespace.wkv_mode, "fp16")
        self.assertEqual(namespace.gpu_memory_utilization, 0.5)
        self.assertEqual(namespace.vllm_env, ["VLLM_WSL2_ENABLE_PIN_MEMORY=1"])
        self.assertIsNone(namespace.host)

    def test_scoreboard_dataset_name_strips_fewshot_and_suite(self) -> None:
        self.assertEqual(eval_run.scoreboard_dataset_name("gsm8k|0"), "gsm8k")
        self.assertEqual(eval_run.scoreboard_dataset_name("extended|mmlu|5"), "mmlu")
        self.assertEqual(eval_run.scoreboard_dataset_name("apibank_level1"), "apibank_level1")

    def test_scoreboard_model_name_prefers_served_model_name(self) -> None:
        loaded = load_example_config()
        self.assertEqual(eval_run.scoreboard_model_name(eval_run_args(), loaded), "g1g-1.5b")
        self.assertEqual(
            eval_run.scoreboard_model_name(
                eval_run_args(lighteval_model_name="custom-name"), loaded
            ),
            "custom-name",
        )


def _postgres_connection() -> dict[str, object] | None:
    try:
        import asyncpg  # noqa: F401
    except ImportError:
        return None
    import asyncio
    import getpass

    async def probe() -> dict[str, object] | None:
        for port in (int(os.environ.get("PGPORT") or 0) or None, 5432, 5433):
            if port is None:
                continue
            kwargs = {
                "user": os.environ.get("PGUSER") or getpass.getuser(),
                "host": os.environ.get("PGHOST") or "/var/run/postgresql",
                "database": os.environ.get("PGDATABASE") or "postgres",
                "port": port,
            }
            try:
                conn = await asyncpg.connect(**kwargs)
            except Exception:
                continue
            await conn.close()
            return kwargs
        return None

    return asyncio.run(probe())


_POSTGRES_KWARGS = _postgres_connection()


@unittest.skipUnless(_POSTGRES_KWARGS, "PostgreSQL is not reachable")
class ScoreboardIngestionTests(unittest.TestCase):
    def test_ingest_records_scores_for_each_task(self) -> None:
        import asyncio
        import uuid

        import asyncpg

        kwargs = dict(_POSTGRES_KWARGS)
        db_name = f"helicopter_scoreboard_test_{uuid.uuid4().hex[:12]}"

        async def create_db() -> None:
            conn = await asyncpg.connect(**kwargs)
            try:
                await conn.execute(f'CREATE DATABASE "{db_name}"')
            finally:
                await conn.close()

        async def drop_db() -> None:
            conn = await asyncpg.connect(**kwargs)
            try:
                await conn.execute(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = $1 AND pid <> pg_backend_pid()
                    """,
                    db_name,
                )
                await conn.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
            finally:
                await conn.close()

        asyncio.run(create_db())
        scoreboard_path = ROOT / "src/scoreboard-server"
        if str(scoreboard_path) not in sys.path:
            sys.path.insert(0, str(scoreboard_path))
        db_env = {
            "SCOREBOARD_DB_HOST": str(kwargs["host"]),
            "SCOREBOARD_DB_PORT": str(kwargs["port"]),
            "SCOREBOARD_DB_USER": str(kwargs["user"]),
            "SCOREBOARD_DB_NAME": db_name,
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                results_path = Path(tmp) / "results_2026-07-06T00-00-00.000000.json"
                results_path.write_text(
                    json.dumps(
                        {
                            "config_general": {"model_name": "openai/unit-test-model"},
                            "results": {
                                "gsm8k|0": {"extractive_match": 0.5},
                                "mmlu|0": {"acc": 0.25},
                                "all": {"extractive_match": 0.5, "acc": 0.25},
                            },
                        }
                    )
                )
                with mock.patch.dict(os.environ, db_env):
                    from scoreboard_server.db.connection import close_db, init_db
                    from scoreboard_server.db.settings import DatabaseSettings

                    async def prepare_schema() -> None:
                        await init_db(DatabaseSettings.from_env(), generate_schemas=True)
                        await close_db()

                    asyncio.run(prepare_schema())
                    recorded = asyncio.run(
                        eval_run._ingest_scoreboard_results(
                            result_files=[str(results_path)],
                            model_name="unit-test-model",
                            root=ROOT,
                        )
                    )

                    self.assertEqual(len(recorded), 2)

                    async def read_back() -> list[dict[str, object]]:
                        await init_db(DatabaseSettings.from_env())
                        try:
                            from scoreboard_server.db.repository import ScoreboardStore

                            store = ScoreboardStore(settings=DatabaseSettings.from_env())
                            return await store.list_latest_scores_for_space()
                        finally:
                            await close_db()

                    rows = asyncio.run(read_back())
        finally:
            asyncio.run(drop_db())

        datasets = {str(row.get("dataset") or row.get("benchmark_name")): row for row in rows}
        self.assertIn("gsm8k", datasets)
        self.assertIn("mmlu", datasets)
        gsm8k_metrics = datasets["gsm8k"].get("metrics")
        self.assertEqual(gsm8k_metrics, {"extractive_match": 0.5})


if __name__ == "__main__":
    unittest.main()
