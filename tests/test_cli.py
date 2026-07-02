from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from helicopter_cli import commands, config, env


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
    }
    values.update(overrides)
    return Namespace(**values)


def lighteval_suite_args(**overrides: object) -> Namespace:
    values = {
        "model": "g1d-0.4b",
        "suite": "rwkv_skills",
        "mapped_only": False,
        "field": None,
        "benchmark": None,
        "model_args": None,
        "lighteval_model_name": None,
        "base_url": None,
        "provider": None,
        "api_key": None,
        "concurrent_requests": None,
        "max_model_length": None,
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
        options = command_options(plan.command)
        self.assertEqual(options["--output-dir"], str(ROOT / "results/lighteval"))
        self.assertEqual(options["--max-samples"], "3")
        self.assertEqual(options["--dataset-loading-processes"], "1")
        self.assertTrue(options["--save-details"])
        self.assertEqual(plan.env["OPENAI_API_KEY"], "EMPTY")

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

    def test_rwkv_skills_lighteval_suite_manifest_covers_registry_count(self) -> None:
        suite_path = ROOT / "configs/lighteval/rwkv_skills.toml"

        with suite_path.open("rb") as file:
            suite = tomllib.load(file)

        self.assertEqual(suite["source_count"], 95)
        self.assertEqual(len(suite["benchmarks"]), 95)
        self.assertIn("gsm8k", suite["benchmarks"])
        self.assertEqual(suite["benchmarks"]["human_eval"]["lighteval_tasks"], ["rwkv_skills:human_eval"])
        self.assertEqual(suite["benchmarks"]["include"]["lighteval_tasks"], ["rwkv_skills:include"])
        self.assertEqual(suite["benchmarks"]["human_eval_cn"]["lighteval_tasks"], ["rwkv_skills:human_eval_cn"])
        self.assertEqual(
            suite["benchmarks"]["human_eval_fix"]["lighteval_tasks"],
            ["rwkv_skills:human_eval_fix"],
        )
        self.assertEqual(
            suite["benchmarks"]["human_eval_plus"]["lighteval_tasks"],
            ["rwkv_skills:human_eval_plus"],
        )
        self.assertEqual(suite["benchmarks"]["mbpp"]["lighteval_tasks"], ["rwkv_skills:mbpp"])
        self.assertEqual(suite["benchmarks"]["mbpp_plus"]["lighteval_tasks"], ["rwkv_skills:mbpp_plus"])
        self.assertEqual(suite["benchmarks"]["longcodeqa"]["lighteval_tasks"], ["rwkv_skills:longcodeqa"])
        self.assertEqual(suite["benchmarks"]["apibank_l1"]["lighteval_tasks"], ["rwkv_skills:apibank_l1"])
        self.assertEqual(suite["benchmarks"]["apibank_l2"]["lighteval_tasks"], ["rwkv_skills:apibank_l2"])
        self.assertEqual(suite["benchmarks"]["apibank_level1"]["lighteval_tasks"], ["rwkv_skills:apibank_level1"])
        self.assertEqual(suite["benchmarks"]["apibank_level2"]["lighteval_tasks"], ["rwkv_skills:apibank_level2"])
        self.assertEqual(
            suite["benchmarks"]["toolalpaca_eval_real"]["lighteval_tasks"],
            ["rwkv_skills:toolalpaca_eval_real"],
        )
        self.assertEqual(
            suite["benchmarks"]["toolalpaca_eval_simulated"]["lighteval_tasks"],
            ["rwkv_skills:toolalpaca_eval_simulated"],
        )
        self.assertEqual(
            suite["benchmarks"]["complexfuncbench_official"]["lighteval_tasks"],
            ["rwkv_skills:complexfuncbench_official"],
        )
        self.assertEqual(
            suite["benchmarks"]["complexfuncbench_subset"]["lighteval_tasks"],
            ["rwkv_skills:complexfuncbench_subset"],
        )
        self.assertEqual(suite["benchmarks"]["bfcl_v3"]["lighteval_tasks"], ["rwkv_skills:bfcl_v3"])
        self.assertEqual(suite["benchmarks"]["browsecomp"]["lighteval_tasks"], ["rwkv_skills:browsecomp"])
        self.assertEqual(suite["benchmarks"]["browsecomp_plus"]["lighteval_tasks"], ["rwkv_skills:browsecomp_plus"])
        self.assertEqual(suite["benchmarks"]["browsecomp_zh"]["lighteval_tasks"], ["rwkv_skills:browsecomp_zh"])
        self.assertEqual(suite["benchmarks"]["bfcl_simple_python"]["lighteval_tasks"], ["rwkv_skills:bfcl_simple_python"])
        self.assertEqual(suite["benchmarks"]["bfcl_multiple"]["lighteval_tasks"], ["rwkv_skills:bfcl_multiple"])
        self.assertEqual(suite["benchmarks"]["bfcl_exec_simple_ast"]["lighteval_tasks"], ["rwkv_skills:bfcl_exec_simple_ast"])
        self.assertEqual(
            suite["benchmarks"]["bfcl_exec_multiple_ast"]["lighteval_tasks"],
            ["rwkv_skills:bfcl_exec_multiple_ast"],
        )
        self.assertEqual(suite["benchmarks"]["bfcl_exec_simple"]["lighteval_tasks"], ["rwkv_skills:bfcl_exec_simple"])
        self.assertEqual(suite["benchmarks"]["bfcl_exec_multiple"]["lighteval_tasks"], ["rwkv_skills:bfcl_exec_multiple"])
        self.assertEqual(suite["benchmarks"]["bfcl_exec_parallel"]["lighteval_tasks"], ["rwkv_skills:bfcl_exec_parallel"])
        self.assertEqual(
            suite["benchmarks"]["bfcl_exec_parallel_multiple"]["lighteval_tasks"],
            ["rwkv_skills:bfcl_exec_parallel_multiple"],
        )
        self.assertEqual(len(suite["benchmarks"]["longbench"]["lighteval_tasks"]), 21)
        self.assertEqual(len(suite["benchmarks"]["longbench_qa"]["lighteval_tasks"]), 9)
        self.assertEqual(len(suite["benchmarks"]["longbench_qa_balanced"]["lighteval_tasks"]), 9)
        self.assertIn("rwkv_skills:longbench_narrativeqa", suite["benchmarks"]["longbench"]["lighteval_tasks"])
        self.assertIn("rwkv_skills:longbench_repobench_p", suite["benchmarks"]["longbench"]["lighteval_tasks"])
        self.assertEqual(suite["benchmarks"]["gsm8k"]["lighteval_tasks"], ["gsm8k"])
        self.assertEqual(
            suite["benchmarks"]["gaokao2023en"]["lighteval_tasks"],
            ["rwkv_skills:gaokao2023en"],
        )
        self.assertEqual(
            suite["benchmarks"]["supergpqa"]["lighteval_tasks"],
            ["rwkv_skills:supergpqa"],
        )
        self.assertEqual(
            suite["benchmarks"]["answer_judge"]["lighteval_tasks"],
            ["rwkv_skills:answer_judge"],
        )
        self.assertEqual(
            suite["benchmarks"]["omni_math"]["lighteval_tasks"],
            ["rwkv_skills:omni_math"],
        )
        self.assertEqual(
            suite["benchmarks"]["college_math"]["lighteval_tasks"],
            ["rwkv_skills:college_math"],
        )
        self.assertEqual(
            suite["benchmarks"]["comp_math_24_25"]["lighteval_tasks"],
            ["rwkv_skills:comp_math_24_25"],
        )
        self.assertEqual(suite["benchmarks"]["mawps"]["lighteval_tasks"], ["rwkv_skills:mawps"])
        self.assertEqual(suite["benchmarks"]["svamp"]["lighteval_tasks"], ["rwkv_skills:svamp"])
        self.assertEqual(len(suite["benchmarks"]["polymath"]["lighteval_tasks"]), 18)
        self.assertIn("rwkv_skills:polymath_zh", suite["benchmarks"]["polymath"]["lighteval_tasks"])
        self.assertEqual(len(suite["benchmarks"]["mmmlu"]["lighteval_tasks"]), 14)
        self.assertIn("rwkv_skills:mmmlu_zh", suite["benchmarks"]["mmmlu"]["lighteval_tasks"])
        self.assertEqual(len(suite["benchmarks"]["wmt24pp"]["lighteval_tasks"]), 5)
        self.assertIn("rwkv_skills:wmt24pp_ja_JP", suite["benchmarks"]["wmt24pp"]["lighteval_tasks"])

    def test_lighteval_suite_requires_explicit_mapped_only_for_partial_mapping(self) -> None:
        loaded_config = load_example_config()

        with self.assertRaises(SystemExit) as raised:
            commands.build_lighteval_suite_plan(
                lighteval_suite_args(benchmark=["gsm8k", "mcp_bench"]),
                root=ROOT,
                env={},
                config=loaded_config,
            )

        self.assertIn("without direct LightEval tasks", str(raised.exception))

    def test_lighteval_suite_mapped_only_uses_0_4b_endpoint_model(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_lighteval_suite_plan(
            lighteval_suite_args(mapped_only=True, benchmark=["gsm8k", "aime24"], max_samples=2),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        self.assertEqual(plan.command[1:5], ["-m", "lighteval", "endpoint", "litellm"])
        self.assertIn("model_name=openai/g1d-0.4b", plan.command[5])
        self.assertEqual(set(plan.command[6].split(",")), {"gsm8k", "aime24"})
        self.assertEqual(command_options(plan.command)["--max-samples"], "2")

    def test_lighteval_suite_uses_suite_custom_tasks(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_lighteval_suite_plan(
            lighteval_suite_args(mapped_only=True, benchmark=["gaokao2023en"], max_samples=1),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        self.assertEqual(plan.command[6], "rwkv_skills:gaokao2023en")
        self.assertEqual(
            command_options(plan.command)["--custom-tasks"],
            str(ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py"),
        )

    def test_lighteval_suite_enables_multilingual_when_needed(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_lighteval_suite_plan(
            lighteval_suite_args(mapped_only=True, benchmark=["ceval"], max_samples=1),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        self.assertEqual(plan.command[6], "ceval_zho_mcf")
        self.assertIn("--load-tasks-multilingual", plan.command)

    def test_lighteval_suite_maps_mmmlu_to_generative_custom_tasks(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_lighteval_suite_plan(
            lighteval_suite_args(mapped_only=True, benchmark=["mmmlu"], max_samples=1),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        tasks = set(plan.command[6].split(","))
        self.assertEqual(len(tasks), 14)
        self.assertIn("rwkv_skills:mmmlu_ar", tasks)
        self.assertIn("rwkv_skills:mmmlu_zh", tasks)
        self.assertEqual(
            command_options(plan.command)["--custom-tasks"],
            str(ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py"),
        )

    def test_lighteval_multilingual_registry_loads_when_eval_deps_installed(self) -> None:
        if importlib.util.find_spec("lighteval") is None:
            self.skipTest("LightEval is not installed")

        self.assertIsNotNone(importlib.util.find_spec("language_data"))

        from lighteval.tasks.registry import Registry

        registry = Registry(load_multilingual=True)

        self.assertTrue(any(name.startswith("ceval_zho_mcf:") for name in registry._task_registry))

    def test_rwkv_skills_custom_lighteval_tasks_load_when_eval_deps_installed(self) -> None:
        if importlib.util.find_spec("lighteval") is None:
            self.skipTest("LightEval is not installed")

        from lighteval.tasks.registry import Registry

        custom_tasks = ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py"
        registry = Registry(custom_tasks=str(custom_tasks))

        self.assertIn("rwkv_skills:human_eval", registry._task_registry)
        self.assertIn("rwkv_skills:include", registry._task_registry)
        self.assertIn("rwkv_skills:human_eval_cn", registry._task_registry)
        self.assertIn("rwkv_skills:human_eval_fix", registry._task_registry)
        self.assertIn("rwkv_skills:human_eval_plus", registry._task_registry)
        self.assertIn("rwkv_skills:mbpp", registry._task_registry)
        self.assertIn("rwkv_skills:mbpp_plus", registry._task_registry)
        self.assertIn("rwkv_skills:longcodeqa", registry._task_registry)
        self.assertIn("rwkv_skills:longbench_narrativeqa", registry._task_registry)
        self.assertIn("rwkv_skills:longbench_repobench_p", registry._task_registry)
        self.assertIn("rwkv_skills:apibank_l1", registry._task_registry)
        self.assertIn("rwkv_skills:apibank_l2", registry._task_registry)
        self.assertIn("rwkv_skills:apibank_level1", registry._task_registry)
        self.assertIn("rwkv_skills:apibank_level2", registry._task_registry)
        self.assertIn("rwkv_skills:toolalpaca_eval_real", registry._task_registry)
        self.assertIn("rwkv_skills:toolalpaca_eval_simulated", registry._task_registry)
        self.assertIn("rwkv_skills:complexfuncbench_official", registry._task_registry)
        self.assertIn("rwkv_skills:complexfuncbench_subset", registry._task_registry)
        self.assertIn("rwkv_skills:bfcl_v3", registry._task_registry)
        self.assertIn("rwkv_skills:browsecomp", registry._task_registry)
        self.assertIn("rwkv_skills:browsecomp_plus", registry._task_registry)
        self.assertIn("rwkv_skills:browsecomp_zh", registry._task_registry)
        self.assertIn("rwkv_skills:bfcl_simple_python", registry._task_registry)
        self.assertIn("rwkv_skills:bfcl_multiple", registry._task_registry)
        self.assertIn("rwkv_skills:bfcl_exec_simple_ast", registry._task_registry)
        self.assertIn("rwkv_skills:bfcl_exec_multiple_ast", registry._task_registry)
        self.assertIn("rwkv_skills:bfcl_exec_simple", registry._task_registry)
        self.assertIn("rwkv_skills:bfcl_exec_multiple", registry._task_registry)
        self.assertIn("rwkv_skills:bfcl_exec_parallel", registry._task_registry)
        self.assertIn("rwkv_skills:bfcl_exec_parallel_multiple", registry._task_registry)
        self.assertIn("rwkv_skills:algebra222", registry._task_registry)
        self.assertIn("rwkv_skills:gaokao2023en", registry._task_registry)
        self.assertIn("rwkv_skills:amc23", registry._task_registry)
        self.assertIn("rwkv_skills:beyond_aime", registry._task_registry)
        self.assertIn("rwkv_skills:brumo25", registry._task_registry)
        self.assertIn("rwkv_skills:hmmt_feb25", registry._task_registry)
        self.assertIn("rwkv_skills:answer_judge", registry._task_registry)
        self.assertIn("rwkv_skills:math_odyssey", registry._task_registry)
        self.assertIn("rwkv_skills:mawps", registry._task_registry)
        self.assertIn("rwkv_skills:omni_math", registry._task_registry)
        self.assertIn("rwkv_skills:college_math", registry._task_registry)
        self.assertIn("rwkv_skills:comp_math_24_25", registry._task_registry)
        self.assertIn("rwkv_skills:svamp", registry._task_registry)
        self.assertIn("rwkv_skills:polymath_en", registry._task_registry)
        self.assertIn("rwkv_skills:polymath_zh", registry._task_registry)
        self.assertIn("rwkv_skills:supergpqa", registry._task_registry)
        self.assertIn("rwkv_skills:mmmlu_ar", registry._task_registry)
        self.assertIn("rwkv_skills:mmmlu_zh", registry._task_registry)
        self.assertIn("rwkv_skills:wmt24pp_de_DE", registry._task_registry)
        self.assertIn("rwkv_skills:wmt24pp_ja_JP", registry._task_registry)

    def test_rwkv_skills_multiple_choice_metric_extracts_answer_letters(self) -> None:
        if importlib.util.find_spec("lighteval") is None:
            self.skipTest("LightEval is not installed")

        custom_tasks = ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py"
        spec = importlib.util.spec_from_file_location("rwkv_skills_lighteval_tasks", custom_tasks)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertEqual(module._extract_choice_letter("答案: B", max_choices=4), "B")
        self.assertEqual(module._extract_choice_letter("Answer: C", max_choices=4), "C")
        self.assertEqual(module._extract_choice_letter("D. No", max_choices=4), "D")
        self.assertIsNone(module._extract_choice_letter("Option E", max_choices=4))
        self.assertEqual(module._extract_judgement("Judgement: Yes"), "yes")
        self.assertEqual(module._extract_judgement("No, the answer does not match."), "no")

    def test_rwkv_skills_longcodeqa_prompt_budgets_context_and_scores_letter(self) -> None:
        if importlib.util.find_spec("lighteval") is None:
            self.skipTest("LightEval is not installed")

        custom_tasks = ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py"
        spec = importlib.util.spec_from_file_location("rwkv_skills_lighteval_tasks", custom_tasks)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        repo_text = "irrelevant implementation detail\n" * 400
        repo_text += "\ndef target_status():\n    return 'deprecated'\n"
        question = (
            "Question:\n"
            "What is the current status returned by `target_status`?\n"
            "A) active\n"
            "B) deprecated\n"
            "C) experimental\n"
            "D) unknown\n"
        )
        with mock.patch.dict(os.environ, {"RWKV_LIGHTEVAL_LONGCODEQA_MAX_PROMPT_CHARS": "2200"}):
            doc = module.longcodeqa_prompt(
                {
                    "repo": "demo/repo",
                    "repo_text": repo_text,
                    "question": question,
                    "prompt_goal": "Use the repository context to answer.",
                    "correct_letter": "B",
                },
                "rwkv_skills:longcodeqa",
            )

        self.assertLessEqual(len(doc.query), 2200)
        self.assertIn("target_status", doc.query)
        self.assertEqual(doc.choices, [" A", " B", " C", " D"])
        self.assertEqual(doc.gold_index, 1)
        self.assertEqual(
            module.extract_longcodeqa_answer('{"name":"final_answer","arguments":{"answer":"B"}}', allowed_letters=("A", "B", "C", "D")),
            "B",
        )
        response = type("Response", (), {"final_text": ['{"arguments":{"answer":"B"}}']})()
        self.assertEqual(module.LongCodeQALetterMatch().compute(doc, response), 1.0)

    def test_rwkv_skills_longbench_prompt_budgets_context_and_scores_f1(self) -> None:
        if importlib.util.find_spec("lighteval") is None:
            self.skipTest("LightEval is not installed")

        custom_tasks = ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py"
        spec = importlib.util.spec_from_file_location("rwkv_skills_lighteval_tasks", custom_tasks)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        context = "noise\n" * 500
        context += "\nThe target project status is deprecated and stable.\n"
        with mock.patch.dict(os.environ, {"RWKV_LIGHTEVAL_LONGCODEQA_MAX_PROMPT_CHARS": "2200"}):
            doc = module.longbench_prompt(
                {
                    "_id": "row-1",
                    "dataset": "narrativeqa",
                    "input": "What is the target project status?",
                    "context": context,
                    "answers": ["deprecated and stable"],
                    "all_classes": [],
                    "language": "en",
                    "length": len(context),
                },
                "rwkv_skills:longbench_narrativeqa",
            )

        self.assertLessEqual(len(doc.query), 2200)
        self.assertIn("target project status", doc.query)
        response = type("Response", (), {"final_text": ['{"answer":"deprecated and stable"}']})()
        self.assertEqual(module.LongBenchExactMatch().compute(doc, response), 1.0)
        self.assertEqual(module.LongBenchF1().compute(doc, response), 1.0)

    def test_rwkv_skills_bfcl_ast_prompt_loads_and_scores_with_official_checker(self) -> None:
        if importlib.util.find_spec("lighteval") is None:
            self.skipTest("LightEval is not installed")

        custom_tasks = ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py"
        spec = importlib.util.spec_from_file_location("rwkv_skills_lighteval_tasks", custom_tasks)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for dataset_name in (
            "bfcl_simple_python",
            "bfcl_multiple",
            "bfcl_exec_simple_ast",
            "bfcl_exec_multiple_ast",
        ):
            rows = module._load_bfcl_ast_rows(dataset_name)
            self.assertGreater(len(rows), 0)
            doc = module.bfcl_ast_prompt(rows[0], f"rwkv_skills:{dataset_name}")
            self.assertIn("Output JSON schema", doc.query)
            self.assertIn("\n\nJSON:", doc.query)
            response = type(
                "Response",
                (),
                {"final_text": [module.json.dumps(rows[0]["expected_tool_calls"], ensure_ascii=False)]},
            )()
            self.assertEqual(module.BfclAstAccuracy().compute(doc, response), 1.0, dataset_name)

    def test_rwkv_skills_bfcl_exec_prompt_loads_and_scores_by_execution(self) -> None:
        if importlib.util.find_spec("lighteval") is None:
            self.skipTest("LightEval is not installed")

        custom_tasks = ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py"
        spec = importlib.util.spec_from_file_location("rwkv_skills_lighteval_tasks", custom_tasks)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for dataset_name in (
            "bfcl_exec_simple",
            "bfcl_exec_multiple",
            "bfcl_exec_parallel",
            "bfcl_exec_parallel_multiple",
        ):
            rows = module._load_bfcl_exec_rows(dataset_name)
            self.assertGreater(len(rows), 0)
            doc = module.bfcl_exec_prompt(rows[0], f"rwkv_skills:{dataset_name}")
            self.assertIn("execution results", doc.query)
            self.assertIn("\n\nJSON:", doc.query)
            response = type(
                "Response",
                (),
                {"final_text": [module.json.dumps(rows[0]["expected_tool_calls"], ensure_ascii=False)]},
            )()
            self.assertEqual(module.BfclExecAccuracy().compute(doc, response), 1.0, dataset_name)

    def test_rwkv_skills_apibank_prompt_loads_and_scores_with_official_sandbox(self) -> None:
        if importlib.util.find_spec("lighteval") is None:
            self.skipTest("LightEval is not installed")

        custom_tasks = ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py"
        spec = importlib.util.spec_from_file_location("rwkv_skills_lighteval_tasks", custom_tasks)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for dataset_name in ("apibank_l1", "apibank_l2", "apibank_level1", "apibank_level2"):
            rows = module._load_apibank_rows(dataset_name)
            self.assertGreater(len(rows), 0)
            doc = module.apibank_prompt(rows[0], f"rwkv_skills:{dataset_name}")
            self.assertIn("API-Bank date convention", doc.query)
            self.assertIn("\n\nJSON:", doc.query)
            response = type(
                "Response",
                (),
                {"final_text": [module.json.dumps(rows[0]["expected_tool_calls"], ensure_ascii=False)]},
            )()
            self.assertEqual(module.ApiBankAccuracy().compute(doc, response), 1.0, dataset_name)

    def test_rwkv_skills_toolalpaca_prompt_loads_and_scores_request_structure(self) -> None:
        if importlib.util.find_spec("lighteval") is None:
            self.skipTest("LightEval is not installed")

        custom_tasks = ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py"
        spec = importlib.util.spec_from_file_location("rwkv_skills_lighteval_tasks", custom_tasks)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for dataset_name in ("toolalpaca_eval_real", "toolalpaca_eval_simulated"):
            rows = module._load_toolalpaca_rows(dataset_name)
            self.assertGreater(len(rows), 0)
            doc = module.toolalpaca_prompt(rows[0], f"rwkv_skills:{dataset_name}")
            self.assertIn("For multiple required tool calls", doc.query)
            self.assertIn("\n\nJSON:", doc.query)
            response = type(
                "Response",
                (),
                {"final_text": [module.json.dumps(rows[0]["expected_tool_calls"], ensure_ascii=False)]},
            )()
            self.assertEqual(module.ToolAlpacaAccuracy().compute(doc, response), 1.0, dataset_name)

    def test_rwkv_skills_complexfuncbench_prompt_loads_and_scores_golden_turns(self) -> None:
        if importlib.util.find_spec("lighteval") is None:
            self.skipTest("LightEval is not installed")

        custom_tasks = ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py"
        spec = importlib.util.spec_from_file_location("rwkv_skills_lighteval_tasks", custom_tasks)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for dataset_name in ("complexfuncbench_official", "complexfuncbench_subset"):
            rows = module._load_complexfuncbench_rows(dataset_name)
            self.assertGreater(len(rows), 0)
            doc = module.complexfuncbench_prompt(rows[0], f"rwkv_skills:{dataset_name}")
            self.assertIn("ComplexFuncBench", doc.query)
            self.assertIn("tool_turns", doc.query)
            self.assertIn("\n\nJSON:", doc.query)
            response = type(
                "Response",
                (),
                {"final_text": [module.json.dumps({"tool_turns": rows[0]["expected_tool_turns"]}, ensure_ascii=False)]},
            )()
            self.assertEqual(module.ComplexFuncBenchSuccessRate().compute(doc, response), 1.0, dataset_name)
            self.assertEqual(module.ComplexFuncBenchCallAccuracy().compute(doc, response), 1.0, dataset_name)

    def test_rwkv_skills_bfcl_v3_prompt_loads_and_scores_golden_turns(self) -> None:
        if importlib.util.find_spec("lighteval") is None:
            self.skipTest("LightEval is not installed")

        custom_tasks = ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py"
        spec = importlib.util.spec_from_file_location("rwkv_skills_lighteval_tasks", custom_tasks)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        rows = module._load_bfcl_v3_rows("bfcl_v3")
        self.assertGreater(len(rows), 0)
        self.assertTrue(any(row["metadata"]["category"] == "multi_turn_miss_func" for row in rows))
        doc = module.bfcl_v3_prompt(rows[0], "rwkv_skills:bfcl_v3")
        self.assertIn("BFCL v3", doc.query)
        self.assertIn("tool_turns", doc.query)
        self.assertIn("\n\nJSON:", doc.query)
        response = type(
            "Response",
            (),
            {"final_text": [module.json.dumps({"tool_turns": rows[0]["expected_tool_turns"]}, ensure_ascii=False)]},
        )()
        self.assertEqual(module.BfclV3SuccessRate().compute(doc, response), 1.0)
        self.assertEqual(module.BfclV3CallAccuracy().compute(doc, response), 1.0)

    def test_rwkv_skills_browsecomp_prompt_loads_and_scores_golden_answers(self) -> None:
        if importlib.util.find_spec("lighteval") is None:
            self.skipTest("LightEval is not installed")

        custom_tasks = ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py"
        spec = importlib.util.spec_from_file_location("rwkv_skills_lighteval_tasks", custom_tasks)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for dataset_name in ("browsecomp", "browsecomp_plus", "browsecomp_zh"):
            rows = module._load_browsecomp_rows(dataset_name)
            self.assertGreater(len(rows), 0)
            doc = module.browsecomp_prompt(rows[0], f"rwkv_skills:{dataset_name}")
            self.assertIn("BrowseComp", doc.query)
            self.assertIn(rows[0]["question"][:32], doc.query)
            answer_text = (
                f"最终答案: {rows[0]['answer']}"
                if dataset_name == "browsecomp_zh"
                else f"Exact Answer: {rows[0]['answer']}"
            )
            response = type(
                "Response",
                (),
                {"final_text": [answer_text]},
            )()
            self.assertEqual(module.BrowseCompExactMatch().compute(doc, response), 1.0, dataset_name)
            self.assertEqual(module.BrowseCompContainsMatch().compute(doc, response), 1.0, dataset_name)
            self.assertEqual(module.BrowseCompF1().compute(doc, response), 1.0, dataset_name)

    def test_rwkv_skills_svamp_prompt_combines_body_question_and_normalizes_answer(self) -> None:
        if importlib.util.find_spec("lighteval") is None:
            self.skipTest("LightEval is not installed")

        custom_tasks = ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py"
        spec = importlib.util.spec_from_file_location("rwkv_skills_lighteval_tasks", custom_tasks)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        doc = module.svamp_prompt(
            {
                "ID": "chal-1",
                "Body": "Each pack costs 76 dollars",
                "Question": "What is the price after a 25 dollar discount?",
                "Equation": "( 76.0 - 25.0 )",
                "Answer": 51.0,
                "Type": "Subtraction",
            },
            "rwkv_skills:svamp",
        )

        self.assertIn("Each pack costs 76 dollars. What is the price", doc.query)
        self.assertEqual(doc.choices, ["51"])
        self.assertEqual(doc.specific["id"], "chal-1")

    def test_rwkv_skills_wmt24pp_prompt_uses_translation_target(self) -> None:
        if importlib.util.find_spec("lighteval") is None:
            self.skipTest("LightEval is not installed")

        custom_tasks = ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py"
        spec = importlib.util.spec_from_file_location("rwkv_skills_lighteval_tasks", custom_tasks)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        doc = module.wmt24pp_prompt(
            {
                "lp": "en-ja_JP",
                "domain": "news",
                "document_id": "doc-1",
                "segment_id": 7,
                "is_bad_source": False,
                "source": "Good morning",
                "target": "ohayo gozaimasu",
            },
            "rwkv_skills:wmt24pp_ja_JP",
        )

        self.assertEqual(doc.query, "English phrase: Good morning\nJapanese phrase:")
        self.assertEqual(doc.choices, ["ohayo gozaimasu"])
        self.assertIn("Translate English to Japanese", doc.instruction)
        self.assertEqual(doc.specific["segment_id"], 7)

    def test_rwkv_skills_human_eval_prompt_and_executor(self) -> None:
        if importlib.util.find_spec("lighteval") is None:
            self.skipTest("LightEval is not installed")

        custom_tasks = ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py"
        spec = importlib.util.spec_from_file_location("rwkv_skills_lighteval_tasks", custom_tasks)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        problem = {
            "task_id": "HumanEval/test",
            "prompt": "def add(a: int, b: int) -> int:\n",
            "canonical_solution": "    return a + b\n",
            "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
            "entry_point": "add",
        }
        doc = module.human_eval_prompt(problem, "rwkv_skills:human_eval")

        self.assertIn("```python\ndef add", doc.query)
        self.assertEqual(doc.choices, ["    return a + b\n"])
        self.assertEqual(doc.specific["entry_point"], "add")
        self.assertEqual(
            module.extract_code_completion("<think>plan</think>\n```python\n    return a + b\n```"),
            "    return a + b",
        )
        self.assertTrue(module._check_humaneval_correctness(problem, "    return a + b", timeout=1.0)["passed"])
        self.assertTrue(
            module._check_humaneval_correctness(
                problem,
                "```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```",
                timeout=1.0,
            )["passed"]
        )
        self.assertFalse(module._check_humaneval_correctness(problem, "    return a - b", timeout=1.0)["passed"])

    def test_rwkv_skills_human_eval_cn_data_and_fix_prompt(self) -> None:
        if importlib.util.find_spec("lighteval") is None:
            self.skipTest("LightEval is not installed")

        custom_tasks = ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py"
        spec = importlib.util.spec_from_file_location("rwkv_skills_lighteval_tasks", custom_tasks)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        cn_data_file = ROOT / "benchmarks/lighteval_data/human_eval_cn/test.jsonl"
        self.assertTrue(cn_data_file.exists())
        with cn_data_file.open("r", encoding="utf-8") as handle:
            first = handle.readline()
            self.assertEqual(1 + sum(1 for line in handle if line.strip()), 164)

        import json

        cn_row = json.loads(first)
        cn_doc = module.human_eval_prompt(cn_row, "rwkv_skills:human_eval_cn")
        self.assertEqual(cn_doc.specific["entry_point"], "has_close_elements")
        self.assertTrue(
            module._check_humaneval_correctness(
                cn_doc.specific,
                cn_row["canonical_solution"],
                timeout=1.0,
            )["passed"]
        )

        fix_row = {
            "task_id": "Python/fixture",
            "prompt": "def foo(x):\n    \"\"\"Return x plus one.\"\"\"\n",
            "buggy_solution": "    return x - 1\n",
            "canonical_solution": "    return x + 1\n",
            "entry_point": "foo",
            "test": "def check(candidate):\n    assert candidate(1) == 2\ncheck(foo)",
            "bug_type": "missing logic",
            "failure_symptoms": "incorrect output",
        }
        fix_doc = module.human_eval_fix_prompt(fix_row, "rwkv_skills:human_eval_fix")
        self.assertIn("Buggy implementation", fix_doc.query)
        self.assertEqual(fix_doc.specific["prompt"], fix_row["prompt"])
        self.assertFalse(
            module._check_humaneval_correctness(
                fix_doc.specific,
                fix_row["buggy_solution"],
                timeout=1.0,
            )["passed"]
        )
        self.assertTrue(
            module._check_humaneval_correctness(
                fix_doc.specific,
                fix_row["canonical_solution"],
                timeout=1.0,
            )["passed"]
        )

    def test_rwkv_skills_mbpp_prompt_and_base_plus_executors(self) -> None:
        if importlib.util.find_spec("lighteval") is None:
            self.skipTest("LightEval is not installed")

        custom_tasks = ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py"
        spec = importlib.util.spec_from_file_location("rwkv_skills_lighteval_tasks", custom_tasks)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        problem = {
            "task_id": "2",
            "prompt": "Write a function named inc that increments an integer.",
            "code": "def inc(x):\n    return x + 1\n",
            "source_file": "unit",
            "test_imports": [],
            "test_list": ["assert inc(1) == 2"],
            "test": "assert inc(1) == 2\nassert inc(2) == 3\n",
        }
        doc = module.mbpp_prompt(problem, "rwkv_skills:mbpp")

        self.assertIn("Write a Python function", doc.query)
        self.assertEqual(doc.choices, ["def inc(x):\n    return x + 1\n"])
        self.assertEqual(doc.specific["task_id"], "Mbpp/2")
        public_only = "```python\ndef inc(x):\n    return 2\n```"
        correct = "def inc(x):\n    return x + 1\n"
        self.assertTrue(
            module._check_mbpp_correctness(doc.specific, public_only, include_plus_tests=False, timeout=1.0)["passed"]
        )
        self.assertFalse(
            module._check_mbpp_correctness(doc.specific, public_only, include_plus_tests=True, timeout=1.0)["passed"]
        )
        self.assertTrue(
            module._check_mbpp_correctness(doc.specific, correct, include_plus_tests=True, timeout=1.0)["passed"]
        )

    def test_rwkv_skills_comp_math_static_data_is_packaged(self) -> None:
        data_file = ROOT / "benchmarks/lighteval_data/comp_math_24_25/test.jsonl"

        self.assertTrue(data_file.exists())
        with data_file.open("r", encoding="utf-8") as handle:
            self.assertEqual(sum(1 for line in handle if line.strip()), 256)

    def test_rwkv_skills_mawps_static_data_loads_as_test_split(self) -> None:
        if importlib.util.find_spec("datasets") is None:
            self.skipTest("datasets is not installed")

        from datasets import load_dataset

        data_dir = ROOT / "benchmarks/lighteval_data/mawps"
        dataset = load_dataset(str(data_dir))

        self.assertEqual(len(dataset["test"]), 2065)
        self.assertEqual(set(dataset["test"].features), {"input", "target"})

    def test_rwkv_skills_include_static_data_loads_as_test_split(self) -> None:
        if importlib.util.find_spec("datasets") is None:
            self.skipTest("datasets is not installed")

        from datasets import load_dataset

        data_dir = ROOT / "benchmarks/lighteval_data/include"
        dataset = load_dataset(str(data_dir))

        self.assertEqual(len(dataset["test"]), 22639)
        self.assertEqual(
            {"question", "answer", "A", "B", "C", "D", "subset", "source"} - set(dataset["test"].features),
            set(),
        )

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


if __name__ == "__main__":
    unittest.main()
