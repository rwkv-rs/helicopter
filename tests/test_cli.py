from __future__ import annotations

import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from helicopter_cli import __main__ as cli_main
from helicopter_cli import commands, config, env, eval_catalog
from helicopter_eval import (
    agentbench,
    apibank,
    arena_hard,
    bfcl_ast,
    bfcl_exec,
    browsecomp_plus,
    browsecomp,
    catalog_runner,
    code_generation,
    complexfuncbench,
    free_response,
    gsm8k,
    instruction_following,
    longbench,
    longcodeqa,
    mcp_bench,
    multiple_choice,
    toolalpaca,
    translation,
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


def eval_infer_args(**overrides: object) -> Namespace:
    values = {
        "dry_run": True,
        "model_path": None,
        "served_model_name": None,
        "host": None,
        "port": None,
        "wkv_mode": None,
        "emb_device": None,
        "tensor_parallel_size": None,
        "gpu_memory_utilization": None,
        "max_model_len": None,
        "max_num_seqs": None,
        "max_num_batched_tokens": None,
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
        self.assertEqual(plan.shown_env, {"PYTHONPATH": str(ROOT / "src/infer/vllm-rwkv")})
        self.assertEqual(plan.env["PYTHONPATH"], str(ROOT / "src/infer/vllm-rwkv"))
        self.assertEqual({key for key in plan.env if key.startswith("VLLM_")}, set())

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
        self.assertEqual(plan.env["PYTHONPATH"], str(ROOT / "src/infer/vllm-rwkv"))
        self.assertEqual(forbidden_env_keys & plan.env.keys(), set())
        self.assertEqual(forbidden_option_keys & options.keys(), set())

    def test_eval_catalog_matches_rwkv_skills_registry_shape(self) -> None:
        catalog = eval_catalog.load_rwkv_skills_catalog()
        plan = catalog.build_job_plan(names=("all",), fields=None)

        self.assertEqual(len(catalog.benchmarks), 95)
        self.assertEqual(len(catalog.runners), 36)
        self.assertEqual(
            catalog.field_counts(),
            {
                "coding": 12,
                "function_calling": 42,
                "instruction_following": 5,
                "knowledge": 11,
                "maths": 25,
            },
        )
        self.assertEqual(catalog.inference_defaults.engine, "vllm-rwkv")
        self.assertEqual(catalog.inference_defaults.protocol, "vllm")
        self.assertEqual(
            catalog.inference_defaults.model_path,
            "/home/chase/GitHub/rwkv-rs/pth/rwkv7-g1d-0.4b-20260210-ctx8192.pth",
        )
        self.assertGreater(len([row for row in plan if row.status == "ready"]), 0)
        self.assertEqual([row.status for row in plan if row.benchmark == "arena_hard_v2"], ["no_scheduler_job_in_rwkv_skills"])

    def test_eval_infer_plan_uses_local_04_vllm_rwkv_defaults(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_eval_infer_plan(
            eval_infer_args(),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        self.assertEqual(
            plan.command[:3],
            [
                "vllm",
                "serve",
                "/home/chase/GitHub/rwkv-rs/pth/rwkv7-g1d-0.4b-20260210-ctx8192.pth",
            ],
        )
        self.assertEqual(
            command_options(plan.command),
            {
                "--host": "127.0.0.1",
                "--port": "29082",
                "--tokenizer-mode": "rwkv",
                "--load-format": "auto",
                "--served-model-name": "rwkv7-g1d-0.4b-20260210-ctx8192",
                "--max-model-len": "8192",
                "--tensor-parallel-size": "1",
            },
        )
        self.assertEqual(
            plan.shown_env,
            {
                "PYTHONPATH": str(ROOT / "src/infer/vllm-rwkv"),
                "VLLM_RWKV7_EMB_DEVICE": "gpu",
                "VLLM_RWKV7_WKV_MODE": "fp16",
            },
        )

    def test_gsm8k_answer_normalization_matches_final_number(self) -> None:
        self.assertEqual(gsm8k.reference_answer_from_gsm8k("reasoning\n#### 1,234.00"), "1234")
        self.assertEqual(gsm8k.completion_answer("The answer is 17."), "17")
        self.assertEqual(gsm8k.completion_answer("work #### -42"), "-42")

    def test_gsm8k_dry_run_summary_uses_vllm_defaults(self) -> None:
        run_config = gsm8k.Gsm8kRunConfig(
            base_url="http://127.0.0.1:29082",
            model="rwkv7-g1d-0.4b-20260210-ctx8192",
            limit=3,
        )

        self.assertEqual(
            gsm8k.dry_run_summary(run_config),
            {
                "benchmark": "gsm8k",
                "hf_dataset": "openai/gsm8k",
                "hf_config": "main",
                "split": "test",
                "limit": 3,
                "base_url": "http://127.0.0.1:29082",
                "model": "rwkv7-g1d-0.4b-20260210-ctx8192",
                "scoreboard_dataset": "gsm8k_test_limit3",
                "job_name": "free_response_judge",
                "job_id": "helicopter-gsm8k",
            },
        )

    def test_gsm8k_full_run_keeps_formal_dataset_name(self) -> None:
        run_config = gsm8k.Gsm8kRunConfig(
            base_url="http://127.0.0.1:29082",
            model="rwkv7-g1d-0.4b-20260210-ctx8192",
            limit=None,
        )

        self.assertEqual(gsm8k.scoreboard_dataset_name(run_config), "gsm8k_test")

    def test_generic_free_response_dry_run_uses_hf_dataset_config(self) -> None:
        args = Namespace(
            dry_run=True,
            benchmark="toy_math",
            dataset="org/toy_math",
            dataset_config="main",
            question_field="prompt",
            answer_field="target",
            answer_marker="####",
            base_url=None,
            model=None,
            limit=5,
            split="validation",
            temperature=0.0,
            top_p=1.0,
            max_tokens=64,
            timeout_s=30.0,
            job_name="free_response_judge",
            job_id=None,
        )

        with mock.patch.object(cli_main, "print_json") as print_json:
            rc = cli_main.handle_eval_run_free_response(args, root=ROOT)

        self.assertEqual(rc, 0)
        self.assertEqual(
            print_json.call_args.args[0],
            {
                "benchmark": "toy_math",
                "hf_dataset": "org/toy_math",
                "hf_config": "main",
                "split": "validation",
                "limit": 5,
                "base_url": "http://127.0.0.1:29082",
                "model": "rwkv7-g1d-0.4b-20260210-ctx8192",
                "scoreboard_dataset": "toy_math_validation_limit5",
                "job_name": "free_response_judge",
                "job_id": "helicopter-toy_math",
            },
        )

    def test_free_response_completion_sampling_matches_scoreboard_task_config(self) -> None:
        run_config = free_response.FreeResponseRunConfig(
            base_url="http://127.0.0.1:29082",
            model="rwkv7-g1d-0.4b-20260210-ctx8192",
            benchmark="toy_math",
            dataset_name="org/toy_math",
            dataset_config="main",
            question_field="prompt",
            answer_field="target",
            limit=1,
            max_tokens=64,
        )

        self.assertEqual(
            free_response.task_sampling_config(run_config),
            {
                "avg_k": 1,
                "pass_ks": [1],
                "prompt_profile": "helicopter",
                "sampling_config": {"answer": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 64}},
            },
        )

    def test_instruction_following_scores_ifeval_rule(self) -> None:
        sample = instruction_following.InstructionFollowingSample(
            sample_index=0,
            key=1000,
            prompt="Answer without commas.",
            instruction_ids=("punctuation:no_comma",),
            kwargs_list=({},),
        )
        config = instruction_following.InstructionFollowingRunConfig(
            base_url="http://127.0.0.1:29082",
            model="rwkv7-g1d-0.4b-20260210-ctx8192",
            benchmark="ifeval",
            dataset_name="google-research/instruction_following_eval",
            source_url="https://example.invalid/input_data.jsonl",
        )

        self.assertEqual(instruction_following.score_response(sample, "No comma here", config)[:3], (True, 1, 1))
        self.assertEqual(instruction_following.score_response(sample, "No, comma here", config)[:3], (False, 0, 1))

    def test_catalog_runner_marks_direct_hf_specs(self) -> None:
        catalog = eval_catalog.load_rwkv_skills_catalog()
        specs = {
            name: catalog_runner.resolve_catalog_run_spec(catalog.benchmarks_by_name[name])
            for name in (
                "gsm8k",
                "mmlu",
                "mmlu_pro",
                "ceval",
                "cmmlu",
                "gpqa_main",
                "mmmlu",
                "supergpqa",
                "ifeval",
                "ifbench",
                "flores200",
                "wmt24pp",
                "aime24",
                "algebra222",
                "hendrycks_math",
                "hle",
                "math_500",
                "polymath",
                "human_eval",
                "human_eval_plus",
                "mbpp",
                "livecodebench",
                "longcodeqa",
                "longbench",
                "longbench_qa",
                "longbench_qa_balanced",
                "agentbench_db",
                "agentbench_kg",
                "mcp_bench",
                "mcp_bench_single",
                "mcp_bench_multi_2server",
                "mcp_bench_multi_3server",
                "browsecomp",
                "browsecomp_zh",
                "apibank_l1",
                "apibank_level2",
                "bfcl_simple_python",
                "bfcl_exec_multiple_ast",
                "bfcl_exec_simple",
                "bfcl_exec_parallel_multiple",
                "toolalpaca_eval_simulated",
                "toolalpaca_eval_real",
                "complexfuncbench_official",
                "complexfuncbench_subset",
            )
        }

        self.assertEqual(specs["gsm8k"].status, "implemented")
        self.assertEqual(specs["gsm8k"].dataset_name, "openai/gsm8k")
        self.assertEqual(specs["mmlu"].dataset_config, "all")
        self.assertEqual(specs["ceval"].dataset_config, "*")
        self.assertEqual(specs["ceval"].choice_fields, ("A", "B", "C", "D"))
        self.assertEqual(specs["cmmlu"].source_type, "cmmlu_zip")
        self.assertEqual(specs["gpqa_main"].row_adapter, "gpqa")
        self.assertEqual(specs["mmmlu"].source_type, "mmmlu")
        self.assertEqual(specs["supergpqa"].source_split, "train")
        self.assertEqual(specs["ifeval"].kind, "instruction_following")
        self.assertTrue(specs["ifeval"].strict)
        self.assertFalse(specs["ifbench"].strict)
        self.assertEqual(specs["flores200"].kind, "translation")
        self.assertEqual(specs["flores200"].status, "needs_dataset_access")
        self.assertEqual(specs["flores200"].source_type, "hf_flores200")
        self.assertEqual(specs["wmt24pp"].source_type, "hf_wmt24pp")
        self.assertEqual(specs["aime24"].source_type, "package_jsonl")
        self.assertEqual(specs["algebra222"].source_type, "url_csv")
        self.assertEqual(specs["hendrycks_math"].source_type, "qwen_math")
        self.assertEqual(specs["hendrycks_math"].dataset_name, "math")
        self.assertEqual(specs["hle"].status, "needs_dataset_access")
        self.assertEqual(specs["math_500"].source_type, "url_jsonl")
        self.assertEqual(specs["math_500"].row_adapter, "answer_solution")
        self.assertEqual(specs["polymath"].source_type, "polymath")
        self.assertEqual(specs["human_eval"].status, "implemented")
        self.assertEqual(specs["human_eval"].kind, "code_generation")
        self.assertEqual(specs["human_eval"].source_type, "human_eval_url_gzip")
        self.assertEqual(specs["human_eval_plus"].source_type, "human_eval_plus_evalplus")
        self.assertEqual(specs["mbpp"].source_type, "mbpp_evalplus")
        self.assertEqual(specs["livecodebench"].source_type, "livecodebench_hf")
        self.assertEqual(specs["livecodebench"].job_name, "code_livecodebench")
        self.assertEqual(specs["longcodeqa"].kind, "longcodeqa")
        self.assertEqual(specs["longcodeqa"].source_type, "hf_zip")
        self.assertEqual(specs["longbench"].kind, "longbench")
        self.assertEqual(specs["longbench_qa"].row_adapter, "longbench_qa")
        self.assertEqual(specs["longbench_qa_balanced"].row_adapter, "longbench_qa_balanced")
        self.assertEqual(specs["agentbench_db"].kind, "agentbench")
        self.assertEqual(specs["agentbench_db"].source_type, "agentbench_official")
        self.assertEqual(specs["agentbench_kg"].dataset_name, "agentbench_kg")
        self.assertEqual(specs["mcp_bench"].kind, "mcp_bench")
        self.assertEqual(specs["mcp_bench"].source_type, "mcp_bench_official")
        self.assertEqual(specs["mcp_bench_single"].dataset_name, "mcp_bench_single")
        self.assertEqual(specs["mcp_bench_multi_2server"].dataset_name, "mcp_bench_multi_2server")
        self.assertEqual(specs["mcp_bench_multi_3server"].dataset_name, "mcp_bench_multi_3server")
        self.assertEqual(specs["browsecomp"].kind, "browsecomp")
        self.assertEqual(specs["browsecomp"].source_type, "browsecomp_csv")
        self.assertEqual(specs["browsecomp_zh"].source_type, "browsecomp_zh_xlsx")
        self.assertEqual(specs["apibank_l1"].kind, "apibank")
        self.assertEqual(specs["apibank_l1"].row_adapter, "apibank_level1")
        self.assertEqual(specs["apibank_level2"].row_adapter, "apibank_level2")
        self.assertEqual(specs["bfcl_simple_python"].kind, "bfcl_ast")
        self.assertEqual(specs["bfcl_simple_python"].row_adapter, "simple_python")
        self.assertEqual(specs["bfcl_exec_multiple_ast"].row_adapter, "exec_multiple")
        self.assertEqual(specs["bfcl_exec_simple"].kind, "bfcl_exec")
        self.assertEqual(specs["bfcl_exec_simple"].row_adapter, "exec_simple")
        self.assertEqual(specs["bfcl_exec_parallel_multiple"].row_adapter, "exec_parallel_multiple")
        self.assertEqual(specs["toolalpaca_eval_simulated"].kind, "toolalpaca")
        self.assertEqual(specs["toolalpaca_eval_simulated"].source_type, "toolalpaca_git")
        self.assertEqual(specs["toolalpaca_eval_real"].row_adapter, "eval_real")
        self.assertEqual(specs["complexfuncbench_official"].kind, "complexfuncbench")
        self.assertEqual(specs["complexfuncbench_official"].source_type, "hf_complexfuncbench")
        self.assertEqual(specs["complexfuncbench_subset"].dataset_name, "complexfuncbench_subset")

    def test_run_catalog_gsm8k_dry_run_uses_rwkv_dataset_slug(self) -> None:
        args = Namespace(
            dry_run=True,
            benchmark="gsm8k",
            base_url=None,
            model=None,
            limit=2,
        )

        with mock.patch.object(cli_main, "print_json") as print_json:
            rc = cli_main.handle_eval_run_catalog(args, root=ROOT)

        self.assertEqual(rc, 0)
        self.assertEqual(
            print_json.call_args.args[0],
            {
                "benchmark": "gsm8k",
                "hf_dataset": "openai/gsm8k",
                "hf_config": "main",
                "split": "test",
                "limit": 2,
                "base_url": "http://127.0.0.1:29082",
                "model": "rwkv7-g1d-0.4b-20260210-ctx8192",
                "scoreboard_dataset": "gsm8k_test_limit2",
                "job_name": "free_response_judge",
                "job_id": "helicopter-gsm8k",
            },
        )

    def test_runnable_catalog_counts_implemented_specs(self) -> None:
        args = Namespace(benchmark=("all",), field=None, json=True)

        with mock.patch.object(cli_main, "print_json") as print_json:
            rc = cli_main.handle_eval_runnable(args)

        self.assertEqual(rc, 0)
        payload = print_json.call_args.args[0]
        self.assertEqual(payload["count"], 95)
        self.assertEqual(payload["status_counts"]["implemented"], 76)
        self.assertEqual(payload["status_counts"].get("needs_dataset_adapter", 0), 0)
        self.assertEqual(payload["status_counts"]["needs_dataset_access"], 2)
        self.assertEqual(payload["status_counts"]["needs_specialized_runner"], 17)

    def test_run_catalog_human_eval_dry_run_uses_code_generation_runner(self) -> None:
        args = Namespace(
            dry_run=True,
            benchmark="human_eval",
            base_url=None,
            model=None,
            limit=2,
        )

        with mock.patch.object(cli_main, "print_json") as print_json:
            rc = cli_main.handle_eval_run_catalog(args, root=ROOT)

        self.assertEqual(rc, 0)
        self.assertEqual(
            print_json.call_args.args[0],
            {
                "benchmark": "human_eval",
                "dataset_name": "human_eval",
                "source_type": "human_eval_url_gzip",
                "split": "test",
                "limit": 2,
                "base_url": "http://127.0.0.1:29082",
                "model": "rwkv7-g1d-0.4b-20260210-ctx8192",
                "scoreboard_dataset": "human_eval_test_limit2",
                "job_name": "code_human_eval",
                "job_id": "helicopter-human_eval",
            },
        )

    def test_code_generation_extracts_last_python_fence(self) -> None:
        text = "<think>draft</think>\n```text\nignore\n```\n```python\ndef add(a, b):\n    return a + b\n```"

        self.assertEqual(
            code_generation.extract_code_completion(text),
            "def add(a, b):\n    return a + b",
        )

    def test_livecodebench_scores_stdio_sample(self) -> None:
        sample = code_generation.CodeGenerationSample(
            sample_index=0,
            task_id="toy",
            prompt="Read an integer and print it plus one.",
            public_test_cases=[{"input": "1\n", "output": "2\n"}],
            private_test_cases=[],
            metadata={},
        )
        config = code_generation.CodeGenerationRunConfig(
            base_url="http://127.0.0.1:29082",
            model="rwkv7-g1d-0.4b-20260210-ctx8192",
            benchmark="livecodebench",
            dataset_name="livecodebench/code_generation_lite",
            source_type="livecodebench_hf",
            eval_timeout_s=1.0,
        )

        self.assertEqual(
            code_generation.score_completion(sample, "n = int(input())\nprint(n + 1)", config),
            (True, "passed", None),
        )

    def test_livecodebench_scores_call_based_sample(self) -> None:
        sample = code_generation.CodeGenerationSample(
            sample_index=0,
            task_id="toy-call",
            prompt="Implement add(a, b).",
            public_test_cases=[{"input": "1\n2", "output": "3"}],
            private_test_cases=[],
            metadata={"func_name": "add"},
        )
        config = code_generation.CodeGenerationRunConfig(
            base_url="http://127.0.0.1:29082",
            model="rwkv7-g1d-0.4b-20260210-ctx8192",
            benchmark="livecodebench",
            dataset_name="livecodebench/code_generation_lite",
            source_type="livecodebench_hf",
            eval_timeout_s=1.0,
        )

        self.assertEqual(
            code_generation.score_completion(sample, "def add(a, b):\n    return a + b", config),
            (True, "passed", None),
        )

    def test_run_catalog_longcodeqa_dry_run_uses_runner(self) -> None:
        args = Namespace(
            dry_run=True,
            benchmark="longcodeqa",
            base_url=None,
            model=None,
            limit=3,
        )

        with mock.patch.object(cli_main, "print_json") as print_json:
            rc = cli_main.handle_eval_run_catalog(args, root=ROOT)

        self.assertEqual(rc, 0)
        self.assertEqual(
            print_json.call_args.args[0],
            {
                "benchmark": "longcodeqa",
                "source": "hf://Steefano/LCB/LongCodeQA.zip",
                "split": "test",
                "limit": 3,
                "base_url": "http://127.0.0.1:29082",
                "model": "rwkv7-g1d-0.4b-20260210-ctx8192",
                "scoreboard_dataset": "longcodeqa_test_limit3",
                "job_name": "function_longcodebench",
                "job_id": "helicopter-longcodeqa",
            },
        )

    def test_longcodeqa_normalizes_json_and_plain_answers(self) -> None:
        self.assertEqual(longcodeqa.normalize_answer('{"answer":"B"}', allowed_letters=("A", "B")), "B")
        self.assertEqual(longcodeqa.normalize_answer("Final answer: C", allowed_letters=("A", "B", "C")), "C")
        sample = longcodeqa.LongCodeQASample(
            sample_index=0,
            task_id="toy",
            prompt="Question\nA) no\nB) yes",
            repo_text="",
            question="A) no\nB) yes",
            correct_letter="B",
        )
        self.assertEqual(longcodeqa.evaluate_completion(sample, "B"), ("B", True))

    def test_run_catalog_longbench_qa_dry_run_uses_runner(self) -> None:
        args = Namespace(
            dry_run=True,
            benchmark="longbench_qa",
            base_url=None,
            model=None,
            limit=2,
        )

        with mock.patch.object(cli_main, "print_json") as print_json:
            rc = cli_main.handle_eval_run_catalog(args, root=ROOT)

        self.assertEqual(rc, 0)
        payload = print_json.call_args.args[0]
        self.assertEqual(payload["benchmark"], "longbench_qa")
        self.assertEqual(payload["source"], "hf://THUDM/LongBench")
        self.assertEqual(payload["limit"], 2)
        self.assertFalse(payload["balance_by_dataset"])
        self.assertIn("hotpotqa", payload["include_datasets"])
        self.assertEqual(payload["scoreboard_dataset"], "longbench_qa_test_limit2")

    def test_longbench_scores_exact_and_f1(self) -> None:
        self.assertEqual(longbench.normalize_answer("Answer: The Eiffel Tower"), "The Eiffel Tower")
        self.assertTrue(longbench.exact_match("the eiffel tower", "Eiffel Tower"))
        self.assertGreater(longbench.token_f1("red blue", "red green"), 0.0)
        exact, f1, ref = longbench.score_answer("Paris", ("Paris", "Lyon"))
        self.assertTrue(exact)
        self.assertEqual(f1, 1.0)
        self.assertEqual(ref, "Paris")

    def test_run_catalog_browsecomp_dry_run_uses_runner(self) -> None:
        args = Namespace(
            dry_run=True,
            benchmark="browsecomp",
            base_url=None,
            model=None,
            limit=2,
        )

        with mock.patch.object(cli_main, "print_json") as print_json:
            rc = cli_main.handle_eval_run_catalog(args, root=ROOT)

        self.assertEqual(rc, 0)
        payload = print_json.call_args.args[0]
        self.assertEqual(payload["benchmark"], "browsecomp")
        self.assertEqual(payload["source_type"], "browsecomp_csv")
        self.assertEqual(payload["limit"], 2)
        self.assertEqual(payload["scoreboard_dataset"], "browsecomp_test_limit2")
        self.assertEqual(payload["job_name"], "function_browsecomp")

    def test_browsecomp_final_answer_parser_accepts_function_call(self) -> None:
        self.assertEqual(
            browsecomp.parse_final_answer(
                '```json\n{"name":"final_answer","arguments":{"answer":"Ada Lovelace"},"id":"final_answer"}\n```'
            ),
            "Ada Lovelace",
        )
        self.assertEqual(browsecomp.parse_final_answer('{"answer": 42}'), "42")

    def test_run_catalog_apibank_dry_run_uses_runner(self) -> None:
        args = Namespace(
            dry_run=True,
            benchmark="apibank_level1",
            base_url=None,
            model=None,
            limit=2,
        )

        with mock.patch.object(cli_main, "print_json") as print_json:
            rc = cli_main.handle_eval_run_catalog(args, root=ROOT)

        self.assertEqual(rc, 0)
        payload = print_json.call_args.args[0]
        self.assertEqual(payload["benchmark"], "apibank_level1")
        self.assertEqual(payload["level"], 1)
        self.assertEqual(payload["scoreboard_dataset"], "apibank_level1_test_limit2")
        self.assertEqual(payload["job_name"], "function_api_bank")

    def test_apibank_tool_call_decoder_accepts_openai_wrapper(self) -> None:
        decoded = apibank.decode_tool_calls(
            '{"tool_calls":[{"function":{"name":"Search","arguments":"{\\"query\\":\\"rwkv\\"}"}}]}'
        )

        self.assertEqual(decoded, [{"name": "Search", "arguments": {"query": "rwkv"}}])

    def test_run_catalog_bfcl_ast_dry_run_uses_runner(self) -> None:
        args = Namespace(
            dry_run=True,
            benchmark="bfcl_simple_python",
            base_url=None,
            model=None,
            limit=2,
        )

        with mock.patch.object(cli_main, "print_json") as print_json:
            rc = cli_main.handle_eval_run_catalog(args, root=ROOT)

        self.assertEqual(rc, 0)
        payload = print_json.call_args.args[0]
        self.assertEqual(payload["benchmark"], "bfcl_simple_python")
        self.assertEqual(payload["category"], "simple_python")
        self.assertEqual(payload["scoreboard_dataset"], "bfcl_simple_python_test_limit2")
        self.assertEqual(payload["job_name"], "function_bfcl_ast")

    def test_bfcl_ast_scores_ground_truth_call_options(self) -> None:
        sample = bfcl_ast.BfclAstSample(
            sample_index=0,
            task_id="toy",
            instruction="Find triangle area.",
            tools=(),
            expected_tool_calls=(
                bfcl_ast.ToolCallExpectation(
                    name="calculate_triangle_area",
                    arguments={"base": 10, "height": 5},
                    argument_options={"base": (10,), "height": (5,), "unit": ("units", "")},
                ),
            ),
            category="simple_python",
        )

        self.assertEqual(
            bfcl_ast.evaluate_completion(
                sample,
                '{"name":"calculate_triangle_area","arguments":{"base":10,"height":5}}',
            )[:2],
            (True, ""),
        )

    def test_run_catalog_bfcl_exec_dry_run_uses_runner(self) -> None:
        args = Namespace(
            dry_run=True,
            benchmark="bfcl_exec_parallel",
            base_url=None,
            model=None,
            limit=2,
        )

        with mock.patch.object(cli_main, "print_json") as print_json:
            rc = cli_main.handle_eval_run_catalog(args, root=ROOT)

        self.assertEqual(rc, 0)
        payload = print_json.call_args.args[0]
        self.assertEqual(payload["benchmark"], "bfcl_exec_parallel")
        self.assertEqual(payload["category"], "exec_parallel")
        self.assertEqual(payload["scoreboard_dataset"], "bfcl_exec_parallel_test_limit2")
        self.assertEqual(payload["job_name"], "function_bfcl_exec")

    def test_bfcl_exec_scores_executable_exact_result(self) -> None:
        sample = bfcl_exec.BfclExecSample(
            sample_index=0,
            task_id="toy",
            instruction="Compute a binomial probability.",
            tools=(),
            expected_executable_calls=("calc_binomial_probability(n=20, k=5, p=0.6)",),
            execution_result_type=("exact_match",),
            category="exec_simple",
        )

        self.assertEqual(
            bfcl_exec.evaluate_completion(
                sample,
                '{"name":"calc_binomial_probability","arguments":{"n":20,"k":5,"p":0.6}}',
            )[:2],
            (True, ""),
        )
        self.assertEqual(
            bfcl_exec.evaluate_completion(
                sample,
                '{"name":"calc_binomial_probability","arguments":{"n":20,"k":4,"p":0.6}}',
            )[:1],
            (False,),
        )

    def test_bfcl_exec_parallel_scores_calls_order_insensitive(self) -> None:
        sample = bfcl_exec.BfclExecSample(
            sample_index=0,
            task_id="toy-parallel",
            instruction="Run independent calls.",
            tools=(),
            expected_executable_calls=(
                "get_zipcode_by_ip_address(ip_address='192.168.1.1')",
                "calculate_electrostatic_potential_energy(charge=5.0, voltage=10.0)",
            ),
            execution_result_type=("exact_match", "exact_match"),
            category="exec_parallel",
        )

        self.assertEqual(
            bfcl_exec.evaluate_completion(
                sample,
                (
                    "["
                    '{"name":"calculate_electrostatic_potential_energy","arguments":{"charge":5.0,"voltage":10.0}},'
                    '{"name":"get_zipcode_by_ip_address","arguments":{"ip_address":"192.168.1.1"}}'
                    "]"
                ),
            )[:2],
            (True, ""),
        )

    def test_run_catalog_toolalpaca_dry_run_uses_runner(self) -> None:
        args = Namespace(
            dry_run=True,
            benchmark="toolalpaca_eval_simulated",
            base_url=None,
            model=None,
            limit=2,
        )

        with mock.patch.object(cli_main, "print_json") as print_json:
            rc = cli_main.handle_eval_run_catalog(args, root=ROOT)

        self.assertEqual(rc, 0)
        payload = print_json.call_args.args[0]
        self.assertEqual(payload["benchmark"], "toolalpaca_eval_simulated")
        self.assertEqual(payload["source_dataset"], "toolalpaca_eval_simulated")
        self.assertEqual(payload["scoreboard_dataset"], "toolalpaca_eval_simulated_test_limit2")
        self.assertEqual(payload["job_name"], "function_toolalpaca")

    def test_agentbench_loads_manifest_for_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "test.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "task_id": "agentbench_db__00000",
                        "task_name": "dbbench-std",
                        "index": 0,
                        "metadata": {"source_format": "official_agentbench_controller"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = agentbench.AgentBenchRunConfig(
                base_url="http://127.0.0.1:29082",
                model="rwkv7-g1d-0.4b-20260210-ctx8192",
                benchmark="agentbench_db",
                dataset_name="agentbench_db",
                source_path=str(manifest),
            )

            payload = agentbench.dry_run_summary(config)

        self.assertEqual(payload["available_samples"], 1)
        self.assertEqual(payload["scoreboard_dataset"], "agentbench_db_test")
        self.assertTrue(payload["controller_required"])

    def test_agentbench_kg_prompt_and_final_answer_message(self) -> None:
        prompt = agentbench.build_prompt(
            [{"role": "user", "content": "Find the entity id."}],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search the graph.",
                        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                    },
                }
            ],
            allow_final_answer_text=True,
        )
        message = agentbench._assistant_message_from_calls(
            [{"name": "final_answer", "arguments": {"answer": "Final Answer: #42"}}],
            1,
        )

        self.assertIn("final_answer", prompt)
        self.assertEqual(message, {"role": "assistant", "content": "Final Answer: #42"})

    def test_mcp_bench_loads_manifest_for_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "test.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "task_id": "task-1",
                        "instruction": "Use a tool.",
                        "task_file": "mcpbench_tasks_single_runner_format.json",
                        "server_name": "Demo Server",
                        "combination_name": "Single Server: Demo",
                        "combination_type": "single_server",
                        "servers": ["Demo Server"],
                        "runtime_root": str(root),
                        "tasks_root": str(root / "tasks"),
                        "task": {
                            "task_id": "task-1",
                            "task_description": "Use a tool.",
                            "fuzzy_description": "Please use a tool.",
                            "dependency_analysis": "",
                            "distraction_servers": [],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = mcp_bench.McpBenchRunConfig(
                base_url="http://127.0.0.1:29082",
                model="rwkv7-g1d-0.4b-20260210-ctx8192",
                benchmark="mcp_bench_single",
                dataset_name="mcp_bench_single",
                source_path=str(manifest),
                runtime_root=str(root),
            )

            payload = mcp_bench.dry_run_summary(config)

        self.assertEqual(payload["available_samples"], 1)
        self.assertEqual(payload["scoreboard_dataset"], "mcp_bench_single_test")
        self.assertFalse(payload["runtime_ready"])
        self.assertIn("Demo Server", payload["checked_servers"])

    def test_mcp_bench_parses_tool_and_final_answer_decisions(self) -> None:
        tool_decision = mcp_bench.parse_planning_decision('{"name":"Demo Server:lookup","arguments":{"q":"x"}}')
        final_decision = mcp_bench.parse_planning_decision('{"name":"final_answer","arguments":{"answer":"done"}}')

        self.assertTrue(tool_decision.should_continue)
        self.assertEqual(tool_decision.tool_calls[0].server, "Demo Server")
        self.assertEqual(tool_decision.tool_calls[0].tool, "lookup")
        self.assertFalse(final_decision.should_continue)
        self.assertEqual(final_decision.final_answer, "done")

    def test_toolalpaca_loads_official_source_and_scores_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_root = Path(tmp)
            payload = [
                {
                    "Name": "DemoAPI",
                    "Documentation": json.dumps(
                        {
                            "servers": [{"url": "https://example.test"}],
                            "paths": {
                                "/lookup": {
                                    "get": {
                                        "parameters": [
                                            {
                                                "name": "query",
                                                "in": "query",
                                                "required": True,
                                                "schema": {"type": "string"},
                                            }
                                        ]
                                    }
                                }
                            },
                        }
                    ),
                    "Function_Description": {
                        "lookup": (
                            "Lookup a value.\n"
                            'Parameters: {"query": "Required. String. Search query."}\n'
                            "Output: object"
                        )
                    },
                    "Function_Projection": {"lookup": ["/lookup", "get"]},
                    "Instructions": ["Look up alpha"],
                    "Golden_Answers": [[{"Action": "lookup", "Action_Input": '{"query":"alpha"}'}]],
                }
            ]
            (source_root / "eval_simulated.json").write_text(json.dumps(payload), encoding="utf-8")
            (source_root / "eval_real.json").write_text(json.dumps(payload), encoding="utf-8")
            config = toolalpaca.ToolAlpacaRunConfig(
                base_url="http://127.0.0.1:29082",
                model="rwkv7-g1d-0.4b-20260210-ctx8192",
                benchmark="toolalpaca_eval_simulated",
                dataset_name="toolalpaca_eval_simulated",
                source_root=str(source_root),
            )

            sample = toolalpaca.load_samples(config)[0]

        self.assertEqual(sample.task_id, "toolalpaca_eval_simulated__demoapi_000")
        self.assertEqual(
            toolalpaca.evaluate_completion(sample, '[{"name":"lookup","arguments":{"query":"alpha"}}]')[:2],
            (True, ""),
        )
        self.assertEqual(
            toolalpaca.evaluate_completion(sample, '[{"name":"lookup","arguments":{"query":"beta"}}]')[:1],
            (False,),
        )

    def test_run_catalog_wmt24pp_dry_run_uses_translation_runner(self) -> None:
        args = Namespace(
            dry_run=True,
            benchmark="wmt24pp",
            base_url=None,
            model=None,
            limit=2,
        )

        with mock.patch.object(cli_main, "print_json") as print_json:
            rc = cli_main.handle_eval_run_catalog(args, root=ROOT)

        self.assertEqual(rc, 0)
        payload = print_json.call_args.args[0]
        self.assertEqual(payload["benchmark"], "wmt24pp")
        self.assertEqual(payload["hf_dataset"], "google/wmt24pp")
        self.assertEqual(payload["source_type"], "hf_wmt24pp")
        self.assertEqual(payload["scoreboard_dataset"], "wmt24pp_test_limit2")
        self.assertEqual(payload["job_name"], "translation_chrf")
        self.assertEqual(payload["metric"], "chrf")

    def test_translation_chrf_scores_exact_match_higher(self) -> None:
        exact = translation.chrf_score("Bonjour le monde", "Bonjour le monde")
        partial = translation.chrf_score("Salut", "Bonjour le monde")

        self.assertEqual(exact, 1.0)
        self.assertLess(partial, exact)
        self.assertEqual(
            translation.score_completion("Translation: Bonjour le monde", "Bonjour le monde")[:2],
            (1.0, True),
        )

    def test_run_catalog_complexfuncbench_dry_run_uses_runner(self) -> None:
        args = Namespace(
            dry_run=True,
            benchmark="complexfuncbench_official",
            base_url=None,
            model=None,
            limit=2,
        )

        with mock.patch.object(cli_main, "print_json") as print_json:
            rc = cli_main.handle_eval_run_catalog(args, root=ROOT)

        self.assertEqual(rc, 0)
        payload = print_json.call_args.args[0]
        self.assertEqual(payload["benchmark"], "complexfuncbench_official")
        self.assertEqual(payload["source"], "hf://zai-org/ComplexFuncBench")
        self.assertEqual(payload["scoreboard_dataset"], "complexfuncbench_official_test_limit2")
        self.assertEqual(payload["job_name"], "function_complexfuncbench")
        self.assertEqual(payload["runtime"], "local_golden_conversation")
        self.assertEqual(payload["metric"], "tool_call_sequence_exact_match")

    def test_complexfuncbench_loads_official_row_and_scores_sequence(self) -> None:
        row = {
            "id": "case-1",
            "tools": [
                {
                    "name": "SearchHotel",
                    "description": "Search hotels.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}, "adults": {"type": "integer"}},
                    },
                },
                {"name": "BookHotel", "description": "Book hotels.", "parameters": {"type": "object"}},
            ],
            "conversations": [
                {"role": "user", "content": "Find a hotel in Paris for two adults, then book h1."},
                {
                    "role": "assistant",
                    "function_call": [
                        {"name": "SearchHotel", "arguments": {"city": "Paris", "adults": 2}},
                        {"name": "BookHotel", "arguments": {"hotel_id": "h1"}},
                    ],
                },
                {"role": "observation", "content": [{"hotel_id": "h1"}, {"status": "booked"}]},
            ],
        }
        sample = complexfuncbench.sample_from_row(row, sample_index=0)
        assert sample is not None

        self.assertEqual(sample.task_id, "complexfuncbench_official__case-1")
        self.assertIn("final_answer", {tool["name"] for tool in sample.tools})
        self.assertEqual(
            complexfuncbench.evaluate_completion(
                sample,
                (
                    "["
                    '{"name":"SearchHotel","arguments":{"city":"Paris","adults":2}},'
                    '{"name":"BookHotel","arguments":{"hotel_id":"h1"}}'
                    "]"
                ),
            )[:2],
            (True, ""),
        )
        self.assertEqual(
            complexfuncbench.evaluate_completion(
                sample,
                '[{"name":"SearchHotel","arguments":{"city":"Lyon","adults":2}}]',
            )[:1],
            (False,),
        )

    def test_complexfuncbench_episode_advances_observation_per_turn(self) -> None:
        row = {
            "id": "case-2",
            "tools": [
                {"name": "SearchHotel", "description": "Search hotels.", "parameters": {"type": "object"}},
                {"name": "BookHotel", "description": "Book hotels.", "parameters": {"type": "object"}},
            ],
            "conversations": [
                {"role": "user", "content": "Find then book h1."},
                {"role": "assistant", "function_call": [{"name": "SearchHotel", "arguments": {"city": "Paris"}}]},
                {"role": "observation", "content": [{"hotel_id": "h1"}]},
                {"role": "assistant", "function_call": [{"name": "BookHotel", "arguments": {"hotel_id": "h1"}}]},
                {"role": "observation", "content": [{"status": "booked"}]},
            ],
        }
        sample = complexfuncbench.sample_from_row(row, sample_index=0)
        assert sample is not None
        outputs = iter(
            [
                '{"name":"SearchHotel","arguments":{"city":"Paris"}}',
                '{"name":"BookHotel","arguments":{"hotel_id":"h1"}}',
                '{"name":"final_answer","arguments":{"answer":"Booked."}}',
            ]
        )
        prompts: list[str] = []

        def fake_request(_config: object, prompt: str) -> str:
            prompts.append(prompt)
            return next(outputs)

        config = complexfuncbench.ComplexFuncBenchRunConfig(
            base_url="http://127.0.0.1:29082",
            model="rwkv7-g1d-0.4b-20260210-ctx8192",
            benchmark="complexfuncbench_official",
            dataset_name="complexfuncbench_official",
        )
        with mock.patch.object(complexfuncbench, "_request_completion", side_effect=fake_request):
            result = complexfuncbench.evaluate_sample(sample, config)

        self.assertTrue(result.is_passed)
        self.assertNotIn("hotel_id", prompts[0])
        self.assertIn("hotel_id", prompts[1])
        self.assertIn("final_answer", prompts[2])

    def test_multiple_choice_normalizes_list_and_arc_choices(self) -> None:
        list_choices = multiple_choice.normalize_choices(["red", "blue"])
        arc_choices = multiple_choice.normalize_choices({"label": ["A", "B"], "text": ["cat", "dog"]})
        numeric_choices = multiple_choice.normalize_choices(["one", "two"], fallback_labels="12")

        self.assertEqual(list_choices.labels, ("A", "B"))
        self.assertEqual(multiple_choice.reference_answer(1, list_choices), "B")
        self.assertEqual(multiple_choice.reference_answer("dog", arc_choices), "B")
        self.assertEqual(multiple_choice.reference_answer("1", numeric_choices), "1")
        self.assertEqual(multiple_choice.completion_answer("I think the answer is B.", arc_choices.labels), "B")

    def test_multiple_choice_adapters_keep_reference_with_shuffled_choices(self) -> None:
        rng = multiple_choice.random.Random(42)
        gpqa_config = multiple_choice.MultipleChoiceRunConfig(
            base_url="http://127.0.0.1:29082",
            model="rwkv7-g1d-0.4b-20260210-ctx8192",
            benchmark="gpqa_main",
            dataset_name="Idavidrein/gpqa",
            dataset_config="gpqa_main",
            question_field="question",
            choices_field="choices",
            answer_field="answer",
            row_adapter="gpqa",
        )
        gpqa_row = multiple_choice.adapt_choice_row(
            {
                "Question": "q",
                "Incorrect Answer 1": "wrong 1",
                "Incorrect Answer 2": "wrong 2",
                "Incorrect Answer 3": "wrong 3",
                "Correct Answer": "right",
            },
            gpqa_config,
            rng,
        )

        self.assertIsNotNone(gpqa_row)
        assert gpqa_row is not None
        self.assertEqual(gpqa_row.choices.texts[gpqa_row.choices.labels.index(gpqa_row.reference_answer)], "right")

        redux_config = multiple_choice.MultipleChoiceRunConfig(
            base_url="http://127.0.0.1:29082",
            model="rwkv7-g1d-0.4b-20260210-ctx8192",
            benchmark="mmlu_redux",
            dataset_name="edinburgh-dawg/mmlu-redux-2.0",
            dataset_config="abstract_algebra",
            question_field="question",
            choices_field="choices",
            answer_field="answer",
            row_adapter="mmlu_redux",
        )

        self.assertIsNone(
            multiple_choice.adapt_choice_row(
                {"error_type": "bad_question", "question": "q", "choices": ["a", "b"], "answer": 0},
                redux_config,
                multiple_choice.random.Random(42),
            )
        )

    def test_generic_multiple_choice_dry_run_uses_hf_dataset_config(self) -> None:
        args = Namespace(
            dry_run=True,
            benchmark="mmlu",
            dataset="cais/mmlu",
            dataset_config="abstract_algebra",
            question_field="question",
            choices_field="choices",
            choice_field=None,
            answer_field="answer",
            choice_labels="ABCD",
            base_url=None,
            model=None,
            limit=10,
            split="test",
            temperature=0.0,
            top_p=1.0,
            max_tokens=8,
            timeout_s=30.0,
            job_name="multi_choice_plain",
            job_id=None,
        )

        with mock.patch.object(cli_main, "print_json") as print_json:
            rc = cli_main.handle_eval_run_multiple_choice(args, root=ROOT)

        self.assertEqual(rc, 0)
        self.assertEqual(
            print_json.call_args.args[0],
            {
                "benchmark": "mmlu",
                "hf_dataset": "cais/mmlu",
                "hf_config": "abstract_algebra",
                "split": "test",
                "limit": 10,
                "base_url": "http://127.0.0.1:29082",
                "model": "rwkv7-g1d-0.4b-20260210-ctx8192",
                "scoreboard_dataset": "mmlu_test_limit10",
                "job_name": "multi_choice_plain",
                "job_id": "helicopter-mmlu",
            },
        )

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
