from __future__ import annotations

import os
import tempfile
import tomllib
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from helicopter_cli import commands, config, env


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "configs/example.toml"
STRICT_SMOKE_CONFIG = ROOT / "configs/strict-smoke.toml"


def load_example_config() -> dict[str, object]:
    loaded, _ = config.load_config(ROOT, str(EXAMPLE_CONFIG))
    return loaded


def infer_args(**overrides: object) -> Namespace:
    values = {
        "model": "g1g-1.5b",
        "dry_run": True,
        "wkv_mode": None,
        "emb_device": None,
        "allow_fp16_accumulation": None,
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
        "allow_fp16_accumulation": None,
        "num_nodes": None,
        "num_devices": None,
        "override": None,
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
        exists.side_effect = lambda path: (
            True if path == venv_python else original_exists(path)
        )
        return commands.build_takeoff_plan(
            args, root=ROOT, env=loaded_env, config=loaded_config
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
    def test_verl_runtime_dependencies_include_required_runtime_stack(self) -> None:
        manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = {
            dependency
            for dependency in manifest["dependency-groups"]["verl"]
            if isinstance(dependency, str)
        }

        self.assertTrue(
            {"math-verify", "latex2sympy2-extended", "nvtx"}.issubset(dependencies)
        )

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

        model_path, model = config.resolve_model_path(
            loaded_config, "g1g-1.5b", root=ROOT, env=loaded_env
        )

        self.assertEqual(model["served_model_name"], "g1g-1.5b")
        self.assertEqual(
            model_path,
            Path("/weights/RWKV/rwkv7-g1g-1.5b-20260526-ctx8192.pth"),
        )

    def test_strict_checkpoint_path_is_the_model_source_of_truth(self) -> None:
        loaded_config = load_example_config()
        expected = Path("/weights/RWKV/rwkv7/pth/strict-checkpoint.pth")

        model_path, _ = config.resolve_model_path(
            loaded_config,
            "g1g-1.5b",
            root=ROOT,
            env={
                "WEIGHT_PATH": "/weights/RWKV",
                "HELICOPTER_CHECKPOINT_PATH": str(expected),
            },
        )

        self.assertEqual(model_path, expected)


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

    def test_takeoff_plan_uses_verl_module_entrypoint_and_default_overrides(
        self,
    ) -> None:
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
                "verl.trainer.main_ppo",
            ],
        )
        self.assertEqual(
            plan.shown_env,
            {
                "PYTHON": str(venv_python),
                "PYTHONPATH": str(ROOT / "src/infer/vllm-rwkv"),
                "RWKV_LM_PATH": str(ROOT / "src/train/rwkv-lm"),
                "RWKV_MODEL_PATH": "/weights/RWKV/rwkv7-g1g-1.5b-20260526-ctx8192.pth",
                "VLLM_RWKV7_WKV_MODE": "fp32io16",
            },
        )
        self.assertEqual(
            {
                key: overrides[key]
                for key in (
                    "data.train_batch_size",
                    "data.max_prompt_length",
                    "data.max_response_length",
                    "data.seed",
                    "reward.custom_reward_function.path",
                    "actor_rollout_ref.actor.use_dynamic_bsz",
                    "actor_rollout_ref.actor.ppo_mini_batch_size",
                    "actor_rollout_ref.actor.ppo_epochs",
                    "actor_rollout_ref.actor.data_loader_seed",
                    "actor_rollout_ref.model.path",
                    "actor_rollout_ref.rollout.name",
                    "actor_rollout_ref.rollout.checkpoint_engine.backend",
                    "actor_rollout_ref.rollout.top_p",
                    "actor_rollout_ref.rollout.seed",
                    "actor_rollout_ref.hybrid_engine",
                    "trainer.v1.trainer_mode",
                    "trainer.n_gpus_per_node",
                    "trainer.logger",
                    "trainer.total_epochs",
                    "trainer.val_before_train",
                )
            },
            {
                "data.train_batch_size": "56",
                "data.max_prompt_length": "1024",
                "data.max_response_length": "7168",
                "data.seed": "42",
                "reward.custom_reward_function.path": str(
                    ROOT
                    / "src/train/verl-rwkv/examples/rwkv_trainer/math_verify_reward.py"
                ),
                "actor_rollout_ref.actor.use_dynamic_bsz": "False",
                "actor_rollout_ref.actor.ppo_mini_batch_size": "56",
                "actor_rollout_ref.actor.ppo_epochs": "1",
                "actor_rollout_ref.actor.data_loader_seed": "42",
                "actor_rollout_ref.model.path": "/weights/RWKV/rwkv7-g1g-1.5b-20260526-ctx8192.pth",
                "actor_rollout_ref.rollout.name": "vllm",
                "actor_rollout_ref.rollout.checkpoint_engine.backend": "naive",
                "actor_rollout_ref.rollout.top_p": "0.8",
                "actor_rollout_ref.rollout.seed": "42",
                "actor_rollout_ref.hybrid_engine": "True",
                "trainer.v1.trainer_mode": "sync",
                "trainer.n_gpus_per_node": "8",
                "trainer.logger": '["console","file"]',
                "trainer.total_epochs": "2",
                "trainer.val_before_train": "True",
            },
        )
        self.assertEqual(optional_rollout_keys & overrides.keys(), set())
        self.assertEqual(
            overrides[
                "+actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend"
            ],
            "uni",
        )
        self.assertEqual(
            overrides[
                "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V2_MODEL_RUNNER"
            ],
            '"1"',
        )
        self.assertEqual(
            overrides["+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL"],
            '"INFO"',
        )
        self.assertNotIn("rollout.nnodes", overrides)
        self.assertNotIn("rollout.n_gpus_per_node", overrides)

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
                "HELICOPTER_TAKEOFF_ALLOW_FP16_ACCUMULATION": "0",
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
        self.assertNotIn("VLLM_RWKV7_EMB_DEVICE", plan.env)
        self.assertNotIn("VLLM_RWKV7_ALLOW_FP16_ACCUMULATION", plan.env)
        self.assertEqual(plan.env["PYTHONPATH"], str(ROOT / "src/infer/vllm-rwkv"))
        self.assertEqual(forbidden_env_keys & plan.env.keys(), set())
        self.assertEqual(forbidden_override_keys & overrides.keys(), set())

    def test_strict_smoke_limits_dataset_before_filtering_and_pins_seed(self) -> None:
        loaded_config, _ = config.load_config(ROOT, str(STRICT_SMOKE_CONFIG))
        plan = build_takeoff_plan(
            loaded_config,
            args=takeoff_args(model="g1h-7.2b", dataset="dapo_smoke"),
            loaded_env={
                "WEIGHT_PATH": "/weights/RWKV",
                "DATASETS_PATH": "/datasets",
                "HELICOPTER_SEED": "42",
            },
        )
        overrides = hydra_map(plan)

        self.assertEqual(overrides["data.train_max_samples"], "4096")
        self.assertEqual(overrides["data.val_max_samples"], "1024")
        self.assertEqual(overrides["data.seed"], "42")
        self.assertEqual(overrides["actor_rollout_ref.actor.data_loader_seed"], "42")
        self.assertEqual(overrides["actor_rollout_ref.rollout.seed"], "42")

    def test_infer_runtime_env_strips_dotenv_vllm_knobs(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_infer_plan(
            infer_args(),
            root=ROOT,
            env={
                "WEIGHT_PATH": "/weights/RWKV",
                "VLLM_RWKV7_WKV_MODE": "fp32io16",
                "HELICOPTER_INFER_ALLOW_FP16_ACCUMULATION": "0",
                "VLLM_GPU_MEMORY_UTILIZATION": "0.85",
                "VLLM_MAX_NUM_SEQS": "2048",
            },
            config=loaded_config,
        )
        options = command_options(plan.command)
        forbidden_env_keys = {"VLLM_GPU_MEMORY_UTILIZATION", "VLLM_MAX_NUM_SEQS"}
        forbidden_option_keys = {"--gpu-memory-utilization", "--max-num-seqs"}

        self.assertEqual(plan.env["VLLM_RWKV7_WKV_MODE"], "fp32io16")
        self.assertNotIn("VLLM_RWKV7_ALLOW_FP16_ACCUMULATION", plan.env)
        self.assertEqual(forbidden_env_keys & plan.env.keys(), set())
        self.assertEqual(forbidden_option_keys & options.keys(), set())

    def test_infer_fp16_accumulation_cli_false_overrides_environment(self) -> None:
        plan = commands.build_infer_plan(
            infer_args(allow_fp16_accumulation=False),
            root=ROOT,
            env={
                "WEIGHT_PATH": "/weights/RWKV",
                "HELICOPTER_INFER_ALLOW_FP16_ACCUMULATION": "1",
            },
            config=load_example_config(),
        )

        self.assertNotIn("VLLM_RWKV7_ALLOW_FP16_ACCUMULATION", plan.env)

    def test_infer_fp16_wkv_enables_fp16_accumulation_by_default(self) -> None:
        plan = commands.build_infer_plan(
            infer_args(wkv_mode="fp16"),
            root=ROOT,
            env={"WEIGHT_PATH": "/weights/RWKV"},
            config=load_example_config(),
        )

        self.assertEqual(plan.env["VLLM_RWKV7_WKV_MODE"], "fp16")
        self.assertNotIn("VLLM_RWKV7_ALLOW_FP16_ACCUMULATION", plan.env)

    def test_takeoff_high_precision_wkv_disables_fp16_accumulation_by_default(
        self,
    ) -> None:
        plan = commands.build_takeoff_plan(
            takeoff_args(),
            root=ROOT,
            env={"WEIGHT_PATH": "/weights/RWKV"},
            config=load_example_config(),
        )

        self.assertEqual(plan.env["VLLM_RWKV7_WKV_MODE"], "fp32io16")
        self.assertNotIn("VLLM_RWKV7_ALLOW_FP16_ACCUMULATION", plan.env)

    def test_infer_rejects_accumulation_that_conflicts_with_wkv_profile(self) -> None:
        with self.assertRaisesRegex(
            SystemExit, "derives GEMM accumulation from WKV mode"
        ):
            commands.build_infer_plan(
                infer_args(wkv_mode="fp16", allow_fp16_accumulation=False),
                root=ROOT,
                env={"WEIGHT_PATH": "/weights/RWKV"},
                config=load_example_config(),
            )

    def test_infer_fp16_accumulation_rejects_invalid_environment_value(self) -> None:
        with self.assertRaisesRegex(
            SystemExit,
            "HELICOPTER_INFER_ALLOW_FP16_ACCUMULATION must be 0 or 1",
        ):
            commands.build_infer_plan(
                infer_args(),
                root=ROOT,
                env={
                    "WEIGHT_PATH": "/weights/RWKV",
                    "HELICOPTER_INFER_ALLOW_FP16_ACCUMULATION": "true",
                },
                config=load_example_config(),
            )

    def test_takeoff_config_adv_estimator_becomes_hydra_overrides(self) -> None:
        loaded_config = load_example_config()
        takeoff = loaded_config["takeoff"]
        takeoff["grpo"] = {
            **takeoff["grpo"],
            "adv_estimator": "maxrl",
            "reward_manager": "dapo",
        }

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

    def test_takeoff_config_infctx_becomes_rwkv_lm_engine_overrides(self) -> None:
        loaded_config = load_example_config()
        takeoff = loaded_config["takeoff"]
        takeoff["grpo"] = {
            **takeoff["grpo"],
            "ctx_len": 8192,
            "infctx": True,
            "chunk_ctx": 2048,
        }

        overrides = hydra_map(build_takeoff_plan(loaded_config))

        self.assertEqual(
            {
                key: overrides[key]
                for key in (
                    "actor_rollout_ref.actor.engine.ctx_len",
                    "actor_rollout_ref.actor.engine.infctx",
                    "actor_rollout_ref.actor.engine.chunk_ctx",
                    "actor_rollout_ref.ref.engine.ctx_len",
                    "actor_rollout_ref.ref.engine.infctx",
                    "actor_rollout_ref.ref.engine.chunk_ctx",
                )
            },
            {
                "actor_rollout_ref.actor.engine.ctx_len": "8192",
                "actor_rollout_ref.actor.engine.infctx": "True",
                "actor_rollout_ref.actor.engine.chunk_ctx": "2048",
                "actor_rollout_ref.ref.engine.ctx_len": "8192",
                "actor_rollout_ref.ref.engine.infctx": "True",
                "actor_rollout_ref.ref.engine.chunk_ctx": "2048",
            },
        )

    def test_fp32io16_uses_fp16_rollout_without_overriding_native_bf16(self) -> None:
        loaded_config = load_example_config()
        loaded_config["takeoff"]["grpo"]["wkv_mode"] = "fp32io16"

        overrides = hydra_map(build_takeoff_plan(loaded_config))

        self.assertEqual(overrides["actor_rollout_ref.rollout.dtype"], "float16")
        self.assertNotIn("actor_rollout_ref.actor.engine.precision", overrides)
        self.assertNotIn("actor_rollout_ref.actor.engine.dtype", overrides)
        self.assertNotIn("actor_rollout_ref.ref.engine.precision", overrides)
        self.assertNotIn("actor_rollout_ref.ref.engine.dtype", overrides)
        self.assertNotIn("actor_rollout_ref.model.dtype", overrides)

    def test_takeoff_config_sets_validation_sampling_for_non_greedy_eval(self) -> None:
        loaded_config = load_example_config()

        overrides = hydra_map(build_takeoff_plan(loaded_config))

        self.assertEqual(
            {
                key: overrides[key]
                for key in (
                    "actor_rollout_ref.rollout.val_kwargs.do_sample",
                    "actor_rollout_ref.rollout.val_kwargs.temperature",
                    "actor_rollout_ref.rollout.val_kwargs.top_k",
                    "actor_rollout_ref.rollout.val_kwargs.top_p",
                    "actor_rollout_ref.rollout.val_kwargs.n",
                    "+data.apply_chat_template_kwargs.rwkv_generation_prompt",
                    "+data.val_apply_chat_template_kwargs.rwkv_generation_prompt",
                )
            },
            {
                "actor_rollout_ref.rollout.val_kwargs.do_sample": "True",
                "actor_rollout_ref.rollout.val_kwargs.temperature": "1",
                "actor_rollout_ref.rollout.val_kwargs.top_k": "32",
                "actor_rollout_ref.rollout.val_kwargs.top_p": "0.28",
                "actor_rollout_ref.rollout.val_kwargs.n": "4",
                "+data.apply_chat_template_kwargs.rwkv_generation_prompt": "open_think",
                "+data.val_apply_chat_template_kwargs.rwkv_generation_prompt": "open_think",
            },
        )

    def test_takeoff_config_can_override_validation_generation_prompt(self) -> None:
        loaded_config = load_example_config()
        takeoff = loaded_config["takeoff"]
        takeoff["grpo"] = {
            **takeoff["grpo"],
            "val_rwkv_generation_prompt": "fake_think",
        }

        overrides = hydra_map(build_takeoff_plan(loaded_config))

        self.assertEqual(
            overrides["+data.val_apply_chat_template_kwargs.rwkv_generation_prompt"],
            "fake_think",
        )

    def test_takeoff_config_can_override_validation_rollout_n(self) -> None:
        loaded_config = load_example_config()
        takeoff = loaded_config["takeoff"]
        takeoff["grpo"] = {**takeoff["grpo"], "val_n": 2}

        overrides = hydra_map(build_takeoff_plan(loaded_config))

        self.assertEqual(overrides["actor_rollout_ref.rollout.val_kwargs.n"], "2")

    def test_takeoff_config_can_enable_validation_dump_dir(self) -> None:
        loaded_config = load_example_config()
        takeoff = loaded_config["takeoff"]
        takeoff["grpo"] = {
            **takeoff["grpo"],
            "validation_data_dir": "logs/validation/run",
        }

        overrides = hydra_map(build_takeoff_plan(loaded_config))

        self.assertEqual(
            overrides["trainer.validation_data_dir"], "logs/validation/run"
        )

    def test_takeoff_config_can_override_training_rollout_top_p(self) -> None:
        loaded_config = load_example_config()
        takeoff = loaded_config["takeoff"]
        takeoff["grpo"] = {**takeoff["grpo"], "rollout_top_p": 0.65}

        overrides = hydra_map(build_takeoff_plan(loaded_config))

        self.assertEqual(overrides["actor_rollout_ref.rollout.top_p"], "0.65")

    def test_takeoff_rejects_legacy_separate_rollout_gpu_pool(self) -> None:
        loaded_config = load_example_config()
        takeoff = loaded_config["takeoff"]
        takeoff["grpo"] = {
            **takeoff["grpo"],
            "trainer_n_gpus_per_node": 7,
            "rollout_n_gpus_per_node": 1,
            "rollout_data_parallel_size": 1,
            "rollout_pipeline_parallel_size": 1,
        }

        with self.assertRaisesRegex(
            SystemExit,
            "strict on-policy takeoff requires trainer.n_gpus_per_node=8",
        ):
            build_takeoff_plan(loaded_config)

    def test_takeoff_rejects_mismatched_round_and_ppo_mini_batch(self) -> None:
        loaded_config = load_example_config()
        takeoff = loaded_config["takeoff"]
        takeoff["grpo"] = {
            **takeoff["grpo"],
            "train_batch_size": 56,
            "ppo_mini_batch_size": 28,
        }

        with self.assertRaisesRegex(
            SystemExit,
            "ppo_mini_batch_size == data.train_batch_size",
        ):
            build_takeoff_plan(loaded_config)

    def test_takeoff_accepts_response_8192_with_complete_sequence_budget(self) -> None:
        loaded_config = load_example_config()
        takeoff = loaded_config["takeoff"]
        takeoff["grpo"] = {
            **takeoff["grpo"],
            "max_response_length": 8192,
            "ppo_max_token_len_per_gpu": 10240,
            "rollout_ignore_eos": True,
            "actor_optimizer_offload": True,
        }

        overrides = hydra_map(build_takeoff_plan(loaded_config))

        self.assertEqual(overrides["data.max_response_length"], "8192")
        self.assertEqual(
            overrides["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"], "10240"
        )
        self.assertEqual(overrides["actor_rollout_ref.rollout.ignore_eos"], "True")
        self.assertEqual(
            overrides["actor_rollout_ref.actor.engine.optimizer_offload"], "True"
        )

    def test_takeoff_rejects_strict_on_policy_override_regressions(self) -> None:
        loaded_config = load_example_config()
        invalid_overrides = (
            "trainer.v1.trainer_mode=colocate_async",
            "actor_rollout_ref.hybrid_engine=False",
            "actor_rollout_ref.actor.ppo_epochs=2",
            "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl",
            "algorithm.rollout_correction.rollout_is=null",
            "algorithm.rollout_correction.rollout_is=sequence",
            "algorithm.rollout_correction.rollout_is_threshold=4.0",
            "algorithm.rollout_correction.rollout_is_batch_normalize=True",
            "algorithm.rollout_correction.rollout_rs=seq_mean_k1",
            "algorithm.rollout_correction.bypass_mode=True",
            "data.dataloader_num_workers=8",
            "trainer.n_gpus_per_node=7",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
            "actor_rollout_ref.rollout.data_parallel_size=2",
            "actor_rollout_ref.rollout.pipeline_model_parallel_size=2",
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096",
            "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096",
            "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096",
            "actor_rollout_ref.actor.engine.infctx=False",
            "actor_rollout_ref.ref.engine.infctx=False",
            "actor_rollout_ref.actor.engine.chunk_ctx=4096",
            "actor_rollout_ref.ref.engine.chunk_ctx=4096",
            "rollout.n_gpus_per_node=1",
        )

        for override in invalid_overrides:
            with self.subTest(override=override):
                with self.assertRaisesRegex(SystemExit, "strict on-policy takeoff"):
                    build_takeoff_plan(
                        loaded_config,
                        args=takeoff_args(override=[override]),
                    )

    def test_takeoff_rejects_environment_drift_from_state_passing_contract(
        self,
    ) -> None:
        loaded_config = load_example_config()
        invalid_environments = (
            {"PPO_MAX_TOKEN_LEN_PER_GPU": "128"},
            {"RWKV_INFCTX": "0"},
            {"RWKV_CHUNK_CTX": "4096"},
        )

        for mutation in invalid_environments:
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(SystemExit, "strict on-policy takeoff"):
                    build_takeoff_plan(
                        loaded_config,
                        loaded_env={
                            "WEIGHT_PATH": "/weights/RWKV",
                            "DATASETS_PATH": "/datasets",
                            **mutation,
                        },
                    )

    def test_takeoff_allows_equal_larger_round_and_ppo_mini_batch(self) -> None:
        plan = build_takeoff_plan(
            load_example_config(),
            args=takeoff_args(
                override=[
                    "data.train_batch_size=112",
                    "actor_rollout_ref.actor.ppo_mini_batch_size=112",
                ]
            ),
        )

        overrides = hydra_map(plan)
        self.assertEqual(overrides["data.train_batch_size"], "112")
        self.assertEqual(
            overrides["actor_rollout_ref.actor.ppo_mini_batch_size"], "112"
        )

    def test_takeoff_fixes_eight_independent_single_gpu_rollout_replicas(self) -> None:
        overrides = hydra_map(build_takeoff_plan(load_example_config()))

        self.assertEqual(overrides["trainer.n_gpus_per_node"], "8")
        self.assertEqual(
            overrides["actor_rollout_ref.rollout.tensor_model_parallel_size"], "1"
        )
        self.assertEqual(overrides["actor_rollout_ref.rollout.data_parallel_size"], "1")
        self.assertEqual(
            overrides["actor_rollout_ref.rollout.pipeline_model_parallel_size"], "1"
        )

    def test_takeoff_enables_nsys_for_all_colocated_roles_from_config(self) -> None:
        config = load_example_config()
        config["takeoff"]["grpo"]["profiler_tool"] = "nsys"
        config["takeoff"]["grpo"]["profiler_steps"] = [2]

        overrides = hydra_map(build_takeoff_plan(config))

        self.assertEqual(overrides["global_profiler.tool"], "nsys")
        self.assertEqual(overrides["global_profiler.steps"], "[2]")
        self.assertEqual(
            overrides["actor_rollout_ref.actor.profiler.all_ranks"], "True"
        )
        self.assertEqual(overrides["actor_rollout_ref.rollout.profiler.enable"], "True")

    def test_takeoff_rejects_tensor_parallel_even_in_topology_phase(self) -> None:
        topology_override = [
            "actor_rollout_ref.rollout.tensor_model_parallel_size=8",
            "actor_rollout_ref.rollout.pipeline_model_parallel_size=1",
        ]
        with self.assertRaisesRegex(SystemExit, "strict on-policy takeoff"):
            build_takeoff_plan(
                load_example_config(), args=takeoff_args(override=topology_override)
            )

        with self.assertRaisesRegex(SystemExit, "tensor_model_parallel_size=1"):
            build_takeoff_plan(
                load_example_config(),
                args=takeoff_args(override=topology_override),
                loaded_env={
                    "WEIGHT_PATH": "/weights/RWKV",
                    "DATASETS_PATH": "/datasets",
                    "HELICOPTER_RUN_PHASE": "topology",
                    "HELICOPTER_ROLLOUT_TOPOLOGY_EXPERIMENT": "1",
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

        plan = build_takeoff_plan(
            loaded_config, args=takeoff_args(dataset="dapo_math_17k")
        )
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
                "VLLM_RWKV7_WKV_MODE",
            },
        )

    def test_takeoff_defaults_enable_native_reference_without_changing_loss(
        self,
    ) -> None:
        loaded_config = load_example_config()
        overrides = hydra_map(build_takeoff_plan(loaded_config))

        self.assertEqual(
            {
                "actor_rollout_ref.actor.use_kl_loss": overrides[
                    "actor_rollout_ref.actor.use_kl_loss"
                ],
                "actor_rollout_ref.actor.kl_loss_coef": overrides[
                    "actor_rollout_ref.actor.kl_loss_coef"
                ],
            },
            {
                "actor_rollout_ref.actor.use_kl_loss": "True",
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
                allow_fp16_accumulation=None,
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
                allow_fp16_accumulation=None,
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

    def test_takeoff_user_overrides_are_appended_after_generated_overrides(
        self,
    ) -> None:
        loaded_config = load_example_config()
        plan = build_takeoff_plan(
            loaded_config,
            args=takeoff_args(
                override=["trainer.total_epochs=1", "trainer.save_freq=10"]
            ),
        )

        self.assertEqual(hydra_values(plan, "trainer.total_epochs"), ["2", "1"])
        self.assertEqual(hydra_values(plan, "trainer.save_freq"), ["20", "10"])
        self.assertEqual(
            plan.command[-2:], ["trainer.total_epochs=1", "trainer.save_freq=10"]
        )

    def test_takeoff_rejects_missing_default_venv_python(self) -> None:
        config = load_example_config()
        env = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "HELICOPTER_PYTHON",
                "PYTHON",
                "HELICOPTER_VENV",
                "VENV",
                "REMOTE_VENV",
            }
        }
        venv_python = ROOT / ".venv/bin/python"
        original_exists = Path.exists

        with mock.patch.object(Path, "exists", autospec=True) as exists:
            exists.side_effect = lambda path: (
                False if path == venv_python else original_exists(path)
            )
            with self.assertRaises(SystemExit) as raised:
                commands.python_executable(
                    config, root=ROOT, env=env, require_configured=True
                )

        self.assertEqual(
            str(raised.exception),
            f"Python executable not found: {venv_python}; run scripts/install_local.sh "
            "or set HELICOPTER_PYTHON / paths.python",
        )


if __name__ == "__main__":
    unittest.main()
