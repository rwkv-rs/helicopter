from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from helicopter_cli import __main__ as cli
from helicopter_cli import commands, config, env


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "configs/example.toml"


def load_example_config() -> dict[str, object]:
    loaded, _ = config.load_config(ROOT, str(EXAMPLE_CONFIG))
    return loaded


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


class CommandDryRunTests(unittest.TestCase):
    def test_infer_dry_run_prints_vllm_serve_command(self) -> None:
        stdout = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"WEIGHT_PATH": "/weights/RWKV"}, clear=False),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = cli.main(
                [
                    "infer",
                    "--config",
                    str(EXAMPLE_CONFIG),
                    "--env-file",
                    "/does/not/exist",
                    "--dry-run",
                    "g1g-1.5b",
                ]
            )

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("vllm serve", output)
        self.assertIn("--tokenizer-mode rwkv", output)
        self.assertNotIn("VLLM_RWKV7_WKV_MODE=", output)
        self.assertNotIn("VLLM_RWKV7_EMB_DEVICE=", output)

    def test_takeoff_dry_run_prints_verl_command_when_venv_python_exists(self) -> None:
        stdout = io.StringIO()
        venv_python = ROOT / ".venv/bin/python"
        original_exists = Path.exists
        with (
            mock.patch.dict(
                os.environ,
                {"WEIGHT_PATH": "/weights/RWKV", "DATASETS_PATH": "/datasets"},
                clear=False,
            ),
            mock.patch.object(Path, "exists", autospec=True) as exists,
            contextlib.redirect_stdout(stdout),
        ):
            exists.side_effect = lambda path: True if path == venv_python else original_exists(path)
            exit_code = cli.main(
                [
                    "takeoff",
                    "--config",
                    str(EXAMPLE_CONFIG),
                    "--env-file",
                    "/does/not/exist",
                    "--dry-run",
                    "g1g-1.5b",
                    "grpo",
                    "--dataset",
                    "gsm8k",
                ]
            )

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn(f"{venv_python} -m verl.experimental.one_step_off_policy.main_ppo", output)
        self.assertNotIn("run_rwkv7_grpo_vllm.sh", output)
        self.assertIn(f"PYTHON={venv_python}", output)
        self.assertIn("actor_rollout_ref.actor.use_dynamic_bsz=False", output)
        self.assertIn("actor_rollout_ref.model.path=/weights/RWKV/rwkv7-g1g-1.5b-20260526-ctx8192.pth", output)
        self.assertIn("actor_rollout_ref.rollout.name=vllm", output)
        self.assertIn("actor_rollout_ref.hybrid_engine=False", output)
        self.assertIn("trainer.total_epochs=2", output)
        self.assertIn("VLLM_RWKV7_WKV_MODE=fp32io16", output)
        self.assertIn("VLLM_RWKV7_EMB_DEVICE=gpu", output)
        self.assertNotIn("VLLM_RWKV_PATH=", output)
        self.assertNotIn("VLLM_USE_V2_MODEL_RUNNER=", output)
        self.assertNotIn("actor_rollout_ref.rollout.gpu_memory_utilization=", output)
        self.assertNotIn("actor_rollout_ref.rollout.max_num_seqs=", output)
        self.assertNotIn("actor_rollout_ref.rollout.max_num_batched_tokens=", output)

    def test_takeoff_runtime_env_strips_dotenv_vllm_knobs(self) -> None:
        loaded_config = load_example_config()
        venv_python = ROOT / ".venv/bin/python"
        original_exists = Path.exists
        args = Namespace(
            algorithm="grpo",
            model="g1g-1.5b",
            dataset="gsm8k",
            dry_run=True,
            wkv_mode=None,
            emb_device=None,
            num_nodes=None,
            num_devices=None,
            override=None,
        )

        with mock.patch.object(Path, "exists", autospec=True) as exists:
            exists.side_effect = lambda path: True if path == venv_python else original_exists(path)
            plan = commands.build_takeoff_plan(
                args,
                root=ROOT,
                env={
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
                config=loaded_config,
            )

        self.assertEqual(plan.env["VLLM_RWKV7_WKV_MODE"], "fp32io16")
        self.assertEqual(plan.env["VLLM_RWKV7_EMB_DEVICE"], "gpu")
        self.assertEqual(plan.env["PYTHONPATH"], str(ROOT / "src/infer/vllm-rwkv"))
        self.assertNotIn("VLLM_GPU_MEMORY_UTILIZATION", plan.env)
        self.assertNotIn("VLLM_MAX_NUM_SEQS", plan.env)
        self.assertNotIn("VLLM_MAX_NUM_BATCHED_TOKENS", plan.env)
        self.assertNotIn("VLLM_RWKV_PATH", plan.env)
        self.assertNotIn("VLLM_USE_V2_MODEL_RUNNER", plan.env)
        self.assertNotIn("actor_rollout_ref.rollout.gpu_memory_utilization=0.85", plan.command)
        self.assertNotIn("actor_rollout_ref.rollout.max_num_seqs=2048", plan.command)
        self.assertNotIn("actor_rollout_ref.rollout.max_num_batched_tokens=65536", plan.command)

    def test_infer_runtime_env_strips_dotenv_vllm_knobs(self) -> None:
        loaded_config = load_example_config()
        args = Namespace(
            model="g1g-1.5b",
            dry_run=True,
            wkv_mode=None,
            emb_device=None,
            host=None,
            port=None,
            served_model_name=None,
            tensor_parallel_size=None,
            gpu_memory_utilization=None,
            max_model_len=None,
            max_num_seqs=None,
            max_num_batched_tokens=None,
            enable_auto_tool_choice=None,
        )

        plan = commands.build_infer_plan(
            args,
            root=ROOT,
            env={
                "WEIGHT_PATH": "/weights/RWKV",
                "VLLM_RWKV7_WKV_MODE": "fp32io16",
                "VLLM_GPU_MEMORY_UTILIZATION": "0.85",
                "VLLM_MAX_NUM_SEQS": "2048",
            },
            config=loaded_config,
        )

        self.assertEqual(plan.env["VLLM_RWKV7_WKV_MODE"], "fp32io16")
        self.assertNotIn("VLLM_GPU_MEMORY_UTILIZATION", plan.env)
        self.assertNotIn("VLLM_MAX_NUM_SEQS", plan.env)
        self.assertNotIn("--gpu-memory-utilization", plan.command)
        self.assertNotIn("--max-num-seqs", plan.command)

    def test_takeoff_config_adv_estimator_becomes_hydra_overrides(self) -> None:
        loaded_config = load_example_config()
        takeoff = loaded_config["takeoff"]
        takeoff["grpo"] = {**takeoff["grpo"], "adv_estimator": "maxrl", "reward_manager": "dapo"}
        venv_python = ROOT / ".venv/bin/python"
        original_exists = Path.exists
        args = Namespace(
            algorithm="grpo",
            model="g1g-1.5b",
            dataset="gsm8k",
            dry_run=True,
            wkv_mode=None,
            emb_device=None,
            num_nodes=None,
            num_devices=None,
            override=None,
        )

        with mock.patch.object(Path, "exists", autospec=True) as exists:
            exists.side_effect = lambda path: True if path == venv_python else original_exists(path)
            plan = commands.build_takeoff_plan(
                args,
                root=ROOT,
                env={"WEIGHT_PATH": "/weights/RWKV", "DATASETS_PATH": "/datasets"},
                config=loaded_config,
            )

        self.assertIn("algorithm.adv_estimator=maxrl", plan.command)
        self.assertIn("reward.reward_manager.name=dapo", plan.command)

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
        venv_python = ROOT / ".venv/bin/python"
        original_exists = Path.exists
        args = Namespace(
            algorithm="grpo",
            model="g1g-1.5b",
            dataset="gsm8k",
            dry_run=True,
            wkv_mode=None,
            emb_device=None,
            num_nodes=None,
            num_devices=None,
            override=None,
        )

        with mock.patch.object(Path, "exists", autospec=True) as exists:
            exists.side_effect = lambda path: True if path == venv_python else original_exists(path)
            plan = commands.build_takeoff_plan(
                args,
                root=ROOT,
                env={"WEIGHT_PATH": "/weights/RWKV", "DATASETS_PATH": "/datasets"},
                config=loaded_config,
            )

        self.assertIn("trainer.n_gpus_per_node=7", plan.command)
        self.assertIn("rollout.n_gpus_per_node=1", plan.command)
        self.assertIn("actor_rollout_ref.rollout.n_gpus_per_node=1", plan.command)
        self.assertIn("actor_rollout_ref.rollout.data_parallel_size=1", plan.command)
        self.assertIn("actor_rollout_ref.rollout.pipeline_model_parallel_size=1", plan.command)

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
        venv_python = ROOT / ".venv/bin/python"
        original_exists = Path.exists
        args = Namespace(
            algorithm="grpo",
            model="g1g-1.5b",
            dataset="dapo_math_17k",
            dry_run=True,
            wkv_mode=None,
            emb_device=None,
            num_nodes=None,
            num_devices=None,
            override=None,
        )

        with mock.patch.object(Path, "exists", autospec=True) as exists:
            exists.side_effect = lambda path: True if path == venv_python else original_exists(path)
            plan = commands.build_takeoff_plan(
                args,
                root=ROOT,
                env={"WEIGHT_PATH": "/weights/RWKV", "DATASETS_PATH": "/datasets"},
                config=loaded_config,
            )

        self.assertIn("data.train_files=['/datasets/DAPO/dapo-math-17k.parquet']", plan.command)
        self.assertIn(
            "data.val_files=['/datasets/AIME24/test.parquet','/datasets/AIME25/test.parquet']",
            plan.command,
        )
        self.assertNotIn(
            "['/datasets/AIME24/test.parquet','/datasets/AIME25/test.parquet']",
            plan.shown_env.values(),
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

        self.assertIn(
            f"data.train_files=['{missing_dataset_root}/DAPO/dapo-math-17k.parquet']",
            plan.command,
        )

    def test_takeoff_user_overrides_are_appended_after_generated_overrides(self) -> None:
        loaded_config = load_example_config()
        venv_python = ROOT / ".venv/bin/python"
        original_exists = Path.exists
        args = Namespace(
            algorithm="grpo",
            model="g1g-1.5b",
            dataset="gsm8k",
            dry_run=True,
            wkv_mode=None,
            emb_device=None,
            num_nodes=None,
            num_devices=None,
            override=["trainer.total_epochs=1", "trainer.save_freq=10"],
        )

        with mock.patch.object(Path, "exists", autospec=True) as exists:
            exists.side_effect = lambda path: True if path == venv_python else original_exists(path)
            plan = commands.build_takeoff_plan(
                args,
                root=ROOT,
                env={"WEIGHT_PATH": "/weights/RWKV", "DATASETS_PATH": "/datasets"},
                config=loaded_config,
            )

        generated_total_epochs = plan.command.index("trainer.total_epochs=2")
        user_total_epochs = plan.command.index("trainer.total_epochs=1")
        user_save_freq = plan.command.index("trainer.save_freq=10")
        self.assertLess(generated_total_epochs, user_total_epochs)
        self.assertEqual(plan.command[-2:], ["trainer.total_epochs=1", "trainer.save_freq=10"])
        self.assertLess(plan.command.index("trainer.save_freq=20"), user_save_freq)

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

        self.assertIn("Python executable not found", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
