from __future__ import annotations

import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from helicopter_cli import commands, config, env
from helicopter_cli import __main__ as cli_main
from helicopter_cli.__main__ import build_parser


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "configs/example.toml"


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


class DotenvTests(unittest.TestCase):
    def test_load_dotenv_supports_simple_export_and_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "PLAIN=value\nexport EXPORTED=enabled\nQUOTED='space value'\n",
                encoding="utf-8",
            )

            self.assertEqual(
                env.load_dotenv(env_file),
                {
                    "PLAIN": "value",
                    "EXPORTED": "enabled",
                    "QUOTED": "space value",
                },
            )

    def test_command_environment_wins_over_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env.local").write_text(
                "WEIGHT_PATH=/from-file\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"WEIGHT_PATH": "/from-env"}):
                loaded, _ = env.load_env(root, ".env.local")

            self.assertEqual(loaded["WEIGHT_PATH"], "/from-env")


class EvaluationCliTests(unittest.TestCase):
    def test_eval_accepts_config_env_file_and_dry_run(self) -> None:
        args = build_parser().parse_args(
            [
                "eval",
                "--config",
                "configs/eval/maxrl_math.toml",
                "--env-file",
                ".env.remote",
                "--dry-run",
            ]
        )

        self.assertEqual(args.command, "eval")
        self.assertEqual(args.evaluator, "lighteval")
        self.assertEqual(args.config, "configs/eval/maxrl_math.toml")
        self.assertEqual(args.env_file, ".env.remote")
        self.assertTrue(args.dry_run)

    def test_eval_dispatches_lm_eval_to_its_isolated_environment(self) -> None:
        root = Path("/workspace/helicopter")
        with (
            mock.patch.object(cli_main, "find_root", return_value=root),
            mock.patch.object(cli_main, "find_env_path", return_value=None),
            mock.patch.object(
                cli_main,
                "load_env",
                return_value=(
                    {
                        "HELICOPTER_LM_EVAL_PYTHON": ".venv-lm-eval/bin/python"
                    },
                    None,
                ),
            ),
            mock.patch.object(cli_main.os, "access", return_value=True),
            mock.patch.object(cli_main, "run_command", return_value=0) as run,
        ):
            result = cli_main.main(
                [
                    "eval",
                    "--evaluator",
                    "lm-eval",
                    "--config",
                    "configs/eval/lm_eval_ppl.toml",
                    "--dry-run",
                ]
            )

        self.assertEqual(result, 0)
        command = run.call_args.args[0]
        self.assertEqual(command[0], str(root / ".venv-lm-eval/bin/python"))
        self.assertEqual(command[1:3], ["-m", "helicopter_lm_eval"])
        self.assertEqual(command[-1], "--dry-run")


class InferPlanTests(unittest.TestCase):
    def test_example_config_builds_vllm_command(self) -> None:
        loaded, _ = config.load_config(ROOT, str(EXAMPLE_CONFIG))

        plan = commands.build_infer_plan(
            infer_args(),
            root=ROOT,
            env={"WEIGHT_PATH": "/weights/RWKV"},
            config=loaded,
        )

        self.assertEqual(
            plan.command[:3],
            [
                "vllm",
                "serve",
                "/weights/RWKV/rwkv7-g1g-1.5b-20260526-ctx8192.pth",
            ],
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
                "--max-num-seqs": "2560",
                "--max-num-batched-tokens": "2560",
            },
        )
        self.assertEqual(
            plan.shown_env,
            {
                "VLLM_USE_V2_MODEL_RUNNER": "1",
                "VLLM_RWKV7_WKV_MODE": "fp32io16",
            },
        )
        self.assertEqual(
            {key for key in plan.env if key.startswith("VLLM_")},
            {
                "VLLM_USE_V2_MODEL_RUNNER",
                "VLLM_RWKV7_WKV_MODE",
            },
        )
        self.assertEqual(plan.cwd, ROOT)

    def test_checkpoint_environment_override_is_respected(self) -> None:
        loaded, _ = config.load_config(ROOT, str(EXAMPLE_CONFIG))

        model_path, _ = config.resolve_model_path(
            loaded,
            "g1g-1.5b",
            root=ROOT,
            env={"HELICOPTER_CHECKPOINT_PATH": "/weights/selected.pth"},
        )

        self.assertEqual(model_path, Path("/weights/selected.pth"))

    def test_runtime_env_strips_dotenv_vllm_knobs(self) -> None:
        loaded, _ = config.load_config(ROOT, str(EXAMPLE_CONFIG))
        plan = commands.build_infer_plan(
            infer_args(),
            root=ROOT,
            env={
                "WEIGHT_PATH": "/weights/RWKV",
                "VLLM_RWKV7_WKV_MODE": "fp32io16",
                "HELICOPTER_INFER_ALLOW_FP16_ACCUMULATION": "0",
                "VLLM_GPU_MEMORY_UTILIZATION": "0.85",
                "VLLM_MAX_NUM_SEQS": "2048",
                "VLLM_USE_RAPID_SAMPLER": "1",
            },
            config=loaded,
        )

        options = command_options(plan.command)
        self.assertNotIn("VLLM_RWKV7_ALLOW_FP16_ACCUMULATION", plan.env)
        self.assertNotIn("VLLM_GPU_MEMORY_UTILIZATION", plan.env)
        self.assertNotIn("VLLM_MAX_NUM_SEQS", plan.env)
        self.assertNotIn("--gpu-memory-utilization", options)
        self.assertEqual(options["--max-num-seqs"], "2560")
        self.assertEqual(options["--max-num-batched-tokens"], "2560")
        self.assertNotIn("VLLM_USE_RAPID_SAMPLER", plan.env)
        self.assertEqual(plan.env["VLLM_USE_V2_MODEL_RUNNER"], "1")

    def test_cli_capacity_override_wins_over_model_profile(self) -> None:
        loaded, _ = config.load_config(ROOT, str(EXAMPLE_CONFIG))
        plan = commands.build_infer_plan(
            infer_args(max_num_seqs=64, max_num_batched_tokens=512),
            root=ROOT,
            env={"WEIGHT_PATH": "/weights/RWKV"},
            config=loaded,
        )

        options = command_options(plan.command)
        self.assertEqual(options["--max-num-seqs"], "64")
        self.assertEqual(options["--max-num-batched-tokens"], "512")

    def test_fp16_accumulation_cli_false_overrides_environment(self) -> None:
        loaded, _ = config.load_config(ROOT, str(EXAMPLE_CONFIG))
        plan = commands.build_infer_plan(
            infer_args(allow_fp16_accumulation=False),
            root=ROOT,
            env={
                "WEIGHT_PATH": "/weights/RWKV",
                "HELICOPTER_INFER_ALLOW_FP16_ACCUMULATION": "1",
            },
            config=loaded,
        )

        self.assertNotIn("VLLM_RWKV7_ALLOW_FP16_ACCUMULATION", plan.env)

    def test_fp16_wkv_enables_matching_accumulation_by_default(self) -> None:
        loaded, _ = config.load_config(ROOT, str(EXAMPLE_CONFIG))
        plan = commands.build_infer_plan(
            infer_args(wkv_mode="fp16"),
            root=ROOT,
            env={"WEIGHT_PATH": "/weights/RWKV"},
            config=loaded,
        )

        self.assertEqual(plan.env["VLLM_RWKV7_WKV_MODE"], "fp16")
        self.assertNotIn("VLLM_RWKV7_ALLOW_FP16_ACCUMULATION", plan.env)

    def test_rejects_accumulation_that_conflicts_with_wkv_profile(self) -> None:
        loaded, _ = config.load_config(ROOT, str(EXAMPLE_CONFIG))
        with self.assertRaisesRegex(
            SystemExit, "derives GEMM accumulation from WKV mode"
        ):
            commands.build_infer_plan(
                infer_args(wkv_mode="fp16", allow_fp16_accumulation=False),
                root=ROOT,
                env={"WEIGHT_PATH": "/weights/RWKV"},
                config=loaded,
            )

    def test_rejects_invalid_accumulation_environment_value(self) -> None:
        loaded, _ = config.load_config(ROOT, str(EXAMPLE_CONFIG))
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
                config=loaded,
            )


class TakeoffPlanTests(unittest.TestCase):
    def test_parser_exposes_only_config_and_forwarded_overrides(self) -> None:
        args = build_parser().parse_args(
            [
                "takeoff",
                "--config",
                "maxrl.toml",
                "--override",
                "trainer.save_freq=10",
            ]
        )

        self.assertEqual(args.config, "maxrl.toml")
        self.assertEqual(args.override, ["trainer.save_freq=10"])
        self.assertFalse(hasattr(args, "model"))
        self.assertFalse(hasattr(args, "dataset"))
        self.assertFalse(hasattr(args, "algorithm"))

    def test_takeoff_is_a_transparent_verl_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "maxrl.toml"
            config_path.write_text("[algorithm]\nname = 'maxrl'\n", encoding="utf-8")
            for relative in (
                "src/train/verl-rwkv",
                "src/train/rwkv-lm",
                "src/infer/vllm-rwkv",
                ".venv/bin",
            ):
                (root / relative).mkdir(parents=True)
            python = root / ".venv/bin/python"
            python.write_text("#!/bin/sh\n", encoding="utf-8")
            python.chmod(0o755)
            args = Namespace(
                config=str(config_path),
                override=["trainer.save_freq=10"],
                dry_run=True,
            )

            plan = commands.build_takeoff_plan(args, root=root, env={})

        self.assertEqual(plan.cwd, root / "src/train/verl-rwkv")
        self.assertEqual(
            plan.command,
            [
                str(python),
                "-m",
                "verl.trainer.maxrl",
                "--config",
                str(config_path),
                "--override",
                "trainer.save_freq=10",
                "--dry-run",
            ],
        )
        self.assertEqual(plan.env["RWKV_LM_PATH"], str(root / "src/train/rwkv-lm"))
        self.assertEqual(plan.env["PYTHONPATH"], str(root / "src/infer/vllm-rwkv"))
        self.assertEqual(plan.env["HELICOPTER_PRODUCT_ROOT"], str(root))
        self.assertFalse(any("data.train_batch_size=" in item for item in plan.command))


if __name__ == "__main__":
    unittest.main()
