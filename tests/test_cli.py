from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
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
        self.assertIn("VLLM_RWKV7_WKV_MODE=fp16", output)
        self.assertIn("VLLM_RWKV7_EMB_DEVICE=gpu", output)

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
        self.assertIn("bash", output)
        self.assertIn("run_rwkv7_grpo_vllm.sh", output)
        self.assertIn(f"PYTHON={venv_python}", output)
        self.assertIn("RWKV_USE_DYNAMIC_BSZ=False", output)
        self.assertIn("VLLM_RWKV7_WKV_MODE=fp32io16", output)

    def test_takeoff_rejects_missing_default_venv_python(self) -> None:
        config = load_example_config()
        env = {**os.environ}

        with self.assertRaises(SystemExit) as raised:
            commands.python_executable(config, root=ROOT, env=env, require_configured=True)

        self.assertIn("Python executable not found", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
