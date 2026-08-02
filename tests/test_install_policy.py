from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING_COMPONENTS = "rwkv-lm,vllm-rwkv,verl-rwkv,lighteval,lm-eval,dev"


class InstallPolicyTests(unittest.TestCase):
    def test_lm_eval_suite_extras_are_locked(self) -> None:
        manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))

        self.assertIn(
            "lm-eval[ifeval,longbench]==0.4.12",
            manifest["dependency-groups"]["lm-eval"],
        )
        self.assertIn(
            "datasets==3.6.0",
            manifest["dependency-groups"]["lm-eval"],
        )
        self.assertTrue(
            any(
                package["name"] == "datasets" and package["version"] == "3.6.0"
                for package in lock["package"]
            )
        )
        conflicts = manifest["tool"]["uv"]["conflicts"]
        self.assertIn(
            [{"group": "lighteval"}, {"group": "lm-eval"}],
            conflicts,
        )
        packages = {package["name"] for package in lock["package"]}
        self.assertGreaterEqual(
            packages,
            {"fuzzywuzzy", "immutabledict", "jieba", "langdetect", "nltk", "rouge"},
        )

    def test_verl_runtime_dependency_is_locked(self) -> None:
        manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))

        self.assertIn(
            "nvidia-ml-py>=12.560.30",
            manifest["dependency-groups"]["verl-rwkv"],
        )
        self.assertTrue(
            any(package["name"] == "nvidia-ml-py" for package in lock["package"])
        )

    def test_training_install_defaults_are_consistent(self) -> None:
        self.assertIn(
            f"INSTALL_COMPONENTS={TRAINING_COMPONENTS}",
            (ROOT / ".env.example").read_text(encoding="utf-8"),
        )
        for script in ("install_local.sh", "install_remote.sh"):
            source = (ROOT / "scripts" / script).read_text(encoding="utf-8")
            self.assertIn(
                f'INSTALL_COMPONENTS="${{INSTALL_COMPONENTS:-{TRAINING_COMPONENTS}}}"',
                source,
            )
            self.assertIn("INSTALL_COMPONENTS=full is disabled", source)

        local = (ROOT / "scripts/install_local.sh").read_text(encoding="utf-8")
        remote = (ROOT / "scripts/install_remote.sh").read_text(encoding="utf-8")
        self.assertIn('--group lm-eval', local)
        self.assertIn(".venv-lm-eval/", remote)

        launcher = (ROOT / "scripts/run_rwkv_vllm.sh").read_text(encoding="utf-8")
        self.assertNotIn("/home/creator", launcher)
        self.assertIn('RWKV_MODEL_PATH:-${model_path}', launcher)
        self.assertIn('VLLM_RWKV7_WKV_MODE:-fp16', launcher)

        workflow = (ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8")
        self.assertIn("--group test --group lm-eval", workflow)


if __name__ == "__main__":
    unittest.main()
