from __future__ import annotations

import os
import subprocess
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class InstallPolicyTests(unittest.TestCase):
    def test_feature_adds_only_the_rwkv_hf_capability_group(self) -> None:
        manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        groups = manifest["dependency-groups"]

        self.assertEqual(
            set(groups),
            {"dev", "rwkv-hf", "rwkv-lm", "verl-liger", "verl-rwkv", "vllm-rwkv"},
        )
        self.assertEqual(groups["dev"], ["pre-commit", "pytest", "pytest-asyncio"])
        self.assertIn("causal-conv1d==1.6.2.post1", groups["rwkv-hf"])
        self.assertIn("flash-linear-attention==0.5.1", groups["rwkv-hf"])
        self.assertNotIn("full", groups)
        self.assertEqual(manifest["tool"]["uv"]["default-groups"], [])

    def test_any2rwkv_declares_its_direct_runtime_dependencies(self) -> None:
        manifest = tomllib.loads(
            (ROOT / "src" / "train" / "any2rwkv" / "pyproject.toml").read_text(
                encoding="utf-8"
            )
        )

        dependencies = set(manifest["project"]["dependencies"])
        self.assertIn("datasets>=5.0.0", dependencies)
        self.assertIn("numpy>=2.0.0", dependencies)
        self.assertIn("rwkv7-hf-adapter==0.5.0", dependencies)
        self.assertEqual(
            manifest["tool"]["uv"]["sources"]["rwkv7-hf-adapter"],
            {"path": "../rwkv-hf", "editable": True},
        )

    def test_installers_accept_rwkv_hf_and_reject_full(self) -> None:
        accepted = subprocess.run(
            ["bash", str(ROOT / "scripts/install_local.sh")],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "1",
                "INSTALL_COMPONENTS": "rwkv-hf,dev",
                "UPDATE_UV": "0",
                "VLLM_TARGET_DEVICE": "cpu",
            },
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertIn("src/train/rwkv-hf", accepted.stdout)
        self.assertIn("src/train/any2rwkv", accepted.stdout)

        remote = (ROOT / "scripts/install_remote.sh").read_text(encoding="utf-8")
        self.assertIn("rwkv-hf", remote)

        for script in ("install_local.sh", "install_remote.sh"):
            source = ROOT / "scripts" / script
            rejected = subprocess.run(
                ["bash", str(source)],
                cwd=ROOT,
                env={
                    **os.environ,
                    "ENV_FILE": "/dev/null",
                    "INSTALL_COMPONENTS": "full",
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("full is disabled", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
