from __future__ import annotations

import os
import subprocess
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class InstallPolicyTests(unittest.TestCase):
    def test_dependency_groups_are_capability_scoped(self) -> None:
        manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        groups = manifest["dependency-groups"]

        self.assertEqual(
            set(groups),
            {"dev", "rwkv-lm", "verl-liger", "verl-rwkv", "vllm-rwkv"},
        )
        self.assertEqual(manifest["tool"]["uv"]["default-groups"], [])
        self.assertIn("pre-commit", groups["dev"])
        self.assertIn("nvidia-ml-py>=12.560.30", groups["verl-rwkv"])
        self.assertNotIn("full", groups)

    def test_vllm_group_matches_rwkv_requirements(self) -> None:
        manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        requirements = {
            line.strip()
            for line in (
                ROOT / "src" / "infer" / "vllm-rwkv" / "requirements" / "rwkv.txt"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith(("#", "-"))
        }

        self.assertEqual(set(manifest["dependency-groups"]["vllm-rwkv"]), requirements)

    def test_installers_reject_full_before_external_actions(self) -> None:
        disabled = (
            ("INSTALL_COMPONENTS", "full"),
            ("INSTALL_PROFILE", "full"),
            ("HELICOPTER_VLLM_BUILD_PROFILE", "full"),
            ("VLLM_BUILD_PROFILE", "full"),
        )
        for script in ("install_local.sh", "install_remote.sh"):
            for variable, value in disabled:
                with self.subTest(script=script, variable=variable):
                    environment = {
                        **os.environ,
                        "ENV_FILE": "/dev/null",
                        "INSTALL_COMPONENTS": "dev",
                        variable: value,
                    }
                    result = subprocess.run(
                        ["bash", str(ROOT / "scripts" / script)],
                        cwd=ROOT,
                        env=environment,
                        text=True,
                        capture_output=True,
                        check=False,
                    )

                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("full is disabled", result.stderr)


if __name__ == "__main__":
    unittest.main()
