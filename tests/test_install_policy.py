from __future__ import annotations

import os
import subprocess
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_GROUPS = {"dev", "rwkv-lm", "verl-liger", "verl-rwkv", "vllm-rwkv"}


class InstallPolicyTests(unittest.TestCase):
    def test_root_manifest_keeps_the_main_dependency_groups(self) -> None:
        manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(set(manifest["dependency-groups"]), MAIN_GROUPS)
        self.assertEqual(
            manifest["dependency-groups"]["dev"],
            ["pre-commit", "pytest", "pytest-asyncio"],
        )
        self.assertEqual(manifest["tool"]["uv"]["default-groups"], [])
        self.assertNotIn("full", manifest["dependency-groups"])

    def test_child_components_own_their_manifests_and_locks(self) -> None:
        for relative in (
            "src/eval/lighteval/pyproject.toml",
            "src/eval/lighteval/uv.lock",
            "src/scoreboard-server/pyproject.toml",
            "src/scoreboard-server/uv.lock",
            "src/scoreboard-client/package.json",
            "src/scoreboard-client/bun.lock",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)

        lighteval = tomllib.loads(
            (ROOT / "src/eval/lighteval/pyproject.toml").read_text(encoding="utf-8")
        )
        dependencies = set(lighteval["project"]["dependencies"])
        self.assertTrue(any(item.startswith("httpx>=") for item in dependencies))
        self.assertTrue(any(item.startswith("openai>=") for item in dependencies))
        self.assertTrue(any(item.startswith("lighteval @ git+") for item in dependencies))

    def test_local_installer_routes_lighteval_outside_root_groups(self) -> None:
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/install_local.sh")],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "1",
                "INSTALL_COMPONENTS": "lighteval,dev",
                "UPDATE_UV": "0",
                "UV_UPGRADE": "0",
                "VLLM_TARGET_DEVICE": "cpu",
            },
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--group dev", result.stdout)
        self.assertNotIn("--group lighteval", result.stdout)
        self.assertIn("--project", result.stdout)
        self.assertIn("src/eval/lighteval", result.stdout)
        self.assertIn("--active", result.stdout)

    def test_installers_reject_full_before_external_actions(self) -> None:
        for script in ("install_local.sh", "install_remote.sh"):
            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / script)],
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
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("full is disabled", result.stderr)

    def test_local_installer_routes_scoreboard_components(self) -> None:
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/install_local.sh")],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "1",
                "INSTALL_COMPONENTS": "scoreboard-server,scoreboard-client,dev",
                "UPDATE_UV": "0",
                "UV_UPGRADE": "0",
                "VLLM_TARGET_DEVICE": "cpu",
            },
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--group scoreboard-server", result.stdout)
        self.assertNotIn("--group scoreboard-client", result.stdout)
        self.assertIn("src/scoreboard-server", result.stdout)
        self.assertIn("src/scoreboard-client", result.stdout)
        self.assertIn("--frozen-lockfile", result.stdout)


if __name__ == "__main__":
    unittest.main()
