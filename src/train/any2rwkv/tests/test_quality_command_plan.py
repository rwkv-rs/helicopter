from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class QualityCommandPlanTests(unittest.TestCase):
    def test_plan_pins_checkouts_and_emits_frozen_task_commands(self) -> None:
        package = Path(__file__).resolve().parents[1]
        suite = json.loads((package / "manifests" / "quality-suite.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkouts = {}
            for name, revision in (
                ("ruler", suite["ruler"]["revision"]),
                ("nemo", suite["ruler"]["implementation_revision"]),
                ("lm_eval", suite["downstream"]["revision"]),
            ):
                checkout = root / name
                checkout.mkdir()
                subprocess.run(["git", "init", "-q", str(checkout)], check=True)
                subprocess.run(["git", "-C", str(checkout), "config", "user.name", "test"], check=True)
                subprocess.run(["git", "-C", str(checkout), "config", "user.email", "test@example.com"], check=True)
                (checkout / "revision").write_text(revision, encoding="utf-8")
                subprocess.run(["git", "-C", str(checkout), "add", "revision"], check=True)
                subprocess.run(
                    ["git", "-C", str(checkout), "commit", "-q", "--date=2000-01-01T00:00:00Z", "-m", name],
                    check=True,
                    env={**__import__("os").environ, "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z"},
                )
                # The command validates the full SHA. Replace the fixture's expected
                # revision with its deterministic local commit without weakening production checks.
                checkouts[name] = (checkout, subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip())
            local_suite = root / "quality-suite.json"
            suite["ruler"]["revision"] = checkouts["ruler"][1]
            suite["ruler"]["implementation_revision"] = checkouts["nemo"][1]
            suite["downstream"]["revision"] = checkouts["lm_eval"][1]
            local_suite.write_text(json.dumps(suite), encoding="utf-8")
            tokenizer = root / "tokenizer"
            tokenizer.mkdir()
            output = root / "plan"
            subprocess.run(
                [
                    sys.executable, str(package / "scripts" / "build_quality_command_plan.py"),
                    "--quality-suite", str(local_suite), "--ruler-checkout", str(checkouts["ruler"][0]),
                    "--nemo-skills-checkout", str(checkouts["nemo"][0]), "--lm-eval-checkout", str(checkouts["lm_eval"][0]),
                    "--model", "fixture/model", "--tokenizer", str(tokenizer), "--base-url", "http://127.0.0.1:8000",
                    "--cluster", "local", "--role", "student", "--target", "proxy", "--output", str(output),
                ],
                check=True,
            )
            plan = json.loads((output / "quality-command-plan.json").read_text(encoding="utf-8"))
            ruler_eval = [row for row in plan["commands"] if row["suite"] == "ruler" and row["stage"] == "evaluate"]
            downstream = [row for row in plan["commands"] if row["suite"] == "downstream"]
            self.assertEqual(len(ruler_eval), 4)
            self.assertEqual(len(downstream), 6)
            self.assertTrue(all("--log_samples" in row["argv"] for row in downstream))


if __name__ == "__main__":
    unittest.main()
