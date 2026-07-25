from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from any2rwkv.artifacts import checkpoint_sha256
from any2rwkv.evaluate import P0_REQUIRED
from any2rwkv.evaluator_runner import read_p0_evidence
from any2rwkv.p0_runner import P0ValidationInputs, _PYTEST_EVIDENCE, run_p0_validation


class P0RunnerTests(unittest.TestCase):
    def test_producer_runs_checks_and_binds_artifact_contents_to_student(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            student = root / "student"
            package_root = root / "package"
            student.mkdir()
            package_root.mkdir()
            (student / "config.json").write_text("{}\n", encoding="utf-8")
            (student / "model.safetensors").write_bytes(b"weights")
            (run_dir / "mapping-coverage.json").parent.mkdir(parents=True)
            (run_dir / "mapping-coverage.json").write_text(
                json.dumps({"source_coverage": 1.0, "target_coverage": 1.0}),
                encoding="utf-8",
            )
            for relative in {
                nodeid.split("::", 1)[0]
                for nodeids in _PYTEST_EVIDENCE.values()
                for nodeid in nodeids
            }:
                path = package_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# frozen test source\n", encoding="utf-8")
            kernel = root / "kernel.json"
            kernel.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "kernel": "rwkv-lm/RWKV7_STATEPASSING_CLAMPW_CUDA",
                        "gradient_finite": [True] * 7,
                    }
                ),
                encoding="utf-8",
            )
            roundtrip = {
                "model_sha256": checkpoint_sha256(student),
                "loading_info": {
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "mismatched_keys": [],
                    "error_msgs": [],
                },
                "greedy_digest": "a" * 64,
                "single_greedy_digest": "a" * 64,
                "batch_greedy_digest": "a" * 64,
                "logits_digest": "b" * 64,
                "ppl": 2.0,
                "backend": "transformers",
                "transformers_version": "fixture",
                "prompt_count": 32,
                "new_tokens": 128,
                "shards": {"tensor_count": 2, "shard_count": 1},
                "strict_reload": True,
                "single_batch_greedy_equal": True,
                "batch_isolation": {"passed": True},
                "full_chunked_cache": {"passed": True},
                "state_reset": {"passed": True},
                "passed": True,
            }
            completed = subprocess.CompletedProcess(["pytest"], 0, "passed", "")

            def write_roundtrip(_inputs, output):
                output.write_text(json.dumps(roundtrip), encoding="utf-8")
                return roundtrip

            with (
                mock.patch("any2rwkv.p0_runner._run_pytest", return_value=completed),
                mock.patch(
                    "any2rwkv.p0_runner._run_roundtrip",
                    side_effect=write_roundtrip,
                ),
            ):
                manifest = run_p0_validation(
                    P0ValidationInputs(run_dir, student, kernel, package_root)
                )
            self.assertEqual(set(manifest["evidence"]), set(P0_REQUIRED))
            inference = json.loads(
                (run_dir / "transformers-inference.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(inference["single_batch_greedy"])
            self.assertTrue(inference["full_chunked_cache"])
            self.assertTrue(inference["state_reset"])
            self.assertTrue(inference["batch_isolation"])
            self.assertTrue(
                all(
                    read_p0_evidence(
                        run_dir / "p0-evidence.json",
                        student_sha256=manifest["student_sha256"],
                    ).values()
                )
            )


if __name__ == "__main__":
    unittest.main()
