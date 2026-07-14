from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from any2rwkv.artifacts import checkpoint_sha256
from any2rwkv.fixture import write_fixture
from any2rwkv.nvfp4_acceptance import validate_nvfp4_acceptance


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class NVFP4AcceptanceTests(unittest.TestCase):
    def test_quality_and_performance_labels_require_all_bound_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bf16 = write_fixture(root / "bf16", layers=4)
            nvfp4 = write_fixture(root / "nvfp4", layers=4)
            nvfp4_config = json.loads((nvfp4 / "config.json").read_text(encoding="utf-8"))
            nvfp4_config["quantization_config"] = {"quant_method": "nvfp4"}
            write(nvfp4 / "config.json", nvfp4_config)
            bf16_sha = checkpoint_sha256(bf16)
            nvfp4_sha = checkpoint_sha256(nvfp4)
            bf16_quality = root / "bf16-quality.json"
            quality = root / "quality.json"
            p0 = root / "p0.json"
            service = root / "service.json"
            roundtrip = root / "roundtrip.json"
            performance = root / "performance.json"
            write(bf16_quality, {"gates": {"P2": {"passed": True}}})
            write(
                quality,
                {
                    "binding": {"teacher_sha256": bf16_sha, "student_sha256": nvfp4_sha},
                    "metrics": {"quality_metrics": {"ppl_ratio": 1.04, "mean_token_kl": 0.04, "ruler_ci_lower_ratio": 0.99, "downstream_ci_lower_ratio": 0.99}},
                },
            )
            write(p0, {"student_sha256": nvfp4_sha, "evidence": {"canonical": {"passed": True}}})
            write(service, {"model_sha256": nvfp4_sha, "passed": True})
            write(roundtrip, {"checkpoint": str(nvfp4), "loading_info": {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}, "prompt_count": 32, "new_tokens": 128})
            write(performance, {"bf16_sha256": bf16_sha, "nvfp4_sha256": nvfp4_sha, "decode_throughput_gain": 0.16, "ttft_p95_regression": 0.04, "tpot_p95_regression": 0.04, "peak_memory_reduction": 0.11})
            result = validate_nvfp4_acceptance(
                bf16_checkpoint=bf16, nvfp4_checkpoint=nvfp4,
                bf16_quality_path=bf16_quality, nvfp4_quality_path=quality,
                p0_evidence_path=p0, service_evidence_path=service,
                roundtrip_path=roundtrip, performance_path=performance,
            )
            self.assertTrue(result["quality_compatible"])
            self.assertTrue(result["performance_deliverable"])
            self.assertEqual(result["label"], "nvfp4-performance-deliverable")
            write(service, {"model_sha256": "0" * 64, "passed": True})
            rejected = validate_nvfp4_acceptance(
                bf16_checkpoint=bf16, nvfp4_checkpoint=nvfp4,
                bf16_quality_path=bf16_quality, nvfp4_quality_path=quality,
                p0_evidence_path=p0, service_evidence_path=service,
                roundtrip_path=roundtrip, performance_path=None,
            )
            self.assertFalse(rejected["quality_compatible"])
            self.assertEqual(rejected["label"], "experimental-nvfp4")


if __name__ == "__main__":
    unittest.main()
