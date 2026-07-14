from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from any2rwkv.artifacts import (
    REQUIRED_RUN_FILES,
    default_contract_lock,
    verify_run_bundle,
    verify_scale_gate,
    write_json,
)
from any2rwkv.calibration import file_sha256, read_calibration_manifest
from any2rwkv.cli import _quality_gate_passed
from any2rwkv.evaluate import P0_REQUIRED, QualityMetrics, migration_gate, p0_gate, paired_bootstrap_ratio_ci, quality_gate, validate_disjoint_splits
from any2rwkv.preflight import collect_preflight
from any2rwkv.nvfp4 import EXCLUDED_PATTERNS, _input_device, build_nvfp4_quant_config
from any2rwkv.quantize import nvfp4_performance_gate, nvfp4_policy, nvfp4_quality_gate
from any2rwkv.fixture import write_fixture


class ArtifactTests(unittest.TestCase):
    def test_quality_suite_freezes_ruler_downstream_and_bootstrap_contract(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "manifests"
            / "quality-suite.json"
        )
        suite = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(suite["ruler"]["revision"]), 40)
        self.assertEqual(len(suite["downstream"]["revision"]), 40)
        self.assertEqual(suite["ruler"]["task_count"], 12)
        self.assertEqual(suite["ruler"]["task_count"], len(suite["ruler"]["tasks"]))
        self.assertEqual(len(suite["ruler"]["implementation_revision"]), 40)
        self.assertEqual(suite["ruler"]["seed"], 42)
        self.assertEqual(len(suite["smoke"]["judge_revision"]), 40)
        self.assertEqual(suite["smoke"]["prompt_count"], 32)
        self.assertEqual(suite["bootstrap"]["samples"], 10_000)
        self.assertTrue(suite["quality_separation"]["nvfp4_calibration_use_forbidden"])

    def test_nvfp4_unlock_reads_the_evaluator_gate_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "quality.json"
            path.write_text(
                json.dumps({"gates": {"P1": {"passed": True}}}),
                encoding="utf-8",
            )
            self.assertTrue(_quality_gate_passed(path, "P1"))
            self.assertFalse(_quality_gate_passed(path, "P2"))

    def test_preflight_records_loader_nvfp4_and_hardware_boundaries(self) -> None:
        product_root = Path(__file__).resolve().parents[4]
        result = collect_preflight(product_root)
        self.assertEqual(result["rwkv_hf"]["import_name"], "rwkv7_hf")
        self.assertEqual(
            result["rwkv_lm"]["loader"],
            "any2rwkv.kernel.load_rwkv_lm_kernel",
        )
        self.assertIn("modelopt_available", result["nvfp4"])

    def test_modelopt_nvfp4_exclusions_extend_pattern_mapping(self) -> None:
        class FakeModelOpt:
            NVFP4_DEFAULT_CFG = {
                "quant_cfg": {"*weight_quantizer": {"enable": True}},
                "algorithm": "max",
            }

        config = build_nvfp4_quant_config(FakeModelOpt)
        self.assertTrue(config["quant_cfg"]["*weight_quantizer"]["enable"])
        self.assertTrue(
            all(config["quant_cfg"][pattern] == {"enable": False} for pattern in EXCLUDED_PATTERNS)
        )

    def test_nvfp4_calibration_uses_sharded_embedding_device(self) -> None:
        class Model:
            def get_input_embeddings(self):
                return torch.nn.Embedding(8, 4, device="cpu")

        self.assertEqual(_input_device(Model()), torch.device("cpu"))

    def test_contract_lock_freezes_oracle_bootstrap_sweeps_and_service(self) -> None:
        lock = default_contract_lock()
        self.assertGreaterEqual(lock["oracle"]["fixture_count"], 32)
        self.assertEqual(lock["bootstrap"]["samples"], 10_000)
        self.assertEqual(lock["corrective_sweeps"]["order"], "59..0")
        self.assertEqual(lock["serving"]["warmups"], 20)
        self.assertEqual(lock["serving"]["requests"], 100)

    def test_materialized_contract_lock_hashes_both_canonical_references(self) -> None:
        product_root = Path(__file__).resolve().parents[4]
        references = default_contract_lock(product_root)["oracle"]["reference_files"]
        self.assertEqual(
            set(references),
            {"rwkv7_fp64", "gdn_mapping", "oracle_fixture", "qwen35_gdn_cpu"},
        )
        self.assertTrue(all(len(entry["sha256"]) == 64 for entry in references.values()))

    def test_artifact_bundle_rejects_any_missing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            student_sha = "d" * 64
            write_json(root / "metadata.json", {"schema_version": 1})
            for name in REQUIRED_RUN_FILES:
                if name.endswith(".json"):
                    write_json(root / name, {"schema_version": 1})
                else:
                    (root / name).write_text('{"layer":0}\n', encoding="utf-8")
            write_json(root / "quality.json", {"gates": {"P0": {"passed": True}}})
            write_json(root / "p0-evidence.json", {"student_sha256": student_sha})
            write_json(root / "migration-baselines.json", {"student_sha256": student_sha})
            for name in ("kernel-oracle.json", "loss-bridge.json", "resume-parity.json"):
                write_json(
                    root / name,
                    {"passed": True, "student_sha256": student_sha},
                )
            self.assertEqual(len(verify_run_bundle(root)), len(REQUIRED_RUN_FILES) + 1)
            (root / "quality.json").unlink()
            with self.assertRaisesRegex(ValueError, "quality.json"):
                verify_run_bundle(root)

    def test_scale_gate_requires_p1_migration_and_student_bound_service(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            student_sha = "d" * 64
            write_json(root / "metadata.json", {"schema_version": 1})
            for name in REQUIRED_RUN_FILES:
                if name.endswith(".json"):
                    write_json(root / name, {"schema_version": 1})
                else:
                    (root / name).write_text('{"layer":0}\n', encoding="utf-8")
            write_json(
                root / "quality.json",
                {
                    "gates": {
                        "P0": {"passed": True},
                        "migration": {"passed": True},
                        "P1": {"passed": True},
                    }
                },
            )
            write_json(root / "p0-evidence.json", {"student_sha256": student_sha})
            write_json(
                root / "migration-baselines.json", {"student_sha256": student_sha}
            )
            for name in ("kernel-oracle.json", "loss-bridge.json", "resume-parity.json"):
                write_json(
                    root / name,
                    {"passed": True, "student_sha256": student_sha},
                )
            write_json(
                root / "vllm-service.json",
                {
                    "schema_version": 1,
                    "passed": True,
                    "model_sha256": student_sha,
                    "warmups": 20,
                    "requests": 100,
                },
            )
            write_json(
                root / "smoke-rubric.json",
                {"schema_version": 1, "passed": True, "student_sha256": student_sha},
            )
            self.assertEqual(
                verify_scale_gate(root)["student_sha256"], student_sha
            )
            service = json.loads(
                (root / "vllm-service.json").read_text(encoding="utf-8")
            )
            service["model_sha256"] = "e" * 64
            write_json(root / "vllm-service.json", service)
            with self.assertRaisesRegex(ValueError, "student-bound"):
                verify_scale_gate(root)


class GateTests(unittest.TestCase):
    def metrics(self) -> QualityMetrics:
        return QualityMetrics(1.1, 0.2, (0.96,) * 60, (0.08,) * 60, 0.95, 0.90, 0.85, 0.94, 3.0)

    def test_p1_p2_and_migration_gates(self) -> None:
        self.assertTrue(p0_gate({name: True for name in P0_REQUIRED}).passed)
        self.assertFalse(p0_gate({}).passed)
        self.assertTrue(quality_gate(self.metrics(), level="P1").passed)
        self.assertTrue(quality_gate(self.metrics(), level="P2").passed)
        self.assertTrue(migration_gate({"random": 3, "naive_copy": 2, "mapped": 1.5, "activation_fitted": 1.2, "layerwise_distilled": 1}).passed)

    def test_nvfp4_policy_and_independent_quality_performance_gates(self) -> None:
        decisions = nvfp4_policy((
            ("model.layers.0.attn.r_proj.weight", "bfloat16", 2),
            ("model.layers.0.mlp.experts.gate_up_proj", "bfloat16", 3),
            ("model.embeddings.weight", "bfloat16", 2),
            ("model.layers.0.attn.r_k", "float32", 2),
        ))
        self.assertTrue(decisions[0].quantized)
        self.assertTrue(decisions[1].quantized)
        self.assertFalse(decisions[2].quantized)
        self.assertFalse(decisions[3].quantized)
        self.assertTrue(nvfp4_quality_gate(ppl_increase=0.04, mean_kl=0.04, ruler_ratio=0.99, downstream_ratio=0.99))
        self.assertTrue(nvfp4_performance_gate(throughput_gain=0.16, ttft_p95_regression=0.04, tpot_p95_regression=0.04, memory_reduction=0.11))

    def test_calibration_manifest_binds_disjoint_rows_by_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "calibration.jsonl"
            data.write_text('{"text":"alpha"}\n{"text":"beta"}\n', encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "split": "nvfp4_calibration",
                        "data_file": data.name,
                        "sha256": file_sha256(data),
                        "row_count": 2,
                        "text_field": "text",
                        "max_length": 128,
                        "batch_size": 1,
                    }
                ),
                encoding="utf-8",
            )
            loaded = read_calibration_manifest(manifest)
            self.assertEqual(loaded.texts(), ("alpha", "beta"))
            data.write_text('{"text":"changed"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                read_calibration_manifest(manifest)

    def test_paired_bootstrap_and_split_exclusivity_are_frozen(self) -> None:
        lower, upper = paired_bootstrap_ratio_ci([0.9, 1.0, 0.8], [1.0, 1.0, 1.0], samples=10_000)
        self.assertLessEqual(lower, upper)
        splits = {name: [name] for name in ("distill_train", "nvfp4_calibration", "validation", "ruler", "downstream", "smoke")}
        self.assertEqual(validate_disjoint_splits(splits)["validation"], 1)
        splits["ruler"] = ["validation"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_disjoint_splits(splits)

    def test_calibration_builder_decodes_only_frozen_calibration_split(self) -> None:
        script = Path(__file__).parents[1] / "scripts" / "build_calibration_manifest.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = write_fixture(root / "tokenizer", layers=4)
            packed = root / "nvfp4_calibration.jsonl"
            packed.write_text(
                json.dumps(
                    {
                        "row_id": "calibration-0",
                        "split": "nvfp4_calibration",
                        "input_ids": [1, 2, 3, 4],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            prepared = root / "data-splits.json"
            prepared.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "prepared",
                        "splits": {
                            "nvfp4_calibration": {
                                "path": packed.name,
                                "sha256": file_sha256(packed),
                                "row_count": 1,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "calibration"
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--prepared-manifest",
                    str(prepared),
                    "--tokenizer-path",
                    str(tokenizer),
                    "--output-dir",
                    str(output),
                    "--max-length",
                    "128",
                ],
                check=True,
            )
            manifest = read_calibration_manifest(
                output / "calibration-manifest.json"
            )
            self.assertEqual(manifest.row_count, 1)
            self.assertEqual(manifest.max_length, 128)
            self.assertTrue(manifest.texts()[0])


if __name__ == "__main__":
    unittest.main()
