from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from any2rwkv.evaluate import P0_REQUIRED
from any2rwkv.calibration import file_sha256
from any2rwkv.evaluator_runner import (
    EvaluationSample,
    EvaluatorConfig,
    PairedSampleScore,
    read_evaluation_manifest,
    read_migration_baselines,
    read_paired_scores,
    read_p0_evidence,
    run_evaluator,
)
from any2rwkv.fixture import write_fixture


class TinyMixer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.projection = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        mixed = self.projection(hidden)
        state = mixed.unsqueeze(1).unsqueeze(-1)
        return mixed, state


class TinyLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.attn = TinyMixer(hidden)

    def forward(self, hidden: Tensor) -> Tensor:
        mixed, _ = self.attn(self.input_layernorm(hidden))
        return hidden + mixed


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 11, hidden: int = 4, layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(TinyLayer(hidden) for _ in range(layers))
        self.model.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
    ):
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        logits = self.lm_head(self.model.norm(hidden))
        cache = (torch.tensor([input_ids.shape[1]]),) if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=cache)


class TinyTokenizer:
    def __call__(self, text: str, return_tensors: str):
        ids = [1 + (ord(character) % 9) for character in text] or [1]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return " ".join(str(value) for value in ids)


class EvaluatorRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.teacher = TinyLM().eval()
        self.student = TinyLM().eval()
        self.student.load_state_dict(self.teacher.state_dict())
        self.samples = (
            EvaluationSample("sample-a", (1, 2, 3, 4, 5)),
            EvaluationSample("sample-b", (5, 4, 3, 2, 1)),
        )
        self.prompts = tuple(f"prompt {index}" for index in range(32))
        self.config = EvaluatorConfig(
            split="validation",
            seed=23,
            tokenizer_sha256="a" * 64,
            dataset_sha256="b" * 64,
            teacher_sha256="c" * 64,
            student_sha256="d" * 64,
            burn_in_tokens=2,
            layer_schedule=(0, 1),
            smoke_new_tokens=3,
            bootstrap_samples=128,
        )

    def test_real_metrics_are_hash_bound_and_missing_suites_fail_p2(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "quality.json"
            result = run_evaluator(
                teacher=self.teacher,
                student=self.student,
                tokenizer=TinyTokenizer(),
                samples=self.samples,
                smoke_prompts=self.prompts,
                config=self.config,
                p0_evidence={name: True for name in P0_REQUIRED},
                output_path=output,
            )
            self.assertTrue(output.is_file())
        self.assertEqual(result["metrics"]["warmed"]["ppl_ratio"], 1.0)
        self.assertEqual(result["metrics"]["warmed"]["mean_token_kl"], 0.0)
        self.assertEqual(len(result["raw_sample_metrics"]["cold"]), 2)
        self.assertEqual(result["smoke"]["pass_count"], 32)
        self.assertEqual(result["external_evaluations"]["ruler"]["status"], "not_run")
        self.assertEqual(result["external_evaluations"]["downstream"]["status"], "not_run")
        self.assertTrue(result["gates"]["P0"]["passed"])
        self.assertTrue(result["gates"]["P1"]["passed"])
        self.assertFalse(result["gates"]["P2"]["passed"])
        self.assertEqual(result["gates"]["P2"]["failures"], ["ruler:not_run", "downstream:not_run"])
        self.assertEqual(len(result["binding"]["evaluator_input_sha256"]), 64)
        for kind in ("intermediate", "state", "output"):
            rows = result["metrics"]["warmed"]["layers"][kind]
            self.assertTrue(all(row["status"] == "run" for row in rows))
            self.assertTrue(all(row["normalized_mse"] == 0.0 for row in rows))

    def test_external_scores_preserve_bootstrap_inputs_and_enable_real_p2(self) -> None:
        scores = tuple(
            PairedSampleScore(f"score-{index}", 1.0, 0.98, "task-a" if index % 2 else "task-b")
            for index in range(8)
        )
        result = run_evaluator(
            teacher=self.teacher,
            student=self.student,
            tokenizer=TinyTokenizer(),
            samples=self.samples,
            smoke_prompts=self.prompts,
            config=self.config,
            p0_evidence={name: True for name in P0_REQUIRED},
            ruler_scores=scores,
            downstream_scores=scores,
        )
        ruler = result["external_evaluations"]["ruler"]
        self.assertEqual(ruler["status"], "run")
        self.assertEqual(ruler["bootstrap"]["samples"], 128)
        self.assertEqual(len(ruler["raw_sample_scores"]), 8)
        self.assertTrue(result["gates"]["P2"]["passed"])

    def test_p0_evidence_requires_hash_and_checkpoint_bound_artifacts(self) -> None:
        student_sha = "d" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "result.json"
            artifact.write_text(
                json.dumps(
                    {
                        "kind": "placeholder",
                        "passed": True,
                        "student_sha256": student_sha,
                    }
                ),
                encoding="utf-8",
            )
            evidence = {
                name: {
                    "path": f"{name}.json",
                    "sha256": "",
                    "student_sha256": student_sha,
                    "passed": True,
                }
                for name in P0_REQUIRED
            }
            for name in P0_REQUIRED:
                item = root / f"{name}.json"
                item.write_text(
                    json.dumps(
                        {
                            "kind": name,
                            "passed": True,
                            "student_sha256": student_sha,
                        }
                    ),
                    encoding="utf-8",
                )
                evidence[name]["sha256"] = file_sha256(item)
            manifest = root / "p0.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "student_sha256": student_sha,
                        "evidence": evidence,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                all(read_p0_evidence(manifest, student_sha256=student_sha).values())
            )
            (root / f"{P0_REQUIRED[0]}.json").write_text(
                '{"passed":false}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                read_p0_evidence(manifest, student_sha256=student_sha)

    def test_migration_matrix_is_checkpoint_bound_and_complete(self) -> None:
        student_sha = "e" * 64
        rows = {
            name: {"mean_token_kl": value}
            for name, value in {
                "random": 3.0,
                "naive_copy": 2.0,
                "mapped": 1.5,
                "activation_fitted": 1.2,
                "layerwise_distilled": 1.0,
            }.items()
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "migration-baselines.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "student_sha256": student_sha,
                        "baselines": rows,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                read_migration_baselines(path, student_sha256=student_sha)["mapped"],
                1.5,
            )
            with self.assertRaisesRegex(ValueError, "different student"):
                read_migration_baselines(path, student_sha256="f" * 64)

    def test_paired_score_builder_binds_suite_runner_and_checkpoints(self) -> None:
        teacher_sha = "a" * 64
        student_sha = "b" * 64
        runner_revision = "c" * 40
        script = Path(__file__).parents[1] / "scripts" / "build_paired_scores.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = root / "teacher.jsonl"
            student = root / "student.jsonl"
            teacher.write_text(
                '\n'.join(
                    json.dumps({"sample_id": name, "group": "task-a", "score": score})
                    for name, score in (("sample-1", 1.0), ("sample-2", 0.5))
                )
                + "\n",
                encoding="utf-8",
            )
            student.write_text(
                '\n'.join(
                    json.dumps({"sample_id": name, "group": "task-a", "score": score})
                    for name, score in (("sample-1", 0.9), ("sample-2", 0.4))
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "paired"
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--suite",
                    "ruler",
                    "--teacher",
                    str(teacher),
                    "--student",
                    str(student),
                    "--teacher-sha256",
                    teacher_sha,
                    "--student-sha256",
                    student_sha,
                    "--runner-repository",
                    "https://github.com/NVIDIA/RULER.git",
                    "--runner-revision",
                    runner_revision,
                    "--output",
                    str(output),
                ],
                check=True,
            )
            manifest = output / "manifest.json"
            scores = read_paired_scores(
                manifest,
                expected_suite="ruler",
                teacher_sha256=teacher_sha,
                student_sha256=student_sha,
            )
            self.assertEqual([score.sample_id for score in scores or ()], ["sample-1", "sample-2"])
            with self.assertRaisesRegex(ValueError, "different student"):
                read_paired_scores(
                    manifest,
                    expected_suite="ruler",
                    teacher_sha256=teacher_sha,
                    student_sha256="d" * 64,
                )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["runner_revision"] = "unpinned"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pinned runner revision"):
                read_paired_scores(manifest, expected_suite="ruler")

    def test_evaluation_manifest_builder_binds_prepared_splits_and_32_prompts(self) -> None:
        script = Path(__file__).parents[1] / "scripts" / "build_evaluation_manifest.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = write_fixture(root / "tokenizer", layers=4)
            validation = root / "validation.jsonl"
            smoke = root / "smoke.jsonl"
            validation.write_text(
                "".join(
                    json.dumps(
                        {
                            "row_id": f"validation-{index}",
                            "split": "validation",
                            "input_ids": [1, 2, 3, 4],
                            "burn_in_tokens": 2,
                            "supervised_tokens": 2,
                        }
                    )
                    + "\n"
                    for index in range(2)
                ),
                encoding="utf-8",
            )
            smoke.write_text(
                "".join(
                    json.dumps(
                        {
                            "row_id": f"smoke-{index}",
                            "split": "smoke",
                            "input_ids": [1, 2, 3, 4],
                            "burn_in_tokens": 2,
                            "supervised_tokens": 2,
                        }
                    )
                    + "\n"
                    for index in range(32)
                ),
                encoding="utf-8",
            )
            prepared = root / "data-splits.json"
            prepared.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "prepared",
                        "seed": 17,
                        "packing": {"burn_in_tokens": 2, "supervised_tokens": 2},
                        "splits": {
                            "validation": {
                                "path": validation.name,
                                "sha256": file_sha256(validation),
                                "row_count": 2,
                            },
                            "smoke": {
                                "path": smoke.name,
                                "sha256": file_sha256(smoke),
                                "row_count": 32,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "evaluation"
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
                ],
                check=True,
            )
            samples, prompts, manifest = read_evaluation_manifest(
                output / "evaluation-manifest.json"
            )
            self.assertEqual(len(samples), 2)
            self.assertEqual(len(prompts), 32)
            self.assertEqual(manifest["burn_in_tokens"], 2)
            (output / "smoke-prompts.jsonl").write_text(
                "tampered\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                read_evaluation_manifest(output / "evaluation-manifest.json")


if __name__ == "__main__":
    unittest.main()
