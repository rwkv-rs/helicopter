from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from any2rwkv.distill import (
    ActiveLayerTrainer,
    HybridReplacementRunner,
    LossWeights,
    SweepController,
    load_sharded_training_checkpoint,
    load_training_checkpoint,
    progressive_schedule,
    save_sharded_training_checkpoint,
    save_training_checkpoint,
)
from any2rwkv.errors import CoverageError
from any2rwkv.calibration import file_sha256
from any2rwkv.distill_runner import (
    _initial_trainable_names,
    _write_baseline_result,
    read_distillation_plan,
    read_distillation_texts,
    read_packed_token_rows,
)
from any2rwkv.mapping import MappingLedger, SourceDisposition, SourceEntry, TargetEntry, TargetProvenance, finalize_fitted_mapping


class MappingTests(unittest.TestCase):
    def test_resident_trainable_names_target_adapter_rwkv_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "warm-start-plan.json").write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "target": "model.layers.0.attn.r_proj.weight",
                                "provenance": "initialized",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                _initial_trainable_names(root, 1),
                [{"rwkv.r_proj.weight"}],
            )

    def test_baseline_result_is_written_atomically_and_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "migration-baselines.json"
            binding = {"source_sha256": "a" * 64}
            _write_baseline_result(
                path,
                name="random",
                metrics={"mean_token_kl": 1.25},
                binding=binding,
                token_budget=32,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["binding"], binding)
            self.assertEqual(payload["baselines"]["random"]["token_budget"], 32)

    def test_fitted_provenance_is_committed_only_after_training_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "mapping.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "targets": [
                            {"target": "model.layers.0.attn.r_proj.weight", "provenance": "copied", "sources": ["source.r"], "shape": [2, 2], "dtype": "bfloat16", "evidence": "warm start", "source_hashes": ["a" * 64]},
                            {"target": "model.layers.0.attn.w0", "provenance": "initialized", "sources": [], "shape": [2], "dtype": "bfloat16", "evidence": "seeded", "source_hashes": ["a" * 64]},
                        ],
                        "sources": [],
                    }
                ),
                encoding="utf-8",
            )
            (root / "warm-start-plan.json").write_text(
                json.dumps(
                    {
                        "entries": [
                            {"target": "model.layers.0.attn.r_proj.weight", "provenance": "copied", "is_semantically_lossless": False},
                            {"target": "model.layers.0.attn.w0", "provenance": "initialized", "is_semantically_lossless": False},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (root / "mapping-coverage.json").write_text(
                json.dumps({"source_total": 1, "target_total": 2, "source_coverage": 1.0, "target_coverage": 1.0}),
                encoding="utf-8",
            )
            coverage = finalize_fitted_mapping(
                root, student_sha256="b" * 64, trace_sha256="c" * 64
            )
            mapping = json.loads((root / "mapping.json").read_text(encoding="utf-8"))
            self.assertTrue(all(row["provenance"] == "fitted" for row in mapping["targets"]))
            self.assertEqual(coverage["provenance"]["fitted"], 2)
            self.assertEqual(coverage["fitted_student_sha256"], "b" * 64)

    def test_bidirectional_coverage_and_taxonomy(self) -> None:
        ledger = MappingLedger()
        ledger.add_source(SourceEntry("source.weight", SourceDisposition.CONSUMED, ("target.weight",), "projection fit"))
        ledger.add_target(TargetEntry("target.weight", TargetProvenance.FITTED, ("source.weight",), (2, 2), "float32", "ridge-v1", ("abc",)))
        coverage = ledger.validate(["source.weight"], ["target.weight"])
        self.assertEqual(coverage["source_coverage"], 1.0)
        self.assertEqual(coverage["target_coverage"], 1.0)
        with self.assertRaises(CoverageError):
            ledger.validate(["source.weight", "missing"], ["target.weight"])

    def test_mapping_edges_and_disposition_are_semantically_bidirectional(self) -> None:
        ledger = MappingLedger()
        ledger.add_source(
            SourceEntry(
                "source.weight",
                SourceDisposition.CONSUMED,
                (),
                "invalid missing target edge",
            )
        )
        ledger.add_target(
            TargetEntry(
                "target.weight",
                TargetProvenance.FITTED,
                ("source.weight",),
                (2, 2),
                "float32",
                "ridge-v1",
                ("abc",),
            )
        )
        with self.assertRaisesRegex(CoverageError, "missing-reverse-edge"):
            ledger.validate(["source.weight"], ["target.weight"])


class DistillationInvariantTests(unittest.TestCase):
    def make_layers(self):
        torch.manual_seed(3)
        return nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(60)])

    def run_forward(self, layers, value):
        hidden = value
        for layer in layers:
            hidden = torch.tanh(layer(hidden))
        return hidden

    def test_only_active_layer_gets_gradient_weight_and_optimizer_state(self) -> None:
        layers = self.make_layers()
        trainer = ActiveLayerTrainer(layers, lr=1e-2)
        trainer.activate(17)
        before = [copy.deepcopy(layer.state_dict()) for layer in layers]
        output = self.run_forward(layers, torch.ones(2, 4))
        trainer.step(output.square().mean())
        for index, layer in enumerate(layers):
            changed = any(not torch.equal(value, before[index][name]) for name, value in layer.state_dict().items())
            self.assertEqual(changed, index == 17)
            self.assertEqual(bool(trainer.optimizers[index].state), index == 17)
            if index != 17:
                self.assertTrue(all(parameter.grad is None for parameter in layer.parameters()))

    def test_resume_matches_uninterrupted_updates(self) -> None:
        initial = self.make_layers()
        uninterrupted = copy.deepcopy(initial)
        resumed = copy.deepcopy(initial)
        first = ActiveLayerTrainer(uninterrupted, lr=1e-2)
        second = ActiveLayerTrainer(resumed, lr=1e-2)
        inputs = [torch.full((2, 4), float(index + 1) / 10) for index in range(4)]
        for value in inputs[:2]:
            first.activate(2)
            first.step(self.run_forward(uninterrupted, value).square().mean())
            second.activate(2)
            second.step(self.run_forward(resumed, value).square().mean())
        checkpoint_layers = copy.deepcopy(resumed.state_dict())
        checkpoint_trainer = copy.deepcopy(second.state_dict())
        resumed = self.make_layers()
        resumed.load_state_dict(checkpoint_layers)
        second = ActiveLayerTrainer(resumed, lr=1e-2)
        second.load_state_dict(checkpoint_trainer)
        for value in inputs[2:]:
            first.step(self.run_forward(uninterrupted, value).square().mean())
            second.step(self.run_forward(resumed, value).square().mean())
        for left, right in zip(uninterrupted.parameters(), resumed.parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(progressive_schedule()[0], tuple(range(60)))
        self.assertEqual(progressive_schedule()[1], tuple(reversed(range(60))))

    def test_atomic_checkpoint_restores_scaler_rng_cursor_and_metadata(self) -> None:
        layers = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)])
        trainer = ActiveLayerTrainer(layers)
        trainer.activate(1)
        trainer.data_cursor = 13
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "layer.pt"
            digest = save_training_checkpoint(path, layers, trainer, metadata={"layout": ["source", "rwkv7"]})
            restored_layers = copy.deepcopy(layers)
            restored = ActiveLayerTrainer(restored_layers)
            metadata = load_training_checkpoint(path, restored_layers, restored)
            self.assertEqual(len(digest), 64)
            self.assertEqual(restored.data_cursor, 13)
            self.assertEqual(restored.scaler_state["enabled"], False)
            self.assertEqual(metadata["layout"], ["source", "rwkv7"])

    def test_sharded_checkpoint_rewrites_only_active_layer_and_restores_accumulation(self) -> None:
        layers = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(3)])
        trainer = ActiveLayerTrainer(layers, lr=1e-2)
        trainer.activate(1)
        value = torch.full((2, 4), 0.25)
        self.assertFalse(
            trainer.backward(self.run_forward(layers, value).square().mean(), accumulation_steps=2)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_sharded_training_checkpoint(
                root, layers, trainer, metadata={"next_visit": 7, "consumed": 11}
            )
            self.assertEqual(
                len(list((root / "layers").iterdir())), 1
            )
            restored_layers = copy.deepcopy(layers)
            restored = ActiveLayerTrainer(restored_layers, lr=1e-2)
            metadata = load_sharded_training_checkpoint(root, restored_layers, restored)
            self.assertEqual(restored.accumulation_step, 1)
            self.assertEqual(restored.active_layer, 1)
            self.assertEqual(metadata, {"next_visit": 7, "consumed": 11})
            self.assertTrue(
                any(parameter.grad is not None for parameter in restored_layers[1].parameters())
            )

    def test_loss_stages_are_explicit(self) -> None:
        self.assertEqual(LossWeights.for_stage("signals").token_kl, 0.0)
        self.assertGreater(LossWeights.for_stage("global").token_kl, 0.0)

    def test_real_runner_freezes_ordered_stage_budgets_and_hashed_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "seed": 7,
                        "learning_rate": 1e-4,
                        "burn_in_tokens": 2,
                        "supervised_tokens": 4,
                        "accumulation_steps": 2,
                        "stage_tokens_per_layer": {
                            "signals": 8,
                            "block": 16,
                            "global": 32,
                        },
                        "corrective_min_sweeps": 1,
                        "corrective_max_sweeps": 3,
                        "corrective_min_delta": 0.001,
                    }
                ),
                encoding="utf-8",
            )
            parsed_plan = read_distillation_plan(plan_path)
            self.assertEqual(parsed_plan.stage_tokens_per_layer["global"], 32)
            self.assertEqual(
                tuple(parsed_plan.stage_tokens_per_layer),
                ("signals", "block", "global"),
            )
            self.assertEqual(parsed_plan.execution_mode, "resident")
            data = root / "train.jsonl"
            data.write_text('{"text":"one"}\n{"text":"two"}\n', encoding="utf-8")
            manifest = root / "data.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "split": "distill_train",
                        "data_file": data.name,
                        "sha256": file_sha256(data),
                        "row_count": 2,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(read_distillation_texts(manifest), ("one", "two"))

            packed = root / "validation.jsonl"
            packed.write_text(
                json.dumps(
                    {
                        "row_id": "validation-00000000",
                        "split": "validation",
                        "input_ids": [1, 2, 3, 4, 5, 6],
                        "burn_in_tokens": 2,
                        "supervised_tokens": 4,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            packed_manifest = root / "splits.json"
            packed_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "prepared",
                        "splits": {
                            "validation": {
                                "path": packed.name,
                                "sha256": file_sha256(packed),
                                "row_count": 1,
                                "token_count": 6,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                read_packed_token_rows(
                    packed_manifest,
                    split="validation",
                    burn_in_tokens=2,
                    supervised_tokens=4,
                ),
                ((1, 2, 3, 4, 5, 6),),
            )

    def test_resume_preserves_mid_accumulation_gradient_and_sweep_cursor(self) -> None:
        initial = self.make_layers()
        uninterrupted = copy.deepcopy(initial)
        resumed = copy.deepcopy(initial)
        first = ActiveLayerTrainer(uninterrupted, lr=1e-2)
        second = ActiveLayerTrainer(resumed, lr=1e-2)
        first.activate(59)
        second.activate(59)
        first.sweep_index = second.sweep_index = 1
        first.visit_cursor = second.visit_cursor = 0
        value = torch.full((2, 4), 0.25)
        self.assertFalse(first.backward(self.run_forward(uninterrupted, value).square().mean(), accumulation_steps=2))
        self.assertFalse(second.backward(self.run_forward(resumed, value).square().mean(), accumulation_steps=2))
        checkpoint_layers = copy.deepcopy(resumed.state_dict())
        checkpoint_trainer = copy.deepcopy(second.state_dict())
        resumed = self.make_layers()
        resumed.load_state_dict(checkpoint_layers)
        second = ActiveLayerTrainer(resumed, lr=1e-2)
        second.load_state_dict(checkpoint_trainer)
        self.assertEqual(second.accumulation_step, 1)
        self.assertEqual(second.sweep_index, 1)
        self.assertTrue(first.backward(self.run_forward(uninterrupted, value).square().mean(), accumulation_steps=2))
        self.assertTrue(second.backward(self.run_forward(resumed, value).square().mean(), accumulation_steps=2))
        for left, right in zip(uninterrupted.parameters(), resumed.parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_corrective_sweep_stops_on_whole_sweep_delta_and_selects_lowest_kl(self) -> None:
        controller = SweepController(min_sweeps=1, max_sweeps=3, min_delta=0.01)
        first = controller.complete(start_checkpoint="full", end_checkpoint="sweep-0", validation_kl=0.8, token_budget=100)
        second = controller.complete(start_checkpoint="sweep-0", end_checkpoint="sweep-1", validation_kl=0.795, token_budget=100)
        self.assertFalse(first["stop"])
        self.assertTrue(second["stop"])
        self.assertEqual(second["selected_checkpoint"], "sweep-1")

    def test_activation_fit_can_freeze_algebraic_parameters_by_name(self) -> None:
        layers = nn.ModuleList([nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))])
        trainer = ActiveLayerTrainer(layers, lr=1e-2)
        before = copy.deepcopy(layers.state_dict())
        trainer.activate(0, trainable_names={"1.weight", "1.bias"})
        loss = layers[0](torch.ones(2, 4)).square().mean()
        trainer.backward(loss)
        self.assertIsNone(layers[0][0].weight.grad)
        self.assertIsNone(layers[0][0].bias.grad)
        self.assertIsNotNone(layers[0][1].weight.grad)
        torch.testing.assert_close(layers.state_dict()["0.0.weight"], before["0.0.weight"])
        torch.testing.assert_close(layers.state_dict()["0.0.bias"], before["0.0.bias"])

    def test_frozen_teacher_suffix_keeps_global_gradient_bridge(self) -> None:
        torch.manual_seed(9)
        teachers = [nn.Linear(4, 4, bias=False) for _ in range(4)]
        students = [nn.Linear(4, 4, bias=False) for _ in range(4)]
        head = nn.Linear(4, 8, bias=False)
        teacher_before = [copy.deepcopy(layer.state_dict()) for layer in teachers]
        runner = HybridReplacementRunner(teachers, students, head)
        for index, layer in enumerate(runner.student_layers):
            layer.requires_grad_(index == 2)
        active_output, logits = runner.isolated(torch.ones(2, 4), active_layer=2)
        logits.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in runner.student_layers[2].parameters()))
        self.assertTrue(all(parameter.grad is None for layer in runner.teacher_layers for parameter in layer.parameters()))
        self.assertTrue(all(parameter.grad is None for index, layer in enumerate(runner.student_layers) if index != 2 for parameter in layer.parameters()))
        for index, layer in enumerate(runner.teacher_layers):
            for name, value in layer.state_dict().items():
                torch.testing.assert_close(value, teacher_before[index][name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
