from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from any2rwkv.distill import (
    ActiveLayerTrainer,
    LossWeights,
    LossBreakdown,
    SweepController,
    load_sharded_training_checkpoint,
    load_training_checkpoint,
    progressive_schedule,
    save_sharded_training_checkpoint,
    save_training_checkpoint,
)
from any2rwkv.errors import ContractError, CoverageError
from any2rwkv.artifacts import file_sha256
from any2rwkv.distill_runner import (
    read_distillation_plan,
    read_distillation_texts,
    read_packed_token_rows,
    validate_distributed_row_capacity,
    validate_training_control_evidence,
)
from any2rwkv.recipes.qwen35_to_rwkv7 import Qwen35ToRWKV7Recipe
from any2rwkv.mapping import MappingLedger, SourceDisposition, SourceEntry, TargetEntry, TargetProvenance, finalize_fitted_mapping


class MappingTests(unittest.TestCase):
    def test_eight_rank_plan_rejects_idle_or_overcommitted_row_sets(self) -> None:
        rows = tuple((index, index + 1) for index in range(8))
        plan = SimpleNamespace(distributed_world_size=8, activation_fit_rows=8)
        validate_distributed_row_capacity(plan, rows, rows)
        with self.assertRaisesRegex(ContractError, "distill_train"):
            validate_distributed_row_capacity(plan, rows[:7], rows)
        with self.assertRaisesRegex(ContractError, "validation"):
            validate_distributed_row_capacity(plan, rows, rows[:7])
        plan.activation_fit_rows = 9
        with self.assertRaisesRegex(ContractError, "activation_fit_rows"):
            validate_distributed_row_capacity(plan, rows, rows)

    def test_loss_breakdown_keeps_nonleaf_autograd_graph(self) -> None:
        leaf = torch.tensor(2.0, requires_grad=True)
        nonleaf = leaf.square()
        losses = LossBreakdown(*(nonleaf for _ in range(6)))
        total = losses.weighted(LossWeights.for_stage("signals"))
        total.backward()
        self.assertIsNotNone(leaf.grad)
        self.assertGreater(float(leaf.grad), 0.0)

    def test_streamed_trainable_names_use_native_mixer_namespace(self) -> None:
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
                Qwen35ToRWKV7Recipe._initial_trainable_names(root, 1),
                [{"r_proj.weight"}],
            )

    def test_zero_bonus_is_not_trainable_during_local_distillation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "warm-start-plan.json").write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "target": "model.layers.0.attn.r_k",
                                "provenance": "initialized",
                                "local_trainable": False,
                            },
                            {
                                "target": "model.layers.0.attn.r_proj.weight",
                                "provenance": "initialized",
                                "local_trainable": True,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                Qwen35ToRWKV7Recipe._initial_trainable_names(root, 1),
                [{"r_proj.weight"}],
            )

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

    def test_real_runner_freezes_layer_major_epochs_and_hashed_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "classification": "fixture-only",
                        "evidence_tier": "fixture",
                        "seed": 7,
                        "learning_rate": 1e-4,
                        "local_learning_rate_by_mixer_kind": {
                            "linear_attention": 2e-5
                        },
                        "burn_in_tokens": 2,
                        "supervised_tokens": 4,
                        "accumulation_steps": 2,
                        "micro_batch_size": 2,
                        "cache_shard_rows": 4,
                        "checkpoint_interval_micro_batches": 8,
                        "activation_fit_functional_steps": 8,
                        "activation_fit_functional_learning_rate": 0.001,
                        "layer_min_epochs": 3,
                        "layer_max_epochs": 6,
                        "layer_min_delta": 0.001,
                        "layer_patience": 2,
                        "corrective_min_sweeps": 1,
                        "corrective_max_sweeps": 3,
                        "corrective_min_delta": 0.001,
                        "local_loss_weights": {
                            "mixer_mse": 1.0,
                            "block_mse": 1.0,
                            "cosine": 0.1,
                        },
                        "global_loss_weights": {
                            "token_kl": 1.0,
                            "shifted_ce": 0.25,
                        },
                        "training_control_evidence": {
                            "status": "fixture-only",
                            "artifact_sha256": None,
                        },
                        "cache_teacher_layers": False,
                        "corrective_resident_model_max_bytes": 1_000_000_000,
                    }
                ),
                encoding="utf-8",
            )
            parsed_plan = read_distillation_plan(plan_path)
            self.assertEqual(parsed_plan.layer_min_epochs, 3)
            self.assertEqual(parsed_plan.layer_max_epochs, 6)
            self.assertEqual(parsed_plan.layer_patience, 2)
            self.assertEqual(
                parsed_plan.local_learning_rate_by_mixer_kind,
                (("linear_attention", 2e-5),),
            )
            invalid_profile_plan = json.loads(
                plan_path.read_text(encoding="utf-8")
            )
            invalid_profile_plan["local_learning_rate_by_mixer_kind"] = {
                "linear_attention": 0.0
            }
            plan_path.write_text(
                json.dumps(invalid_profile_plan), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ContractError,
                "local_learning_rate_by_mixer_kind",
            ):
                read_distillation_plan(plan_path)
            invalid_profile_plan["local_learning_rate_by_mixer_kind"] = {
                "linear_attention": 2e-5
            }
            plan_path.write_text(
                json.dumps(invalid_profile_plan), encoding="utf-8"
            )
            self.assertEqual(parsed_plan.execution_mode, "streamed_layer_store")
            self.assertFalse(parsed_plan.cache_teacher_layers)
            self.assertFalse(
                parsed_plan.activation_fit_attention_time_mix_ablation
            )
            self.assertIsNone(parsed_plan.exploratory_layer_limit)
            self.assertEqual(
                parsed_plan.corrective_resident_model_max_bytes,
                1_000_000_000,
            )
            legacy_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            legacy_plan.pop("activation_fit_functional_steps")
            legacy_plan.pop("activation_fit_functional_learning_rate")
            legacy_plan["activation_fit_time_mix_steps"] = 12
            legacy_plan["activation_fit_time_mix_learning_rate"] = 5e-4
            plan_path.write_text(json.dumps(legacy_plan), encoding="utf-8")
            parsed_legacy_plan = read_distillation_plan(plan_path)
            self.assertEqual(parsed_legacy_plan.activation_fit_functional_steps, 12)
            self.assertEqual(
                parsed_legacy_plan.activation_fit_functional_learning_rate,
                5e-4,
            )
            conflicting_plan = dict(legacy_plan)
            conflicting_plan["activation_fit_functional_steps"] = 16
            plan_path.write_text(json.dumps(conflicting_plan), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "aliases conflict"):
                read_distillation_plan(plan_path)
            legacy_plan.pop("activation_fit_time_mix_steps")
            legacy_plan.pop("activation_fit_time_mix_learning_rate")
            legacy_plan["activation_fit_functional_steps"] = 8
            legacy_plan["activation_fit_functional_learning_rate"] = 0.001
            plan_path.write_text(json.dumps(legacy_plan), encoding="utf-8")
            validate_training_control_evidence(parsed_plan)
            enabled_ablation_plan = json.loads(
                plan_path.read_text(encoding="utf-8")
            )
            enabled_ablation_plan[
                "activation_fit_attention_time_mix_ablation"
            ] = True
            plan_path.write_text(
                json.dumps(enabled_ablation_plan), encoding="utf-8"
            )
            self.assertTrue(
                read_distillation_plan(
                    plan_path
                ).activation_fit_attention_time_mix_ablation
            )
            enabled_ablation_plan[
                "activation_fit_attention_time_mix_ablation"
            ] = 1
            plan_path.write_text(
                json.dumps(enabled_ablation_plan), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ContractError, "attention_time_mix_ablation"
            ):
                read_distillation_plan(plan_path)
            plan_path.write_text(json.dumps(legacy_plan), encoding="utf-8")
            blocked_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            blocked_plan["classification"] = "arbitrary-human-readable-label"
            blocked_plan["evidence_tier"] = "p1"
            blocked_plan["distributed_world_size"] = 8
            blocked_plan["max_cached_layer_input_bytes_per_rank"] = 1_000_000
            blocked_plan["learning_rate_schedule"] = "warmup-constant"
            blocked_plan["optimizer"] = {
                "name": "adamw",
                "learning_rate": 1e-4,
                "final_learning_rate": 1e-4,
                "warmup_steps": 10,
                "betas": [0.9, 0.99],
                "epsilon": 1e-8,
                "weight_decay": 0.1,
            }
            plan_path.write_text(json.dumps(blocked_plan), encoding="utf-8")
            with self.assertRaisesRegex(
                ContractError, "explicit activation-fit controls"
            ):
                read_distillation_plan(plan_path)
            blocked_plan.update(
                {
                    "activation_fit_rows": 8,
                    "activation_fit_ridge": 0.001,
                    "activation_fit_functional_steps": 32,
                    "activation_fit_functional_learning_rate": 0.0003,
                }
            )
            blocked_limited_plan = dict(blocked_plan)
            blocked_limited_plan["exploratory_layer_limit"] = 1
            plan_path.write_text(
                json.dumps(blocked_limited_plan), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ContractError, "only allowed for exploratory evidence"
            ):
                read_distillation_plan(plan_path)
            invalid_null_plan = dict(blocked_plan)
            invalid_null_plan["exploratory_layer_limit"] = None
            plan_path.write_text(
                json.dumps(invalid_null_plan), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ContractError, "must be a JSON integer"
            ):
                read_distillation_plan(plan_path)
            plan_path.write_text(json.dumps(blocked_plan), encoding="utf-8")
            with self.assertRaisesRegex(
                ContractError, "training controls are calibrated"
            ):
                validate_training_control_evidence(
                    read_distillation_plan(plan_path)
                )
            plan_path.write_text(json.dumps(blocked_plan), encoding="utf-8")
            invalid_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            invalid_plan["stage_tokens_per_layer"] = {"signals": 8, "block": 16, "global": 32}
            plan_path.write_text(json.dumps(invalid_plan), encoding="utf-8")
            with self.assertRaisesRegex(
                ContractError,
                "forbidden",
            ):
                read_distillation_plan(plan_path)

            single_gpu_real_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            single_gpu_real_plan.pop("stage_tokens_per_layer", None)
            single_gpu_real_plan["evidence_tier"] = "exploratory"
            single_gpu_real_plan["distributed_world_size"] = 1
            plan_path.write_text(json.dumps(single_gpu_real_plan), encoding="utf-8")
            with self.assertRaisesRegex(
                ContractError,
                "suffix-free layer-major contract",
            ):
                read_distillation_plan(plan_path)
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

    def test_corrective_first_sweep_can_roll_back_to_pre_sweep_baseline(self) -> None:
        controller = SweepController(
            min_sweeps=1,
            max_sweeps=1,
            min_delta=0.01,
            baseline_validation_kl=0.7,
            baseline_checkpoint="pre-sweep",
        )
        result = controller.complete(
            start_checkpoint="pre-sweep",
            end_checkpoint="sweep-0",
            validation_kl=0.8,
            token_budget=100,
        )
        self.assertAlmostEqual(result["delta"], -0.1)
        self.assertTrue(result["stop"])
        self.assertEqual(result["selected_checkpoint"], "pre-sweep")

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

if __name__ == "__main__":
    unittest.main()
