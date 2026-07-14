from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from any2rwkv.checkpoint import read_checkpoint
from any2rwkv.fixture import write_fixture
from any2rwkv.mapping import TargetProvenance
from any2rwkv.migration import kv_expand, kv_repeat
from any2rwkv.migration_init import (
    TensorOperation,
    WarmStartTensorProvider,
    WarmStartVariant,
    materialize_warm_start,
    plan_warm_start,
)
from any2rwkv.target import rwkv7_mixer_specs


class MigrationInitializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.fixture_path = write_fixture(Path(self.temporary_directory.name) / "qwen35", layers=4)
        self.source = read_checkpoint(self.fixture_path, require_final_layers=False)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_gdn_constrained_materializes_exact_tensor_values_with_provenance(self) -> None:
        specs = rwkv7_mixer_specs(0, hidden_size=64, head_dim=16)
        plan = plan_warm_start(self.source, specs, variant=WarmStartVariant.GDN_CONSTRAINED)
        materialized = materialize_warm_start(self.source, specs, plan)
        source_tensors = load_file(self.fixture_path / "model.safetensors")
        packed = source_tensors["model.layers.0.linear_attn.in_proj_qkv.weight"]
        entries = {entry.target: entry for entry in plan.entries}

        expected = {
            "r": packed[:64] / 4.0,
            "k": packed[64:128],
            "v": packed[128:192],
            "o": source_tensors["model.layers.0.linear_attn.out_proj.weight"],
        }
        for role, value in expected.items():
            name = f"model.layers.0.attn.{role}_proj.weight"
            torch.testing.assert_close(materialized[name], value.to(torch.bfloat16), rtol=0, atol=0)
            self.assertFalse(entries[name].is_semantically_lossless)
        self.assertEqual(entries["model.layers.0.attn.r_proj.weight"].provenance, TargetProvenance.ALGEBRAIC)
        self.assertEqual(entries["model.layers.0.attn.r_proj.weight"].operation, TensorOperation.SCALE)
        for role in ("k", "v", "o"):
            self.assertEqual(entries[f"model.layers.0.attn.{role}_proj.weight"].provenance, TargetProvenance.COPIED)
        self.assertEqual(plan.errors, ())

    def test_layer_bounded_provider_matches_bulk_materialization(self) -> None:
        specs = tuple(
            spec
            for layer_index in range(4)
            for spec in rwkv7_mixer_specs(layer_index, hidden_size=64, head_dim=16)
        )
        plan = plan_warm_start(self.source, specs, variant=WarmStartVariant.MAPPED)
        expected = materialize_warm_start(self.source, specs, plan)
        provider = WarmStartTensorProvider(self.source, specs, plan)
        actual = {spec.name: provider(spec) for spec in specs}
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def test_full_attention_kv_baselines_are_numerically_distinct_and_group_aware(self) -> None:
        specs = rwkv7_mixer_specs(3, hidden_size=64, head_dim=16)
        repeated_plan = plan_warm_start(self.source, specs, variant="kv_repeat")
        expanded_plan = plan_warm_start(self.source, specs, variant="kv_expand")
        repeated = materialize_warm_start(self.source, specs, repeated_plan)
        expanded = materialize_warm_start(self.source, specs, expanded_plan)
        source_tensors = load_file(self.fixture_path / "model.safetensors")
        entries = {entry.target: entry for entry in repeated_plan.entries}

        query = source_tensors["model.layers.3.self_attn.q_proj.weight"]
        expected_query = query.reshape(4, 32, 64)[:, :16].flatten(0, 1)
        torch.testing.assert_close(
            repeated["model.layers.3.attn.r_proj.weight"],
            expected_query.to(torch.bfloat16),
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            entries["model.layers.3.attn.r_proj.weight"].operation,
            TensorOperation.HEADWISE_QUERY_SLICE,
        )

        for role in ("k", "v"):
            source_value = source_tensors[f"model.layers.3.self_attn.{role}_proj.weight"]
            name = f"model.layers.3.attn.{role}_proj.weight"
            torch.testing.assert_close(
                repeated[name],
                kv_repeat(source_value, num_query_heads=4, num_kv_heads=2).to(torch.bfloat16),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                expanded[name],
                kv_expand(source_value, num_query_heads=4, num_kv_heads=2).to(torch.bfloat16),
                rtol=0,
                atol=0,
            )
            self.assertFalse(torch.equal(repeated[name], expanded[name]))
            self.assertEqual(entries[name].provenance, TargetProvenance.ALGEBRAIC)
            self.assertFalse(entries[name].is_semantically_lossless)
        expanded_entries = {entry.target: entry for entry in expanded_plan.entries}
        self.assertEqual(
            expanded_entries["model.layers.3.attn.k_proj.weight"].provenance,
            TargetProvenance.INITIALIZED,
        )
        kv_rows = [
            error
            for error in repeated_plan.errors
            if error.target.endswith(".k_proj.weight")
        ]
        self.assertEqual([row.head_index for row in kv_rows], [0, 1, 2, 3])
        self.assertEqual([row.group_index for row in kv_rows], [0, 0, 1, 1])
        self.assertTrue(all(row.code == "recurrent_state_semantics_changed" for row in kv_rows))

    def test_gdn_source_defined_key_head_repeat_is_materialized(self) -> None:
        checkpoint_path = self.fixture_path / "model.safetensors"
        tensors = load_file(checkpoint_path)
        packed_name = "model.layers.0.linear_attn.in_proj_qkv.weight"
        original = tensors[packed_name]
        query = original[:32].clone()
        key = original[64:96].clone()
        value = original[128:192].clone()
        tensors[packed_name] = torch.cat((query, key, value))
        save_file(tensors, checkpoint_path)
        config_path = self.fixture_path / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["linear_num_key_heads"] = 2
        config_path.write_text(json.dumps(config), encoding="utf-8")
        source = read_checkpoint(self.fixture_path, require_final_layers=False)

        specs = rwkv7_mixer_specs(0, hidden_size=64, head_dim=16)
        plan = plan_warm_start(source, specs, variant="gdn_constrained")
        materialized = materialize_warm_start(source, specs, plan)
        entries = {entry.target: entry for entry in plan.entries}
        expected = {
            "r": kv_repeat(query, num_query_heads=4, num_kv_heads=2) / 4.0,
            "k": kv_repeat(key, num_query_heads=4, num_kv_heads=2),
        }
        for role, value in expected.items():
            name = f"model.layers.0.attn.{role}_proj.weight"
            torch.testing.assert_close(materialized[name], value.to(torch.bfloat16), rtol=0, atol=0)
            self.assertEqual(entries[name].operation, TensorOperation.KV_REPEAT)
            self.assertEqual(entries[name].provenance, TargetProvenance.ALGEBRAIC)

    def test_naive_gqa_shape_mismatch_is_initialized_with_per_head_group_errors(self) -> None:
        specs = rwkv7_mixer_specs(3, hidden_size=64, head_dim=16)
        plan = plan_warm_start(self.source, specs, variant="naive_copy")
        entries = {entry.target: entry for entry in plan.entries}
        for role in ("k", "v"):
            name = f"model.layers.3.attn.{role}_proj.weight"
            self.assertEqual(entries[name].provenance, TargetProvenance.INITIALIZED)
            rows = [error for error in plan.errors if error.target == name]
            self.assertEqual([row.head_index for row in rows], [0, 1, 2, 3])
            self.assertEqual([row.group_index for row in rows], [0, 0, 1, 1])
            self.assertTrue(all(row.code == "projection_shape_mismatch" for row in rows))

    def test_changed_gdn_state_geometry_keeps_flat_projection_warm_start_nonlossless(self) -> None:
        specs = rwkv7_mixer_specs(0, hidden_size=64, head_dim=64)
        plan = plan_warm_start(self.source, specs, variant="gdn_constrained")
        projections = [entry for entry in plan.entries if "_proj.weight" in entry.target]
        self.assertTrue(projections)
        self.assertTrue(any(entry.provenance == TargetProvenance.ALGEBRAIC for entry in projections))
        self.assertTrue(any(entry.provenance == TargetProvenance.COPIED for entry in projections))
        self.assertTrue(all(not entry.is_semantically_lossless for entry in projections))
        rows = [error for error in plan.errors if error.code == "semantic_geometry_mismatch"]
        self.assertEqual({row.layer_index for row in rows}, {0})
        self.assertEqual({row.head_index for row in rows}, {0})
        self.assertTrue(all("trace fitting is required" in row.message for row in rows))

    def test_naive_copy_keeps_tensor_copy_separate_from_lossless_semantics(self) -> None:
        specs = rwkv7_mixer_specs(0, hidden_size=64, head_dim=64)
        plan = plan_warm_start(self.source, specs, variant="naive_copy")
        projections = [entry for entry in plan.entries if "_proj.weight" in entry.target]
        self.assertTrue(all(entry.provenance == TargetProvenance.COPIED for entry in projections))
        self.assertTrue(all(not entry.is_semantically_lossless for entry in projections))
        rows = [error for error in plan.errors if error.code == "semantic_geometry_mismatch"]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all("not lossless" in row.message for row in rows))

    def test_mapped_warm_start_does_not_claim_fitted_before_training(self) -> None:
        specs = tuple(
            spec
            for layer in range(4)
            for spec in rwkv7_mixer_specs(layer, hidden_size=64, head_dim=16)
        )
        plan = plan_warm_start(self.source, specs, variant="mapped")
        self.assertNotIn(TargetProvenance.FITTED, {entry.provenance for entry in plan.entries})
        self.assertTrue(
            any(
                entry.provenance in {TargetProvenance.COPIED, TargetProvenance.ALGEBRAIC}
                and not entry.is_semantically_lossless
                for entry in plan.entries
            )
        )

    def test_ambiguous_gqa_layout_is_rejected_before_materialization(self) -> None:
        ambiguous = replace(
            self.source,
            config={**self.source.config, "num_key_value_heads": 3},
        )
        specs = rwkv7_mixer_specs(3, hidden_size=64, head_dim=16)
        with self.assertRaisesRegex(ValueError, "ambiguous GQA layout"):
            plan_warm_start(ambiguous, specs, variant="kv_repeat")

    def test_head_factored_projection_has_explicit_reshape_provenance(self) -> None:
        checkpoint_path = self.fixture_path / "model.safetensors"
        tensors = load_file(checkpoint_path)
        output_name = "model.layers.3.self_attn.o_proj.weight"
        expected = tensors[output_name].clone()
        tensors[output_name] = tensors[output_name].reshape(4, 16, 64)
        save_file(tensors, checkpoint_path)
        source = read_checkpoint(self.fixture_path, require_final_layers=False)
        specs = rwkv7_mixer_specs(3, hidden_size=64, head_dim=16)
        plan = plan_warm_start(source, specs, variant="kv_repeat")
        entry = next(item for item in plan.entries if item.target.endswith(".o_proj.weight"))
        self.assertEqual(entry.operation, TensorOperation.RESHAPE)
        self.assertEqual(entry.provenance, TargetProvenance.COPIED)
        materialized = materialize_warm_start(source, specs, plan)
        torch.testing.assert_close(
            materialized["model.layers.3.attn.o_proj.weight"],
            expected.to(torch.bfloat16),
            rtol=0,
            atol=0,
        )

    def test_all_baseline_variants_and_seeded_random_materialization_are_deterministic(self) -> None:
        specs = rwkv7_mixer_specs(0, hidden_size=64, head_dim=16)
        variants = {variant.value for variant in WarmStartVariant}
        self.assertEqual(
            variants,
            {
                "random",
                "naive_copy",
                "gdn_constrained",
                "kv_repeat",
                "kv_expand",
                "mapped",
            },
        )
        for variant in WarmStartVariant:
            variant_plan = plan_warm_start(self.source, specs, variant=variant)
            self.assertEqual(variant_plan.variant, variant)
            self.assertEqual({entry.target for entry in variant_plan.entries}, {spec.name for spec in specs})
        first_plan = plan_warm_start(self.source, specs, variant="random")
        second_plan = plan_warm_start(self.source, specs, variant="random")
        self.assertEqual(first_plan.to_dict(), second_plan.to_dict())
        first = materialize_warm_start(self.source, specs, first_plan, seed=17)
        second = materialize_warm_start(self.source, specs, second_plan, seed=17)
        self.assertEqual(first.keys(), second.keys())
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)

    def test_materializer_rejects_checkpoint_changed_after_planning(self) -> None:
        specs = rwkv7_mixer_specs(0, hidden_size=64, head_dim=16)
        plan = plan_warm_start(self.source, specs, variant="random")
        checkpoint_path = self.fixture_path / "model.safetensors"
        tensors = load_file(checkpoint_path)
        tensors["model.embed_tokens.weight"] = tensors["model.embed_tokens.weight"] + 1
        save_file(tensors, checkpoint_path)
        with self.assertRaisesRegex(ValueError, "hashes changed"):
            materialize_warm_start(self.source, specs, plan)


if __name__ == "__main__":
    unittest.main()
