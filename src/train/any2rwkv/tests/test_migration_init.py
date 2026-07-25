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
from any2rwkv.export import (
    BF16_VALUE_RESIDUAL_DISABLED_LOGIT,
    BF16_VALUE_RESIDUAL_EPSILON,
)
from any2rwkv.mapping import SourceDisposition, TargetProvenance
from any2rwkv.migration import kv_expand, kv_repeat
from any2rwkv.migration_init import (
    TensorOperation,
    WarmStartTensorProvider,
    WarmStartVariant,
    _gdn_conv_time_mix,
    apply_warm_start_plan,
    materialize_warm_start,
    plan_warm_start,
)
from any2rwkv.target import build_zero_step_ledger, rwkv7_mixer_specs


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
        self.assertEqual(set(entries), {spec.name for spec in specs})
        self.assertTrue(all(entry.local_trainable is True for entry in entries.values()))
        for role in ("r", "k", "v"):
            name = f"model.layers.0.attn.x_{role}"
            self.assertEqual(
                entries[name].operation, TensorOperation.GDN_CONV_TIME_MIX
            )
            self.assertEqual(
                set((entries[name].source, *entries[name].auxiliary_sources)),
                {
                    "model.layers.0.linear_attn.in_proj_qkv.weight",
                    "model.layers.0.linear_attn.conv1d.weight",
                },
            )
            self.assertTrue(torch.all(materialized[name] >= 0))
            self.assertTrue(torch.all(materialized[name] <= 1))
        for role in ("w", "a", "g"):
            name = f"model.layers.0.attn.x_{role}"
            self.assertEqual(entries[name].operation, TensorOperation.ZERO)
            self.assertTrue(entries[name].local_trainable)
            torch.testing.assert_close(
                materialized[name], torch.zeros_like(materialized[name]), rtol=0, atol=0
            )
        gate_down_name = "model.layers.0.attn.g_lora.lora.0.weight"
        gate_source_name = "model.layers.0.linear_attn.in_proj_z.weight"
        gate_down_entry = entries[gate_down_name]
        self.assertEqual(
            gate_down_entry.operation, TensorOperation.EVEN_ROW_SUBSAMPLE
        )
        self.assertEqual(gate_down_entry.source, gate_source_name)
        source_gate = source_tensors[gate_source_name]
        gate_indices = torch.linspace(
            0,
            source_gate.shape[0] - 1,
            materialized[gate_down_name].shape[0],
            dtype=torch.float64,
        ).round().to(torch.long)
        torch.testing.assert_close(
            materialized[gate_down_name],
            source_gate.index_select(0, gate_indices).to(torch.bfloat16),
            rtol=0,
            atol=0,
        )
        self.assertTrue(entries["model.layers.0.attn.r_k"].local_trainable)
        key_scale = "model.layers.0.attn.k_k"
        write_interpolation = "model.layers.0.attn.k_a"
        for target in (key_scale, write_interpolation):
            self.assertTrue(entries[target].local_trainable)
            self.assertTrue(entries[target].is_semantically_lossless)
            self.assertEqual(entries[target].provenance, TargetProvenance.ALGEBRAIC)
        torch.testing.assert_close(
            materialized[key_scale],
            torch.ones_like(materialized[key_scale]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            materialized[write_interpolation],
            torch.zeros_like(materialized[write_interpolation]),
            rtol=0,
            atol=0,
        )

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

        beta = source_tensors["model.layers.0.linear_attn.in_proj_b.weight"]
        decay_input = source_tensors["model.layers.0.linear_attn.in_proj_a.weight"]
        decay_scale = source_tensors["model.layers.0.linear_attn.A_log"].float().exp()
        decay_bias = source_tensors["model.layers.0.linear_attn.dt_bias"].float()
        zero_decay = torch.exp(
            -decay_scale * torch.nn.functional.softplus(decay_bias)
        )
        zero_erase = (0.5 * zero_decay).clamp(1e-6, 1 - 1e-6)
        denominator = 1 - zero_erase
        expected_erase_gradient = (
            (0.5 / denominator).unsqueeze(-1) * beta.float()
            - (
                decay_scale * torch.sigmoid(decay_bias) / denominator
            ).unsqueeze(-1)
            * decay_input.float()
        )
        a_down = materialized["model.layers.0.attn.a_lora.lora.0.weight"]
        torch.testing.assert_close(
            a_down[:4], expected_erase_gradient.to(torch.bfloat16), rtol=0, atol=0
        )
        self.assertEqual(torch.count_nonzero(a_down[4:]).item(), 0)
        expected_up = torch.zeros((64, 64), dtype=torch.bfloat16)
        for head in range(4):
            expected_up[head * 16 : (head + 1) * 16, head] = 1
        torch.testing.assert_close(
            materialized["model.layers.0.attn.a_lora.lora.2.weight"],
            expected_up,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            materialized["model.layers.0.attn.a_lora.lora.2.bias"],
            torch.logit(zero_erase).repeat_interleave(16).to(torch.bfloat16),
            rtol=0,
            atol=0,
        )
        expected_norm = source_tensors[
            "model.layers.0.linear_attn.norm.weight"
        ].repeat(4)
        torch.testing.assert_close(
            materialized["model.layers.0.attn.g_norm.weight"],
            expected_norm.to(torch.bfloat16),
            rtol=0,
            atol=0,
        )
        self.assertFalse(
            entries[
                "model.layers.0.attn.a_lora.lora.0.weight"
            ].is_semantically_lossless
        )
        for name in (
            "model.layers.0.attn.v_proj.weight",
            "model.layers.0.attn.a_lora.lora.0.weight",
            "model.layers.0.attn.a_lora.lora.2.weight",
            "model.layers.0.attn.a_lora.lora.2.bias",
            "model.layers.0.attn.w_lora.lora.2.weight",
            "model.layers.0.attn.w_lora.lora.2.bias",
        ):
            self.assertFalse(entries[name].is_semantically_lossless, name)
        self.assertFalse(
            entries["model.layers.0.attn.g_norm.weight"].is_semantically_lossless
        )
        w_down = materialized["model.layers.0.attn.w_lora.lora.0.weight"]
        torch.testing.assert_close(w_down[:4], decay_input.to(torch.bfloat16), rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(w_down[4:]).item(), 0)
        self.assertEqual(
            entries["model.layers.0.attn.w_lora.lora.2.weight"].operation,
            TensorOperation.DECAY_LINEAR_UP,
        )
        self.assertEqual(
            torch.count_nonzero(
                materialized["model.layers.0.attn.w_lora.lora.2.weight"]
            ).item(),
            0,
        )
        self.assertTrue(
            torch.isfinite(
                materialized["model.layers.0.attn.w_lora.lora.2.bias"]
            ).all()
        )

    def test_gdn_conv_time_mix_finds_best_native_two_tap_direction(self) -> None:
        projection = torch.eye(2)
        conv = torch.zeros(2, 1, 4)
        conv[0, 0, -2:] = 1
        conv[1, 0, -1] = 1
        actual = _gdn_conv_time_mix(
            projection, conv, target_shape=(1, 1, 2)
        )
        torch.testing.assert_close(
            actual,
            torch.tensor([[[0.5, 0.0]]]),
            rtol=0,
            atol=0,
        )

    def test_full_attention_current_token_time_mix_is_zero_initialized(self) -> None:
        specs = rwkv7_mixer_specs(3, hidden_size=64, head_dim=16)
        plan = plan_warm_start(self.source, specs, variant=WarmStartVariant.MAPPED)
        materialized = materialize_warm_start(self.source, specs, plan)
        entries = {entry.target: entry for entry in plan.entries}
        for role in ("r", "w", "k", "v", "a", "g"):
            name = f"model.layers.3.attn.x_{role}"
            self.assertEqual(entries[name].operation, TensorOperation.ZERO)
            self.assertTrue(entries[name].is_semantically_lossless)
            torch.testing.assert_close(
                materialized[name], torch.zeros_like(materialized[name]), rtol=0, atol=0
            )

    def test_full_attention_keeps_native_key_controls_locally_trainable(self) -> None:
        specs = rwkv7_mixer_specs(3, hidden_size=64, head_dim=16)
        plan = plan_warm_start(self.source, specs, variant=WarmStartVariant.MAPPED)
        materialized = materialize_warm_start(self.source, specs, plan)
        entries = {entry.target: entry for entry in plan.entries}
        self.assertTrue(entries["model.layers.3.attn.k_k"].local_trainable)
        self.assertTrue(entries["model.layers.3.attn.k_a"].local_trainable)
        torch.testing.assert_close(
            materialized["model.layers.3.attn.k_k"],
            torch.ones_like(materialized["model.layers.3.attn.k_k"]),
            rtol=0,
            atol=0,
        )
        for name in ("k_a", "r_k"):
            target = f"model.layers.3.attn.{name}"
            self.assertEqual(entries[target].operation, TensorOperation.ZERO)
            torch.testing.assert_close(
                materialized[target],
                torch.zeros_like(materialized[target]),
                rtol=0,
                atol=0,
            )

    def test_full_attention_starts_from_source_compatible_linear_attention_core(self) -> None:
        specs = rwkv7_mixer_specs(3, hidden_size=64, head_dim=16)
        plan = plan_warm_start(self.source, specs, variant=WarmStartVariant.MAPPED)
        materialized = materialize_warm_start(self.source, specs, plan)
        entries = {entry.target: entry for entry in plan.entries}
        source = load_file(self.fixture_path / "model.safetensors")

        for family in ("w_lora", "a_lora"):
            down = f"model.layers.3.attn.{family}.lora.0.weight"
            up = f"model.layers.3.attn.{family}.lora.2.weight"
            bias = f"model.layers.3.attn.{family}.lora.2.bias"
            self.assertGreater(torch.count_nonzero(materialized[down]).item(), 0)
            torch.testing.assert_close(
                materialized[up], torch.zeros_like(materialized[up]), rtol=0, atol=0
            )
            self.assertEqual(
                entries[bias].operation, TensorOperation.DISABLED_SIGMOID_BIAS
            )
            torch.testing.assert_close(
                materialized[bias],
                torch.full_like(
                    materialized[bias], BF16_VALUE_RESIDUAL_DISABLED_LOGIT
                ),
                rtol=0,
                atol=0,
            )

        query = source["model.layers.3.self_attn.q_proj.weight"]
        source_gate = query.reshape(4, 32, 64)[:, 16:].flatten(0, 1)
        gate_down = "model.layers.3.attn.g_lora.lora.0.weight"
        indices = torch.linspace(
            0,
            source_gate.shape[0] - 1,
            materialized[gate_down].shape[0],
            dtype=torch.float64,
        ).round().to(torch.long)
        self.assertEqual(
            entries[gate_down].operation,
            TensorOperation.HEADWISE_QUERY_GATE_SUBSAMPLE,
        )
        torch.testing.assert_close(
            materialized[gate_down],
            source_gate.index_select(0, indices).to(torch.bfloat16),
            rtol=0,
            atol=0,
        )
        gate_up = "model.layers.3.attn.g_lora.lora.2.weight"
        self.assertEqual(
            entries[gate_up].operation,
            TensorOperation.HEADWISE_QUERY_GATE_RECONSTRUCTION,
        )
        self.assertGreater(torch.count_nonzero(materialized[gate_up]).item(), 0)
        torch.testing.assert_close(
            materialized[gate_up].float().sum(dim=1),
            torch.ones(materialized[gate_up].shape[0]),
            rtol=0,
            atol=2e-2,
        )
        selected_source_rows = torch.unique(indices, sorted=True)
        expected_selected = torch.zeros(
            selected_source_rows.numel(),
            materialized[gate_down].shape[0],
            dtype=torch.bfloat16,
        )
        for row_index, source_row in enumerate(selected_source_rows.tolist()):
            basis_column = int((indices == source_row).nonzero()[0].item())
            expected_selected[row_index, basis_column] = 1
        torch.testing.assert_close(
            materialized[gate_up].index_select(0, selected_source_rows),
            expected_selected,
            rtol=0,
            atol=0,
        )
        zero_features = torch.sigmoid(
            torch.zeros(1, materialized[gate_down].shape[1])
            @ materialized[gate_down].float().T
        )
        zero_gate = zero_features @ materialized[gate_up].float().T
        torch.testing.assert_close(
            zero_gate,
            torch.full_like(zero_gate, 0.5),
            rtol=0,
            atol=1e-2,
        )

    def test_joint_erase_warm_start_beats_legacy_beta_only_near_zero_input(self) -> None:
        specs = rwkv7_mixer_specs(0, hidden_size=64, head_dim=16)
        plan = plan_warm_start(
            self.source, specs, variant=WarmStartVariant.GDN_CONSTRAINED
        )
        materialized = materialize_warm_start(self.source, specs, plan)
        source = load_file(self.fixture_path / "model.safetensors")
        prefix = "model.layers.0.linear_attn"
        generator = torch.Generator().manual_seed(20260716)
        hidden = torch.randn(256, 64, generator=generator) * 0.02
        beta_logits = hidden @ source[f"{prefix}.in_proj_b.weight"].float().T
        decay_input = hidden @ source[f"{prefix}.in_proj_a.weight"].float().T
        beta = torch.sigmoid(beta_logits)
        decay = torch.exp(
            -source[f"{prefix}.A_log"].float().exp()
            * torch.nn.functional.softplus(
                decay_input + source[f"{prefix}.dt_bias"].float()
            )
        )
        expected = (beta * decay).repeat_interleave(16, dim=-1)

        down = materialized[
            "model.layers.0.attn.a_lora.lora.0.weight"
        ].float()
        up = materialized[
            "model.layers.0.attn.a_lora.lora.2.weight"
        ].float()
        bias = materialized[
            "model.layers.0.attn.a_lora.lora.2.bias"
        ].float()
        initialized = torch.sigmoid((hidden @ down.T) @ up.T + bias)
        legacy_beta_only = beta.repeat_interleave(16, dim=-1)

        initialized_mse = (initialized - expected).square().mean()
        legacy_mse = (legacy_beta_only - expected).square().mean()
        self.assertLess(float(initialized_mse), float(legacy_mse))

    def test_joint_erase_records_and_uses_every_auxiliary_source(self) -> None:
        specs = rwkv7_mixer_specs(0, hidden_size=64, head_dim=16)
        plan = plan_warm_start(
            self.source, specs, variant=WarmStartVariant.GDN_CONSTRAINED
        )
        source_names = tuple(self.source.tensor_names())
        shard_hashes = tuple(
            self.source.file_hashes[path.name] for path in self.source.shards
        )
        ledger, _, target_names = build_zero_step_ledger(
            source_names,
            layer_count=self.source.contract.num_hidden_layers,
            hidden_size=self.source.contract.hidden_size,
            head_dim=16,
            source_shard_hashes=shard_hashes,
        )
        apply_warm_start_plan(ledger, plan)
        ledger.validate(source_names, target_names)

        target = "model.layers.0.attn.a_lora.lora.0.weight"
        prefix = "model.layers.0.linear_attn"
        expected_sources = {
            f"{prefix}.in_proj_b.weight",
            f"{prefix}.in_proj_a.weight",
            f"{prefix}.A_log",
            f"{prefix}.dt_bias",
        }
        self.assertEqual(set(ledger.targets[target].sources), expected_sources)
        for source_name in expected_sources:
            self.assertEqual(
                ledger.sources[source_name].disposition,
                SourceDisposition.CONSUMED,
            )
            self.assertIn(target, ledger.sources[source_name].targets)

        baseline = materialize_warm_start(self.source, specs, plan)[target]
        checkpoint_path = self.fixture_path / "model.safetensors"
        original_tensors = load_file(checkpoint_path)
        try:
            for source_name in sorted(expected_sources):
                perturbed = {
                    name: value.clone() for name, value in original_tensors.items()
                }
                perturbed[source_name] = perturbed[source_name] + 0.125
                save_file(perturbed, checkpoint_path)
                perturbed_source = read_checkpoint(
                    self.fixture_path, require_final_layers=False
                )
                perturbed_plan = plan_warm_start(
                    perturbed_source,
                    specs,
                    variant=WarmStartVariant.GDN_CONSTRAINED,
                )
                actual = materialize_warm_start(
                    perturbed_source, specs, perturbed_plan
                )[target]
                self.assertFalse(
                    torch.equal(actual, baseline),
                    f"auxiliary source did not affect materialization: {source_name}",
                )
        finally:
            save_file(original_tensors, checkpoint_path)

    def test_source_compatible_warm_start_initially_disables_native_value_residual(self) -> None:
        for layer_index in (1, 3):
            specs = rwkv7_mixer_specs(layer_index, hidden_size=64, head_dim=16)
            plan = plan_warm_start(
                self.source, specs, variant=WarmStartVariant.MAPPED
            )
            materialized = materialize_warm_start(self.source, specs, plan)
            entries = {entry.target: entry for entry in plan.entries}
            prefix = f"model.layers.{layer_index}.attn.v_lora.lora"
            down = f"{prefix}.0.weight"
            up = f"{prefix}.2.weight"
            bias = f"{prefix}.2.bias"

            self.assertGreater(torch.count_nonzero(materialized[down]).item(), 0)
            self.assertEqual(torch.count_nonzero(materialized[up]).item(), 0)
            coefficient = torch.sigmoid(materialized[bias].float())
            self.assertLessEqual(
                coefficient.max().item(),
                BF16_VALUE_RESIDUAL_EPSILON * 1.02,
            )
            self.assertEqual(entries[up].provenance, TargetProvenance.ALGEBRAIC)
            self.assertEqual(entries[bias].provenance, TargetProvenance.ALGEBRAIC)
            for name in (down, up, bias):
                self.assertTrue(entries[name].local_trainable)

    def test_gdn_norm_geometry_mismatch_defers_to_fitting_instead_of_aborting(self) -> None:
        config_path = self.fixture_path / "config.json"
        config = json.loads(config_path.read_text())
        config["linear_value_head_dim"] = 32
        config_path.write_text(json.dumps(config), encoding="utf-8")
        tensor_path = self.fixture_path / "model.safetensors"
        tensors = load_file(tensor_path)
        generator = torch.Generator().manual_seed(17)
        prefix = "model.layers.0.linear_attn"
        tensors[f"{prefix}.in_proj_qkv.weight"] = torch.randn(
            256, 64, generator=generator
        ) * 0.02
        tensors[f"{prefix}.in_proj_z.weight"] = torch.randn(
            128, 64, generator=generator
        ) * 0.02
        tensors[f"{prefix}.conv1d.weight"] = torch.randn(
            256, 1, 4, generator=generator
        ) * 0.02
        tensors[f"{prefix}.norm.weight"] = torch.ones(32)
        tensors[f"{prefix}.out_proj.weight"] = torch.randn(
            64, 128, generator=generator
        ) * 0.02
        save_file(tensors, tensor_path)
        source = read_checkpoint(self.fixture_path, require_final_layers=False)

        plan = plan_warm_start(
            source,
            rwkv7_mixer_specs(0, hidden_size=64, head_dim=16),
            variant=WarmStartVariant.MAPPED,
        )
        entries = {entry.target: entry for entry in plan.entries}
        norm = entries["model.layers.0.attn.g_norm.weight"]
        self.assertEqual(norm.provenance, TargetProvenance.INITIALIZED)
        self.assertEqual(norm.operation, TensorOperation.INITIALIZE)
        self.assertTrue(
            any(error.code == "normalization_geometry_mismatch" for error in plan.errors)
        )

    def test_gdn_expanded_recurrent_width_preserves_source_value_geometry(self) -> None:
        fixture_path = write_fixture(
            Path(self.temporary_directory.name) / "expanded-gdn",
            layers=1,
            config_overrides={
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "head_dim": 16,
                "linear_num_key_heads": 4,
                "linear_num_value_heads": 8,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
            },
        )
        source = read_checkpoint(fixture_path, require_final_layers=False)
        specs = rwkv7_mixer_specs(
            0,
            hidden_size=64,
            attention_hidden_size=128,
            head_dim=16,
        )
        plan = plan_warm_start(source, specs, variant=WarmStartVariant.MAPPED)
        materialized = materialize_warm_start(source, specs, plan)
        source_tensors = load_file(fixture_path / "model.safetensors")
        packed = source_tensors["model.layers.0.linear_attn.in_proj_qkv.weight"]

        expected_read = kv_repeat(
            packed[:64], num_query_heads=8, num_kv_heads=4
        ) / 4.0
        expected_key = kv_repeat(
            packed[64:128], num_query_heads=8, num_kv_heads=4
        )
        torch.testing.assert_close(
            materialized["model.layers.0.attn.r_proj.weight"],
            expected_read.to(torch.bfloat16),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            materialized["model.layers.0.attn.k_proj.weight"],
            expected_key.to(torch.bfloat16),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            materialized["model.layers.0.attn.v_proj.weight"],
            packed[128:256].to(torch.bfloat16),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            materialized["model.layers.0.attn.o_proj.weight"],
            source_tensors["model.layers.0.linear_attn.out_proj.weight"].to(
                torch.bfloat16
            ),
            rtol=0,
            atol=0,
        )

        erase_up = materialized["model.layers.0.attn.a_lora.lora.2.weight"]
        expected_erase_up = torch.zeros((128, 64), dtype=torch.bfloat16)
        for head in range(8):
            expected_erase_up[head * 16 : (head + 1) * 16, head] = 1
        torch.testing.assert_close(erase_up, expected_erase_up, rtol=0, atol=0)
        self.assertEqual(
            materialized["model.layers.0.attn.g_norm.weight"].shape,
            (128,),
        )
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
