from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from any2rwkv.checkpoint import read_checkpoint
from any2rwkv.artifacts import write_json
from any2rwkv.configuration_any2rwkv import Any2RWKVProxyConfig
from any2rwkv.contract import build_target_config
from any2rwkv.distill import chunked_token_kl, normalized_mse, token_kl
from any2rwkv.distributed import DistributedContext, _gradient_buckets
from any2rwkv.fixture import write_fixture
from any2rwkv.export import export_hf_checkpoint
from any2rwkv.errors import ContractError
from any2rwkv.layer_store import LayerTensorStore
from any2rwkv.migration_init import (
    WarmStartTensorProvider,
    WarmStartVariant,
    apply_warm_start_plan,
    plan_warm_start,
)
from any2rwkv.migration import qwen35_l2_normalize
from any2rwkv.mixer import ProjectionBoundaryRWKV7Attention
from any2rwkv.mixer_store import RWKV7MixerLayerStore
from any2rwkv.streaming_training import ActiveLayerOptimizer
from any2rwkv.streamed_teacher import (
    Qwen35TeacherLayerLoader,
    StreamedQwen35HybridExecutor,
    StreamedQwen35Teacher,
)
from any2rwkv.target import build_zero_step_ledger, rwkv7_mixer_specs


class LayerTensorStoreTests(unittest.TestCase):
    def test_qwen35_l2_normalize_uses_additive_squared_norm_epsilon(self) -> None:
        value = torch.tensor([[0.0, 1e-4, -2e-4]], dtype=torch.float32)
        expected = value / torch.sqrt(value.square().sum(-1, keepdim=True) + 1e-6)

        actual = qwen35_l2_normalize(value)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(
            torch.equal(actual, torch.nn.functional.normalize(value, dim=-1))
        )

    def test_warmup_cosine_schedule_has_frozen_bounds(self) -> None:
        optimizer = ActiveLayerOptimizer(
            learning_rate=1e-3,
            learning_rate_schedule="warmup-cosine",
            warmup_steps=2,
            total_steps=6,
            min_learning_rate_ratio=0.1,
        )
        self.assertAlmostEqual(optimizer._learning_rate_multiplier(0), 0.5)
        self.assertAlmostEqual(optimizer._learning_rate_multiplier(1), 1.0)
        self.assertAlmostEqual(optimizer._learning_rate_multiplier(2), 1.0)
        self.assertAlmostEqual(optimizer._learning_rate_multiplier(5), 0.1)

    def test_warmup_constant_rejects_unused_final_learning_rate(self) -> None:
        with self.assertRaisesRegex(
            ContractError, "require identical initial/final learning rates"
        ):
            ActiveLayerOptimizer(
                learning_rate=1e-3,
                final_learning_rate=1e-4,
                learning_rate_schedule="warmup-constant",
                warmup_steps=2,
                total_steps=6,
            )

    def test_gradient_buckets_bound_collectives_and_keep_dtype_groups(self) -> None:
        gradients = [
            torch.zeros(4, dtype=torch.float32),
            torch.zeros(4, dtype=torch.float32),
            torch.zeros(4, dtype=torch.float16),
        ]
        buckets = _gradient_buckets(gradients, max_bytes=20)
        self.assertEqual([len(bucket) for bucket in buckets], [1, 1, 1])
        self.assertIs(buckets[0][0], gradients[0])

    def test_distributed_reductions_preserve_strided_in_place_semantics(
        self,
    ) -> None:
        context = DistributedContext(rank=0, local_rank=0, world_size=8)
        source = torch.arange(36, dtype=torch.float64).reshape(2, 3, 6)
        summed = source.clone().transpose(1, 2)
        maximized = source.clone().transpose(1, 2)
        expected_sum = summed.clone() + 7
        self.assertEqual(summed.stride(), (18, 1, 6))
        self.assertFalse(summed.is_contiguous())
        self.assertFalse(maximized.is_contiguous())

        def reduce(contiguous, *, op):
            self.assertTrue(contiguous.is_contiguous())
            if op is torch.distributed.ReduceOp.SUM:
                contiguous.add_(7)
            elif op is torch.distributed.ReduceOp.MAX:
                contiguous.fill_(5)
            else:
                self.fail(f"unexpected reduction: {op}")

        with mock.patch(
            "any2rwkv.distributed.dist.all_reduce",
            side_effect=reduce,
        ):
            summed_returned = context.all_reduce_sum(summed)
            maximized_returned = context.all_reduce_max(maximized)

        self.assertIs(summed_returned, summed)
        self.assertIs(maximized_returned, maximized)
        torch.testing.assert_close(summed, expected_sum)
        torch.testing.assert_close(maximized, torch.full_like(maximized, 5))

    def test_fixture_layers_are_indexed_and_loaded_without_mutating_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_dir = Path(temporary) / "source"
            write_fixture(source_dir, layers=60)
            checkpoint = read_checkpoint(source_dir)
            before = dict(checkpoint.file_hashes)
            store = LayerTensorStore(checkpoint)
            self.assertEqual(store.num_layers, 60)
            layer = store.load_layer(37)
            self.assertTrue(layer)
            self.assertTrue(all("layers.37." in name for name in layer))
            self.assertFalse(any(name.startswith("mtp.") for name in layer))
            self.assertFalse(any("layers.36." in name for name in layer))
            self.assertEqual(store.verify_source_shards(), {
                shard.name: before[shard.name] for shard in checkpoint.shards
            })

    def test_streamed_teacher_layer_strictly_loads_one_fixture_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_dir = Path(temporary) / "source"
            write_fixture(source_dir, layers=60)
            loader = Qwen35TeacherLayerLoader(read_checkpoint(source_dir))
            loaded = loader.load_layer(41, device="cpu", dtype=torch.float32)
            self.assertEqual(loaded.layer_index, 41)
            self.assertGreater(loaded.source_tensor_bytes, 0)
            self.assertTrue(all(not parameter.requires_grad for parameter in loaded.module.parameters()))
            with loader.layer_lease(41):
                with self.assertRaisesRegex(
                    ContractError,
                    "does not permit overlapping execution",
                ):
                    with loader.layer_lease(41):
                        pass

    def test_cached_teacher_reads_each_source_layer_once_across_forwards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_dir = Path(temporary) / "source"
            write_fixture(source_dir, layers=4)
            teacher = StreamedQwen35Teacher(
                read_checkpoint(source_dir, require_final_layers=False),
                device="cpu",
                dtype=torch.float32,
                cache_layers=True,
            )
            input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
            with mock.patch.object(
                teacher.loader.tensor_store,
                "load_layer",
                wraps=teacher.loader.tensor_store.load_layer,
            ) as load_layer:
                teacher.forward(input_ids)
                teacher.forward(input_ids)
            self.assertEqual(load_layer.call_count, 4)

    def test_streamed_teacher_matches_resident_fixture_forward(self) -> None:
        from transformers import AutoModelForCausalLM
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeRMSNormGated,
            torch_chunk_gated_delta_rule,
            torch_recurrent_gated_delta_rule,
        )

        with tempfile.TemporaryDirectory() as temporary:
            source_dir = Path(temporary) / "source"
            write_fixture(source_dir, layers=60)
            resident = AutoModelForCausalLM.from_pretrained(
                source_dir, torch_dtype=torch.float32
            ).eval()
            # causal-conv1d exposes a CUDA-only custom op even when the
            # Transformers reference model is intentionally resident on CPU.
            # Select Transformers' mathematically equivalent torch fallback.
            for module in resident.modules():
                if hasattr(module, "causal_conv1d_fn"):
                    module.causal_conv1d_fn = None
                if hasattr(module, "chunk_gated_delta_rule"):
                    module.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
                if hasattr(module, "recurrent_gated_delta_rule"):
                    module.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
                if (
                    hasattr(module, "norm")
                    and module.norm.__class__.__name__ == "FusedRMSNormGated"
                ):
                    replacement = Qwen3_5MoeRMSNormGated(
                        module.norm.hidden_size, eps=module.norm.eps
                    )
                    replacement.weight.data.copy_(module.norm.weight.data)
                    module.norm = replacement
            streamed = StreamedQwen35Teacher(
                read_checkpoint(source_dir), device="cpu", dtype=torch.float32
            )
            input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
            with torch.inference_mode():
                expected = resident(input_ids=input_ids, use_cache=False).logits
                actual = streamed.forward(
                    input_ids,
                    capture_layer_index=37,
                )
            torch.testing.assert_close(actual.logits, expected, rtol=1e-5, atol=1e-5)
            self.assertIsNotNone(actual.active_layer_input)
            self.assertIsNotNone(actual.active_mixer_output)
            self.assertIsNotNone(actual.active_block_output)

    def test_cached_layer_local_loads_neither_prefix_nor_suffix_during_optimization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_dir = Path(temporary) / "source"
            write_fixture(source_dir, layers=4)
            source = read_checkpoint(source_dir, require_final_layers=False)
            config = Any2RWKVProxyConfig(
                **build_target_config(source.config, require_final_layers=False)
            )
            source_text = source.config.get("text_config", source.config)
            rope = source_text.get("rope_parameters", {})
            mixer = ProjectionBoundaryRWKV7Attention(
                config,
                2,
                source_used_rope=source_text["layer_types"][2] == "full_attention",
                rotary_dim=16,
                rope_theta=float(rope.get("rope_theta", 10_000.0)),
            )
            teacher = StreamedQwen35Teacher(
                source,
                device="cpu",
                dtype=torch.float32,
                load_output_head=False,
            )
            loaded = teacher.loader.load_layer(2, device="cpu", dtype=torch.float32)
            hidden = teacher.embed_input_ids(
                torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
            )
            shared = torch.randn_like(hidden)
            executor = StreamedQwen35HybridExecutor(teacher)
            with mock.patch.object(
                teacher.loader,
                "load_layer",
                side_effect=AssertionError("cached local forward must not load any layer"),
            ):
                output = executor.forward_cached_layer_local(
                    hidden,
                    shared_states=shared,
                    active_layer_index=2,
                    active_mixer=mixer,
                    loaded_layer=loaded,
                )
            signals = output.teacher_signals
            self.assertIsNotNone(signals)
            assert signals is not None
            source_mixer = loaded.module.linear_attn
            beta = signals["beta"]
            decay = signals["decay"]
            raw_query = signals["q"].view(
                *signals["q"].shape[:2],
                source_mixer.num_v_heads,
                source_mixer.head_k_dim,
            )
            raw_key = signals["k"].view_as(raw_query)
            raw_value = signals["v"].view(
                *signals["v"].shape[:2],
                source_mixer.num_v_heads,
                source_mixer.head_v_dim,
            )
            torch.testing.assert_close(
                signals["recurrent_r"],
                (
                    qwen35_l2_normalize(raw_query.float())
                    * source_mixer.head_k_dim**-0.5
                )
                .to(raw_query.dtype)
                .flatten(2),
            )
            torch.testing.assert_close(
                signals["write_key"],
                qwen35_l2_normalize(raw_key.float())
                .to(raw_key.dtype)
                .flatten(2),
            )
            torch.testing.assert_close(
                signals["write_value"],
                (raw_value * beta.unsqueeze(-1)).flatten(2),
            )
            torch.testing.assert_close(
                signals["erase_rate"],
                (beta * decay).repeat_interleave(
                    source_mixer.head_v_dim, dim=-1
                ),
            )
            normalized_mse(
                output.student_block_output,
                output.teacher_block_output,
            ).backward()
            self.assertTrue(
                any(
                    parameter.grad is not None and torch.count_nonzero(parameter.grad)
                    for parameter in mixer.parameters()
                )
            )
            self.assertIsNone(teacher.norm_weight)
            self.assertIsNone(teacher.lm_head_weight)

    def test_full_attention_teacher_traces_use_the_actual_post_norm_mixer_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_dir = Path(temporary) / "source"
            write_fixture(source_dir, layers=4)
            source = read_checkpoint(source_dir, require_final_layers=False)
            config = Any2RWKVProxyConfig(
                **build_target_config(source.config, require_final_layers=False)
            )
            source_text = source.config.get("text_config", source.config)
            rope = source_text.get("rope_parameters", {})
            mixer = ProjectionBoundaryRWKV7Attention(
                config,
                3,
                source_used_rope=True,
                rotary_dim=16,
                rope_theta=float(rope.get("rope_theta", 10_000.0)),
            )
            teacher = StreamedQwen35Teacher(
                source,
                device="cpu",
                dtype=torch.float32,
                load_output_head=False,
            )
            loaded = teacher.loader.load_layer(3, device="cpu", dtype=torch.float32)
            hidden = teacher.embed_input_ids(
                torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
            )
            executor = StreamedQwen35HybridExecutor(teacher)
            output = executor.forward_cached_layer_local(
                hidden,
                shared_states=torch.randn_like(hidden),
                active_layer_index=3,
                active_mixer=mixer,
                loaded_layer=loaded,
            )
            signals = output.teacher_signals
            self.assertIsNotNone(signals)
            assert signals is not None

            source_mixer = loaded.module.self_attn
            normalized = loaded.module.input_layernorm(hidden)
            input_shape = normalized.shape[:-1]
            head_dim = int(source_mixer.head_dim)
            hidden_shape = (*input_shape, -1, head_dim)
            query, gate = torch.chunk(
                source_mixer.q_proj(normalized).view(
                    *input_shape, -1, head_dim * 2
                ),
                2,
                dim=-1,
            )
            query = source_mixer.q_norm(query.view(hidden_shape)).flatten(2)
            key = source_mixer.k_norm(
                source_mixer.k_proj(normalized).view(hidden_shape)
            )
            value = source_mixer.v_proj(normalized).view(hidden_shape)
            groups = int(source_mixer.num_key_value_groups)
            if groups > 1:
                key = key.repeat_interleave(groups, dim=2)
                value = value.repeat_interleave(groups, dim=2)

            torch.testing.assert_close(signals["mixer_input"], normalized)
            torch.testing.assert_close(signals["q"], query)
            torch.testing.assert_close(signals["k"], key.flatten(2))
            torch.testing.assert_close(signals["v"], value.flatten(2))
            torch.testing.assert_close(
                signals["gate"], torch.sigmoid(gate.flatten(2))
            )

            raw_query, raw_gate = torch.chunk(
                source_mixer.q_proj(hidden).view(
                    *input_shape, -1, head_dim * 2
                ),
                2,
                dim=-1,
            )
            self.assertFalse(
                torch.allclose(
                    signals["q"],
                    source_mixer.q_norm(raw_query.view(hidden_shape)).flatten(2),
                )
            )
            self.assertFalse(
                torch.allclose(
                    signals["gate"], torch.sigmoid(raw_gate.flatten(2))
                )
            )


class ChunkedTokenKLTests(unittest.TestCase):
    def test_chunked_value_and_gradient_match_full_vocab_objective(self) -> None:
        generator = torch.Generator().manual_seed(20260714)
        teacher = torch.randn(2, 5, 37, generator=generator, dtype=torch.float64)
        full_student = torch.randn(2, 5, 37, generator=generator, dtype=torch.float64, requires_grad=True)
        chunked_student = full_student.detach().clone().requires_grad_(True)
        full = token_kl(full_student, teacher)
        chunked = chunked_token_kl(
            chunked_student,
            teacher,
            vocab_chunk_size=7,
        )
        full_gradient = torch.autograd.grad(full, full_student)[0]
        chunked_gradient = torch.autograd.grad(chunked, chunked_student)[0]
        torch.testing.assert_close(chunked, full, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(chunked_gradient, full_gradient, rtol=2e-5, atol=2e-6)

    def test_token_kl_is_mean_per_token_not_sequence_sum(self) -> None:
        student = torch.tensor([[[1.0, -1.0]]])
        teacher = torch.tensor([[[0.0, 0.0]]])
        single = token_kl(student, teacher)
        repeated = token_kl(student.repeat(1, 7, 1), teacher.repeat(1, 7, 1))
        torch.testing.assert_close(repeated, single)


class ActiveLayerOptimizerTests(unittest.TestCase):
    def test_user_adamw_baseline_uses_ten_actual_warmup_updates(self) -> None:
        module = torch.nn.Linear(2, 2, bias=False)
        optimizer = ActiveLayerOptimizer(
            optimizer_name="adamw",
            learning_rate=1e-6,
            final_learning_rate=1e-6,
            adam_betas=(0.9, 0.99),
            weight_decay=0.1,
            learning_rate_schedule="warmup-constant",
            warmup_steps=10,
            total_steps=20,
            gradient_clip_norm=1.0,
        )
        optimizer.activate(0, module)

        applied = []
        for _ in range(11):
            optimizer.backward(
                module(torch.ones(1, 2)).square().mean(),
                accumulation_steps=1,
            )
            applied.append(optimizer.last_learning_rate)

        expected = [1e-7 * step for step in range(1, 11)] + [1e-6]
        for actual, target in zip(applied, expected, strict=True):
            self.assertAlmostEqual(actual, target, places=15)

    def test_explicit_adamw_contract_controls_param_group_and_snapshot_identity(self) -> None:
        module = torch.nn.Linear(4, 4, bias=False)
        optimizer = ActiveLayerOptimizer(
            optimizer_name="adamw",
            learning_rate=1e-6,
            final_learning_rate=1e-6,
            adam_betas=(0.9, 0.99),
            adam_epsilon=1e-8,
            weight_decay=0.1,
            learning_rate_schedule="warmup-constant",
            warmup_steps=10,
            total_steps=20,
            gradient_clip_norm=1.0,
        )
        optimizer.activate(0, module)
        group = optimizer.optimizer.param_groups[0]
        self.assertEqual(group["betas"], (0.9, 0.99))
        self.assertEqual(group["eps"], 1e-8)
        self.assertEqual(group["weight_decay"], 0.1)
        snapshot = optimizer.release()

        incompatible = ActiveLayerOptimizer(
            learning_rate=1e-6,
            final_learning_rate=1e-6,
            adam_betas=(0.9, 0.999),
            weight_decay=0.1,
            learning_rate_schedule="warmup-constant",
            warmup_steps=10,
            total_steps=20,
        )
        with self.assertRaisesRegex(ContractError, "snapshot contract differs"):
            incompatible.activate(0, module, snapshot=snapshot)

    def test_bf16_module_updates_through_fp32_master_weights_and_moments(self) -> None:
        torch.manual_seed(23)
        module = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        before = module.weight.detach().clone()
        optimizer = ActiveLayerOptimizer(learning_rate=1e-2)
        optimizer.activate(0, module)

        value = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4) / 8
        self.assertTrue(
            optimizer.backward(
                module(value).float().square().mean(),
                accumulation_steps=1,
            )
        )

        self.assertFalse(torch.equal(module.weight, before))
        self.assertIsNotNone(optimizer.master_parameters)
        self.assertTrue(
            all(
                parameter.dtype == torch.float32
                for parameter in optimizer.master_parameters or ()
            )
        )
        state_tensors = [
            value
            for state in optimizer.optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor) and value.numel() > 1
        ]
        self.assertTrue(state_tensors)
        self.assertTrue(all(value.dtype == torch.float32 for value in state_tensors))

    def test_accumulation_weights_unequal_tail_by_sample_count(self) -> None:
        torch.manual_seed(29)
        initial = torch.nn.Linear(3, 2, bias=False)
        accumulated = torch.nn.Linear(3, 2, bias=False)
        combined = torch.nn.Linear(3, 2, bias=False)
        accumulated.load_state_dict(initial.state_dict())
        combined.load_state_dict(initial.state_dict())
        full = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 7

        weighted = ActiveLayerOptimizer(learning_rate=1e-2)
        weighted.activate(2, accumulated)
        self.assertFalse(
            weighted.backward(
                accumulated(full[:3]).square().mean(),
                accumulation_steps=2,
                sample_weight=3,
            )
        )
        self.assertTrue(
            weighted.backward(
                accumulated(full[3:]).square().mean(),
                accumulation_steps=2,
                sample_weight=1,
            )
        )

        reference = ActiveLayerOptimizer(learning_rate=1e-2)
        reference.activate(2, combined)
        self.assertTrue(
            reference.backward(
                combined(full).square().mean(),
                accumulation_steps=1,
                sample_weight=4,
            )
        )
        torch.testing.assert_close(accumulated.weight, combined.weight)

    def test_snapshot_rejects_accidental_trainable_set_expansion(self) -> None:
        module = torch.nn.Linear(4, 4, bias=True)
        signals = ActiveLayerOptimizer(learning_rate=1e-2)
        signals.activate(3, module, trainable_names={"weight"})
        snapshot = signals.release()

        block = ActiveLayerOptimizer(learning_rate=1e-2)
        with self.assertRaisesRegex(
            ContractError, "trainable parameter set differs"
        ):
            block.activate(3, module, snapshot=snapshot, trainable_names=None)

    def test_optimizer_reports_gradient_and_relative_update_telemetry(self) -> None:
        module = torch.nn.Linear(4, 4, bias=False)
        optimizer = ActiveLayerOptimizer(learning_rate=1e-3)
        optimizer.activate(0, module)
        self.assertTrue(
            optimizer.backward(
                module(torch.ones(2, 4)).square().mean(), accumulation_steps=1
            )
        )

        telemetry = optimizer.telemetry()
        self.assertGreater(telemetry["last_gradient_l2"], 0.0)
        self.assertGreaterEqual(
            telemetry["max_gradient_l2"], telemetry["last_gradient_l2"]
        )
        self.assertGreater(telemetry["max_parameter_gradient_l2"], 0.0)
        self.assertEqual(telemetry["worst_gradient_parameter"], "weight")
        self.assertGreater(telemetry["last_learning_rate"], 0.0)
        self.assertGreater(telemetry["max_parameter_update_l2"], 0.0)
        self.assertEqual(telemetry["worst_absolute_update_parameter"], "weight")
        self.assertGreater(telemetry["max_parameter_update_relative_l2"], 0.0)
        self.assertEqual(telemetry["worst_update_parameter"], "weight")

    def test_optimizer_clips_fp32_master_gradient_before_adamw(self) -> None:
        module = torch.nn.Linear(4, 4, bias=False)
        optimizer = ActiveLayerOptimizer(
            learning_rate=1e-3,
            gradient_clip_norm=0.25,
        )
        optimizer.activate(0, module)
        self.assertTrue(
            optimizer.backward(
                module(torch.ones(2, 4)).square().mean(), accumulation_steps=1
            )
        )

        telemetry = optimizer.telemetry()
        self.assertGreater(telemetry["last_gradient_l2"], 0.25)
        self.assertAlmostEqual(telemetry["last_applied_gradient_l2"], 0.25)
        self.assertEqual(telemetry["gradient_clip_count"], 1)
        self.assertLess(telemetry["min_gradient_clip_scale"], 1.0)

    def test_optimizer_bounds_transactional_relative_update(self) -> None:
        module = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            module.weight.fill_(1.0)
        before = module.weight.detach().clone()
        optimizer = ActiveLayerOptimizer(
            learning_rate=1.0,
            max_parameter_update_relative_l2=0.1,
        )
        optimizer.activate(0, module)
        self.assertTrue(
            optimizer.backward(
                module(torch.ones(2, 4)).square().mean(), accumulation_steps=1
            )
        )

        applied_relative = float(
            (module.weight.detach() - before).norm() / before.norm()
        )
        telemetry = optimizer.telemetry()
        self.assertLessEqual(applied_relative, 0.1 + 1e-6)
        self.assertGreater(telemetry["max_parameter_update_relative_l2"], 0.1)
        self.assertLessEqual(
            telemetry["max_applied_parameter_update_relative_l2"], 0.1 + 1e-6
        )
        self.assertEqual(telemetry["update_trust_region_count"], 1)
        self.assertLess(telemetry["min_update_trust_scale"], 1.0)

    def test_zero_initialized_parameter_does_not_freeze_transactional_update(self) -> None:
        module = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            module.weight.zero_()
        optimizer = ActiveLayerOptimizer(
            learning_rate=1e-3,
            max_parameter_update_relative_l2=0.1,
        )
        optimizer.activate(0, module)
        self.assertTrue(
            optimizer.backward(
                (module(torch.ones(2, 4)) - 1).square().mean(),
                accumulation_steps=1,
            )
        )

        first = optimizer.telemetry()
        self.assertGreater(float(module.weight.detach().norm()), 0.0)
        self.assertEqual(first["relative_update_exempt_parameter_count"], 1)
        self.assertEqual(first["worst_relative_update_exempt_parameter"], "weight")
        self.assertGreater(first["max_relative_update_exempt_l2"], 0.0)
        self.assertEqual(first["update_trust_region_count"], 0)
        snapshot = optimizer.release()

        resumed = ActiveLayerOptimizer(
            learning_rate=1e-3,
            max_parameter_update_relative_l2=0.1,
        )
        resumed.activate(0, module, snapshot=snapshot)
        self.assertTrue(
            resumed.backward(
                (module(torch.ones(2, 4)) - 1).square().mean(),
                accumulation_steps=1,
            )
        )
        self.assertEqual(
            resumed.telemetry()["relative_update_exempt_parameter_count"], 1
        )
        self.assertEqual(resumed.telemetry()["update_trust_region_count"], 0)

    def test_mid_accumulation_snapshot_restores_without_resident_optimizer_list(self) -> None:
        torch.manual_seed(31)
        initial = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        uninterrupted = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        resumed = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        uninterrupted.load_state_dict(initial.state_dict())
        resumed.load_state_dict(initial.state_dict())
        value = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)

        reference = ActiveLayerOptimizer(learning_rate=1e-2)
        reference.activate(17, uninterrupted)
        self.assertFalse(
            reference.backward(
                uninterrupted(value).float().square().mean(), accumulation_steps=2
            )
        )
        self.assertTrue(
            reference.backward(
                uninterrupted(value).float().square().mean(), accumulation_steps=2
            )
        )

        first = ActiveLayerOptimizer(learning_rate=1e-2)
        first.activate(17, resumed)
        self.assertFalse(
            first.backward(
                resumed(value).float().square().mean(), accumulation_steps=2
            )
        )
        snapshot = first.release()
        self.assertFalse(first.is_active)
        self.assertIsNone(first.optimizer)
        second = ActiveLayerOptimizer(learning_rate=1e-2)
        second.activate(17, resumed, snapshot=snapshot)
        self.assertTrue(
            second.backward(
                resumed(value).float().square().mean(), accumulation_steps=2
            )
        )
        torch.testing.assert_close(resumed.weight, uninterrupted.weight, rtol=0, atol=0)

        incompatible = ActiveLayerOptimizer(learning_rate=1e-2)
        with self.assertRaisesRegex(ContractError, "parameter signature differs"):
            incompatible.activate(
                17,
                torch.nn.Linear(4, 5, bias=False),
                snapshot=snapshot,
            )

    def test_bf16_epoch_release_preserves_fp32_master_weight_residuals(self) -> None:
        torch.manual_seed(37)
        initial = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        uninterrupted = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        resumed = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        uninterrupted.load_state_dict(initial.state_dict())
        resumed.load_state_dict(initial.state_dict())
        first_value = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4) / 7
        second_value = torch.arange(8, 16, dtype=torch.bfloat16).reshape(2, 4) / 11

        reference = ActiveLayerOptimizer(learning_rate=1e-3)
        reference.activate(5, uninterrupted)
        self.assertTrue(
            reference.backward(
                uninterrupted(first_value).float().square().mean(),
                accumulation_steps=1,
            )
        )
        self.assertTrue(
            reference.backward(
                uninterrupted(second_value).float().square().mean(),
                accumulation_steps=1,
            )
        )

        first_epoch = ActiveLayerOptimizer(learning_rate=1e-3)
        first_epoch.activate(5, resumed)
        self.assertTrue(
            first_epoch.backward(
                resumed(first_value).float().square().mean(), accumulation_steps=1
            )
        )
        snapshot = first_epoch.release()
        self.assertTrue(snapshot.master_parameters)
        self.assertTrue(
            any(
                not torch.equal(master, module_parameter.float())
                for master, module_parameter in zip(
                    snapshot.master_parameters, resumed.parameters(), strict=True
                )
            )
        )

        second_epoch = ActiveLayerOptimizer(learning_rate=1e-3)
        second_epoch.activate(5, resumed, snapshot=snapshot)
        self.assertTrue(
            second_epoch.backward(
                resumed(second_value).float().square().mean(), accumulation_steps=1
            )
        )
        torch.testing.assert_close(resumed.weight, uninterrupted.weight, rtol=0, atol=0)
        for actual, expected in zip(
            second_epoch.master_parameters or (),
            reference.master_parameters or (),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class RWKV7MixerLayerStoreTests(unittest.TestCase):
    def test_overlay_roundtrip_replaces_only_one_hash_bound_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            base_dir = root / "zero-step"
            source_path = write_fixture(source_dir, layers=4)
            source = read_checkpoint(source_path, require_final_layers=False)
            target_config = build_target_config(
                source.config, require_final_layers=False
            )
            specs = tuple(
                spec
                for layer_index in range(4)
                for spec in rwkv7_mixer_specs(
                    layer_index, hidden_size=64, head_dim=16
                )
            )
            plan = plan_warm_start(source, specs, variant=WarmStartVariant.MAPPED)
            export_hf_checkpoint(
                source,
                base_dir,
                target_config=target_config,
                target_specs=specs,
                target_tensor_provider=WarmStartTensorProvider(source, specs, plan),
            )
            store = RWKV7MixerLayerStore(base_dir, root / "overlays")
            mixer = store.load_mixer(2, device="cpu", dtype=torch.float32)
            with torch.no_grad():
                next(mixer.parameters()).add_(0.125)
            expected = {
                name: tensor.detach().clone() for name, tensor in mixer.state_dict().items()
            }
            metadata = store.save_mixer(2, mixer, cursor={"visit": 7})
            self.assertEqual(metadata["cursor"], {"visit": 7})
            restored = store.load_mixer(2, device="cpu", dtype=torch.float32)
            for name, tensor in restored.state_dict().items():
                torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
            untouched = store.load_mixer(1, device="cpu", dtype=torch.float32)
            self.assertFalse((root / "overlays/layer-001.safetensors").exists())
            self.assertTrue(untouched.state_dict())

    def test_all_layer_fingerprint_changes_with_selected_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = read_checkpoint(
                write_fixture(root / "source", layers=4), require_final_layers=False
            )
            target_config = build_target_config(source.config, require_final_layers=False)
            specs = tuple(
                spec
                for layer_index in range(4)
                for spec in rwkv7_mixer_specs(
                    layer_index, hidden_size=64, head_dim=16
                )
            )
            plan = plan_warm_start(source, specs, variant=WarmStartVariant.MAPPED)
            export_hf_checkpoint(
                source,
                root / "zero-step",
                target_config=target_config,
                target_specs=specs,
                target_tensor_provider=WarmStartTensorProvider(source, specs, plan),
            )
            store = RWKV7MixerLayerStore(root / "zero-step", root / "overlays")
            for layer_index in range(4):
                mixer = store.load_mixer(layer_index, device="cpu", dtype=torch.float32)
                store.save_mixer(layer_index, mixer, cursor={"visit": 0})
            before = store.fingerprint()
            mixer = store.load_mixer(2, device="cpu", dtype=torch.float32)
            with torch.no_grad():
                next(mixer.parameters()).add_(0.5)
            store.save_mixer(2, mixer, cursor={"visit": 1})
            self.assertNotEqual(store.fingerprint(), before)

    def test_materialized_checkpoint_strictly_contains_all_selected_overlays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = read_checkpoint(
                write_fixture(root / "source", layers=4), require_final_layers=False
            )
            target_config = build_target_config(source.config, require_final_layers=False)
            specs = tuple(
                spec
                for layer_index in range(4)
                for spec in rwkv7_mixer_specs(
                    layer_index, hidden_size=64, head_dim=16
                )
            )
            plan = plan_warm_start(source, specs, variant=WarmStartVariant.MAPPED)
            export_hf_checkpoint(
                source,
                root / "zero-step",
                target_config=target_config,
                target_specs=specs,
                target_tensor_provider=WarmStartTensorProvider(source, specs, plan),
            )
            source_names = tuple(source.tensor_names())
            shard_hashes = tuple(
                source.file_hashes[path.name] for path in source.shards
            )
            ledger, _, target_names = build_zero_step_ledger(
                source_names,
                layer_count=source.contract.num_hidden_layers,
                hidden_size=source.contract.hidden_size,
                head_dim=16,
                source_shard_hashes=shard_hashes,
            )
            apply_warm_start_plan(ledger, plan)
            ledger.write(root / "mapping.json")
            write_json(
                root / "mapping-coverage.json",
                ledger.validate(source_names, target_names),
            )
            write_json(root / "warm-start-plan.json", plan.to_dict())
            store = RWKV7MixerLayerStore(root / "zero-step", root / "overlays")
            expected = {}
            for layer_index in range(4):
                mixer = store.load_mixer(layer_index, device="cpu", dtype=torch.float32)
                with torch.no_grad():
                    next(mixer.parameters()).add_(layer_index + 0.25)
                expected[layer_index] = {
                    name: tensor.detach().clone()
                    for name, tensor in mixer.state_dict().items()
                }
                store.save_mixer(layer_index, mixer, cursor={"layer": layer_index})
            (root / "layer-convergence.json").write_text(
                json.dumps({"schema_version": 2, "epochs": [{"layer": 0}]}) + "\n",
                encoding="utf-8",
            )
            activation_fit = root / "activation-fit"
            activation_fit.mkdir()
            (activation_fit / "time-mix-layer-000.json").write_text(
                json.dumps(
                    {
                        "layer": 0,
                        "status": "accepted",
                        "boundary": "source-mixer-output-to-native-time-mix-v1",
                        "optimizer": "Adam",
                        "train_cache_binding": {"split": "distill_train"},
                        "validation_cache_binding": {"split": "validation"},
                        "selected_parameter_sha256": {"x_r": "a" * 64},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            materialized = store.materialize_checkpoint(
                root / "materialized", fitted_evidence_root=root
            )
            payload = json.loads((materialized / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["any2rwkv"]["training_stage"], "layerwise-local-complete")
            mapping = json.loads(
                (materialized / "mapping.json").read_text(encoding="utf-8")
            )
            trained_targets = [
                row
                for row in mapping["targets"]
                if "activation_fit_manifest_sha256=" in row["evidence"]
            ]
            self.assertTrue(trained_targets)
            self.assertTrue(
                all(row["provenance"] == "fitted" for row in trained_targets)
            )
            provenance = json.loads(
                (materialized / "activation-fit-provenance.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(provenance["activation_fit_reports"]), 1)
            self.assertEqual(
                provenance["mixer_overlay_fingerprint"],
                payload["any2rwkv"]["mixer_overlay_fingerprint"],
            )
            reloaded = RWKV7MixerLayerStore(materialized, root / "reload-overlays")
            for layer_index in range(4):
                mixer = reloaded.load_mixer(layer_index, device="cpu", dtype=torch.float32)
                for name, tensor in mixer.state_dict().items():
                    torch.testing.assert_close(
                        tensor, expected[layer_index][name], rtol=0, atol=0
                    )

    def test_sweep_snapshot_restores_selected_all_layer_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            base_dir = root / "zero-step"
            source = read_checkpoint(
                write_fixture(source_dir, layers=4), require_final_layers=False
            )
            target_config = build_target_config(source.config, require_final_layers=False)
            specs = tuple(
                spec
                for layer_index in range(4)
                for spec in rwkv7_mixer_specs(
                    layer_index, hidden_size=64, head_dim=16
                )
            )
            plan = plan_warm_start(source, specs, variant=WarmStartVariant.MAPPED)
            export_hf_checkpoint(
                source,
                base_dir,
                target_config=target_config,
                target_specs=specs,
                target_tensor_provider=WarmStartTensorProvider(source, specs, plan),
            )
            store = RWKV7MixerLayerStore(base_dir, root / "overlays")
            expected = {}
            for layer_index in range(4):
                mixer = store.load_mixer(layer_index, device="cpu", dtype=torch.float32)
                with torch.no_grad():
                    next(mixer.parameters()).add_(layer_index + 1)
                expected[layer_index] = {
                    name: tensor.detach().clone() for name, tensor in mixer.state_dict().items()
                }
                store.save_mixer(layer_index, mixer, cursor={"sweep": 0})
            snapshot = store.snapshot(root / "sweeps/sweep-00")
            changed = store.load_mixer(2, device="cpu", dtype=torch.float32)
            with torch.no_grad():
                next(changed.parameters()).mul_(0)
            store.save_mixer(2, changed, cursor={"sweep": 1})
            store.restore_snapshot(snapshot)
            for layer_index in range(4):
                restored = store.load_mixer(layer_index, device="cpu", dtype=torch.float32)
                for name, tensor in restored.state_dict().items():
                    torch.testing.assert_close(
                        tensor, expected[layer_index][name], rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
