from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from any2rwkv.checkpoint import read_checkpoint
from any2rwkv.configuration_any2rwkv import Any2RWKVProxyConfig
from any2rwkv.contract import build_target_config
from any2rwkv.distill import chunked_token_kl, token_kl
from any2rwkv.fixture import write_fixture
from any2rwkv.export import export_hf_checkpoint
from any2rwkv.errors import ContractError
from any2rwkv.layer_store import LayerTensorStore
from any2rwkv.migration_init import (
    WarmStartTensorProvider,
    WarmStartVariant,
    materialize_warm_start,
    plan_warm_start,
)
from any2rwkv.mixer import ProjectionBoundaryRWKV7Attention
from any2rwkv.mixer_store import RWKV7MixerLayerStore
from any2rwkv.streaming_training import ActiveLayerOptimizer
from any2rwkv.streamed_teacher import (
    Qwen35TeacherLayerLoader,
    StreamedQwen35HybridExecutor,
    StreamedQwen35Teacher,
)
from any2rwkv.streamed_distill_runner import (
    _WarmStartMixerProvider,
    _commit_streamed_generation,
    _head_partition_errors,
    _estimate_streamed_workload,
    _load_committed_generation,
    _read_streamed_baselines,
    _supervised_prediction_window,
    _write_streamed_baseline,
)
from any2rwkv.target import rwkv7_mixer_specs


class LayerTensorStoreTests(unittest.TestCase):
    def test_signals_layer_one_derives_v_first_from_frozen_layer_zero_shadow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = read_checkpoint(
                write_fixture(root / "source", layers=4), require_final_layers=False
            )
            target_payload = build_target_config(
                source.config, require_final_layers=False
            )
            config = Any2RWKVProxyConfig(**target_payload)
            source_text = source.config.get("text_config", source.config)
            source_types = source_text["layer_types"]
            rope = source_text.get("rope_parameters", {})
            rotary_dim = int(
                source_text.get("head_dim", 16)
                * float(rope.get("partial_rotary_factor", 1.0))
            )
            rotary_dim -= rotary_dim % 2
            mixers = [
                ProjectionBoundaryRWKV7Attention(
                    config,
                    layer_index,
                    source_used_rope=source_types[layer_index] == "full_attention",
                    rotary_dim=rotary_dim,
                    rope_theta=float(rope.get("rope_theta", 10_000.0)),
                )
                for layer_index in range(4)
            ]
            teacher = StreamedQwen35Teacher(
                source, device="cpu", dtype=torch.float32
            )
            output = StreamedQwen35HybridExecutor(teacher).forward(
                torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
                active_layer_index=1,
                active_mixer=mixers[1],
                converted_layer_indices=set(),
                frozen_mixer_provider=lambda layer_index: mixers[layer_index],
            )
            self.assertEqual(output.active_mixer_output.shape, (1, 4, 64))
            with self.assertRaisesRegex(ContractError, "does not permit padding"):
                StreamedQwen35HybridExecutor(teacher).forward(
                    torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
                    attention_mask=torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
                    active_layer_index=1,
                    active_mixer=mixers[1],
                    converted_layer_indices=set(),
                    frozen_mixer_provider=lambda layer_index: mixers[layer_index],
                )

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

    def test_streamed_teacher_matches_resident_fixture_forward(self) -> None:
        from transformers import AutoModelForCausalLM

        with tempfile.TemporaryDirectory() as temporary:
            source_dir = Path(temporary) / "source"
            write_fixture(source_dir, layers=60)
            resident = AutoModelForCausalLM.from_pretrained(
                source_dir, torch_dtype=torch.float32
            ).eval()
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
            self.assertIsNotNone(actual.active_recurrent_state)

    def test_streamed_hybrid_reloads_suffix_and_bridges_gradient_to_active_mixer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_dir = Path(temporary) / "source"
            write_fixture(source_dir, layers=4)
            source = read_checkpoint(source_dir, require_final_layers=False)
            target_payload = build_target_config(
                source.config,
                require_final_layers=False,
            )
            config = Any2RWKVProxyConfig(**target_payload)
            source_text = source.config.get("text_config", source.config)
            source_types = source_text["layer_types"]
            rope = source_text.get("rope_parameters", {})
            rotary_dim = int(
                source_text.get("head_dim", 16)
                * float(rope.get("partial_rotary_factor", 1.0))
            )
            rotary_dim -= rotary_dim % 2
            mixers = [
                ProjectionBoundaryRWKV7Attention(
                    config,
                    layer_index,
                    source_used_rope=source_types[layer_index] == "full_attention",
                    rotary_dim=rotary_dim,
                    rope_theta=float(rope.get("rope_theta", 10_000.0)),
                )
                for layer_index in range(4)
            ]
            specs = tuple(
                spec
                for layer_index in range(4)
                for spec in rwkv7_mixer_specs(layer_index, hidden_size=64)
            )
            plan = plan_warm_start(source, specs, variant=WarmStartVariant.MAPPED)
            tensors = materialize_warm_start(source, specs, plan)
            for layer_index, mixer in enumerate(mixers):
                prefix = f"model.layers.{layer_index}.attn."
                mixer.load_state_dict(
                    {
                        name: tensors[prefix + name]
                        for name in mixer.state_dict()
                    },
                    strict=True,
                )
            teacher = StreamedQwen35Teacher(
                source, device="cpu", dtype=torch.float32
            )
            executor = StreamedQwen35HybridExecutor(teacher)
            active_layer_index = 2
            output = executor.forward(
                torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
                active_layer_index=active_layer_index,
                active_mixer=mixers[active_layer_index],
                converted_layer_indices={0, 1, 3},
                frozen_mixer_provider=lambda layer_index: mixers[layer_index],
            )
            output.logits.square().mean().backward()
            self.assertTrue(
                any(
                    parameter.grad is not None
                    and torch.count_nonzero(parameter.grad)
                    for parameter in mixers[active_layer_index].parameters()
                )
            )
            for layer_index in (0, 1, 3):
                self.assertTrue(
                    all(parameter.grad is None for parameter in mixers[layer_index].parameters())
                )


class ChunkedTokenKLTests(unittest.TestCase):
    def test_scale_estimate_counts_teacher_student_and_suffix_weight_movement(self) -> None:
        class Plan:
            supervised_tokens = 512
            stage_tokens_per_layer = {
                "signals": 512,
                "block": 1024,
                "global": 2048,
            }
            corrective_max_sweeps = 1
            accumulation_steps = 1

        estimate = _estimate_streamed_workload(
            source_weight_bytes=4_000,
            num_layers=4,
            plan=Plan(),
        )
        self.assertEqual(estimate["teacher_full_forwards"], 44)
        self.assertEqual(estimate["student_full_forwards"], 44)
        self.assertEqual(estimate["suffix_layer_backward_reloads"], 66)
        self.assertEqual(estimate["layer_zero_shadow_loads"], 3)
        self.assertEqual(estimate["estimated_weight_bytes_moved"], 509_000)

    def test_supervised_prediction_window_keeps_exact_target_budget(self) -> None:
        logits = torch.arange(1 * 7 * 3).reshape(1, 7, 3)
        tokens = torch.arange(7).reshape(1, 7)
        selected, labels = _supervised_prediction_window(
            logits, tokens, start=2, targets=4
        )
        torch.testing.assert_close(selected, logits[:, 1:5])
        torch.testing.assert_close(labels, tokens[:, 2:6])
        self.assertEqual(labels.numel(), 4)
        perfect = torch.full((1, 7, 10), -20.0)
        for position in range(6):
            perfect[0, position, position + 1] = 20.0
        aligned_logits, aligned_labels = _supervised_prediction_window(
            perfect, tokens, start=2, targets=4
        )
        loss = torch.nn.functional.cross_entropy(
            aligned_logits.reshape(-1, aligned_logits.shape[-1]),
            aligned_labels.reshape(-1),
        )
        self.assertLess(float(loss), 1e-6)

    def test_head_partition_errors_report_each_target_head(self) -> None:
        teacher = torch.ones(1, 2, 8)
        student = teacher.clone()
        student[..., 2:4] = 2
        errors = _head_partition_errors(student, teacher, heads=4)
        self.assertEqual(errors, [0.0, 1.0, 0.0, 0.0])

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


class ActiveLayerOptimizerTests(unittest.TestCase):
    def test_mid_accumulation_snapshot_restores_without_resident_optimizer_list(self) -> None:
        torch.manual_seed(31)
        initial = torch.nn.Linear(4, 4, bias=False)
        uninterrupted = torch.nn.Linear(4, 4, bias=False)
        resumed = torch.nn.Linear(4, 4, bias=False)
        uninterrupted.load_state_dict(initial.state_dict())
        resumed.load_state_dict(initial.state_dict())
        value = torch.arange(8, dtype=torch.float32).reshape(2, 4)

        reference = ActiveLayerOptimizer(learning_rate=1e-2)
        reference.activate(17, uninterrupted)
        self.assertFalse(reference.backward(uninterrupted(value).square().mean(), accumulation_steps=2))
        self.assertTrue(reference.backward(uninterrupted(value).square().mean(), accumulation_steps=2))

        first = ActiveLayerOptimizer(learning_rate=1e-2)
        first.activate(17, resumed)
        self.assertFalse(first.backward(resumed(value).square().mean(), accumulation_steps=2))
        snapshot = first.release()
        self.assertFalse(first.is_active)
        self.assertIsNone(first.optimizer)
        second = ActiveLayerOptimizer(learning_rate=1e-2)
        second.activate(17, resumed, snapshot=snapshot)
        self.assertTrue(second.backward(resumed(value).square().mean(), accumulation_steps=2))
        torch.testing.assert_close(resumed.weight, uninterrupted.weight, rtol=0, atol=0)


class RWKV7MixerLayerStoreTests(unittest.TestCase):
    def test_generation_commit_restores_one_atomic_mixer_optimizer_cursor_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = read_checkpoint(
                write_fixture(root / "source", layers=4), require_final_layers=False
            )
            target_config = build_target_config(source.config, require_final_layers=False)
            specs = tuple(
                spec
                for layer_index in range(4)
                for spec in rwkv7_mixer_specs(layer_index, hidden_size=64)
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
            mixer = store.load_mixer(2, device="cpu", dtype=torch.float32)
            optimizer = ActiveLayerOptimizer(learning_rate=1e-3)
            optimizer.activate(2, mixer)
            snapshot = optimizer.release()
            progress_dir = root / "progress"
            trace_path = root / "active-layer-trace.jsonl"
            progress = {
                "schema_version": 2,
                "binding": {"source": "a" * 64},
                "next_visit": 7,
                "active_layer": 2,
                "consumed": 512,
                "data_cursor": 9,
            }
            _commit_streamed_generation(
                progress_dir=progress_dir,
                trace_path=trace_path,
                mixer_store=store,
                active_layer=2,
                mixer=mixer,
                optimizer_snapshot=snapshot,
                phase="microstep",
                committed_layer_state={},
                progress=progress,
                trace_row={"visit_index": 7, "data_cursor": 9},
            )
            expected = {
                name: tensor.detach().clone() for name, tensor in mixer.state_dict().items()
            }
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write('{"orphan": true}\n')
            changed = store.load_mixer(2, device="cpu", dtype=torch.float32)
            with torch.no_grad():
                next(changed.parameters()).add_(3)
            store.save_mixer(2, changed, cursor={"orphan": True})
            orphan = store.load_mixer(3, device="cpu", dtype=torch.float32)
            with torch.no_grad():
                next(orphan.parameters()).add_(7)
            store.save_mixer(3, orphan, cursor={"uncommitted_next_layer": True})
            (progress_dir / "optimizer-layer-003.pt").write_bytes(b"orphan")
            restored_progress, restored_optimizer = _load_committed_generation(
                progress_dir,
                trace_path=trace_path,
                mixer_store=store,
                progress_dir=progress_dir,
            )
            self.assertEqual(restored_progress["next_visit"], 7)
            self.assertEqual(restored_optimizer.layer_index, 2)
            self.assertEqual(
                trace_path.read_text(encoding="utf-8"),
                '{"data_cursor": 9, "visit_index": 7}\n',
            )
            restored = store.load_mixer(2, device="cpu", dtype=torch.float32)
            for name, tensor in restored.state_dict().items():
                torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
            self.assertFalse((root / "overlays/layer-003.safetensors").exists())
            self.assertFalse((progress_dir / "optimizer-layer-003.pt").exists())

    def test_streamed_warm_start_provider_matches_mapped_checkpoint_one_layer_at_a_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = read_checkpoint(
                write_fixture(root / "source", layers=4),
                require_final_layers=False,
            )
            base_dir = root / "zero-step"
            target_config = build_target_config(source.config, require_final_layers=False)
            specs = tuple(
                spec
                for index in range(4)
                for spec in rwkv7_mixer_specs(index, hidden_size=64)
            )
            mapped = plan_warm_start(source, specs, variant=WarmStartVariant.MAPPED)
            export_hf_checkpoint(
                source,
                base_dir,
                target_config=target_config,
                target_specs=specs,
                target_tensor_provider=WarmStartTensorProvider(source, specs, mapped),
            )
            checkpoint_store = RWKV7MixerLayerStore(base_dir, root / "overlay")
            provider = _WarmStartMixerProvider(
                source, base_dir, WarmStartVariant.MAPPED
            )
            for index in range(4):
                expected = checkpoint_store.load_mixer(
                    index, device="cpu", dtype=torch.float32
                )
                actual = provider.load_mixer(
                    index, device="cpu", dtype=torch.float32
                )
                for name, tensor in actual.state_dict().items():
                    torch.testing.assert_close(
                        tensor, expected.state_dict()[name], rtol=0, atol=0
                    )

    def test_streamed_baseline_artifact_is_protocol_bound_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "migration-baselines.json"
            binding = {"source": "a" * 64, "rows": 8}
            _write_streamed_baseline(
                path,
                binding=binding,
                name="random",
                metrics={"mean_token_kl": 3.0},
                token_budget=0,
            )
            self.assertEqual(
                _read_streamed_baselines(path, binding=binding), {"random"}
            )
            with self.assertRaisesRegex(ContractError, "different frozen protocol"):
                _read_streamed_baselines(path, binding={"source": "changed"})

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
                for spec in rwkv7_mixer_specs(layer_index, hidden_size=64)
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
                for spec in rwkv7_mixer_specs(layer_index, hidden_size=64)
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
                for spec in rwkv7_mixer_specs(layer_index, hidden_size=64)
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
