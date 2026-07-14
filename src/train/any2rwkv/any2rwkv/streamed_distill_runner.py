from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

from .artifacts import checkpoint_sha256, file_sha256, write_json
from .checkpoint import CheckpointManifest
from .distill import (
    DEFAULT_VOCAB_CHUNK_SIZE,
    LossBreakdown,
    LossWeights,
    SweepController,
    chunked_token_kl,
    normalized_mse,
)
from .errors import ContractError
from .export import export_hf_checkpoint
from .migration_init import (
    WarmStartTensorProvider,
    WarmStartVariant,
    plan_warm_start,
)
from .mapping import finalize_fitted_mapping
from .layer_store import LayerTensorStore
from .mixer_store import RWKV7MixerFactory, RWKV7MixerLayerStore
from .streamed_teacher import StreamedQwen35HybridExecutor, StreamedQwen35Teacher
from .streaming_training import ActiveLayerOptimizer, ActiveLayerOptimizerSnapshot
from .target import layer_index, rwkv7_mixer_specs


class _WarmStartMixerProvider:
    """Materialize one warm-start mixer at a time directly from source shards."""

    def __init__(
        self,
        source: CheckpointManifest,
        zero_step_dir: Path,
        variant: WarmStartVariant,
    ) -> None:
        self.factory = RWKV7MixerFactory.from_checkpoint_config(zero_step_dir)
        self.specs = tuple(
            spec
            for index in range(source.contract.num_hidden_layers)
            for spec in rwkv7_mixer_specs(index, hidden_size=source.contract.hidden_size)
        )
        self.by_layer = {
            index: tuple(spec for spec in self.specs if layer_index(spec.name) == index)
            for index in range(source.contract.num_hidden_layers)
        }
        plan = plan_warm_start(source, self.specs, variant=variant)
        self.tensor_provider = WarmStartTensorProvider(source, self.specs, plan)

    def load_mixer(self, index: int, *, device, dtype):
        mixer = self.factory.create(index, device="cpu", dtype=dtype)
        prefix = f"model.layers.{index}.attn."
        state = {
            spec.name.removeprefix(prefix): self.tensor_provider(spec)
            for spec in self.by_layer[index]
        }
        incompatible = mixer.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ContractError(
                f"streamed warm-start mixer {index} strict load failed: "
                f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
            )
        return mixer.to(device=device, dtype=dtype)


def run_streamed_distillation(
    *,
    source_manifest: CheckpointManifest,
    run_dir: Path,
    token_rows: Sequence[tuple[int, ...]],
    validation_rows: Sequence[tuple[int, ...]],
    plan: Any,
    dataset_manifest: Path,
    training_config: Path,
    resume: Path | None,
) -> dict[str, object]:
    """Scale runner with one source layer, mixer, and optimizer resident at a time."""
    if not torch.cuda.is_available():
        raise ContractError("streamed layer-store distillation requires CUDA")
    if not plan.activation_checkpointing:
        raise ContractError("streamed layer-store execution requires reloadable suffix checkpointing")
    zero_step_dir = run_dir / "checkpoint-zero-step"
    if not zero_step_dir.is_dir():
        raise ContractError("streamed distillation requires checkpoint-zero-step")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    mixer_store = RWKV7MixerLayerStore(
        zero_step_dir,
        run_dir / "streamed-mixer-layers",
    )
    num_layers = source_manifest.contract.num_hidden_layers
    visits = [
        (False, layer, stage, budget)
        for stage, budget in plan.stage_tokens_per_layer.items()
        for layer in range(num_layers)
    ]
    progressive_visits = len(visits)
    visits.extend(
        (True, layer, "rollout", plan.stage_tokens_per_layer["global"])
        for _ in range(plan.corrective_max_sweeps)
        for layer in reversed(range(num_layers))
    )
    progress_dir = run_dir / "streamed-checkpoints"
    progress_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "active-layer-trace.jsonl"
    cache_teacher_layers = bool(getattr(plan, "cache_teacher_layers", False))
    source_weight_bytes = sum(
        shard.stat().st_size for shard in source_manifest.shards
    )
    max_teacher_cache_bytes = getattr(plan, "max_teacher_cache_bytes", None)
    estimated_teacher_layer_bytes = LayerTensorStore(
        source_manifest
    ).estimated_layer_bytes(dtype)
    if cache_teacher_layers and (
        max_teacher_cache_bytes is None
        or estimated_teacher_layer_bytes > max_teacher_cache_bytes
    ):
        raise ContractError(
            "teacher layer cache exceeds max_teacher_cache_bytes before CUDA allocation: "
            f"required={estimated_teacher_layer_bytes} limit={max_teacher_cache_bytes}"
        )
    binding = {
        "source_config_sha256": source_manifest.file_hashes["config.json"],
        "source_shard_sha256": {
            shard.name: source_manifest.file_hashes[shard.name]
            for shard in source_manifest.shards
        },
        "dataset_manifest_sha256": file_sha256(dataset_manifest),
        "training_config_sha256": file_sha256(training_config),
        "zero_step_checkpoint_sha256": checkpoint_sha256(zero_step_dir),
        "warm_start_plan_sha256": file_sha256(run_dir / "warm-start-plan.json"),
        "streamed_runner_sha256": file_sha256(Path(__file__)),
        "streamed_teacher_sha256": file_sha256(
            Path(__file__).with_name("streamed_teacher.py")
        ),
        "streaming_training_sha256": file_sha256(
            Path(__file__).with_name("streaming_training.py")
        ),
        "execution_mode": "streamed_layer_store",
        "teacher_layer_cache": "cuda" if cache_teacher_layers else "none",
    }
    execution_estimate = _estimate_streamed_workload(
        source_weight_bytes=source_weight_bytes,
        num_layers=num_layers,
        plan=plan,
        baseline_rows=min(8, len(validation_rows)),
        target_mixer_bytes=max(mixer_store.estimated_mixer_bytes(dtype)),
    )
    execution_estimate["limit_weight_bytes_moved"] = (
        plan.max_estimated_weight_bytes_moved
    )
    execution_estimate["max_teacher_cache_bytes"] = max_teacher_cache_bytes
    execution_estimate["estimated_teacher_layer_bytes"] = (
        estimated_teacher_layer_bytes
    )
    execution_estimate["accepted"] = (
        plan.max_estimated_weight_bytes_moved is None
        or execution_estimate["estimated_weight_bytes_moved"]
        <= plan.max_estimated_weight_bytes_moved
    )
    write_json(run_dir / "execution-estimate.json", execution_estimate)
    if not execution_estimate["accepted"]:
        raise ContractError(
            "streamed distillation execution estimate exceeds the frozen weight-movement "
            "budget; use a distributed resident/expert-sharded backend or reduce the plan"
        )
    torch.cuda.reset_peak_memory_stats(device)
    teacher = StreamedQwen35Teacher(
        source_manifest,
        device=device,
        dtype=dtype,
        cache_layers=cache_teacher_layers,
    )
    executor = StreamedQwen35HybridExecutor(teacher)
    baseline_rows = validation_rows[: min(8, len(validation_rows))]
    if not baseline_rows:
        raise ContractError("streamed migration baseline split is empty")
    baseline_path = run_dir / "migration-baselines.json"
    baseline_binding = {
        "schema": "any2rwkv-migration-baseline-v1",
        **binding,
        "seed": plan.seed,
        "burn_in_tokens": plan.burn_in_tokens,
        "supervised_tokens": plan.supervised_tokens,
        "rows": len(baseline_rows),
        "precision": "bf16-io-fp32-state",
    }
    existing_baselines = _read_streamed_baselines(
        baseline_path, binding=baseline_binding
    )
    static_variants = (
        ("random", WarmStartVariant.RANDOM),
        ("naive_copy", WarmStartVariant.NAIVE_COPY),
        ("gdn_algebraic", WarmStartVariant.GDN_CONSTRAINED),
        ("kv_repeat", WarmStartVariant.KV_REPEAT),
        ("kv_expand", WarmStartVariant.KV_EXPAND),
        ("mapped", WarmStartVariant.MAPPED),
    )
    for name, variant in static_variants:
        if name in existing_baselines:
            continue
        provider = _WarmStartMixerProvider(source_manifest, zero_step_dir, variant)
        _write_streamed_baseline(
            baseline_path,
            binding=baseline_binding,
            name=name,
            metrics=_score_streamed_candidate(
                teacher,
                executor,
                provider,
                baseline_rows,
                plan,
                device,
                dtype,
            ),
            token_budget=0,
        )
        del provider
        torch.cuda.empty_cache()
    if cache_teacher_layers and teacher.loader.cached_layer_count < num_layers:
        cache_input_ids, cache_attention_mask, cache_position_ids = _encode_tokens(
            baseline_rows[0],
            plan.burn_in_tokens + plan.supervised_tokens,
            device,
        )
        teacher.forward(
            cache_input_ids,
            attention_mask=cache_attention_mask,
            position_ids=cache_position_ids,
        )
    residency_path = run_dir / "teacher-residency.json"
    residency = _teacher_residency(
        teacher,
        device=device,
        max_teacher_cache_bytes=max_teacher_cache_bytes,
        max_cuda_reserved_bytes=plan.max_cuda_reserved_bytes,
    )
    write_json(residency_path, residency)
    if not residency["accepted"]:
        raise ContractError(
            "resident teacher exceeds the frozen cache or CUDA reserved-memory limit"
        )
    next_visit = 0
    consumed = 0
    data_cursor = 0
    committed_layer_state: dict[str, dict[str, object]] = {}
    resumed_optimizer_snapshot = None
    if resume is not None:
        progress, resumed_optimizer_snapshot = _load_committed_generation(
            resume,
            trace_path=trace_path,
            mixer_store=mixer_store,
            progress_dir=progress_dir,
        )
        if progress["binding"] != binding:
            raise ContractError("streamed resume binding differs from source/data/plan")
        next_visit = int(progress["next_visit"])
        consumed = int(progress["consumed"])
        data_cursor = int(progress["data_cursor"])
        committed_layer_state = dict(progress.get("committed_layer_state", {}))
        torch.set_rng_state(progress["torch_rng"])
        torch.cuda.set_rng_state_all(progress["cuda_rng"])
        random.setstate(progress["python_rng"])
    else:
        if (progress_dir / "latest.json").exists():
            raise ContractError(
                "streamed run already has progress; pass --resume instead of starting over"
            )
        # A crash before the first latest.json contains no committed update.
        # Discard any half-published overlay/generation and restart from zero-step.
        mixer_store.discard_all_overlays()
        trace_path.unlink(missing_ok=True)
        if progress_dir.exists():
            shutil.rmtree(progress_dir)
        progress_dir.mkdir(parents=True)
        torch.manual_seed(plan.seed)
        torch.cuda.manual_seed_all(plan.seed)
        random.seed(plan.seed)
    trainable_names = _initial_trainable_names(run_dir, num_layers)
    controller = SweepController(
        min_sweeps=plan.corrective_min_sweeps,
        max_sweeps=plan.corrective_max_sweeps,
        min_delta=plan.corrective_min_delta,
    )
    sweep_path = run_dir / "corrective-sweeps.json"
    if sweep_path.is_file():
        controller.history = list(
            json.loads(sweep_path.read_text(encoding="utf-8")).get("sweeps", [])
        )
    total_length = plan.burn_in_tokens + plan.supervised_tokens
    committed_sweep_boundary = progressive_visits + len(controller.history) * num_layers
    stop = bool(
        controller.history
        and controller.history[-1].get("stop", False)
        and next_visit >= committed_sweep_boundary
    )
    for visit_index, (fully_recurrent, active_layer, stage, token_budget) in enumerate(visits):
        if stop:
            break
        if visit_index < next_visit:
            continue
        sweep_index = (
            None
            if visit_index < progressive_visits
            else (visit_index - progressive_visits) // num_layers
        )
        mixer = mixer_store.load_mixer(active_layer, device=device, dtype=dtype)
        optimizer = ActiveLayerOptimizer(learning_rate=plan.learning_rate)
        optimizer_snapshot = (
            resumed_optimizer_snapshot
            if resumed_optimizer_snapshot is not None
            and visit_index == next_visit
            and int(progress.get("active_layer", -1)) == active_layer
            else _load_optimizer_snapshot(
                progress_dir / f"optimizer-layer-{active_layer:03d}.pt"
            )
        )
        resumed_optimizer_snapshot = None
        signal_names = trainable_names[active_layer] if stage == "signals" else None
        optimizer.activate(
            active_layer,
            mixer,
            snapshot=optimizer_snapshot,
            trainable_names=signal_names,
        )
        snapshot = optimizer_snapshot
        if visit_index != next_visit:
            consumed = 0
        converted_layers = (
            set(range(num_layers)) - {active_layer}
            if fully_recurrent
            else (set() if stage == "signals" else set(range(active_layer)))
        )
        while consumed < token_budget or optimizer.accumulation_step:
            tokens = token_rows[data_cursor % len(token_rows)]
            input_ids, attention_mask, position_ids = _encode_tokens(
                tokens, total_length, device
            )
            teacher_output = teacher.forward(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                capture_layer_index=active_layer,
            )
            student_output = executor.forward(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                active_layer_index=active_layer,
                active_mixer=mixer,
                converted_layer_indices=converted_layers,
                frozen_mixer_provider=lambda index: mixer_store.load_mixer(
                    index, device=device, dtype=dtype
                ),
            )
            start = plan.burn_in_tokens
            stop_token = start + plan.supervised_tokens
            student_mixer = student_output.active_mixer_output[:, start:stop_token]
            teacher_mixer = teacher_output.active_mixer_output[:, start:stop_token].to(device)
            student_block = student_output.active_block_output[:, start:stop_token]
            teacher_block = teacher_output.active_block_output[:, start:stop_token].to(device)
            student_logits, labels = _supervised_prediction_window(
                student_output.logits,
                input_ids,
                start=start,
                targets=plan.supervised_tokens,
            )
            teacher_logits, teacher_labels = _supervised_prediction_window(
                teacher_output.logits.to(device),
                input_ids,
                start=start,
                targets=plan.supervised_tokens,
            )
            if not torch.equal(labels, teacher_labels):
                raise ContractError("teacher/student supervised token windows differ")
            teacher_state = teacher_output.active_recurrent_state
            state_supervision = "not_available_for_source_attention"
            state_mse = student_logits.new_zeros(())
            if teacher_state is not None:
                teacher_state = teacher_state.to(device=device, dtype=torch.float32)
                if teacher_state.shape != student_output.active_state.shape:
                    transposed = teacher_state.transpose(-1, -2)
                    if transposed.shape == student_output.active_state.shape:
                        teacher_state = transposed
                    else:
                        teacher_state = None
                        state_supervision = (
                            "not_available_for_changed_head_state_geometry:"
                            f"teacher={tuple(transposed.transpose(-1, -2).shape)}:"
                            f"student={tuple(student_output.active_state.shape)}"
                        )
                if teacher_state is not None:
                    state_mse = normalized_mse(
                        student_output.active_state.float(), teacher_state
                    )
                    state_supervision = "gdn-final-recurrent-state"
            cosine = 1 - torch.nn.functional.cosine_similarity(
                student_block.flatten(0, -2),
                teacher_block.flatten(0, -2),
                dim=-1,
            ).mean()
            head_errors = _head_partition_errors(
                student_mixer,
                teacher_mixer,
                heads=mixer.num_heads,
            )
            losses = LossBreakdown(
                intermediate_mse=normalized_mse(student_mixer, teacher_mixer),
                state_mse=state_mse,
                block_mse=normalized_mse(student_block, teacher_block),
                cosine=cosine,
                token_kl=chunked_token_kl(
                    student_logits,
                    teacher_logits,
                    vocab_chunk_size=DEFAULT_VOCAB_CHUNK_SIZE,
                ),
                shifted_ce=torch.nn.functional.cross_entropy(
                    student_logits.reshape(-1, student_logits.shape[-1]),
                    labels.reshape(-1),
                ),
                rollout=(
                    normalized_mse(student_logits, teacher_logits)
                    if fully_recurrent
                    else student_logits.new_zeros(())
                ),
            )
            loss_weights = LossWeights.for_stage(stage)
            if teacher_state is None:
                loss_weights = replace(loss_weights, state_mse=0.0)
            total = losses.weighted(loss_weights)
            stepped = optimizer.backward(
                total,
                accumulation_steps=plan.accumulation_steps,
            )
            if (
                plan.max_cuda_reserved_bytes is not None
                and torch.cuda.max_memory_reserved(device)
                > plan.max_cuda_reserved_bytes
            ):
                write_json(
                    residency_path,
                    _teacher_residency(
                        teacher,
                        device=device,
                        max_teacher_cache_bytes=max_teacher_cache_bytes,
                        max_cuda_reserved_bytes=plan.max_cuda_reserved_bytes,
                    ),
                )
                raise ContractError(
                    "distillation exceeds max_cuda_reserved_bytes during active-layer training"
                )
            gradient_norm = math.sqrt(
                sum(
                    float(parameter.grad.detach().float().square().sum())
                    for parameter in mixer.parameters()
                    if parameter.grad is not None
                )
            )
            data_cursor += 1
            consumed += int(attention_mask[:, start:stop_token].sum())
            trace_row = {
                    **binding,
                    "visit_index": visit_index,
                    "active_layer": active_layer,
                    "stage": stage,
                    "fully_recurrent": fully_recurrent,
                    "consumed_tokens": consumed,
                    "data_cursor": data_cursor,
                    "optimizer_stepped": stepped,
                    "gradient_norm": gradient_norm,
                    "state_supervision": state_supervision,
                    "state_loss_effective": teacher_state is not None,
                    "loss_weights": {
                        name: float(value)
                        for name, value in loss_weights.__dict__.items()
                    },
                    "head_error_space": "target-hidden-channel-partitions-after-source-output-projection",
                    "head_normalized_mse": head_errors,
                    "losses": {
                        name: float(value.detach())
                        for name, value in losses.__dict__.items()
                    },
                }
            snapshot = optimizer.release()
            committed_layer_state = _commit_streamed_generation(
                progress_dir=progress_dir,
                trace_path=trace_path,
                mixer_store=mixer_store,
                active_layer=active_layer,
                mixer=mixer,
                optimizer_snapshot=snapshot,
                phase="microstep",
                committed_layer_state=committed_layer_state,
                progress={
                    "schema_version": 2,
                    "binding": binding,
                    "next_visit": visit_index,
                    "active_layer": active_layer,
                    "consumed": consumed,
                    "data_cursor": data_cursor,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "python_rng": random.getstate(),
                },
                trace_row=trace_row,
            )
            if consumed < token_budget or snapshot.accumulation_step:
                mixer = mixer_store.load_mixer(active_layer, device=device, dtype=dtype)
                optimizer = ActiveLayerOptimizer(learning_rate=plan.learning_rate)
                optimizer.activate(
                    active_layer,
                    mixer,
                    snapshot=snapshot,
                    trainable_names=signal_names,
                )
        completed_consumed = consumed
        stage_boundary = not fully_recurrent and active_layer == num_layers - 1
        sweep_boundary = fully_recurrent and active_layer == 0
        evidence_boundary = stage_boundary or sweep_boundary
        consumed = 0
        next_visit = visit_index + 1
        if snapshot is None:
            raise ContractError("streamed visit completed without an optimizer snapshot")
        committed_layer_state = _commit_streamed_generation(
            progress_dir=progress_dir,
            trace_path=trace_path,
            mixer_store=mixer_store,
            active_layer=active_layer,
            mixer=mixer,
            optimizer_snapshot=snapshot,
            phase=(
                f"{stage}-data-complete"
                if evidence_boundary
                else "visit-complete"
            ),
            committed_layer_state=committed_layer_state,
            progress={
                "schema_version": 2,
                "binding": binding,
                "next_visit": visit_index if evidence_boundary else next_visit,
                "active_layer": active_layer,
                "consumed": completed_consumed if evidence_boundary else 0,
                "data_cursor": data_cursor,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(),
            },
            trace_row=None,
        )
        if not fully_recurrent and active_layer == num_layers - 1:
            baseline_name = {
                "signals": "activation_fitted",
                "block": "hybrid_progressive",
                "global": "first_fully_recurrent",
            }[stage]
            stage_budget = {
                "signals": plan.stage_tokens_per_layer["signals"],
                "block": (
                    plan.stage_tokens_per_layer["signals"]
                    + plan.stage_tokens_per_layer["block"]
                ),
                "global": sum(plan.stage_tokens_per_layer.values()),
            }[stage] * num_layers
            _write_streamed_baseline(
                baseline_path,
                binding=baseline_binding,
                name=baseline_name,
                metrics=_score_streamed_candidate(
                    teacher,
                    executor,
                    mixer_store,
                    baseline_rows,
                    plan,
                    device,
                    dtype,
                ),
                token_budget=stage_budget,
            )
            pre_sweep_fingerprint = None
            if stage == "global":
                pre_sweep = mixer_store.snapshot(
                    run_dir / "streamed-sweeps" / "pre-sweep"
                )
                pre_sweep_fingerprint = mixer_store.snapshot_fingerprint(pre_sweep)
            committed_layer_state = _commit_streamed_generation(
                progress_dir=progress_dir,
                trace_path=trace_path,
                mixer_store=mixer_store,
                active_layer=active_layer,
                mixer=mixer,
                optimizer_snapshot=snapshot,
                phase=f"{stage}-complete",
                committed_layer_state=committed_layer_state,
                progress={
                    "schema_version": 2,
                    "binding": binding,
                    "next_visit": next_visit,
                    "active_layer": active_layer,
                    "consumed": 0,
                    "data_cursor": data_cursor,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "python_rng": random.getstate(),
                    "pre_sweep_fingerprint": pre_sweep_fingerprint,
                },
                trace_row=None,
            )
        if fully_recurrent and active_layer == 0:
            sweep_metrics = _score_streamed_candidate(
                teacher,
                executor,
                mixer_store,
                baseline_rows,
                plan,
                device,
                dtype,
            )
            validation_kl = sweep_metrics["mean_token_kl"]
            sweep_snapshot = mixer_store.snapshot(
                run_dir / "streamed-sweeps" / f"sweep-{sweep_index:02d}"
            )
            start_checkpoint = (
                str(run_dir / "streamed-sweeps" / "pre-sweep")
                if sweep_index == 0
                else str(
                    run_dir
                    / "streamed-sweeps"
                    / f"sweep-{sweep_index - 1:02d}"
                )
            )
            if sweep_index < len(controller.history):
                sweep = dict(controller.history[sweep_index])
                if sweep.get("end_checkpoint") != str(sweep_snapshot):
                    raise ContractError("persisted corrective sweep checkpoint differs")
            else:
                sweep = controller.complete(
                    start_checkpoint=start_checkpoint,
                    end_checkpoint=str(sweep_snapshot),
                    validation_kl=validation_kl,
                    token_budget=plan.stage_tokens_per_layer["global"] * num_layers,
                )
                sweep["start_checkpoint_fingerprint"] = mixer_store.snapshot_fingerprint(
                    Path(start_checkpoint)
                )
                sweep["end_checkpoint_fingerprint"] = mixer_store.snapshot_fingerprint(
                    sweep_snapshot
                )
                controller.history[-1] = sweep
            write_json(sweep_path, {"schema_version": 1, "sweeps": controller.history})
            _write_streamed_baseline(
                baseline_path,
                binding=baseline_binding,
                name=f"corrective_sweep_{sweep_index:02d}",
                metrics=sweep_metrics,
                token_budget=(
                    sum(plan.stage_tokens_per_layer.values())
                    + (sweep_index + 1) * plan.stage_tokens_per_layer["global"]
                )
                * num_layers,
            )
            stop = bool(sweep["stop"])
            committed_layer_state = _commit_streamed_generation(
                progress_dir=progress_dir,
                trace_path=trace_path,
                mixer_store=mixer_store,
                active_layer=active_layer,
                mixer=mixer,
                optimizer_snapshot=snapshot,
                phase=f"sweep-{sweep_index:02d}-complete",
                committed_layer_state=committed_layer_state,
                progress={
                    "schema_version": 2,
                    "binding": binding,
                    "next_visit": next_visit,
                    "active_layer": active_layer,
                    "consumed": 0,
                    "data_cursor": data_cursor,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "python_rng": random.getstate(),
                    "sweep_index": sweep_index,
                    "sweep_checkpoint_fingerprint": sweep.get(
                        "end_checkpoint_fingerprint"
                    ),
                },
                trace_row=None,
            )
            if stop:
                break
    if controller.history:
        selected = Path(str(controller.history[-1]["selected_checkpoint"]))
        mixer_store.restore_snapshot(selected)
    trained_dir = run_dir / "checkpoint-trained-bf16"
    _export_streamed_checkpoint(
        source_manifest,
        zero_step_dir,
        mixer_store,
        trained_dir,
    )
    _write_streamed_baseline(
        baseline_path,
        binding=baseline_binding,
        name="layerwise_distilled",
        metrics=_score_streamed_candidate(
            teacher,
            executor,
            mixer_store,
            baseline_rows,
            plan,
            device,
            dtype,
        ),
        token_budget=(
            sum(plan.stage_tokens_per_layer.values())
            + len(controller.history) * plan.stage_tokens_per_layer["global"]
        )
        * num_layers,
    )
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    trained_sha = checkpoint_sha256(trained_dir)
    baseline_payload["student_sha256"] = trained_sha
    write_json(baseline_path, baseline_payload)
    finalize_fitted_mapping(
        run_dir,
        student_sha256=trained_sha,
        trace_sha256=file_sha256(trace_path),
    )
    residency = _teacher_residency(
        teacher,
        device=device,
        max_teacher_cache_bytes=max_teacher_cache_bytes,
        max_cuda_reserved_bytes=plan.max_cuda_reserved_bytes,
    )
    write_json(residency_path, residency)
    return {
        "status": "distilled-bf16-streamed",
        "checkpoint": str(trained_dir),
        "checkpoint_sha256": checkpoint_sha256(trained_dir),
        "next_visit": next_visit,
        "data_cursor": data_cursor,
        "corrective_sweeps": controller.history,
        "stopped_by_sweep_gate": stop,
        "execution_mode": "streamed_layer_store",
        "resident_contract": {
            "source_decoder_layers": (
                num_layers if cache_teacher_layers else 1
            ),
            "teacher_layer_bytes": residency["teacher_layer_bytes"],
            "teacher_global_bytes": residency["teacher_global_bytes"],
            "cuda_peak_allocated_bytes": residency["cuda_peak_allocated_bytes"],
            "cuda_peak_reserved_bytes": residency["cuda_peak_reserved_bytes"],
            "target_mixers": 1,
            "optimizers": 1,
            "frozen_suffix": "reentrant-reload-on-backward",
        },
    }


def _teacher_residency(
    teacher,
    *,
    device,
    max_teacher_cache_bytes,
    max_cuda_reserved_bytes,
):
    teacher_layer_bytes = teacher.loader.cached_layer_bytes
    peak_reserved = torch.cuda.max_memory_reserved(device)
    accepted = (
        (max_teacher_cache_bytes is None or teacher_layer_bytes <= max_teacher_cache_bytes)
        and (
            max_cuda_reserved_bytes is None
            or peak_reserved <= max_cuda_reserved_bytes
        )
    )
    return {
        "schema_version": 1,
        "teacher_layer_cache": "cuda" if teacher.loader.cache_layers else "none",
        "cached_decoder_layers": teacher.loader.cached_layer_count,
        "teacher_layer_bytes": teacher_layer_bytes,
        "teacher_global_bytes": teacher.resident_global_bytes,
        "max_teacher_cache_bytes": max_teacher_cache_bytes,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_peak_reserved_bytes": peak_reserved,
        "max_cuda_reserved_bytes": max_cuda_reserved_bytes,
        "accepted": accepted,
    }


def _initial_trainable_names(run_dir: Path, num_layers: int) -> list[set[str]]:
    payload = json.loads((run_dir / "warm-start-plan.json").read_text(encoding="utf-8"))
    result = [set() for _ in range(num_layers)]
    for entry in payload.get("entries", []):
        index = layer_index(str(entry.get("target", "")))
        if index is None or (
            entry.get("provenance") not in {"fitted", "initialized"}
            and entry.get("is_semantically_lossless") is not False
        ):
            continue
        prefix = f"model.layers.{index}.attn."
        result[index].add(str(entry["target"]).removeprefix(prefix))
    if any(not names for names in result):
        raise ContractError("streamed signal stage has a layer with no fitted/initialized parameters")
    return result


def _encode_tokens(tokens, total_length, device):
    if len(tokens) != total_length:
        raise ContractError("streamed packed row length differs from frozen plan")
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(total_length, device=device).unsqueeze(0)
    return input_ids, attention_mask, position_ids


def _estimate_streamed_workload(
    *,
    source_weight_bytes,
    num_layers,
    plan,
    baseline_rows=0,
    target_mixer_bytes=None,
):
    def rounded_microbatches(tokens):
        raw = math.ceil(tokens / plan.supervised_tokens)
        return (
            math.ceil(raw / plan.accumulation_steps)
            * plan.accumulation_steps
        )

    batches = {
        stage: rounded_microbatches(tokens)
        for stage, tokens in plan.stage_tokens_per_layer.items()
    }
    progressive_batches = sum(batches.values()) * num_layers
    sweep_batches = (
        batches["global"] * num_layers * plan.corrective_max_sweeps
    )
    teacher_forwards = progressive_batches + sweep_batches
    student_forwards = teacher_forwards
    suffix_layer_reloads = 0
    for stage_batches in batches.values():
        suffix_layer_reloads += stage_batches * sum(range(num_layers))
    suffix_layer_reloads += (
        batches["global"]
        * sum(range(num_layers))
        * plan.corrective_max_sweeps
    )
    per_layer_bytes = math.ceil(source_weight_bytes / num_layers)
    if target_mixer_bytes is None:
        target_mixer_bytes = per_layer_bytes
    baseline_evaluations = 6 + 3 + plan.corrective_max_sweeps + 1
    baseline_forwards = baseline_evaluations * baseline_rows
    shadow_layer_loads = batches["signals"] * max(0, num_layers - 1)
    generation_writes = teacher_forwards * per_layer_bytes * 2
    target_mixer_reloads = (
        student_forwards * num_layers + suffix_layer_reloads
    )
    target_mixer_read_bytes = target_mixer_reloads * target_mixer_bytes
    cache_teacher_layers = bool(getattr(plan, "cache_teacher_layers", False))
    if cache_teacher_layers:
        teacher_source_reads = source_weight_bytes
        student_source_shell_reads = 0
        suffix_source_reloads = 0
        baseline_teacher_reads = 0
        baseline_student_reads = baseline_forwards * source_weight_bytes
        estimated = (
            teacher_source_reads
            + baseline_student_reads
            + shadow_layer_loads * per_layer_bytes
            + target_mixer_read_bytes
            + generation_writes
        )
        estimator = "resident-teacher-streamed-target-conservative-v2"
        assumptions = [
            "teacher globals and layers are read once and retained on the teacher device",
            "hybrid source-shell forwards and checkpointed suffix recomputation reuse cached teacher layers",
            "each warm-start baseline student evaluation conservatively counts one full source scan",
            "mixer plus optimizer generation writes bounded by two average source layers",
            "target mixer reads use the largest materialized target mixer tensor size",
            "does not count activation, filesystem cache, snapshot/export, or network bytes",
        ]
    else:
        teacher_source_reads = teacher_forwards * source_weight_bytes
        student_source_shell_reads = student_forwards * source_weight_bytes
        suffix_source_reloads = suffix_layer_reloads * per_layer_bytes
        baseline_teacher_reads = baseline_forwards * source_weight_bytes
        baseline_student_reads = baseline_forwards * source_weight_bytes
        estimated = (
            teacher_source_reads
            + student_source_shell_reads
            + suffix_source_reloads
            + baseline_teacher_reads
            + baseline_student_reads
            + shadow_layer_loads * per_layer_bytes
            + target_mixer_read_bytes
            + generation_writes
        )
        estimator = "streamed-layer-store-conservative-v2"
        assumptions = [
            "one full source checkpoint scan per teacher forward",
            "one full source-shell scan per student forward",
            "one average source layer reload per checkpointed suffix backward",
            "target mixer reads use the largest materialized target mixer tensor size",
            "mixer plus optimizer generation writes bounded by two average source layers",
            "does not count activation, filesystem cache, snapshot/export, or network bytes",
        ]
    return {
        "schema_version": 1,
        "estimator": estimator,
        "teacher_layer_cache": "cuda" if cache_teacher_layers else "none",
        "source_weight_bytes": source_weight_bytes,
        "target_mixer_bytes_upper_bound": target_mixer_bytes,
        "num_layers": num_layers,
        "microbatches_per_layer": batches,
        "teacher_full_forwards": teacher_forwards,
        "student_full_forwards": student_forwards,
        "suffix_layer_backward_reloads": suffix_layer_reloads,
        "baseline_full_forwards": baseline_forwards,
        "layer_zero_shadow_loads": shadow_layer_loads,
        "estimated_teacher_source_read_bytes": teacher_source_reads,
        "estimated_student_source_shell_read_bytes": student_source_shell_reads,
        "estimated_suffix_source_reload_bytes": suffix_source_reloads,
        "estimated_baseline_teacher_read_bytes": baseline_teacher_reads,
        "estimated_baseline_student_read_bytes": baseline_student_reads,
        "estimated_target_mixer_read_bytes": target_mixer_read_bytes,
        "estimated_generation_write_bytes": generation_writes,
        "estimated_weight_bytes_moved": estimated,
        "assumptions": assumptions,
    }


def _supervised_prediction_window(logits, input_ids, *, start, targets):
    """Align exactly ``targets`` next-token predictions after prefix burn-in."""
    if start < 1 or targets < 1:
        raise ContractError(
            "streamed next-token supervision requires burn_in_tokens >= 1 and targets >= 1"
        )
    stop = start + targets
    if stop > input_ids.shape[-1] or stop - 1 > logits.shape[-2]:
        raise ContractError("supervised prediction window exceeds the packed row")
    return logits[:, start - 1 : stop - 1], input_ids[:, start:stop]


def _head_partition_errors(student: Tensor, teacher: Tensor, *, heads: int) -> list[float]:
    if student.shape != teacher.shape or student.shape[-1] % heads:
        raise ContractError(
            "headwise migration error requires aligned hidden outputs divisible by target heads"
        )
    head_dim = student.shape[-1] // heads
    student_heads = student.float().reshape(*student.shape[:-1], heads, head_dim)
    teacher_heads = teacher.float().reshape(*teacher.shape[:-1], heads, head_dim)
    reduce_dims = tuple(range(student_heads.ndim - 2)) + (student_heads.ndim - 1,)
    numerator = (student_heads - teacher_heads).square().sum(dim=reduce_dims)
    denominator = teacher_heads.square().sum(dim=reduce_dims).clamp_min(1e-30)
    return [float(value) for value in (numerator / denominator).detach().cpu()]


def _append_trace(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _atomic_torch_save(path: Path, payload: object) -> str:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    content = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(content).hexdigest()


def _save_optimizer_snapshot(path: Path, snapshot: ActiveLayerOptimizerSnapshot) -> None:
    sha = _atomic_torch_save(path, snapshot)
    write_json(
        path.with_suffix(path.suffix + ".json"),
        {"schema_version": 1, "path": path.name, "sha256": sha},
    )


def _load_optimizer_snapshot(path: Path) -> ActiveLayerOptimizerSnapshot | None:
    if not path.is_file():
        return None
    metadata_path = path.with_suffix(path.suffix + ".json")
    if not metadata_path.is_file():
        raise ContractError(f"streamed optimizer snapshot has no hash binding: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("path") != path.name or metadata.get("sha256") != file_sha256(path):
        raise ContractError(f"streamed optimizer snapshot SHA-256 mismatch: {path}")
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(snapshot, ActiveLayerOptimizerSnapshot):
        raise ContractError(f"invalid streamed optimizer snapshot: {path}")
    return snapshot


def _commit_streamed_generation(
    *,
    progress_dir: Path,
    trace_path: Path,
    mixer_store: RWKV7MixerLayerStore,
    active_layer: int,
    mixer,
    optimizer_snapshot: ActiveLayerOptimizerSnapshot,
    phase: str,
    committed_layer_state: dict[str, dict[str, object]],
    progress: dict[str, object],
    trace_row: dict[str, object] | None,
) -> dict[str, dict[str, object]]:
    cursor = {
        "visit_index": int(progress["next_visit"]),
        "active_layer": active_layer,
        "consumed": int(progress["consumed"]),
        "data_cursor": int(progress["data_cursor"]),
        "phase": phase,
    }
    generation_name = (
        f"v{cursor['visit_index']:04d}-l{active_layer:03d}-"
        f"d{cursor['data_cursor']:012d}-c{cursor['consumed']:012d}-{phase}"
    )
    generations = progress_dir / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    generation = generations / generation_name
    temporary = generations / f".{generation_name}.tmp"
    pointer_path = progress_dir / "latest.json"
    if generation.is_dir() and pointer_path.is_file():
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        manifest_path = generation / "manifest.json"
        if (
            pointer.get("generation") == generation_name
            and manifest_path.is_file()
            and pointer.get("manifest_sha256") == file_sha256(manifest_path)
        ):
            existing_progress = torch.load(
                generation / "progress.pt",
                map_location="cpu",
                weights_only=False,
            )
            if existing_progress.get("commit_cursor") != cursor:
                raise ContractError(
                    "latest streamed generation name matches but cursor differs"
                )
            existing_state = existing_progress.get("committed_layer_state")
            if not isinstance(existing_state, dict):
                raise ContractError("latest streamed generation lacks layer-state map")
            return {
                str(layer): dict(record)
                for layer, record in existing_state.items()
            }
    if temporary.exists():
        shutil.rmtree(temporary)
    if generation.exists():
        shutil.rmtree(generation)
    temporary.mkdir()
    mixer_metadata = mixer_store.save_generation(
        temporary / "mixer",
        active_layer,
        mixer,
        cursor=cursor,
    )
    optimizer_path = temporary / "optimizer.pt"
    _save_optimizer_snapshot(optimizer_path, optimizer_snapshot)
    next_committed_layer_state = {
        str(layer): dict(record)
        for layer, record in committed_layer_state.items()
    }
    next_committed_layer_state[str(active_layer)] = {
        "generation": generation_name,
        "mixer_sha256": mixer_metadata["sha256"],
        "optimizer_sha256": file_sha256(optimizer_path),
        "cursor": cursor,
    }
    progress = {
        **progress,
        "commit_cursor": cursor,
        "committed_layer_state": next_committed_layer_state,
    }
    progress_sha = _atomic_torch_save(temporary / "progress.pt", progress)
    if trace_row is not None:
        (temporary / "trace.json").write_text(
            json.dumps(trace_row, sort_keys=True) + "\n", encoding="utf-8"
        )
    manifest = {
        "schema_version": 1,
        "generation": generation_name,
        "cursor": cursor,
        "mixer_sha256": mixer_metadata["sha256"],
        "optimizer_sha256": file_sha256(optimizer_path),
        "progress_sha256": progress_sha,
        "trace_sha256": (
            file_sha256(temporary / "trace.json") if trace_row is not None else None
        ),
    }
    write_json(temporary / "manifest.json", manifest)
    _fsync_tree(temporary)
    generation.parent.mkdir(parents=True, exist_ok=True)
    temporary.rename(generation)
    generation_directory = os.open(generation.parent, os.O_RDONLY)
    try:
        os.fsync(generation_directory)
    finally:
        os.close(generation_directory)

    mixer_store.restore_generation(
        generation / "mixer", active_layer, expected_cursor=cursor
    )
    published_optimizer = progress_dir / f"optimizer-layer-{active_layer:03d}.pt"
    staged = published_optimizer.with_suffix(published_optimizer.suffix + ".generation")
    shutil.copy2(generation / "optimizer.pt", staged)
    staged.replace(published_optimizer)
    write_json(
        published_optimizer.with_suffix(published_optimizer.suffix + ".json"),
        {
            "schema_version": 1,
            "path": published_optimizer.name,
            "sha256": file_sha256(published_optimizer),
        },
    )
    if trace_row is not None:
        _append_trace(trace_path, trace_row)
    trace_size = trace_path.stat().st_size if trace_path.exists() else 0
    trace_sha256 = file_sha256(trace_path) if trace_path.exists() else None
    write_json(
        progress_dir / "latest.json",
        {
            "schema_version": 2,
            "generation": generation_name,
            "manifest_sha256": file_sha256(generation / "manifest.json"),
            "trace_size": trace_size,
            "trace_sha256": trace_sha256,
        },
    )
    _reclaim_unreferenced_generations(
        generations,
        {
            str(record["generation"])
            for record in next_committed_layer_state.values()
        },
    )
    return next_committed_layer_state


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for directory_path in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        descriptor = os.open(directory_path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reclaim_unreferenced_generations(
    generations: Path, referenced: set[str]
) -> None:
    for candidate in generations.iterdir():
        if candidate.name in referenced:
            continue
        if candidate.is_dir():
            shutil.rmtree(candidate)
        else:
            candidate.unlink(missing_ok=True)
    descriptor = os.open(generations, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_committed_generation(
    path: Path,
    *,
    trace_path: Path,
    mixer_store: RWKV7MixerLayerStore,
    progress_dir: Path,
) -> tuple[dict[str, object], ActiveLayerOptimizerSnapshot]:
    root = path if path.is_dir() else path.parent
    pointer_path = root / "latest.json" if path.is_dir() else path
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("schema_version") != 2:
        raise ContractError("legacy non-transactional streamed progress cannot be resumed")
    generation = root / "generations" / str(pointer["generation"])
    manifest_path = generation / "manifest.json"
    if (
        not manifest_path.is_file()
        or file_sha256(manifest_path) != pointer.get("manifest_sha256")
    ):
        raise ContractError("streamed generation manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cursor = manifest.get("cursor")
    if not isinstance(cursor, dict):
        raise ContractError("streamed generation has no commit cursor")
    progress_path = generation / "progress.pt"
    optimizer_path = generation / "optimizer.pt"
    if (
        file_sha256(progress_path) != manifest.get("progress_sha256")
        or file_sha256(optimizer_path) != manifest.get("optimizer_sha256")
    ):
        raise ContractError("streamed generation progress/optimizer SHA-256 mismatch")
    expected_trace_size = int(pointer.get("trace_size", 0))
    actual_trace_size = trace_path.stat().st_size if trace_path.exists() else 0
    if actual_trace_size < expected_trace_size:
        raise ContractError("committed streamed trace is shorter than latest generation")
    if actual_trace_size > expected_trace_size:
        with trace_path.open("r+b") as handle:
            handle.truncate(expected_trace_size)
            handle.flush()
            os.fsync(handle.fileno())
    actual_trace_sha256 = file_sha256(trace_path) if trace_path.exists() else None
    if actual_trace_sha256 != pointer.get("trace_sha256"):
        raise ContractError("committed streamed trace SHA-256 mismatch")
    progress = torch.load(progress_path, map_location="cpu", weights_only=False)
    if progress.get("commit_cursor") != cursor:
        raise ContractError("streamed generation progress cursor mismatch")
    committed = progress.get("committed_layer_state")
    if not isinstance(committed, dict):
        raise ContractError("streamed generation lacks committed per-layer state")
    for layer in range(mixer_store.factory.config.num_hidden_layers):
        if str(layer) in committed:
            continue
        mixer_store.discard_overlay(layer)
        optimizer_file = progress_dir / f"optimizer-layer-{layer:03d}.pt"
        optimizer_file.unlink(missing_ok=True)
        optimizer_file.with_suffix(optimizer_file.suffix + ".json").unlink(
            missing_ok=True
        )
    for layer_text, record in committed.items():
        if not isinstance(record, dict):
            raise ContractError("invalid committed streamed layer-state record")
        layer = int(layer_text)
        layer_generation = root / "generations" / str(record.get("generation"))
        layer_optimizer = layer_generation / "optimizer.pt"
        layer_cursor = record.get("cursor")
        if (
            not isinstance(layer_cursor, dict)
            or file_sha256(layer_optimizer) != record.get("optimizer_sha256")
        ):
            raise ContractError(f"committed optimizer generation mismatch at layer {layer}")
        restored_metadata = mixer_store.restore_generation(
            layer_generation / "mixer",
            layer,
            expected_cursor=layer_cursor,
        )
        if restored_metadata.get("sha256") != record.get("mixer_sha256"):
            raise ContractError(f"committed mixer generation mismatch at layer {layer}")
        published = progress_dir / f"optimizer-layer-{layer:03d}.pt"
        staged = published.with_suffix(published.suffix + ".resume")
        shutil.copy2(layer_optimizer, staged)
        staged.replace(published)
        write_json(
            published.with_suffix(published.suffix + ".json"),
            {
                "schema_version": 1,
                "path": published.name,
                "sha256": file_sha256(published),
            },
        )
    _reclaim_unreferenced_generations(
        root / "generations",
        {str(record["generation"]) for record in committed.values()},
    )
    active_layer = int(cursor["active_layer"])
    published_optimizer = progress_dir / f"optimizer-layer-{active_layer:03d}.pt"
    optimizer = _load_optimizer_snapshot(published_optimizer)
    if optimizer is None:
        raise ContractError("committed streamed optimizer snapshot is absent")
    return progress, optimizer


def _read_streamed_baselines(
    path: Path,
    *,
    binding: dict[str, object],
) -> set[str]:
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("binding") != binding:
        raise ContractError("existing streamed migration baselines use a different frozen protocol")
    baselines = payload.get("baselines")
    if not isinstance(baselines, dict):
        raise ContractError("streamed migration baseline artifact has no baselines object")
    return set(str(name) for name in baselines)


def _write_streamed_baseline(
    path: Path,
    *,
    binding: dict[str, object],
    name: str,
    metrics: dict[str, float],
    token_budget: int,
) -> None:
    payload = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {"schema_version": 1, "binding": binding, "baselines": {}}
    )
    if payload.get("binding") != binding or not isinstance(payload.get("baselines"), dict):
        raise ContractError("streamed migration baseline binding changed within one run")
    payload["baselines"][name] = {**metrics, "token_budget": int(token_budget)}
    write_json(path, payload)


def _score_streamed_candidate(
    teacher,
    executor,
    mixer_provider,
    rows,
    plan,
    device,
    dtype,
) -> dict[str, float]:
    teacher_ce = []
    student_ce = []
    kl_rows = []
    block_mse_rows = []
    cosine_rows = []
    num_layers = teacher.loader.tensor_store.num_layers
    active_index = num_layers - 1
    with torch.no_grad():
        for tokens in rows:
            input_ids, attention_mask, position_ids = _encode_tokens(
                tokens, plan.burn_in_tokens + plan.supervised_tokens, device
            )
            reference = teacher.forward(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                capture_layer_index=active_index,
            )
            active = mixer_provider.load_mixer(
                active_index, device=device, dtype=dtype
            )
            candidate = executor.forward(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                active_layer_index=active_index,
                active_mixer=active,
                converted_layer_indices=set(range(active_index)),
                frozen_mixer_provider=lambda index: mixer_provider.load_mixer(
                    index, device=device, dtype=dtype
                ),
            )
            start = plan.burn_in_tokens
            reference_logits, labels = _supervised_prediction_window(
                reference.logits.to(device),
                input_ids,
                start=start,
                targets=plan.supervised_tokens,
            )
            candidate_logits, candidate_labels = _supervised_prediction_window(
                candidate.logits,
                input_ids,
                start=start,
                targets=plan.supervised_tokens,
            )
            if not torch.equal(labels, candidate_labels):
                raise ContractError("candidate/reference evaluation windows differ")
            teacher_ce.append(
                torch.nn.functional.cross_entropy(
                    reference_logits.reshape(-1, reference_logits.shape[-1]),
                    labels.reshape(-1),
                )
            )
            student_ce.append(
                torch.nn.functional.cross_entropy(
                    candidate_logits.reshape(-1, candidate_logits.shape[-1]),
                    labels.reshape(-1),
                )
            )
            kl_rows.append(
                chunked_token_kl(
                    candidate_logits,
                    reference_logits,
                    vocab_chunk_size=DEFAULT_VOCAB_CHUNK_SIZE,
                )
            )
            candidate_block = candidate.active_block_output[:, start:]
            reference_block = reference.active_block_output[:, start:].to(device)
            block_mse_rows.append(normalized_mse(candidate_block, reference_block))
            cosine_rows.append(
                torch.nn.functional.cosine_similarity(
                    candidate_block.flatten(0, -2),
                    reference_block.flatten(0, -2),
                    dim=-1,
                ).mean()
            )
    return {
        "teacher_ppl": float(torch.exp(torch.stack(teacher_ce).mean())),
        "student_ppl": float(torch.exp(torch.stack(student_ce).mean())),
        "mean_token_kl": float(torch.stack(kl_rows).mean()),
        "final_block_normalized_mse": float(torch.stack(block_mse_rows).mean()),
        "final_block_cosine": float(torch.stack(cosine_rows).mean()),
    }


def _export_streamed_checkpoint(source, zero_step_dir, mixer_store, destination):
    binding = {
        "schema_version": 1,
        "source_config_sha256": source.file_hashes["config.json"],
        "source_shard_sha256": {
            shard.name: source.file_hashes[shard.name] for shard in source.shards
        },
        "target_config_sha256": file_sha256(zero_step_dir / "config.json"),
        "selected_mixer_fingerprint": mixer_store.fingerprint(),
    }
    binding_path = destination / "any2rwkv-export-binding.json"
    if destination.is_dir():
        _validate_complete_export(destination)
        if not binding_path.is_file() or json.loads(
            binding_path.read_text(encoding="utf-8")
        ) != binding:
            raise ContractError(
                "existing streamed HF export does not match the selected mixer snapshot"
            )
        return
    temporary = destination.with_name(destination.name + ".partial")
    target_config = json.loads((zero_step_dir / "config.json").read_text(encoding="utf-8"))
    specs = tuple(
        spec
        for index in range(source.contract.num_hidden_layers)
        for spec in rwkv7_mixer_specs(index, hidden_size=source.contract.hidden_size)
    )
    cache: dict[str, object] = {"layer": None, "state": {}}

    def provide(spec):
        index = layer_index(spec.name)
        if index is None:
            raise ContractError(f"streamed target spec has no layer: {spec.name}")
        if cache["layer"] != index:
            mixer = mixer_store.load_mixer(index, device="cpu", dtype=torch.bfloat16)
            cache["layer"] = index
            cache["state"] = {
                f"model.layers.{index}.attn.{name}": tensor.detach().cpu()
                for name, tensor in mixer.state_dict().items()
            }
        return cache["state"][spec.name]

    export_hf_checkpoint(
        source,
        temporary,
        target_config=target_config,
        target_specs=specs,
        target_tensor_provider=provide,
        resume_partial=True,
        external_resume_binding=binding,
    )
    _validate_complete_export(temporary)
    write_json(temporary / "any2rwkv-export-binding.json", binding)
    temporary.rename(destination)


def _validate_complete_export(path: Path) -> None:
    config_path = path / "config.json"
    index_path = path / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise ContractError(f"streamed HF export is incomplete: {path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ContractError(f"streamed HF export has no nonempty weight_map: {path}")
    missing = sorted(
        shard for shard in set(weight_map.values()) if not (path / str(shard)).is_file()
    )
    if missing:
        raise ContractError(f"streamed HF export is missing shards: {missing}")
    if (path / ".export-progress.json").exists():
        raise ContractError(f"streamed HF export still has an active progress journal: {path}")
