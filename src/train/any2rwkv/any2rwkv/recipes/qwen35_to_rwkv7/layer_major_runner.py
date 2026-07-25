from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Iterable

import torch
from safetensors import safe_open

from ...artifacts import file_sha256, write_json
from ...core import (
    LayerInputBatch,
    LayerInputCacheEstimate,
    LayerInputCacheReader,
    estimate_layer_input_cache_bytes,
    prepare_distributed_layer_input_cache,
    publish_distributed_layer_input_cache,
    require_layer_input_cache_capacity,
    write_distributed_layer_input_cache_partition,
    write_layer_input_cache,
)
from ...distill import normalized_mse
from ...distributed import DistributedContext
from ...errors import ContractError
from ...layer_schedule import (
    LayerConvergenceState,
    epoch_permutation,
    update_convergence,
)
from ...migration import (
    TraceNormalEquations,
    native_decay_fit_targets,
    solve_teacher_trace_normal_equations,
    teacher_trace_normal_equations,
)
from ...mixer_store import RWKV7MixerLayerStore
from ...streamed_teacher import StreamedQwen35HybridExecutor, StreamedQwen35Teacher
from ...streaming_training import ActiveLayerOptimizer, ActiveLayerOptimizerSnapshot
from ...zero_step_probe import materialize_native_projection
from .gqa_zero_step import (
    GQANativeFitConfig,
    GQANativeFitTrace,
    estimate_gqa_native_streamed_peak_bytes,
    fit_gqa_native_zero_step,
    validate_gqa_native_fit_trace,
)


def performance_profile_cases(
    source_config: dict[str, object],
) -> tuple[dict[str, object], ...]:
    text_config = source_config.get("text_config", source_config)
    if not isinstance(text_config, dict):
        raise ContractError("source config text_config must be an object")
    layer_types = text_config.get("layer_types")
    layer_count = int(text_config.get("num_hidden_layers", 0))
    if (
        not isinstance(layer_types, list)
        or layer_count <= 0
        or len(layer_types) != layer_count
        or not all(isinstance(value, str) and value for value in layer_types)
    ):
        raise ContractError(
            "performance profile requires one mixer kind for every source layer"
        )
    cases: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for layer_index, mixer_kind in enumerate(layer_types):
        input_boundary = (
            "embedding-output" if layer_index == 0 else "recurrent-prefix"
        )
        identity = (input_boundary, mixer_kind)
        if identity in seen:
            continue
        seen.add(identity)
        matching_layers = [
            index
            for index, value in enumerate(layer_types)
            if value == mixer_kind
            and ("embedding-output" if index == 0 else "recurrent-prefix")
            == input_boundary
        ]
        cases.append(
            {
                "profile_case_id": f"{input_boundary}:{mixer_kind}",
                "source_mixer_kind": mixer_kind,
                "input_boundary": input_boundary,
                "representative_layer": layer_index,
                "layer_count": len(matching_layers),
                "transition_count": sum(
                    index + 1 < layer_count for index in matching_layers
                ),
            }
        )
    return tuple(cases)


def prepare_performance_profile_caches(
    *,
    source_manifest,
    run_dir: Path,
    zero_step_dir: Path,
    token_rows: tuple[tuple[int, ...], ...],
    validation_rows: tuple[tuple[int, ...], ...],
    plan,
    initial_trainable: list[set[str]],
    training_config: Path,
    dataset_manifest: Path,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, object]:
    distributed = DistributedContext.initialize()
    planned_world_size = int(getattr(plan, "distributed_world_size", 1))
    if planned_world_size != distributed.world_size:
        raise ContractError(
            "profile-cache plan world size differs from torchrun: "
            f"plan={planned_world_size} runtime={distributed.world_size}"
        )
    device = distributed.device if device is None else torch.device(device)
    if distributed.world_size > 1 and device.type != "cuda":
        raise ContractError("distributed profile-cache preparation requires CUDA")
    dtype = (
        (torch.bfloat16 if device.type == "cuda" else torch.float32)
        if dtype is None
        else dtype
    )
    cases = performance_profile_cases(source_manifest.config)
    representative_layers = tuple(
        int(case["representative_layer"]) for case in cases
    )
    max_layer = max(representative_layers)
    if len(initial_trainable) != source_manifest.contract.num_hidden_layers:
        raise ContractError(
            "profile-cache trainable sets must cover every source layer"
        )
    all_rows = (*token_rows, *validation_rows)
    if not all_rows or any(len(row) != len(all_rows[0]) for row in all_rows):
        raise ContractError(
            "profile-cache preparation requires nonempty fixed-length packed rows"
        )
    cache_root = run_dir / "performance-profile-cache"
    base_binding = _run_binding(
        source_manifest=source_manifest,
        run_dir=run_dir,
        zero_step_dir=zero_step_dir,
        initial_trainable=initial_trainable,
        training_config=training_config,
        dataset_manifest=dataset_manifest,
    )
    row_count = len(token_rows) + len(validation_rows)
    sequence_length = len(all_rows[0])
    base_bytes = (
        row_count
        * sequence_length
        * source_manifest.contract.hidden_size
        * 2
    )
    retained_cache_bytes = base_bytes + max_layer * base_bytes * 2
    required_free_bytes = math.ceil(retained_cache_bytes * 1.10)
    max_cache_bytes = getattr(plan, "max_layer_input_cache_bytes", None)
    if max_cache_bytes is not None and required_free_bytes > int(max_cache_bytes):
        raise ContractError(
            "performance profile caches exceed frozen run limit: "
            f"required={required_free_bytes} limit={max_cache_bytes}"
        )
    if distributed.is_primary:
        capacity = require_layer_input_cache_capacity(
            cache_root,
            LayerInputCacheEstimate(
                current_cache_bytes=retained_cache_bytes,
                next_cache_bytes=0,
                required_free_bytes=required_free_bytes,
            ),
        )
        write_json(run_dir / "performance-profile-cache-capacity.json", capacity)
    distributed.barrier()

    teacher = StreamedQwen35Teacher(
        source_manifest,
        device=device,
        dtype=dtype,
        cache_layers=False,
        load_output_head=False,
    )
    executor = StreamedQwen35HybridExecutor(teacher)
    store = RWKV7MixerLayerStore(zero_step_dir, run_dir / "mixer-overlays")
    prefix_fingerprint = _sha256_json(
        {
            "recipe": "qwen35_to_rwkv7",
            "source_files": source_manifest.file_hashes,
            "boundary": "embedding-output",
        }
    )
    _ensure_embedding_caches(
        teacher=teacher,
        cache_root=cache_root,
        token_rows=token_rows,
        validation_rows=validation_rows,
        shard_rows=plan.cache_shard_rows,
        hidden_size=source_manifest.contract.hidden_size,
        base_binding=base_binding,
        prefix_fingerprint=prefix_fingerprint,
        distributed=distributed,
    )
    for layer_index in range(max_layer):
        train_reader = _open_cache(
            cache_root,
            layer_index,
            "distill_train",
            base_binding,
            prefix_fingerprint,
            max_cached_bytes=int(plan.max_cached_layer_input_bytes_per_rank),
        )
        validation_reader = _open_cache(
            cache_root,
            layer_index,
            "validation",
            base_binding,
            prefix_fingerprint,
            max_cached_bytes=int(plan.max_cached_layer_input_bytes_per_rank),
        )
        loaded_layer = teacher.loader.load_layer(
            layer_index, device=device, dtype=dtype
        )
        mixer = store.load_mixer(layer_index, device=device, dtype=dtype)
        next_prefix_fingerprint = _sha256_json(
            {
                "previous_prefix_fingerprint": prefix_fingerprint,
                "zero_step_checkpoint_sha256": (
                    base_binding["zero_step_checkpoint_sha256"]
                ),
                "layer_index": layer_index,
                "mixer_state_sha256": _sha256_json(
                    {
                        name: _tensor_sha256(tensor)
                        for name, tensor in sorted(mixer.state_dict().items())
                    }
                ),
            }
        )
        _ensure_next_layer_caches(
            executor=executor,
            cache_root=cache_root,
            current_layer=layer_index,
            train_reader=train_reader,
            validation_reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            shard_rows=plan.cache_shard_rows,
            hidden_size=source_manifest.contract.hidden_size,
            base_binding=base_binding,
            next_prefix_fingerprint=next_prefix_fingerprint,
            distributed=distributed,
        )
        del train_reader, validation_reader, loaded_layer, mixer
        gc.collect()
        prefix_fingerprint = next_prefix_fingerprint

    result = {
        "schema_version": 1,
        "status": "prepared",
        "world_size": distributed.world_size,
        "training_config_sha256": file_sha256(training_config),
        "dataset_manifest_sha256": file_sha256(dataset_manifest),
        "cases": [
            {
                **case,
                "train_cache": str(
                    _split_cache_dir(
                        cache_root,
                        int(case["representative_layer"]),
                        "distill_train",
                    )
                ),
                "validation_cache": str(
                    _split_cache_dir(
                        cache_root,
                        int(case["representative_layer"]),
                        "validation",
                    )
                ),
            }
            for case in cases
        ],
    }
    if distributed.is_primary:
        write_json(run_dir / "performance-profile-caches.json", result)
    distributed.barrier()
    return result


def run_suffix_free_layer_major(
    *,
    source_manifest,
    run_dir: Path,
    zero_step_dir: Path,
    token_rows: tuple[tuple[int, ...], ...],
    validation_rows: tuple[tuple[int, ...], ...],
    plan,
    initial_trainable: list[set[str]],
    training_config: Path,
    dataset_manifest: Path,
    resume: Path | None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    progress_callback: Callable[[str, Path], None] | None = None,
) -> dict[str, object]:
    if plan.cache_teacher_layers:
        raise ContractError(
            "layer-major training keeps only the current source layer resident"
        )
    distributed = DistributedContext.initialize()
    planned_world_size = int(getattr(plan, "distributed_world_size", 1))
    if planned_world_size != distributed.world_size:
        raise ContractError(
            "distillation plan distributed_world_size differs from torchrun: "
            f"plan={planned_world_size} runtime={distributed.world_size}"
        )
    if device is None:
        device = distributed.device
    else:
        device = torch.device(device)
        if distributed.world_size > 1 and device.type != "cuda":
            raise ContractError("distributed layer-major training requires CUDA")
    dtype = (
        (torch.bfloat16 if device.type == "cuda" else torch.float32)
        if dtype is None
        else dtype
    )
    num_layers = source_manifest.contract.num_hidden_layers
    if len(initial_trainable) != num_layers:
        raise ContractError("layer-major trainable sets must cover every source layer")
    source_text_config = source_manifest.config.get(
        "text_config", source_manifest.config
    )
    source_layer_types = tuple(source_text_config.get("layer_types", ()))
    learning_rate_profiles = dict(
        getattr(plan, "local_learning_rate_by_mixer_kind", ())
    )
    unknown_learning_rate_profiles = set(learning_rate_profiles) - set(
        source_layer_types
    )
    if unknown_learning_rate_profiles:
        raise ContractError(
            "local learning-rate profile names are absent from source layer types: "
            f"{sorted(unknown_learning_rate_profiles)}"
        )
    exploratory_layer_limit = getattr(plan, "exploratory_layer_limit", None)
    if exploratory_layer_limit is not None:
        exploratory_layer_limit = int(exploratory_layer_limit)
        if not 0 < exploratory_layer_limit < num_layers:
            raise ContractError(
                "exploratory_layer_limit must stop before the final source layer"
            )
    layer_stop = exploratory_layer_limit or num_layers
    cache_root = run_dir / "layer-input-cache"
    base_binding = _run_binding(
        source_manifest=source_manifest,
        run_dir=run_dir,
        zero_step_dir=zero_step_dir,
        initial_trainable=initial_trainable,
        training_config=training_config,
        dataset_manifest=dataset_manifest,
    )
    sequence_length = len(token_rows[0])
    max_cuda_reserved_bytes = getattr(plan, "max_cuda_reserved_bytes", None)
    max_cached_layer_input_bytes_per_rank = int(
        getattr(plan, "max_cached_layer_input_bytes_per_rank", None) or 1024**3
    )
    layer_fixed_epochs = getattr(plan, "layer_fixed_epochs", None)
    if any(len(row) != sequence_length for row in (*token_rows, *validation_rows)):
        raise ContractError(
            "rolling layer-input cache requires fixed-length packed rows"
        )
    estimate = estimate_layer_input_cache_bytes(
        row_count=len(token_rows) + len(validation_rows),
        sequence_length=sequence_length,
        hidden_size=source_manifest.contract.hidden_size,
        dtype_bytes=2,
        current_has_shared_states=True,
        next_has_shared_states=True,
    )
    if (
        plan.max_layer_input_cache_bytes is not None
        and estimate.required_free_bytes > plan.max_layer_input_cache_bytes
    ):
        raise ContractError(
            "layer-input cache exceeds frozen run limit: "
            f"required={estimate.required_free_bytes} "
            f"limit={plan.max_layer_input_cache_bytes}"
        )
    if distributed.is_primary:
        capacity = require_layer_input_cache_capacity(cache_root, estimate)
        write_json(run_dir / "layer-input-cache-capacity.json", capacity)
    distributed.barrier()

    teacher = StreamedQwen35Teacher(
        source_manifest,
        device=device,
        dtype=dtype,
        cache_layers=False,
        load_output_head=False,
    )
    executor = StreamedQwen35HybridExecutor(teacher)
    store = RWKV7MixerLayerStore(zero_step_dir, run_dir / "mixer-overlays")
    initial_prefix_fingerprint = _sha256_json(
        {
            "recipe": "qwen35_to_rwkv7",
            "source_files": source_manifest.file_hashes,
            "boundary": "embedding-output",
        }
    )
    if resume is not None and not resume.is_file():
        raise ContractError(f"explicit resume progress is missing: {resume}")
    progress_path = resume or (run_dir / "layer-major-progress.json")
    progress = _read_progress(progress_path) if progress_path.is_file() else None
    resume_best_generation = None
    if progress is not None:
        _validate_progress_binding(progress, base_binding)
        resume_best_generation = _resolve_best_generation(run_dir, progress)
    resume_generation = (
        _resolve_progress_generation(run_dir, progress)
        if progress is not None
        else None
    )
    _prune_generations_distributed(
        run_dir,
        keep=(resume_generation, resume_best_generation),
        distributed=distributed,
    )
    start_layer = int(progress["active_layer"]) if progress else 0
    if start_layer > layer_stop:
        raise ContractError("resume active layer exceeds the configured layer stop")
    histories = list(progress.get("history", [])) if progress else []
    completed_optimizer_steps = (
        int(progress.get("completed_optimizer_steps", 0)) if progress else 0
    )
    prefix_fingerprint = (
        str(progress["prefix_fingerprint"]) if progress else initial_prefix_fingerprint
    )
    if start_layer == 0:
        cache_transition_started = time.perf_counter()
        _ensure_embedding_caches(
            teacher=teacher,
            cache_root=cache_root,
            token_rows=token_rows,
            validation_rows=validation_rows,
            shard_rows=plan.cache_shard_rows,
            hidden_size=source_manifest.contract.hidden_size,
            base_binding=base_binding,
            prefix_fingerprint=initial_prefix_fingerprint,
            distributed=distributed,
        )
        _write_cache_transition_telemetry(
            run_dir=run_dir,
            source_layer=-1,
            target_layer=0,
            row_count=len(token_rows) + len(validation_rows),
            sequence_length=sequence_length,
            local_wall_seconds=time.perf_counter() - cache_transition_started,
            distributed=distributed,
        )
    if start_layer == layer_stop and exploratory_layer_limit is not None:
        _rank0_filesystem_step(
            distributed,
            "restore exploratory convergence artifact",
            lambda: _write_layer_convergence(
                run_dir, histories=histories, world_size=distributed.world_size
            ),
        )
        return _exploratory_completion(
            completed_layers=layer_stop,
            total_layers=num_layers,
            optimizer_steps=completed_optimizer_steps,
            histories=histories,
            run_dir=run_dir,
        )
    if start_layer == num_layers:
        checkpoint = run_dir / "checkpoint-layerwise-local"
        if not checkpoint.is_dir():
            _rank0_filesystem_step(
                distributed,
                "materialize layerwise checkpoint",
                lambda: store.materialize_checkpoint(
                    checkpoint, fitted_evidence_root=run_dir
                ),
            )
        completion = _local_completion(num_layers, completed_optimizer_steps, histories)
        completion["checkpoint"] = str(checkpoint)
        return completion

    for layer_index in range(start_layer, layer_stop):
        train_reader = _open_cache(
            cache_root,
            layer_index,
            "distill_train",
            base_binding,
            prefix_fingerprint,
            max_cached_bytes=max_cached_layer_input_bytes_per_rank,
        )
        validation_reader = _open_cache(
            cache_root,
            layer_index,
            "validation",
            base_binding,
            prefix_fingerprint,
            max_cached_bytes=max_cached_layer_input_bytes_per_rank,
        )
        full_validation_reader = validation_reader
        train_cache_manifest_sha256 = file_sha256(
            train_reader.cache_dir / "manifest.json"
        )
        validation_cache_manifest_sha256 = file_sha256(
            validation_reader.cache_dir / "manifest.json"
        )
        if (
            progress
            and layer_index == start_layer
            and progress.get("phase") in {"train", "epoch-complete"}
        ):
            progress_cursor = progress.get("generation_cursor", {})
            if (
                progress_cursor.get("train_cache_manifest_sha256")
                != train_cache_manifest_sha256
                or progress_cursor.get("validation_cache_manifest_sha256")
                != validation_cache_manifest_sha256
            ):
                raise ContractError(
                    "resume cache manifests differ from the durable generation cursor"
                )
        if distributed.is_primary:
            _cleanup_stale_layer_caches(cache_root, keep_layer=layer_index)
        distributed.barrier()
        loaded_layer = teacher.loader.load_layer(
            layer_index, device=device, dtype=dtype
        )
        base_mixer = store.load_base_mixer(layer_index, device=device, dtype=dtype)
        frozen_parameter_sha256 = _frozen_parameter_sha256(
            base_mixer, initial_trainable[layer_index]
        )
        del base_mixer
        gc.collect()
        mixer = store.load_mixer(layer_index, device=device, dtype=dtype)
        global_micro_batch_size = plan.micro_batch_size * distributed.world_size
        micro_batches_per_epoch = (
            train_reader.row_count + global_micro_batch_size - 1
        ) // global_micro_batch_size
        optimizer_steps_per_epoch = (
            micro_batches_per_epoch + plan.accumulation_steps - 1
        ) // plan.accumulation_steps
        scheduled_epochs = (
            getattr(plan, "layer_learning_rate_schedule_epochs", None)
            or layer_fixed_epochs
            or plan.layer_max_epochs
        )
        total_optimizer_steps = optimizer_steps_per_epoch * scheduled_epochs
        configured_warmup_steps = getattr(plan, "learning_rate_warmup_steps", None)
        warmup_steps = (
            int(configured_warmup_steps)
            if configured_warmup_steps is not None
            else int(
                total_optimizer_steps * getattr(plan, "learning_rate_warmup_ratio", 0.0)
            )
        )
        source_layer_type = (
            source_layer_types[layer_index]
            if layer_index < len(source_layer_types)
            else None
        )
        gqa_native_geometry = (
            _gqa_native_zero_step_geometry(mixer=mixer, loaded_layer=loaded_layer)
            if source_layer_type == "full_attention"
            else None
        )
        gqa_installation_reader = None
        if gqa_native_geometry is not None:
            gqa_installation_reader, validation_reader = (
                _split_gqa_validation_protocol(
                    validation_reader,
                    world_size=distributed.world_size,
                )
            )
        configured_learning_rate = learning_rate_profiles.get(
            source_layer_type, plan.learning_rate
        )
        plan_final_learning_rate = float(
            getattr(
                plan,
                "final_learning_rate",
                float(plan.learning_rate)
                * float(getattr(plan, "min_learning_rate_ratio", 1.0)),
            )
        )
        final_learning_rate_ratio = plan_final_learning_rate / float(plan.learning_rate)
        configured_final_learning_rate = (
            configured_learning_rate
            if math.isclose(final_learning_rate_ratio, 1.0, rel_tol=1e-12, abs_tol=0.0)
            else configured_learning_rate * final_learning_rate_ratio
        )
        optimizer = ActiveLayerOptimizer(
            optimizer_name=getattr(plan, "optimizer_name", "adamw"),
            learning_rate=configured_learning_rate,
            final_learning_rate=configured_final_learning_rate,
            adam_betas=getattr(plan, "optimizer_betas", (0.9, 0.999)),
            adam_epsilon=getattr(plan, "optimizer_epsilon", 1e-8),
            weight_decay=getattr(plan, "optimizer_weight_decay", 0.0),
            detailed_telemetry_interval_steps=getattr(
                plan, "optimizer_telemetry_interval_steps", 1
            ),
            learning_rate_schedule=getattr(plan, "learning_rate_schedule", "constant"),
            warmup_steps=warmup_steps,
            total_steps=total_optimizer_steps,
            min_learning_rate_ratio=getattr(plan, "min_learning_rate_ratio", 1.0),
            gradient_clip_norm=getattr(plan, "gradient_clip_norm", None),
            max_parameter_update_relative_l2=getattr(
                plan, "max_parameter_update_relative_l2", None
            ),
            gradient_sync=distributed.synchronize_gradients,
        )
        snapshot: ActiveLayerOptimizerSnapshot | None = None
        best_generation = (
            resume_best_generation if progress and layer_index == start_layer else None
        )
        layer_state = LayerConvergenceState()
        first_epoch = 0
        first_row = 0
        resumed_active_generation = False
        if progress and layer_index == start_layer:
            phase = str(progress["phase"])
            if phase in {"train", "epoch-complete"}:
                generation = _resolve_progress_generation(run_dir, progress)
                _verify_generation_integrity(
                    generation,
                    expected_manifest_sha256=str(
                        progress["generation_manifest_sha256"]
                    ),
                )
                cursor = dict(progress["generation_cursor"])
                _rank0_filesystem_step(
                    distributed,
                    "resume generation mixer restore",
                    lambda: store.restore_generation(
                        generation / "mixer", layer_index, expected_cursor=cursor
                    ),
                )
                mixer = store.load_mixer(layer_index, device=device, dtype=dtype)
                snapshot = _load_generation_state(
                    generation,
                    device=device,
                    rank=distributed.rank,
                    world_size=distributed.world_size,
                    expected_cursor=cursor,
                )
                _validate_optimizer_snapshot(snapshot, progress)
                first_epoch = int(progress["epoch_index"])
                first_row = int(progress["next_train_row"])
                layer_state = _state_from_progress(progress)
                resumed_active_generation = True
            elif phase != "layer-ready":
                raise ContractError(f"unsupported layer-major resume phase: {phase}")
        _require_frozen_parameter_sha256(
            mixer,
            frozen_parameter_sha256,
            boundary="resume-or-zero-step-load",
        )
        activation_fit_rows = int(getattr(plan, "activation_fit_rows", 0))
        activation_fit_functional_steps = int(plan.activation_fit_functional_steps)
        activation_fit_functional_learning_rate = float(
            plan.activation_fit_functional_learning_rate
        )
        gqa_activation_fit = None
        if (
            activation_fit_rows
            and source_layer_type in {"linear_attention", "full_attention"}
            and not resumed_active_generation
        ):
            activation_fit_baseline = None
            if source_layer_type == "full_attention":
                gqa_activation_fit = _activation_fit_full_attention(
                    gqa_native_geometry=gqa_native_geometry,
                    gqa_installation_reader=gqa_installation_reader,
                    executor=executor,
                    train_reader=train_reader,
                    validation_reader=validation_reader,
                    mixer=mixer,
                    loaded_layer=loaded_layer,
                    layer_index=layer_index,
                    burn_in_tokens=plan.burn_in_tokens,
                    fit_rows=activation_fit_rows,
                    ridge=float(getattr(plan, "activation_fit_ridge", 1e-3)),
                    functional_steps=activation_fit_functional_steps,
                    functional_learning_rate=(
                        activation_fit_functional_learning_rate
                    ),
                    micro_batch_size=plan.micro_batch_size,
                    loss_weights=plan.local_loss_weights,
                    max_trace_bytes_per_rank=(
                        max_cached_layer_input_bytes_per_rank
                    ),
                    run_time_mix_ablation=bool(
                        getattr(
                            plan,
                            "activation_fit_attention_time_mix_ablation",
                            False,
                        )
                    ),
                    run_dir=run_dir,
                    distributed=distributed,
                )
            if source_layer_type == "linear_attention":
                activation_fit_baseline = _activation_fit_decay_projection(
                    executor=executor,
                    train_reader=train_reader,
                    validation_reader=validation_reader,
                    mixer=mixer,
                    loaded_layer=loaded_layer,
                    layer_index=layer_index,
                    burn_in_tokens=plan.burn_in_tokens,
                    fit_rows=activation_fit_rows,
                    ridge=float(getattr(plan, "activation_fit_ridge", 1e-3)),
                    micro_batch_size=plan.micro_batch_size,
                    loss_weights=plan.local_loss_weights,
                    run_dir=run_dir,
                    distributed=distributed,
                )
                activation_fit_baseline = _activation_fit_native_a_projection(
                    executor=executor,
                    train_reader=train_reader,
                    validation_reader=validation_reader,
                    mixer=mixer,
                    loaded_layer=loaded_layer,
                    layer_index=layer_index,
                    burn_in_tokens=plan.burn_in_tokens,
                    fit_rows=activation_fit_rows,
                    ridge=float(getattr(plan, "activation_fit_ridge", 1e-3)),
                    micro_batch_size=plan.micro_batch_size,
                    loss_weights=plan.local_loss_weights,
                    chain_baseline=activation_fit_baseline,
                    run_dir=run_dir,
                    distributed=distributed,
                )
            if source_layer_type == "linear_attention":
                activation_fit_baseline = _activation_fit_source_qkv_projections(
                    executor=executor,
                    train_reader=train_reader,
                    validation_reader=validation_reader,
                    mixer=mixer,
                    loaded_layer=loaded_layer,
                    layer_index=layer_index,
                    source_layer_type=source_layer_type,
                    burn_in_tokens=plan.burn_in_tokens,
                    fit_rows=activation_fit_rows,
                    ridge=float(getattr(plan, "activation_fit_ridge", 1e-3)),
                    micro_batch_size=plan.micro_batch_size,
                    loss_weights=plan.local_loss_weights,
                    chain_baseline=activation_fit_baseline,
                    run_dir=run_dir,
                    distributed=distributed,
                )
                activation_fit_baseline = _activation_fit_source_gate_projection(
                    executor=executor,
                    train_reader=train_reader,
                    validation_reader=validation_reader,
                    mixer=mixer,
                    loaded_layer=loaded_layer,
                    layer_index=layer_index,
                    source_layer_type=source_layer_type,
                    burn_in_tokens=plan.burn_in_tokens,
                    fit_rows=activation_fit_rows,
                    ridge=float(getattr(plan, "activation_fit_ridge", 1e-3)),
                    functional_steps=activation_fit_functional_steps,
                    functional_learning_rate=(activation_fit_functional_learning_rate),
                    micro_batch_size=plan.micro_batch_size,
                    loss_weights=plan.local_loss_weights,
                    chain_baseline=activation_fit_baseline,
                    run_dir=run_dir,
                    distributed=distributed,
                )
                activation_fit_baseline = _activation_fit_norm_affine(
                    executor=executor,
                    train_reader=train_reader,
                    validation_reader=validation_reader,
                    mixer=mixer,
                    loaded_layer=loaded_layer,
                    layer_index=layer_index,
                    source_layer_type=source_layer_type,
                    burn_in_tokens=plan.burn_in_tokens,
                    fit_rows=activation_fit_rows,
                    ridge=float(getattr(plan, "activation_fit_ridge", 1e-3)),
                    micro_batch_size=plan.micro_batch_size,
                    loss_weights=plan.local_loss_weights,
                    chain_baseline=activation_fit_baseline,
                    run_dir=run_dir,
                    distributed=distributed,
                )
                _activation_fit_output_projection(
                    executor=executor,
                    train_reader=train_reader,
                    validation_reader=validation_reader,
                    mixer=mixer,
                    loaded_layer=loaded_layer,
                    layer_index=layer_index,
                    source_layer_type=source_layer_type,
                    burn_in_tokens=plan.burn_in_tokens,
                    fit_rows=activation_fit_rows,
                    ridge=float(getattr(plan, "activation_fit_ridge", 1e-3)),
                    micro_batch_size=plan.micro_batch_size,
                    loss_weights=plan.local_loss_weights,
                    validation_baseline=activation_fit_baseline,
                    run_dir=run_dir,
                    distributed=distributed,
                )
                _activation_fit_dependency_transaction(
                    executor=executor,
                    train_reader=train_reader,
                    validation_reader=validation_reader,
                    mixer=mixer,
                    loaded_layer=loaded_layer,
                    layer_index=layer_index,
                    burn_in_tokens=plan.burn_in_tokens,
                    fit_rows=activation_fit_rows,
                    ridge=float(getattr(plan, "activation_fit_ridge", 1e-3)),
                    functional_steps=activation_fit_functional_steps,
                    functional_learning_rate=(activation_fit_functional_learning_rate),
                    micro_batch_size=plan.micro_batch_size,
                    loss_weights=plan.local_loss_weights,
                    run_dir=run_dir,
                    distributed=distributed,
                )
        distributed.synchronize_module_parameters(mixer)
        _require_frozen_parameter_sha256(
            mixer,
            frozen_parameter_sha256,
            boundary="post-activation-fit",
        )
        frozen_pre_epoch_validation = None
        if not resumed_active_generation:
            pre_epoch_validation = _validate(
                executor=executor,
                reader=validation_reader,
                mixer=mixer,
                loaded_layer=loaded_layer,
                layer_index=layer_index,
                burn_in_tokens=plan.burn_in_tokens,
                micro_batch_size=plan.micro_batch_size,
                loss_weights=plan.local_loss_weights,
                distributed=distributed,
            )
            optimizer.activate(
                layer_index,
                mixer,
                trainable_names=initial_trainable[layer_index],
            )
            _require_frozen_parameter_sha256(
                mixer,
                frozen_parameter_sha256,
                boundary="pre-epoch-baseline",
            )
            if gqa_activation_fit is not None:
                committed_input_sha256 = _sha256_json(
                    _module_state_hashes(mixer)
                )
                if (
                    committed_input_sha256
                    != gqa_activation_fit["selected_module_state_sha256"]
                ):
                    raise ContractError(
                        "GQA native zero-step selection changed before the "
                        "immutable pre-epoch generation"
                    )
                gqa_activation_fit = {
                    **gqa_activation_fit,
                    "pre_epoch_generation_input_sha256": (
                        committed_input_sha256
                    ),
                }
            baseline_cursor = _cursor(
                layer_index,
                -1,
                0,
                _sha256_json(
                    {
                        "phase": "pre-epoch-baseline",
                        "layer": layer_index,
                        "validation": pre_epoch_validation,
                    }
                ),
                consumed_rows=(),
                train_cache_manifest_sha256=train_cache_manifest_sha256,
                validation_cache_manifest_sha256=validation_cache_manifest_sha256,
                activation_fit_binding=gqa_activation_fit,
            )
            best_generation = _commit_distributed_generation(
                distributed,
                run_dir,
                store,
                mixer,
                optimizer,
                baseline_cursor,
            )
            if gqa_activation_fit is not None:
                generation_state_sha256 = _generation_mixer_state_sha256(
                    best_generation
                    / "mixer"
                    / f"layer-{layer_index:03d}.safetensors",
                    layer_index=layer_index,
                )
                if (
                    generation_state_sha256
                    != gqa_activation_fit["selected_module_state_sha256"]
                ):
                    raise ContractError(
                        "immutable pre-epoch generation differs from the "
                        "selected GQA native zero-step state"
                    )
                gqa_activation_fit = {
                    **gqa_activation_fit,
                    "pre_epoch_generation_state_sha256": (
                        generation_state_sha256
                    ),
                }
            snapshot = optimizer.release()
            layer_state = LayerConvergenceState(
                completed_epochs=0,
                best_metric=float(pre_epoch_validation["loss"]),
                best_epoch=-1,
                best_training_metric=None,
                best_training_epoch=None,
                bad_epochs=0,
            )
            if distributed.is_primary:
                write_json(
                    run_dir / "training-baselines" / f"layer-{layer_index:03d}.json",
                    {
                        "schema_version": 1,
                        "status": "frozen",
                        "layer": layer_index,
                        "boundary": "post-activation-fit-pre-epoch",
                        "validation": pre_epoch_validation,
                        "generation": str(
                            best_generation.relative_to(run_dir.resolve())
                        ),
                        "generation_manifest_sha256": file_sha256(
                            best_generation / "integrity.json"
                        ),
                        "activation_fit": gqa_activation_fit,
                    },
                )
            frozen_pre_epoch_validation = pre_epoch_validation
        else:
            baseline_path = (
                run_dir / "training-baselines" / f"layer-{layer_index:03d}.json"
            )
            if not baseline_path.is_file():
                raise ContractError(
                    "resumed layer is missing its frozen pre-epoch baseline"
                )
            frozen_pre_epoch_validation = json.loads(
                baseline_path.read_text(encoding="utf-8")
            ).get("validation")
            if not isinstance(frozen_pre_epoch_validation, dict):
                raise ContractError(
                    "resumed layer has a malformed frozen pre-epoch baseline"
                )
        convergence_observed = any(
            bool(row.get("convergence_observed", row.get("converged", False)))
            for row in histories
            if int(row.get("layer", -1)) == layer_index
        )

        for epoch_index in range(first_epoch, plan.layer_max_epochs):
            row_position = first_row if epoch_index == first_epoch else 0
            if not optimizer.is_active:
                optimizer.activate(
                    layer_index,
                    mixer,
                    snapshot=snapshot,
                    # The recipe-bound set is stable for the entire local stage.
                    # Qwen3.5→RWKV7 deliberately includes every current-mixer
                    # parameter; only other layers and the teacher stay frozen.
                    # Passing ``None`` would make resume identity ambiguous.
                    trainable_names=initial_trainable[layer_index],
                )
                distributed.validate_trainable_signature(mixer)
                snapshot = None
            permutation, permutation_sha = epoch_permutation(
                row_count=train_reader.row_count,
                seed=plan.seed,
                layer=layer_index,
                epoch=epoch_index,
            )
            if progress and layer_index == start_layer and epoch_index == first_epoch:
                _validate_resume_permutation(
                    progress,
                    permutation=permutation,
                    permutation_sha=permutation_sha,
                )
            segment_start_row = row_position
            segment_start_optimizer_step = optimizer.optimizer_step
            distributed.barrier()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            train_started = time.perf_counter()
            checkpoint_wall_seconds = 0.0
            micro_batch_index = 0
            current_generation: Path | None = None
            while row_position < len(permutation):
                row_stop = min(row_position + global_micro_batch_size, len(permutation))
                trailing = len(permutation) - row_stop
                if 0 < trailing < distributed.world_size:
                    row_stop = len(permutation)
                global_row_indices = permutation[row_position:row_stop]
                row_indices = distributed.shard_rows(global_row_indices)
                cached = train_reader.read_rows(row_indices)
                output = executor.forward_cached_layer_local(
                    cached.hidden_states,
                    shared_states=cached.shared_states,
                    active_layer_index=layer_index,
                    active_mixer=mixer,
                    loaded_layer=loaded_layer,
                )
                will_checkpoint = (
                    plan.checkpoint_interval_micro_batches > 0
                    and (micro_batch_index + 1) % plan.checkpoint_interval_micro_batches
                    == 0
                    and row_stop < len(permutation)
                )
                loss, metrics = _local_loss(
                    output,
                    plan.burn_in_tokens,
                    plan.local_loss_weights,
                    materialize_metrics=will_checkpoint,
                )
                loss_scale = (
                    len(row_indices) * distributed.world_size / len(global_row_indices)
                )
                optimizer.backward(
                    loss * loss_scale,
                    accumulation_steps=plan.accumulation_steps,
                    sample_weight=len(global_row_indices),
                )
                row_position = row_stop
                micro_batch_index += 1
                if (
                    plan.checkpoint_interval_micro_batches > 0
                    and micro_batch_index % plan.checkpoint_interval_micro_batches == 0
                    and row_position < len(permutation)
                ):
                    _require_frozen_parameter_sha256(
                        mixer,
                        frozen_parameter_sha256,
                        boundary="mid-epoch-checkpoint",
                    )
                    metrics = distributed.aggregate_metrics(metrics, len(row_indices))
                    checkpoint_started = time.perf_counter()
                    cursor = _cursor(
                        layer_index,
                        epoch_index,
                        row_position,
                        permutation_sha,
                        consumed_rows=permutation[:row_position],
                        train_cache_manifest_sha256=train_cache_manifest_sha256,
                        validation_cache_manifest_sha256=(
                            validation_cache_manifest_sha256
                        ),
                    )
                    current_generation = _commit_distributed_generation(
                        distributed, run_dir, store, mixer, optimizer, cursor
                    )
                    _rank0_filesystem_step(
                        distributed,
                        "publish layer-major train progress",
                        lambda: (
                            _write_progress(
                                progress_path,
                                phase="train",
                                active_layer=layer_index,
                                epoch_index=epoch_index,
                                next_train_row=row_position,
                                prefix_fingerprint=prefix_fingerprint,
                                generation=current_generation,
                                best_generation=best_generation,
                                generation_cursor=cursor,
                                convergence=layer_state,
                                completed_optimizer_steps=completed_optimizer_steps,
                                active_optimizer_steps=optimizer.optimizer_step,
                                history=histories,
                                base_binding=base_binding,
                                last_train_metrics=metrics,
                            ),
                            _notify_progress(progress_callback, "train", progress_path),
                        ),
                    )
                    _prune_generations_distributed(
                        run_dir,
                        keep=(current_generation, best_generation),
                        distributed=distributed,
                    )
                    checkpoint_wall_seconds += time.perf_counter() - checkpoint_started
            optimizer.flush(accumulation_steps=plan.accumulation_steps)
            _require_frozen_parameter_sha256(
                mixer,
                frozen_parameter_sha256,
                boundary="post-epoch-optimizer",
            )
            cursor = _cursor(
                layer_index,
                epoch_index,
                len(permutation),
                permutation_sha,
                consumed_rows=permutation,
                train_cache_manifest_sha256=train_cache_manifest_sha256,
                validation_cache_manifest_sha256=validation_cache_manifest_sha256,
            )
            checkpoint_started = time.perf_counter()
            current_generation = _commit_distributed_generation(
                distributed, run_dir, store, mixer, optimizer, cursor
            )
            checkpoint_wall_seconds += time.perf_counter() - checkpoint_started
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_wall_seconds = time.perf_counter() - train_started
            validation_started = time.perf_counter()
            validation = _validate(
                executor=executor,
                reader=validation_reader,
                mixer=mixer,
                loaded_layer=loaded_layer,
                layer_index=layer_index,
                burn_in_tokens=plan.burn_in_tokens,
                micro_batch_size=plan.micro_batch_size,
                loss_weights=plan.local_loss_weights,
                distributed=distributed,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            validation_wall_seconds = time.perf_counter() - validation_started
            telemetry_by_rank = distributed.gather_scalar_metrics(
                {
                    "train_wall_seconds": train_wall_seconds,
                    "validation_wall_seconds": validation_wall_seconds,
                    "checkpoint_wall_seconds": checkpoint_wall_seconds,
                    "peak_cuda_allocated_bytes": (
                        torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda"
                        else 0
                    ),
                    "peak_cuda_reserved_bytes": (
                        torch.cuda.max_memory_reserved(device)
                        if device.type == "cuda"
                        else 0
                    ),
                }
            )
            segment_row_count = len(permutation) - segment_start_row
            max_train_wall_seconds = max(telemetry_by_rank["train_wall_seconds"])
            max_validation_wall_seconds = max(
                telemetry_by_rank["validation_wall_seconds"]
            )
            max_checkpoint_wall_seconds = max(
                telemetry_by_rank["checkpoint_wall_seconds"]
            )
            peak_cuda_reserved_bytes = max(
                telemetry_by_rank["peak_cuda_reserved_bytes"]
            )
            telemetry = {
                "scope": "current-process-epoch-segment",
                "world_size": distributed.world_size,
                "per_rank_micro_batch_size": plan.micro_batch_size,
                "global_micro_batch_size": (
                    plan.micro_batch_size * distributed.world_size
                ),
                "accumulation_steps": plan.accumulation_steps,
                "segment_start_row": segment_start_row,
                "segment_row_count": segment_row_count,
                "epoch_row_count": len(permutation),
                "context_token_count": len(permutation) * sequence_length,
                "loss_token_count": len(permutation) * plan.supervised_tokens,
                "segment_optimizer_steps": (
                    optimizer.optimizer_step - segment_start_optimizer_step
                ),
                "active_layer_optimizer_steps": optimizer.optimizer_step,
                "train_wall_seconds_by_rank": list(
                    telemetry_by_rank["train_wall_seconds"]
                ),
                "validation_wall_seconds_by_rank": list(
                    telemetry_by_rank["validation_wall_seconds"]
                ),
                "checkpoint_wall_seconds_by_rank": list(
                    telemetry_by_rank["checkpoint_wall_seconds"]
                ),
                "max_train_wall_seconds": max_train_wall_seconds,
                "max_validation_wall_seconds": max_validation_wall_seconds,
                "max_checkpoint_wall_seconds": max_checkpoint_wall_seconds,
                "checkpoint_fraction_of_train_wall": (
                    max_checkpoint_wall_seconds / max_train_wall_seconds
                ),
                "segment_context_tokens_per_second": (
                    segment_row_count * sequence_length / max_train_wall_seconds
                ),
                "segment_loss_tokens_per_second": (
                    segment_row_count * plan.supervised_tokens / max_train_wall_seconds
                ),
                "peak_cuda_allocated_bytes_by_rank": list(
                    telemetry_by_rank["peak_cuda_allocated_bytes"]
                ),
                "peak_cuda_reserved_bytes_by_rank": list(
                    telemetry_by_rank["peak_cuda_reserved_bytes"]
                ),
                "peak_cuda_reserved_bytes": peak_cuda_reserved_bytes,
                "max_cuda_reserved_bytes": max_cuda_reserved_bytes,
                "optimizer": optimizer.telemetry(),
                "source_mixer_kind": source_layer_type,
                "configured_learning_rate": configured_learning_rate,
                "learning_rate_schedule_epochs": scheduled_epochs,
                "learning_rate_schedule_optimizer_steps": total_optimizer_steps,
            }
            telemetry.update({"layer": layer_index, "epoch": epoch_index})
            if distributed.is_primary:
                _write_epoch_telemetry(run_dir, telemetry)
            memory_limit_exceeded = (
                max_cuda_reserved_bytes is not None
                and peak_cuda_reserved_bytes > max_cuda_reserved_bytes
            )
            decision = update_convergence(
                layer_state,
                metric=validation["loss"],
                min_epochs=plan.layer_min_epochs,
                max_epochs=plan.layer_max_epochs,
                min_delta=plan.layer_min_delta,
                patience=plan.layer_patience,
            )
            regressed_past_frozen_baseline = (
                not decision.improved
                and decision.state.completed_epochs >= plan.layer_min_epochs
                and decision.state.best_epoch is not None
                and decision.state.best_epoch >= 0
                and validation["loss"] >= frozen_pre_epoch_validation["loss"]
            )
            if regressed_past_frozen_baseline and not decision.converged:
                decision = replace(
                    decision,
                    converged=True,
                    exhausted=False,
                    reason="validation-regressed-past-frozen-baseline",
                )
            convergence_observed = convergence_observed or decision.converged
            fixed_budget_complete = (
                layer_fixed_epochs is not None
                and decision.state.completed_epochs >= layer_fixed_epochs
            )
            training_improved = (
                decision.state.best_epoch is not None
                and decision.state.best_epoch >= 0
                and decision.state.best_metric is not None
                and decision.state.best_metric
                < frozen_pre_epoch_validation["loss"] - plan.layer_min_delta
            )
            layer_complete = (
                decision.converged and training_improved
                if layer_fixed_epochs is None
                else (
                    fixed_budget_complete and convergence_observed and training_improved
                )
            )
            if decision.improved:
                best_generation = current_generation
            elif best_generation is None:
                raise ContractError(
                    "layer convergence state references a missing best generation"
                )
            previous_layer_optimizer_steps = max(
                (
                    int(row["training_budget"]["active_layer_optimizer_steps"])
                    for row in histories
                    if int(row.get("layer", -1)) == layer_index
                    and isinstance(row.get("training_budget"), dict)
                ),
                default=0,
            )
            training_budget = {
                "world_size": distributed.world_size,
                "per_rank_micro_batch_size": plan.micro_batch_size,
                "global_micro_batch_size": (
                    plan.micro_batch_size * distributed.world_size
                ),
                "accumulation_steps": plan.accumulation_steps,
                "train_row_count": len(permutation),
                "context_token_count": len(permutation) * sequence_length,
                "loss_token_count": len(permutation) * plan.supervised_tokens,
                "epoch_optimizer_steps": (
                    optimizer.optimizer_step - previous_layer_optimizer_steps
                ),
                "active_layer_optimizer_steps": optimizer.optimizer_step,
            }
            histories.append(
                {
                    "layer": layer_index,
                    "epoch": epoch_index,
                    "train_row_count": len(permutation),
                    "micro_batch_size": plan.micro_batch_size,
                    "permutation_sha256": permutation_sha,
                    "consumed_rows_sha256": cursor["consumed_rows_sha256"],
                    "validation": validation,
                    "training_budget": training_budget,
                    "improved": decision.improved,
                    "training_curve_improved": decision.training_curve_improved,
                    "best_epoch": decision.state.best_epoch,
                    "best_metric": decision.state.best_metric,
                    "best_training_epoch": decision.state.best_training_epoch,
                    "best_training_metric": decision.state.best_training_metric,
                    "bad_epochs": decision.state.bad_epochs,
                    "converged": decision.converged,
                    "convergence_observed": convergence_observed,
                    "fixed_budget_complete": fixed_budget_complete,
                    "exhausted": decision.exhausted,
                    "convergence_reason": decision.reason,
                }
            )
            layer_state = decision.state
            active_optimizer_steps = optimizer.optimizer_step
            first_row = 0
            if memory_limit_exceeded:
                _rank0_filesystem_step(
                    distributed,
                    "publish CUDA-limit progress",
                    lambda: _write_progress(
                        progress_path,
                        phase="epoch-complete",
                        active_layer=layer_index,
                        epoch_index=epoch_index + 1,
                        next_train_row=0,
                        prefix_fingerprint=prefix_fingerprint,
                        generation=current_generation,
                        best_generation=best_generation,
                        generation_cursor=cursor,
                        convergence=layer_state,
                        completed_optimizer_steps=completed_optimizer_steps,
                        active_optimizer_steps=active_optimizer_steps,
                        history=histories,
                        base_binding=base_binding,
                        last_train_metrics=None,
                    ),
                )
                _prune_generations_distributed(
                    run_dir,
                    keep=(current_generation, best_generation),
                    distributed=distributed,
                )
                raise ContractError(
                    "layer-major CUDA reserved memory exceeded the frozen plan limit: "
                    f"observed={int(peak_cuda_reserved_bytes)} "
                    f"limit={max_cuda_reserved_bytes}"
                )
            if layer_complete:
                if best_generation is None:
                    raise ContractError(
                        "converged layer has no immutable best generation"
                    )
                optimizer.release()
                best_cursor = json.loads(
                    (best_generation / "cursor.json").read_text(encoding="utf-8")
                )
                if distributed.is_primary:
                    store.restore_generation(
                        best_generation / "mixer",
                        layer_index,
                        expected_cursor=best_cursor,
                    )
                distributed.barrier()
                del mixer
                gc.collect()
                mixer = store.load_mixer(layer_index, device=device, dtype=dtype)
                next_prefix_fingerprint = _advance_prefix_fingerprint(
                    prefix_fingerprint,
                    best_generation / "mixer" / f"layer-{layer_index:03d}.safetensors",
                )
                if layer_index + 1 < layer_stop:
                    cache_transition_started = time.perf_counter()
                    _ensure_next_layer_caches(
                        executor=executor,
                        cache_root=cache_root,
                        current_layer=layer_index,
                        train_reader=train_reader,
                        validation_reader=full_validation_reader,
                        mixer=mixer,
                        loaded_layer=loaded_layer,
                        shard_rows=plan.cache_shard_rows,
                        hidden_size=source_manifest.contract.hidden_size,
                        base_binding=base_binding,
                        next_prefix_fingerprint=next_prefix_fingerprint,
                        distributed=distributed,
                    )
                    _write_cache_transition_telemetry(
                        run_dir=run_dir,
                        source_layer=layer_index,
                        target_layer=layer_index + 1,
                        row_count=(
                            train_reader.row_count
                            + full_validation_reader.row_count
                        ),
                        sequence_length=sequence_length,
                        local_wall_seconds=(
                            time.perf_counter() - cache_transition_started
                        ),
                        distributed=distributed,
                    )
                completed_optimizer_steps += active_optimizer_steps
                next_phase = (
                    "calibration-complete"
                    if exploratory_layer_limit is not None
                    and layer_index + 1 == layer_stop
                    else (
                        "layer-ready"
                        if layer_index + 1 < num_layers
                        else "local-complete"
                    )
                )
                _rank0_filesystem_step(
                    distributed,
                    "publish layer-ready progress",
                    lambda: (
                        (
                            _write_layer_convergence(
                                run_dir,
                                histories=histories,
                                world_size=distributed.world_size,
                            )
                            if next_phase == "calibration-complete"
                            else None
                        ),
                        _write_progress(
                            progress_path,
                            phase=next_phase,
                            active_layer=layer_index + 1,
                            epoch_index=0,
                            next_train_row=0,
                            prefix_fingerprint=next_prefix_fingerprint,
                            generation=best_generation,
                            best_generation=best_generation,
                            generation_cursor=best_cursor,
                            convergence=LayerConvergenceState(),
                            completed_optimizer_steps=completed_optimizer_steps,
                            active_optimizer_steps=0,
                            history=histories,
                            base_binding=base_binding,
                            last_train_metrics=None,
                        ),
                        _notify_progress(
                            progress_callback,
                            next_phase,
                            progress_path,
                        ),
                    ),
                )
                _prune_generations_distributed(
                    run_dir,
                    keep=(best_generation,),
                    distributed=distributed,
                )
                if distributed.is_primary and layer_index + 1 < layer_stop:
                    shutil.rmtree(_layer_cache_dir(cache_root, layer_index))
                distributed.barrier()
                # The next loop iteration must not load layer i+1 while Python
                # locals still keep layer i resident.  This is also important
                # for CPU cache shards: the per-rank byte LRU belongs only to
                # the active layer and must not survive the transition.
                del (
                    train_reader,
                    validation_reader,
                    full_validation_reader,
                    loaded_layer,
                    mixer,
                    optimizer,
                )
                gc.collect()
                prefix_fingerprint = next_prefix_fingerprint
                progress = None
                break
            _rank0_filesystem_step(
                distributed,
                "publish epoch-complete progress",
                lambda: (
                    _write_progress(
                        progress_path,
                        phase="epoch-complete",
                        active_layer=layer_index,
                        epoch_index=epoch_index + 1,
                        next_train_row=0,
                        prefix_fingerprint=prefix_fingerprint,
                        generation=current_generation,
                        best_generation=best_generation,
                        generation_cursor=cursor,
                        convergence=layer_state,
                        completed_optimizer_steps=completed_optimizer_steps,
                        active_optimizer_steps=active_optimizer_steps,
                        history=histories,
                        base_binding=base_binding,
                        last_train_metrics=None,
                    ),
                    _notify_progress(
                        progress_callback, "epoch-complete", progress_path
                    ),
                ),
            )
            _prune_generations_distributed(
                run_dir,
                keep=(current_generation, best_generation),
                distributed=distributed,
            )
            if (
                layer_fixed_epochs is not None
                and fixed_budget_complete
                and not convergence_observed
            ):
                raise ContractError(
                    f"layer-convergence-failed: layer {layer_index} did not converge "
                    f"within fixed budget of {layer_fixed_epochs} epochs"
                )
            if decision.converged and not training_improved:
                raise ContractError(
                    f"layer-training-no-improvement: layer {layer_index} plateaued "
                    "without beating its frozen post-fit pre-epoch baseline"
                )
            if layer_fixed_epochs is None and decision.exhausted:
                raise ContractError(
                    f"layer-convergence-failed: layer {layer_index} reached "
                    f"{plan.layer_max_epochs} epochs"
                )
        else:
            raise ContractError(
                f"layer-convergence-failed: layer {layer_index} did not converge"
            )

    _rank0_filesystem_step(
        distributed,
        "publish layer convergence artifact",
        lambda: _write_layer_convergence(
            run_dir, histories=histories, world_size=distributed.world_size
        ),
    )
    if exploratory_layer_limit is not None:
        return _exploratory_completion(
            completed_layers=layer_stop,
            total_layers=num_layers,
            optimizer_steps=completed_optimizer_steps,
            histories=histories,
            run_dir=run_dir,
        )
    checkpoint = run_dir / "checkpoint-layerwise-local"
    _rank0_filesystem_step(
        distributed,
        "materialize layerwise checkpoint",
        lambda: store.materialize_checkpoint(checkpoint, fitted_evidence_root=run_dir),
    )
    completion = _local_completion(num_layers, completed_optimizer_steps, histories)
    completion["checkpoint"] = str(checkpoint)
    return completion


def _local_completion(num_layers, optimizer_steps, histories):
    return {
        "status": "layerwise-local-complete",
        "execution_mode": "rolling_layer_input_cache",
        "layers": num_layers,
        "optimizer_steps": optimizer_steps,
        "history": histories,
        "next_stage": "fully-recurrent-global-corrective",
    }


def _exploratory_completion(
    *, completed_layers, total_layers, optimizer_steps, histories, run_dir
):
    return {
        "status": "exploratory-layer-calibration-complete",
        "execution_mode": "rolling_layer_input_cache",
        "layers_completed": completed_layers,
        "total_layers": total_layers,
        "optimizer_steps": optimizer_steps,
        "history": histories,
        "overlay_root": str(run_dir / "mixer-overlays"),
        "next_stage": "compare-training-controls",
    }


def _write_layer_convergence(run_dir, *, histories, world_size) -> None:
    write_json(
        run_dir / "layer-convergence.json",
        {
            "schema_version": 2,
            "schedule": "rolling-cache-layer-major-v1",
            "world_size": world_size,
            "epochs": histories,
        },
    )


def _rank0_filesystem_step(distributed, description, operation) -> None:
    status = None
    if distributed.is_primary:
        try:
            operation()
            status = {"status": "ok"}
        except BaseException as error:
            if distributed.world_size == 1:
                raise
            status = {"status": "error", "error": repr(error)}
    status = distributed.broadcast_object(status)
    if status["status"] != "ok":
        raise ContractError(f"{description} failed: {status['error']}")


def _prune_generations_distributed(run_dir, *, keep, distributed) -> None:
    keep_resolved = {path.resolve() for path in keep if path is not None}

    def prune() -> None:
        root = run_dir / "layer-generations"
        if not root.is_dir():
            return
        for candidate in tuple(root.iterdir()):
            if candidate.resolve() in keep_resolved:
                continue
            if candidate.is_dir():
                shutil.rmtree(candidate)
            elif candidate.exists():
                candidate.unlink()

    _rank0_filesystem_step(
        distributed,
        "prune unreferenced layer generations",
        prune,
    )


def _write_epoch_telemetry(run_dir: Path, telemetry: dict[str, object]) -> None:
    destination = (
        run_dir
        / "training-telemetry"
        / (
            f"l{int(telemetry['layer']):03d}-e{int(telemetry['epoch']):03d}-"
            f"r{int(telemetry['segment_start_row']):09d}-"
            f"{int(telemetry['segment_start_row']) + int(telemetry['segment_row_count']):09d}.json"
        )
    )
    write_json(destination, telemetry)


def _write_cache_transition_telemetry(
    *,
    run_dir,
    source_layer,
    target_layer,
    row_count,
    sequence_length,
    local_wall_seconds,
    distributed,
):
    gathered = distributed.gather_scalar_metrics({"wall_seconds": local_wall_seconds})
    if not distributed.is_primary:
        return
    wall_seconds_by_rank = list(gathered["wall_seconds"])
    max_wall_seconds = max(wall_seconds_by_rank)
    context_token_count = row_count * sequence_length
    write_json(
        run_dir / "cache-transition-telemetry" / f"l{target_layer:03d}.json",
        {
            "schema_version": 1,
            "writer": (
                "distributed-row-sharded"
                if distributed.world_size > 1
                else "single-rank-fixture"
            ),
            "source_layer": source_layer,
            "target_layer": target_layer,
            "world_size": distributed.world_size,
            "row_count": row_count,
            "context_token_count": context_token_count,
            "wall_seconds_by_rank": wall_seconds_by_rank,
            "max_wall_seconds": max_wall_seconds,
            "context_tokens_per_second": context_token_count / max_wall_seconds,
        },
    )


def _ensure_embedding_caches(
    *,
    teacher,
    cache_root,
    token_rows,
    validation_rows,
    shard_rows,
    hidden_size,
    base_binding,
    prefix_fingerprint,
    distributed,
):
    for split, rows in (("distill_train", token_rows), ("validation", validation_rows)):
        destination = _split_cache_dir(cache_root, 0, split)
        binding = _cache_binding(base_binding, split, prefix_fingerprint)

        def batches() -> Iterable[LayerInputBatch]:
            with torch.no_grad():
                rank_start, rank_stop = _rank_cache_row_range(len(rows), distributed)
                for start in range(rank_start, rank_stop, shard_rows):
                    stop = min(start + shard_rows, rank_stop)
                    input_ids = torch.tensor(
                        rows[start:stop], dtype=torch.long, device=teacher.device
                    )
                    hidden = teacher.embed_input_ids(input_ids)
                    yield LayerInputBatch(
                        torch.arange(start, stop, dtype=torch.int64),
                        hidden,
                    )

        _write_cache_across_ranks(
            destination=destination,
            layer_index=0,
            split=split,
            row_count=len(rows),
            sequence_length=len(rows[0]),
            hidden_size=hidden_size,
            has_shared_states=False,
            binding=binding,
            batches=batches(),
            distributed=distributed,
        )


def _ensure_next_layer_caches(
    *,
    executor,
    cache_root,
    current_layer,
    train_reader,
    validation_reader,
    mixer,
    loaded_layer,
    shard_rows,
    hidden_size,
    base_binding,
    next_prefix_fingerprint,
    distributed,
):
    next_layer = current_layer + 1
    for split, reader in (
        ("distill_train", train_reader),
        ("validation", validation_reader),
    ):
        destination = _split_cache_dir(cache_root, next_layer, split)
        binding = _cache_binding(base_binding, split, next_prefix_fingerprint)

        def batches() -> Iterable[LayerInputBatch]:
            rank_start, rank_stop = _rank_cache_row_range(reader.row_count, distributed)
            for start in range(rank_start, rank_stop, shard_rows):
                stop = min(start + shard_rows, rank_stop)
                cached = reader.read_rows(range(start, stop))
                hidden, shared = executor.forward_cached_target_block(
                    cached.hidden_states,
                    shared_states=cached.shared_states,
                    layer_index=current_layer,
                    mixer=mixer,
                    loaded_layer=loaded_layer,
                )
                if shared is None:
                    raise ContractError(
                        "target adapter did not provide required cross-layer shared state"
                    )
                yield LayerInputBatch(
                    torch.arange(start, stop, dtype=torch.int64), hidden, shared
                )

        _write_cache_across_ranks(
            destination=destination,
            layer_index=next_layer,
            split=split,
            row_count=reader.row_count,
            sequence_length=int(reader.manifest["sequence_length"]),
            hidden_size=hidden_size,
            has_shared_states=True,
            binding=binding,
            batches=batches(),
            distributed=distributed,
        )


def _write_cache_across_ranks(
    *,
    destination,
    layer_index,
    split,
    row_count,
    sequence_length,
    hidden_size,
    has_shared_states,
    binding,
    batches,
    distributed,
):
    if distributed.world_size == 1:
        if destination.is_dir():
            LayerInputCacheReader(destination, expected_binding=binding)
            return
        write_layer_input_cache(
            destination,
            layer_index=layer_index,
            split=split,
            row_count=row_count,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            binding=binding,
            batches=batches,
        )
        return
    decision = None
    if distributed.is_primary:
        try:
            decision = {
                "status": "ok",
                "transition": prepare_distributed_layer_input_cache(
                    destination,
                    world_size=distributed.world_size,
                    layer_index=layer_index,
                    split=split,
                    row_count=row_count,
                    sequence_length=sequence_length,
                    hidden_size=hidden_size,
                    has_shared_states=has_shared_states,
                    binding=binding,
                ),
            }
        except BaseException as error:
            decision = {"status": "error", "error": repr(error)}
    decision = distributed.broadcast_object(decision)
    if decision["status"] != "ok":
        raise ContractError(
            "distributed layer-input cache prepare failed: " + decision["error"]
        )
    if decision["transition"]["state"] != "published":
        try:
            partition_status = write_distributed_layer_input_cache_partition(
                destination,
                rank=distributed.rank,
                world_size=distributed.world_size,
                layer_index=layer_index,
                split=split,
                row_count=row_count,
                sequence_length=sequence_length,
                hidden_size=hidden_size,
                has_shared_states=has_shared_states,
                binding=binding,
                batches=batches,
            )
            local_status = {"status": "ok", "partition": partition_status}
        except BaseException as error:
            local_status = {
                "status": "error",
                "rank": distributed.rank,
                "error": repr(error),
            }
        rank_statuses = distributed.all_gather_objects(local_status)
        failures = [row for row in rank_statuses if row["status"] != "ok"]
        if failures:
            raise ContractError(
                "distributed layer-input cache partition failed: "
                + "; ".join(
                    f"rank={row['rank']} error={row['error']}" for row in failures
                )
            )
        publish_status = None
        if distributed.is_primary:
            try:
                publish_distributed_layer_input_cache(
                    destination,
                    world_size=distributed.world_size,
                    layer_index=layer_index,
                    split=split,
                    row_count=row_count,
                    sequence_length=sequence_length,
                    hidden_size=hidden_size,
                    has_shared_states=has_shared_states,
                    binding=binding,
                )
                publish_status = {"status": "ok"}
            except BaseException as error:
                publish_status = {"status": "error", "error": repr(error)}
        publish_status = distributed.broadcast_object(publish_status)
        if publish_status["status"] != "ok":
            raise ContractError(
                "distributed layer-input cache publish failed: "
                + publish_status["error"]
            )
    LayerInputCacheReader(
        destination,
        expected_binding=binding,
        verification="manifest",
    )


def _rank_cache_row_range(row_count, distributed):
    return (
        row_count * distributed.rank // distributed.world_size,
        row_count * (distributed.rank + 1) // distributed.world_size,
    )


def _open_cache(
    cache_root,
    layer_index,
    split,
    base_binding,
    prefix_fingerprint,
    *,
    max_cached_bytes,
):
    return LayerInputCacheReader(
        _split_cache_dir(cache_root, layer_index, split),
        expected_binding=_cache_binding(base_binding, split, prefix_fingerprint),
        max_cached_bytes=max_cached_bytes,
    )


def _layer_cache_dir(cache_root: Path, layer_index: int) -> Path:
    return cache_root / f"layer-{layer_index:03d}"


def _split_cache_dir(cache_root: Path, layer_index: int, split: str) -> Path:
    return _layer_cache_dir(cache_root, layer_index) / split


def _cleanup_stale_layer_caches(cache_root: Path, *, keep_layer: int) -> None:
    for layer_index in range(keep_layer):
        stale = _layer_cache_dir(cache_root, layer_index)
        if stale.exists():
            shutil.rmtree(stale)


def _cache_binding(base_binding, split, prefix_fingerprint):
    binding = {
        **base_binding,
        "split": split,
        "prefix_fingerprint": prefix_fingerprint,
    }
    split_sample_ids = base_binding.get("split_sample_ids_sha256")
    if isinstance(split_sample_ids, dict):
        sample_ids_sha256 = split_sample_ids.get(split)
        if isinstance(sample_ids_sha256, str):
            binding["source_sample_ids_sha256"] = sample_ids_sha256
    return binding


def _run_binding(
    *,
    source_manifest,
    run_dir,
    zero_step_dir,
    initial_trainable,
    training_config,
    dataset_manifest,
):
    from ...distill_runner import _binding_sha256, _zero_step_checkpoint_binding

    warm_start_plan = run_dir / "warm-start-plan.json"
    if not warm_start_plan.is_file():
        raise ContractError("layer-major training requires warm-start-plan.json")
    dataset_payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    split_sample_ids_sha256 = {
        split: str(metadata["source_sample_ids_sha256"])
        for split, metadata in dataset_payload.get("splits", {}).items()
        if isinstance(metadata, dict)
        and isinstance(metadata.get("source_sample_ids_sha256"), str)
    }
    return {
        "recipe": "qwen35_to_rwkv7",
        "source_checkpoint_sha256": _sha256_json(source_manifest.file_hashes),
        "zero_step_checkpoint_sha256": _binding_sha256(
            _zero_step_checkpoint_binding(zero_step_dir)
        ),
        "warm_start_plan_sha256": file_sha256(warm_start_plan),
        "initial_trainable_sha256": _sha256_json(
            [sorted(names) for names in initial_trainable]
        ),
        "training_config_sha256": file_sha256(training_config),
        "dataset_manifest_sha256": file_sha256(dataset_manifest),
        "split_sample_ids_sha256": split_sample_ids_sha256,
    }


def _advance_prefix_fingerprint(prefix_fingerprint: str, mixer_path: Path) -> str:
    return _sha256_json(
        {
            "previous_prefix_fingerprint": prefix_fingerprint,
            "committed_mixer_sha256": file_sha256(mixer_path),
        }
    )


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    if contiguous.dtype == torch.bfloat16:
        contiguous = contiguous.view(torch.uint16)
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _frozen_parameter_sha256(
    mixer: torch.nn.Module, trainable_names: set[str]
) -> dict[str, str]:
    parameters = dict(mixer.named_parameters())
    unknown = sorted(trainable_names - parameters.keys())
    if unknown:
        raise ContractError(
            f"local trainable set contains unknown mixer parameters: {unknown}"
        )
    frozen = {
        name: _tensor_sha256(parameter)
        for name, parameter in parameters.items()
        if name not in trainable_names
    }
    return frozen


def _require_frozen_parameter_sha256(
    mixer: torch.nn.Module,
    expected: dict[str, str],
    *,
    boundary: str,
) -> None:
    parameters = dict(mixer.named_parameters())
    missing = sorted(expected.keys() - parameters.keys())
    changed = sorted(
        name
        for name, digest in expected.items()
        if name in parameters and _tensor_sha256(parameters[name]) != digest
    )
    if missing or changed:
        raise ContractError(
            "locally frozen mixer parameters changed at "
            f"{boundary}: missing={missing} changed={changed}"
        )


def _local_loss(
    output,
    burn_in_tokens: int,
    loss_weights,
    *,
    materialize_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    student_mixer = output.student_mixer_output[:, burn_in_tokens:]
    teacher_mixer = output.teacher_mixer_output[:, burn_in_tokens:].to(
        student_mixer.device
    )
    student_block = output.student_block_output[:, burn_in_tokens:]
    teacher_block = output.teacher_block_output[:, burn_in_tokens:].to(
        student_block.device
    )
    mixer_mse = normalized_mse(student_mixer, teacher_mixer)
    block_mse = normalized_mse(student_block, teacher_block)
    cosine = torch.nn.functional.cosine_similarity(
        student_block.float().flatten(0, -2),
        teacher_block.float().flatten(0, -2),
        dim=-1,
    ).mean()
    weights = loss_weights
    loss = (
        weights.mixer_mse * mixer_mse
        + weights.block_mse * block_mse
        + weights.cosine * (1 - cosine)
    )
    if not materialize_metrics:
        return loss, {}
    return loss, {
        "loss": float(loss.detach()),
        "mixer_normalized_mse": float(mixer_mse.detach()),
        "block_normalized_mse": float(block_mse.detach()),
        "normalized_mse": float(((mixer_mse + block_mse) / 2).detach()),
        "cosine": float(cosine.detach()),
    }


def _validate(
    *,
    executor,
    reader,
    mixer,
    loaded_layer,
    layer_index,
    burn_in_tokens,
    micro_batch_size,
    loss_weights,
    distributed,
):
    local_sums: dict[str, float] = {}
    local_row_count = 0
    global_micro_batch_size = micro_batch_size * distributed.world_size
    with torch.no_grad():
        start = 0
        while start < reader.row_count:
            stop = min(start + global_micro_batch_size, reader.row_count)
            trailing = reader.row_count - stop
            if 0 < trailing < distributed.world_size:
                stop = reader.row_count
            global_rows = tuple(range(start, stop))
            local_rows = distributed.shard_rows(global_rows)
            cached = reader.read_rows(local_rows)
            output = executor.forward_cached_layer_local(
                cached.hidden_states,
                shared_states=cached.shared_states,
                active_layer_index=layer_index,
                active_mixer=mixer,
                loaded_layer=loaded_layer,
            )
            _, metrics = _local_loss(output, burn_in_tokens, loss_weights)
            for key, value in metrics.items():
                local_sums[key] = local_sums.get(key, 0.0) + len(local_rows) * value
            local_row_count += len(local_rows)
            start = stop
    if local_row_count <= 0:
        raise ContractError("validation shard produced no rows")
    local_means = {key: value / local_row_count for key, value in local_sums.items()}
    return distributed.aggregate_metrics(local_means, local_row_count)


def _validate_gdn_head_geometry(*, mixer, loaded_layer) -> None:
    """Reject any target that repartitions the source GDN recurrent state."""
    source_mixer = loaded_layer.module.linear_attn
    source_geometry = (
        int(source_mixer.num_v_heads),
        int(source_mixer.head_k_dim),
        int(source_mixer.head_v_dim),
    )
    target_geometry = (
        int(mixer.num_heads),
        int(mixer.head_dim),
        int(mixer.head_dim),
    )
    if source_geometry != target_geometry:
        raise ContractError(
            "lossy GDN head/state repartition is forbidden: "
            f"source={source_geometry} target={target_geometry}"
        )


class _LayerInputReaderView:
    """A deterministic row-local view over an immutable layer-input cache."""

    def __init__(self, parent, row_indices, *, role: str) -> None:
        self._parent = parent
        self._row_indices = tuple(int(value) for value in row_indices)
        if not self._row_indices:
            raise ContractError("layer-input reader view requires at least one row")
        if len(set(self._row_indices)) != len(self._row_indices):
            raise ContractError("layer-input reader view contains duplicate rows")
        if min(self._row_indices) < 0 or max(self._row_indices) >= parent.row_count:
            raise ContractError("layer-input reader view row is out of range")
        self.cache_dir = parent.cache_dir
        self.manifest = dict(parent.manifest)
        binding = dict(self.manifest.get("binding", {}))
        binding.update(
            {
                "row_subset_role": role,
                "parent_row_count": parent.row_count,
                "parent_row_indices": list(self._row_indices),
                "parent_row_indices_sha256": _sha256_json(
                    list(self._row_indices)
                ),
                "parent_row_identity_sha256": _sha256_json(
                    [
                        {
                            "dataset_manifest_sha256": binding.get(
                                "dataset_manifest_sha256"
                            ),
                            "split": binding.get("split"),
                            "source_sample_ids_sha256": binding.get(
                                "source_sample_ids_sha256"
                            ),
                            "row_index": row_index,
                        }
                        for row_index in self._row_indices
                    ]
                ),
            }
        )
        self.manifest.update(
            {
                "row_count": len(self._row_indices),
                "binding": binding,
            }
        )

    @property
    def row_count(self) -> int:
        return len(self._row_indices)

    def read_rows(self, row_indices) -> LayerInputBatch:
        requested = tuple(int(value) for value in row_indices)
        if not requested:
            raise ContractError("layer-input reader view requires at least one row")
        try:
            parent_rows = tuple(self._row_indices[value] for value in requested)
        except IndexError as error:
            raise ContractError(
                "layer-input reader view row is out of range"
            ) from error
        if min(requested) < 0:
            raise ContractError("layer-input reader view row is out of range")
        batch = self._parent.read_rows(parent_rows)
        return LayerInputBatch(
            row_indices=torch.tensor(
                requested,
                dtype=batch.row_indices.dtype,
                device=batch.row_indices.device,
            ),
            hidden_states=batch.hidden_states,
            shared_states=batch.shared_states,
        )


def _split_gqa_validation_protocol(reader, *, world_size: int):
    """Reserve disjoint installation and epoch-validation row identities."""
    binding = reader.manifest.get("binding")
    if (
        not isinstance(binding, dict)
        or binding.get("split") != "validation"
        or not isinstance(binding.get("dataset_manifest_sha256"), str)
        or not isinstance(binding.get("source_sample_ids_sha256"), str)
        or len(binding["source_sample_ids_sha256"]) != 64
    ):
        raise ContractError(
            "GQA validation protocol requires a sample-identity-bound "
            "validation cache"
        )
    if world_size <= 0 or reader.row_count < 2 * world_size:
        raise ContractError(
            "GQA validation protocol requires at least two rows per rank"
        )
    installation_rows = tuple(range(0, reader.row_count, 2))
    epoch_rows = tuple(range(1, reader.row_count, 2))
    if min(len(installation_rows), len(epoch_rows)) < world_size:
        raise ContractError(
            "GQA validation protocol produced an undersized distributed split"
        )
    return (
        _LayerInputReaderView(
            reader,
            installation_rows,
            role="gqa-native-zero-step-installation",
        ),
        _LayerInputReaderView(
            reader,
            epoch_rows,
            role="layerwise-epoch-selection",
        ),
    )


def _gqa_native_zero_step_geometry(*, mixer, loaded_layer):
    source_mixer = getattr(loaded_layer.module, "self_attn", None)
    if source_mixer is None or not hasattr(source_mixer, "q_proj"):
        return None
    query_heads = int(source_mixer.config.num_attention_heads)
    key_value_heads = int(source_mixer.config.num_key_value_heads)
    source_head_dim = int(source_mixer.head_dim)
    if key_value_heads >= query_heads:
        return None
    if (
        source_head_dim != 2 * int(mixer.head_dim)
        or int(mixer.num_heads) != 2 * query_heads
    ):
        return None
    if (
        int(mixer.rope_num_heads) != query_heads
        or int(mixer.rope_head_dim) != source_head_dim
    ):
        raise ContractError(
            "GQA native zero-step requires preserved source RoPE head geometry"
        )
    return {
        "query_heads": query_heads,
        "key_value_heads": key_value_heads,
        "source_head_dim": source_head_dim,
    }


def _attention_context_lengths(reader, burn_in_tokens):
    sequence_length = int(reader.manifest["sequence_length"])
    supervised = sequence_length - int(burn_in_tokens)
    if supervised < 3:
        raise ContractError(
            "attention activation fitting requires three context lengths"
        )
    return tuple(
        sorted(
            {
                int(burn_in_tokens) + 1,
                int(burn_in_tokens) + max(2, supervised // 2),
                sequence_length,
            }
        )
    )


def _iter_activation_fit_batches(
    reader, local_rows, *, source_layer_type, burn_in_tokens
):
    """Yield full GDN batches or deterministic ragged attention context groups."""
    if source_layer_type != "full_attention":
        yield reader.read_rows(local_rows)
        return
    lengths = _attention_context_lengths(reader, burn_in_tokens)
    grouped = {length: [] for length in lengths}
    for row in local_rows:
        grouped[lengths[int(row) % len(lengths)]].append(int(row))
    for length in lengths:
        rows = grouped[length]
        if not rows:
            continue
        cached = reader.read_rows(rows)
        yield LayerInputBatch(
            cached.row_indices,
            cached.hidden_states[:, :length],
            (
                None
                if cached.shared_states is None
                else cached.shared_states[:, :length]
            ),
        )


def _require_independent_activation_fit_caches(train_reader, validation_reader) -> None:
    train_binding = train_reader.manifest.get("binding")
    validation_binding = validation_reader.manifest.get("binding")
    if (
        not isinstance(train_binding, dict)
        or not isinstance(validation_binding, dict)
        or train_binding.get("split") != "distill_train"
        or validation_binding.get("split") != "validation"
        or train_binding == validation_binding
    ):
        raise ContractError(
            "activation fitting requires non-empty, mutually exclusive "
            "distill_train and validation cache bindings"
        )


def _activation_fit_source_qkv_projections(
    *,
    executor,
    train_reader,
    validation_reader,
    mixer,
    loaded_layer,
    layer_index,
    source_layer_type,
    burn_in_tokens,
    fit_rows,
    ridge,
    micro_batch_size,
    loss_weights,
    chain_baseline,
    run_dir,
    distributed,
    fit_stage="initial",
    defer_chain_acceptance=False,
):
    """Absorb source post-activation Q/K/V traces into native projections."""
    _require_independent_activation_fit_caches(train_reader, validation_reader)
    fit_rows = int(fit_rows)
    if fit_rows > train_reader.row_count or fit_rows < distributed.world_size:
        raise ContractError("QKV activation-fit rows violate distributed capacity")
    is_gdn = source_layer_type == "linear_attention"
    source_mixer = getattr(
        loaded_layer.module, "linear_attn" if is_gdn else "self_attn", None
    )
    if source_mixer is None:
        raise ContractError("QKV activation fitting requires the selected source mixer")
    target_width = int(mixer.r_proj.out_features)
    if is_gdn:
        source_heads = int(source_mixer.num_v_heads)
        source_key_dim = int(source_mixer.head_k_dim)
        source_value_dim = int(source_mixer.head_v_dim)
    else:
        source_heads = int(source_mixer.config.num_attention_heads)
        source_key_dim = source_value_dim = int(source_mixer.head_dim)
    source_widths = {
        "r": source_heads * source_key_dim,
        "k": source_heads * source_key_dim,
        "v": source_heads * source_value_dim,
    }
    report_prefix = "gdn" if is_gdn else "attention"
    boundary = (
        "gdn-headnorm-equivalent-signals-to-native-projections-v4"
        if is_gdn
        else "attention-norm-qkv-to-native-projections-v1"
    )
    report_name = (
        f"{report_prefix}-qkv-layer-{layer_index:03d}.json"
        if fit_stage == "initial"
        else f"{report_prefix}-qkv-{fit_stage}-layer-{layer_index:03d}.json"
    )
    report_path = run_dir / "activation-fit" / report_name
    if is_gdn:
        _validate_gdn_head_geometry(mixer=mixer, loaded_layer=loaded_layer)
    if any(width != target_width for width in source_widths.values()):
        raise ContractError(
            "QKV activation-fit source recurrence-signal geometry must equal "
            f"the native target width: source={source_widths} target={target_width}"
        )
    validation_before = _validate_source_qkv_projections(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        source_layer_type=source_layer_type,
        distributed=distributed,
    )
    layer_validation_before = _validate(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        loss_weights=loss_weights,
        distributed=distributed,
    )
    originals = {
        role: getattr(mixer, f"{role}_proj").weight.detach().clone()
        for role in ("r", "k", "v")
    }
    solver_metrics: dict[str, object] = {}
    statistics_hashes: dict[str, object] = {}
    rank_contributions: dict[str, object] = {}
    proposed_weight_hashes: dict[str, str] = {}
    teacher_roles = (
        (("r", "recurrent_r"), ("k", "write_key"), ("v", "write_value"))
        if is_gdn
        else (("r", "q"), ("k", "k"), ("v", "v"))
    )
    for role, teacher_name in teacher_roles:
        gram = rhs = target_squared_sum = None
        local_row_count = 0
        local_row_indices: list[int] = []
        local_tokens = 0
        global_micro_batch_size = micro_batch_size * distributed.world_size
        start = 0
        while start < fit_rows:
            stop = min(start + global_micro_batch_size, fit_rows)
            if 0 < fit_rows - stop < distributed.world_size:
                stop = fit_rows
            local_rows = distributed.shard_rows(tuple(range(start, stop)))
            local_row_count += len(local_rows)
            local_row_indices.extend(local_rows)
            for cached in _iter_activation_fit_batches(
                train_reader,
                local_rows,
                source_layer_type=source_layer_type,
                burn_in_tokens=burn_in_tokens,
            ):
                with torch.no_grad():
                    output = executor.forward_cached_layer_local(
                        cached.hidden_states,
                        shared_states=cached.shared_states,
                        active_layer_index=layer_index,
                        active_mixer=mixer,
                        loaded_layer=loaded_layer,
                    )
                if (
                    output.teacher_signals is None
                    or teacher_name not in output.teacher_signals
                ):
                    raise ContractError(
                        f"source mixer did not expose {teacher_name} trace"
                    )
                features = output.student_signals[f"mixed_{role}"][
                    :, burn_in_tokens:
                ].flatten(0, 1)
                targets = (
                    output.teacher_signals[teacher_name][:, burn_in_tokens:]
                    .flatten(0, 1)
                    .float()
                )
                if is_gdn and role == "r":
                    targets = targets * float(source_key_dim) ** 0.5
                if role == "r" and not is_gdn:
                    targets = targets / float(source_key_dim) ** 0.5
                stats = teacher_trace_normal_equations(features, targets)
                gram = stats.gram if gram is None else gram + stats.gram
                rhs = stats.rhs if rhs is None else rhs + stats.rhs
                target_squared_sum = (
                    stats.target_squared_sum
                    if target_squared_sum is None
                    else target_squared_sum + stats.target_squared_sum
                )
                local_tokens += stats.tokens
            start = stop
        if gram is None or rhs is None or target_squared_sum is None:
            raise ContractError("GDN QKV fitting produced no sufficient statistics")
        rank_contributions[role] = list(
            distributed.all_gather_objects(
                {
                    "rank": distributed.rank,
                    "rows": local_row_count,
                    "tokens": local_tokens,
                    "row_indices_sha256": _sha256_json(local_row_indices),
                    "local_gram_sha256": _tensor_sha256(gram),
                    "local_rhs_sha256": _tensor_sha256(rhs),
                    "local_target_squared_sum_sha256": _tensor_sha256(
                        target_squared_sum
                    ),
                }
            )
        )
        distributed.all_reduce_sum(gram)
        distributed.all_reduce_sum(rhs)
        distributed.all_reduce_sum(target_squared_sum)
        token_count = torch.tensor(local_tokens, dtype=torch.int64, device=gram.device)
        distributed.all_reduce_sum(token_count)
        fitted_weight = torch.empty_like(
            getattr(mixer, f"{role}_proj").weight, dtype=torch.float32
        )
        metrics = None
        if distributed.is_primary:
            fit = solve_teacher_trace_normal_equations(
                TraceNormalEquations(
                    gram, rhs, target_squared_sum, int(token_count.item())
                ),
                ridge=ridge,
            )
            fitted_weight.copy_(fit.weight)
            metrics = {
                "normalized_mse": fit.normalized_mse,
                "cosine": fit.cosine,
                "tokens": int(token_count.item()),
            }
            statistics_hashes[role] = {
                "gram_sha256": _tensor_sha256(gram),
                "rhs_sha256": _tensor_sha256(rhs),
                "target_squared_sum_sha256": _tensor_sha256(target_squared_sum),
            }
        distributed.broadcast_tensor(fitted_weight)
        metrics = distributed.broadcast_object(metrics)
        solver_metrics[role] = metrics
        proposed_weight_hashes[role] = _tensor_sha256(
            fitted_weight.to(getattr(mixer, f"{role}_proj").weight.dtype)
        )
        with torch.no_grad():
            getattr(mixer, f"{role}_proj").weight.copy_(
                fitted_weight.to(getattr(mixer, f"{role}_proj").weight.dtype)
            )
    validation_after = _validate_source_qkv_projections(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        source_layer_type=source_layer_type,
        distributed=distributed,
    )
    layer_validation_after = _validate(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        loss_weights=loss_weights,
        distributed=distributed,
    )
    before_score = sum(
        float(validation_before[role]["normalized_mse"]) for role in ("r", "k", "v")
    )
    after_score = sum(
        float(validation_after[role]["normalized_mse"]) for role in ("r", "k", "v")
    )
    every_head_non_regressed = all(
        float(after) <= float(before)
        for role in ("r", "k", "v")
        for before, after in zip(
            validation_before[role]["per_target_head_normalized_mse"],
            validation_after[role]["per_target_head_normalized_mse"],
            strict=True,
        )
    )
    improved = (
        math.isfinite(after_score)
        and after_score < before_score
        and every_head_non_regressed
        and layer_validation_after["mixer_normalized_mse"]
        <= layer_validation_before["mixer_normalized_mse"]
    )
    if not improved and not defer_chain_acceptance:
        with torch.no_grad():
            for role, weight in originals.items():
                getattr(mixer, f"{role}_proj").weight.copy_(weight)
    selected_parameter_hashes = distributed.all_gather_objects(
        {
            "rank": distributed.rank,
            "weights": {
                role: _tensor_sha256(getattr(mixer, f"{role}_proj").weight)
                for role in ("r", "k", "v")
            },
        }
    )
    if any(
        row["weights"] != selected_parameter_hashes[0]["weights"]
        for row in selected_parameter_hashes[1:]
    ):
        raise ContractError("QKV-fit parameters differ across distributed ranks")
    if distributed.is_primary:
        write_json(
            report_path,
            {
                "schema_version": 1,
                "status": (
                    "deferred"
                    if defer_chain_acceptance
                    else ("accepted" if improved else "rejected")
                ),
                "layer": layer_index,
                "boundary": boundary,
                "fit_stage": fit_stage,
                "selection_scope": (
                    "dependency-transaction" if defer_chain_acceptance else "component"
                ),
                "world_size": distributed.world_size,
                "fit_rows": fit_rows,
                "ridge": ridge,
                "source_geometry": {
                    "query_heads": source_heads,
                    "key_value_heads": (
                        int(source_mixer.num_k_heads)
                        if is_gdn
                        else int(source_mixer.config.num_key_value_heads)
                    ),
                    "key_head_dim": source_key_dim,
                    "value_head_dim": source_value_dim,
                    "kv_repeat": (
                        int(source_mixer.num_v_heads // source_mixer.num_k_heads)
                        if is_gdn
                        else int(source_mixer.num_key_value_groups)
                    ),
                },
                "target_geometry": {
                    "heads": int(mixer.num_heads),
                    "head_dim": int(mixer.head_dim),
                },
                "target_signals": {
                    role: (
                        "recurrent_r*sqrt(head_dim)"
                        if is_gdn and role == "r"
                        else teacher_name
                    )
                    for role, teacher_name in teacher_roles
                },
                "recurrent_r_equivalence": (
                    "L2-normalized query without the source 1/sqrt(head_dim) "
                    "factor; positive per-head scale is removed by the "
                    "post-recurrence headwise normalization"
                    if is_gdn
                    else None
                ),
                "train_cache_binding": train_reader.manifest.get("binding"),
                "validation_cache_binding": validation_reader.manifest.get("binding"),
                "fit_context_lengths": (
                    list(_attention_context_lengths(train_reader, burn_in_tokens))
                    if not is_gdn
                    else [int(train_reader.manifest["sequence_length"])]
                ),
                "validation_context_lengths": (
                    list(_attention_context_lengths(validation_reader, burn_in_tokens))
                    if not is_gdn
                    else [int(validation_reader.manifest["sequence_length"])]
                ),
                "solver": "bias-free-normal-equations-ridge-v1",
                "rank_contributions": rank_contributions,
                "fit": solver_metrics,
                "validation_before": validation_before,
                "validation_after": validation_after,
                "layer_validation_before": layer_validation_before,
                "layer_validation_after": layer_validation_after,
                "chain_acceptance_baseline": chain_baseline,
                "every_head_non_regressed": every_head_non_regressed,
                "normal_equations": statistics_hashes,
                "original_weight_sha256": {
                    role: _tensor_sha256(originals[role]) for role in ("r", "k", "v")
                },
                "proposed_weight_sha256": proposed_weight_hashes,
                "selected_weight_sha256": {
                    role: _tensor_sha256(getattr(mixer, f"{role}_proj").weight)
                    for role in ("r", "k", "v")
                },
                "selected_parameter_hashes": list(selected_parameter_hashes),
            },
        )
    distributed.barrier()
    if improved or defer_chain_acceptance:
        return chain_baseline or layer_validation_before
    return chain_baseline


def _activation_fit_native_a_projection(
    *,
    executor,
    train_reader,
    validation_reader,
    mixer,
    loaded_layer,
    layer_index,
    burn_in_tokens,
    fit_rows,
    ridge,
    micro_batch_size,
    loss_weights,
    chain_baseline,
    run_dir,
    distributed,
):
    """Fit native ``a_lora`` to the GDN erase rate ``beta * decay``."""
    _require_independent_activation_fit_caches(train_reader, validation_reader)
    report_path = run_dir / "activation-fit" / f"gdn-a-layer-{layer_index:03d}.json"
    source_mixer = loaded_layer.module.linear_attn
    source_heads = int(source_mixer.num_v_heads)
    target_heads = int(mixer.num_heads)
    _validate_gdn_head_geometry(mixer=mixer, loaded_layer=loaded_layer)
    source_beta_weight = source_mixer.in_proj_b.weight.detach().float()
    source_decay_weight = source_mixer.in_proj_a.weight.detach().float()
    if source_beta_weight.shape != source_decay_weight.shape or (
        source_beta_weight.shape[0] != source_heads
    ):
        raise ContractError("source GDN beta/decay projection basis is malformed")
    validation_before = _validate_source_erase_rate_projection(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        distributed=distributed,
    )
    layer_validation_before = _validate(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        loss_weights=loss_weights,
        distributed=distributed,
    )
    fit_rows = int(fit_rows)
    if fit_rows > train_reader.row_count or fit_rows < distributed.world_size:
        raise ContractError(
            "erase-rate activation-fit rows violate distributed capacity"
        )
    gram = rhs = target_squared_sum = None
    local_row_count = 0
    local_row_indices: list[int] = []
    local_tokens = 0
    global_micro_batch_size = micro_batch_size * distributed.world_size
    start = 0
    while start < fit_rows:
        stop = min(start + global_micro_batch_size, fit_rows)
        if 0 < fit_rows - stop < distributed.world_size:
            stop = fit_rows
        local_rows = distributed.shard_rows(tuple(range(start, stop)))
        local_row_count += len(local_rows)
        local_row_indices.extend(local_rows)
        for cached in _iter_activation_fit_batches(
            train_reader,
            local_rows,
            source_layer_type="linear_attention",
            burn_in_tokens=burn_in_tokens,
        ):
            with torch.no_grad():
                output = executor.forward_cached_layer_local(
                    cached.hidden_states,
                    shared_states=cached.shared_states,
                    active_layer_index=layer_index,
                    active_mixer=mixer,
                    loaded_layer=loaded_layer,
                )
            if (
                output.teacher_signals is None
                or "erase_rate" not in output.teacher_signals
            ):
                raise ContractError(
                    "source GDN did not expose the beta*decay erase rate"
                )
            mixed_a = output.student_signals["mixed_a"][:, burn_in_tokens:].flatten(
                0, 1
            )
            features = torch.cat(
                (
                    torch.nn.functional.linear(mixed_a.float(), source_beta_weight),
                    torch.nn.functional.linear(mixed_a.float(), source_decay_weight),
                ),
                dim=-1,
            )
            features = torch.cat((features, torch.ones_like(features[:, :1])), dim=-1)
            targets = torch.logit(
                (
                    output.teacher_signals["beta"][:, burn_in_tokens:]
                    * output.teacher_signals["decay"][:, burn_in_tokens:]
                )
                .float()
                .clamp(1e-6, 1.0 - 1e-6)
            ).flatten(0, 1)
            stats = teacher_trace_normal_equations(features, targets)
            gram = stats.gram if gram is None else gram + stats.gram
            rhs = stats.rhs if rhs is None else rhs + stats.rhs
            target_squared_sum = (
                stats.target_squared_sum
                if target_squared_sum is None
                else target_squared_sum + stats.target_squared_sum
            )
            local_tokens += stats.tokens
        start = stop
    if gram is None or rhs is None or target_squared_sum is None:
        raise ContractError("erase-rate fitting produced no sufficient statistics")
    rank_contributions = distributed.all_gather_objects(
        {
            "rank": distributed.rank,
            "rows": local_row_count,
            "tokens": local_tokens,
            "row_indices_sha256": _sha256_json(local_row_indices),
            "local_gram_sha256": _tensor_sha256(gram),
            "local_rhs_sha256": _tensor_sha256(rhs),
            "local_target_squared_sum_sha256": _tensor_sha256(target_squared_sum),
        }
    )
    distributed.all_reduce_sum(gram)
    distributed.all_reduce_sum(rhs)
    distributed.all_reduce_sum(target_squared_sum)
    token_count = torch.tensor(local_tokens, dtype=torch.int64, device=gram.device)
    distributed.all_reduce_sum(token_count)
    down = mixer.a_lora.lora[0]
    up = mixer.a_lora.lora[2]
    if up.bias is None:
        raise ContractError("native erase-rate up projection requires bias")
    if down.out_features < source_heads:
        raise ContractError(
            "native a_lora rank is smaller than the source value-head count"
        )
    original_down_weight = down.weight.detach().clone()
    original_up_weight = up.weight.detach().clone()
    original_up_bias = up.bias.detach().clone()
    fitted_down_weight = torch.zeros_like(original_down_weight, dtype=torch.float32)
    fitted_up_weight = torch.zeros_like(original_up_weight, dtype=torch.float32)
    fitted_up_bias = torch.empty_like(original_up_bias, dtype=torch.float32)
    fit_metrics = None
    if distributed.is_primary:
        fit = solve_teacher_trace_normal_equations(
            TraceNormalEquations(
                gram, rhs, target_squared_sum, int(token_count.item())
            ),
            ridge=ridge,
        )
        basis_weight = fit.weight[:, :-1]
        fitted_down_weight[:source_heads].copy_(
            basis_weight[:, :source_heads] @ source_beta_weight
            + basis_weight[:, source_heads:] @ source_decay_weight
        )
        for head in range(source_heads):
            fitted_up_weight[
                head * mixer.head_dim : (head + 1) * mixer.head_dim,
                head,
            ] = 1
        fitted_up_bias.copy_(fit.weight[:, -1].repeat_interleave(mixer.head_dim))
        fit_metrics = {
            "erase_logit_normalized_mse": fit.normalized_mse,
            "erase_logit_cosine": fit.cosine,
            "tokens": int(token_count.item()),
        }
    distributed.broadcast_tensor(fitted_down_weight)
    distributed.broadcast_tensor(fitted_up_weight)
    distributed.broadcast_tensor(fitted_up_bias)
    fit_metrics = distributed.broadcast_object(fit_metrics)
    with torch.no_grad():
        down.weight.copy_(fitted_down_weight.to(down.weight.dtype))
        up.weight.copy_(fitted_up_weight.to(up.weight.dtype))
        up.bias.copy_(fitted_up_bias.to(up.bias.dtype))
    validation_after = _validate_source_erase_rate_projection(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        distributed=distributed,
    )
    layer_validation_after = _validate(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        loss_weights=loss_weights,
        distributed=distributed,
    )
    every_head_non_regressed = all(
        float(after) <= float(before)
        for before, after in zip(
            validation_before["per_head_normalized_mse"],
            validation_after["per_head_normalized_mse"],
            strict=True,
        )
    )
    improved = (
        math.isfinite(validation_after["normalized_mse"])
        and validation_after["normalized_mse"] < validation_before["normalized_mse"]
        and every_head_non_regressed
        and layer_validation_after["mixer_normalized_mse"]
        <= layer_validation_before["mixer_normalized_mse"]
    )
    if not improved:
        with torch.no_grad():
            down.weight.copy_(original_down_weight)
            up.weight.copy_(original_up_weight)
            up.bias.copy_(original_up_bias)
    selected_hashes = distributed.all_gather_objects(
        {
            "rank": distributed.rank,
            "down_weight_sha256": _tensor_sha256(down.weight),
            "up_weight_sha256": _tensor_sha256(up.weight),
            "up_bias_sha256": _tensor_sha256(up.bias),
        }
    )
    if any(
        row["down_weight_sha256"] != selected_hashes[0]["down_weight_sha256"]
        or row["up_weight_sha256"] != selected_hashes[0]["up_weight_sha256"]
        or row["up_bias_sha256"] != selected_hashes[0]["up_bias_sha256"]
        for row in selected_hashes[1:]
    ):
        raise ContractError("erase-rate fit parameters differ across distributed ranks")
    if distributed.is_primary:
        write_json(
            report_path,
            {
                "schema_version": 1,
                "status": "accepted" if improved else "rejected",
                "layer": layer_index,
                "fit_stage": "initial",
                "selection_scope": "component",
                "boundary": "gdn-beta-times-decay-to-native-erase-v3",
                "world_size": distributed.world_size,
                "fit_rows": fit_rows,
                "source_heads": source_heads,
                "target_heads": target_heads,
                "ridge": ridge,
                "solver": "source-beta-decay-basis-ridge-v1",
                "basis_width": source_heads * 2,
                "train_cache_binding": train_reader.manifest.get("binding"),
                "validation_cache_binding": validation_reader.manifest.get("binding"),
                "rank_contributions": list(rank_contributions),
                "fit": fit_metrics,
                "validation_before": validation_before,
                "validation_after": validation_after,
                "layer_validation_before": layer_validation_before,
                "layer_validation_after": layer_validation_after,
                "every_head_non_regressed": every_head_non_regressed,
                "chain_acceptance_baseline": chain_baseline,
                "normal_equations": {
                    "gram_sha256": _tensor_sha256(gram),
                    "rhs_sha256": _tensor_sha256(rhs),
                    "target_squared_sum_sha256": _tensor_sha256(target_squared_sum),
                },
                "original_down_weight_sha256": _tensor_sha256(original_down_weight),
                "original_up_weight_sha256": _tensor_sha256(original_up_weight),
                "original_up_bias_sha256": _tensor_sha256(original_up_bias),
                "proposed_down_weight_sha256": _tensor_sha256(
                    fitted_down_weight.to(down.weight.dtype)
                ),
                "proposed_up_weight_sha256": _tensor_sha256(
                    fitted_up_weight.to(up.weight.dtype)
                ),
                "proposed_up_bias_sha256": _tensor_sha256(
                    fitted_up_bias.to(up.bias.dtype)
                ),
                "selected_down_weight_sha256": _tensor_sha256(down.weight),
                "selected_up_weight_sha256": _tensor_sha256(up.weight),
                "selected_up_bias_sha256": _tensor_sha256(up.bias),
                "selected_parameter_hashes": list(selected_hashes),
            },
        )
    distributed.barrier()
    if improved:
        return chain_baseline or layer_validation_before
    return chain_baseline


def _activation_fit_source_gate_projection(
    *,
    executor,
    train_reader,
    validation_reader,
    mixer,
    loaded_layer,
    layer_index,
    source_layer_type,
    burn_in_tokens,
    fit_rows,
    ridge,
    functional_steps,
    functional_learning_rate,
    micro_batch_size,
    loss_weights,
    chain_baseline,
    run_dir,
    distributed,
    fit_stage="initial",
    defer_chain_acceptance=False,
):
    """Fit native gate up-projection to source post-activation gate traces."""
    is_gdn = source_layer_type == "linear_attention"
    source_mixer = getattr(
        loaded_layer.module, "linear_attn" if is_gdn else "self_attn", None
    )
    if source_mixer is None:
        raise ContractError(
            "gate activation fitting requires the selected source mixer"
        )
    target_width = int(mixer.g_lora.lora[2].out_features)
    source_width = (
        int(source_mixer.value_dim)
        if is_gdn
        else int(source_mixer.config.num_attention_heads * source_mixer.head_dim)
    )
    prefix = "gdn" if is_gdn else "attention"
    boundary = (
        "gdn-silu-z-to-native-gate-up-v1"
        if is_gdn
        else "attention-sigmoid-q-gate-to-native-gate-up-v1"
    )
    report_name = (
        f"{prefix}-gate-layer-{layer_index:03d}.json"
        if fit_stage == "initial"
        else f"{prefix}-gate-{fit_stage}-layer-{layer_index:03d}.json"
    )
    report_path = run_dir / "activation-fit" / report_name
    if source_width != target_width:
        raise ContractError(
            "source/target gate width mismatch is forbidden: "
            f"source={source_width} target={target_width}"
        )
    validation_before = _validate_source_gate_projection(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        source_layer_type=source_layer_type,
        distributed=distributed,
    )
    layer_validation_before = _validate(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        loss_weights=loss_weights,
        distributed=distributed,
    )
    fit_rows = int(fit_rows)
    if fit_rows > train_reader.row_count or fit_rows < distributed.world_size:
        raise ContractError("gate activation-fit rows violate distributed capacity")
    global_micro_batch_size = micro_batch_size * distributed.world_size

    def collect_gate_statistics():
        gram = rhs = target_squared_sum = None
        local_tokens = 0
        start = 0
        while start < fit_rows:
            stop = min(start + global_micro_batch_size, fit_rows)
            if 0 < fit_rows - stop < distributed.world_size:
                stop = fit_rows
            local_rows = distributed.shard_rows(tuple(range(start, stop)))
            for cached in _iter_activation_fit_batches(
                train_reader,
                local_rows,
                source_layer_type=source_layer_type,
                burn_in_tokens=burn_in_tokens,
            ):
                with torch.no_grad():
                    output = executor.forward_cached_layer_local(
                        cached.hidden_states,
                        shared_states=cached.shared_states,
                        active_layer_index=layer_index,
                        active_mixer=mixer,
                        loaded_layer=loaded_layer,
                    )
                if (
                    output.teacher_signals is None
                    or "gate" not in output.teacher_signals
                ):
                    raise ContractError("source mixer did not expose gate trace")
                features = output.student_signals["g_features"][
                    :, burn_in_tokens:
                ].flatten(0, 1)
                targets = output.teacher_signals["gate"][:, burn_in_tokens:].flatten(
                    0, 1
                )
                stats = teacher_trace_normal_equations(features, targets)
                gram = stats.gram if gram is None else gram + stats.gram
                rhs = stats.rhs if rhs is None else rhs + stats.rhs
                target_squared_sum = (
                    stats.target_squared_sum
                    if target_squared_sum is None
                    else target_squared_sum + stats.target_squared_sum
                )
                local_tokens += stats.tokens
            start = stop
        if gram is None or rhs is None or target_squared_sum is None:
            raise ContractError("gate fitting produced no sufficient statistics")
        distributed.all_reduce_sum(gram)
        distributed.all_reduce_sum(rhs)
        distributed.all_reduce_sum(target_squared_sum)
        token_count = torch.tensor(local_tokens, dtype=torch.int64, device=gram.device)
        distributed.all_reduce_sum(token_count)
        return gram, rhs, target_squared_sum, token_count

    gram, rhs, target_squared_sum, token_count = collect_gate_statistics()
    gate_down = mixer.g_lora.lora[0].weight
    gate_up = mixer.g_lora.lora[2].weight
    original_down_weight = gate_down.detach().clone()
    original_weight = gate_up.detach().clone()
    fitted_weight = torch.empty_like(original_weight, dtype=torch.float32)
    fit_metrics = None
    if distributed.is_primary:
        fit = solve_teacher_trace_normal_equations(
            TraceNormalEquations(
                gram, rhs, target_squared_sum, int(token_count.item())
            ),
            ridge=ridge,
        )
        fitted_weight.copy_(fit.weight)
        fit_metrics = {
            "normalized_mse": fit.normalized_mse,
            "cosine": fit.cosine,
            "tokens": int(token_count.item()),
        }
    distributed.broadcast_tensor(fitted_weight)
    fit_metrics = distributed.broadcast_object(fit_metrics)
    with torch.no_grad():
        mixer.g_lora.lora[2].weight.copy_(
            fitted_weight.to(mixer.g_lora.lora[2].weight.dtype)
        )
    validation_after = _validate_source_gate_projection(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        source_layer_type=source_layer_type,
        distributed=distributed,
    )
    layer_validation_after = _validate(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        loss_weights=loss_weights,
        distributed=distributed,
    )
    if defer_chain_acceptance:
        improved = (
            math.isfinite(validation_after["normalized_mse"])
            and validation_after["normalized_mse"] < validation_before["normalized_mse"]
            and layer_validation_after["mixer_normalized_mse"]
            <= layer_validation_before["mixer_normalized_mse"]
        )
    else:
        improved = _retain_gate_fit_candidate(
            weight=gate_up,
            original_weight=original_weight,
            validation_before=validation_before,
            validation_after=validation_after,
            layer_validation_before=layer_validation_before,
            layer_validation_after=layer_validation_after,
        )
    functional_fit = {
        "status": "not-run" if not improved else "rejected",
        "steps": int(functional_steps),
        "learning_rate": float(functional_learning_rate),
        "gradient_scope": ["g_lora.lora.0.weight"],
        "fit_loss_history": [],
    }
    if improved or defer_chain_acceptance:
        ridge_down_weight = gate_down.detach().clone()
        ridge_up_weight = gate_up.detach().clone()
        ridge_validation_after = validation_after
        ridge_layer_validation_after = layer_validation_after
        original_requires_grad = {
            name: parameter.requires_grad
            for name, parameter in mixer.named_parameters()
        }
        optimizer = None
        try:
            for parameter in mixer.parameters():
                parameter.requires_grad_(False)
            gate_down.requires_grad_(True)
            distributed.validate_trainable_signature(mixer)
            optimizer = torch.optim.Adam(
                [gate_down], lr=float(functional_learning_rate)
            )
            for _ in range(int(functional_steps)):
                optimizer.zero_grad(set_to_none=True)
                local_rows_seen = 0
                local_loss_sum = 0.0
                start = 0
                while start < fit_rows:
                    stop = min(start + global_micro_batch_size, fit_rows)
                    if 0 < fit_rows - stop < distributed.world_size:
                        stop = fit_rows
                    local_rows = distributed.shard_rows(tuple(range(start, stop)))
                    for cached in _iter_activation_fit_batches(
                        train_reader,
                        local_rows,
                        source_layer_type=source_layer_type,
                        burn_in_tokens=burn_in_tokens,
                    ):
                        output = executor.forward_cached_layer_local(
                            cached.hidden_states,
                            shared_states=cached.shared_states,
                            active_layer_index=layer_index,
                            active_mixer=mixer,
                            loaded_layer=loaded_layer,
                        )
                        if (
                            output.teacher_signals is None
                            or "gate" not in output.teacher_signals
                        ):
                            raise ContractError(
                                "source mixer did not expose gate trace"
                            )
                        student_gate = output.student_signals["gate"][
                            :, burn_in_tokens:
                        ].float()
                        teacher_gate = output.teacher_signals["gate"][
                            :, burn_in_tokens:
                        ].float()
                        loss = (student_gate - teacher_gate).square().mean()
                        batch_rows = int(cached.hidden_states.shape[0])
                        (loss * batch_rows).backward()
                        local_rows_seen += batch_rows
                        local_loss_sum += float(loss.detach()) * batch_rows
                    start = stop
                if local_rows_seen <= 0 or gate_down.grad is None:
                    raise ContractError(
                        "functional gate fitting produced no local gradient"
                    )
                gate_down.grad.mul_(distributed.world_size / fit_rows)
                distributed.synchronize_gradients(mixer)
                torch.nn.utils.clip_grad_norm_([gate_down], max_norm=1.0)
                optimizer.step()
                loss_total = torch.tensor(
                    [local_loss_sum, float(local_rows_seen)],
                    dtype=torch.float64,
                    device=gate_down.device,
                )
                distributed.all_reduce_sum(loss_total)
                functional_fit["fit_loss_history"].append(
                    float((loss_total[0] / loss_total[1].clamp_min(1)).item())
                )
        except Exception:
            with torch.no_grad():
                gate_down.copy_(ridge_down_weight)
                gate_up.copy_(ridge_up_weight)
            raise
        finally:
            for name, parameter in mixer.named_parameters():
                parameter.requires_grad_(original_requires_grad[name])
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

        (
            functional_gram,
            functional_rhs,
            functional_target_squared_sum,
            functional_token_count,
        ) = collect_gate_statistics()
        functional_fitted_weight = torch.empty_like(
            original_weight, dtype=torch.float32
        )
        functional_fit_metrics = None
        if distributed.is_primary:
            functional_solution = solve_teacher_trace_normal_equations(
                TraceNormalEquations(
                    functional_gram,
                    functional_rhs,
                    functional_target_squared_sum,
                    int(functional_token_count.item()),
                ),
                ridge=ridge,
            )
            functional_fitted_weight.copy_(functional_solution.weight)
            functional_fit_metrics = {
                "normalized_mse": functional_solution.normalized_mse,
                "cosine": functional_solution.cosine,
                "tokens": int(functional_token_count.item()),
            }
        distributed.broadcast_tensor(functional_fitted_weight)
        functional_fit_metrics = distributed.broadcast_object(functional_fit_metrics)
        with torch.no_grad():
            gate_up.copy_(functional_fitted_weight.to(gate_up.dtype))
        functional_validation_after = _validate_source_gate_projection(
            executor=executor,
            reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            burn_in_tokens=burn_in_tokens,
            micro_batch_size=micro_batch_size,
            source_layer_type=source_layer_type,
            distributed=distributed,
        )
        functional_layer_validation_after = _validate(
            executor=executor,
            reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            burn_in_tokens=burn_in_tokens,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            distributed=distributed,
        )
        functional_improved = (
            math.isfinite(functional_validation_after["normalized_mse"])
            and functional_validation_after["normalized_mse"]
            < ridge_validation_after["normalized_mse"]
            and functional_layer_validation_after["mixer_normalized_mse"]
            <= ridge_layer_validation_after["mixer_normalized_mse"]
        )
        functional_fit.update(
            {
                "status": "accepted" if functional_improved else "rejected",
                "validation_before": ridge_validation_after,
                "validation_after": functional_validation_after,
                "layer_validation_before": ridge_layer_validation_after,
                "layer_validation_after": functional_layer_validation_after,
                "ridge_refit": functional_fit_metrics,
            }
        )
        if functional_improved:
            gram = functional_gram
            rhs = functional_rhs
            target_squared_sum = functional_target_squared_sum
            token_count = functional_token_count
            fitted_weight = functional_fitted_weight
            fit_metrics = functional_fit_metrics
            validation_after = functional_validation_after
            layer_validation_after = functional_layer_validation_after
        else:
            with torch.no_grad():
                gate_down.copy_(ridge_down_weight)
                gate_up.copy_(ridge_up_weight)
    selected_parameter_hashes = distributed.all_gather_objects(
        {
            "rank": distributed.rank,
            "down_weight_sha256": _tensor_sha256(gate_down),
            "up_weight_sha256": _tensor_sha256(gate_up),
        }
    )
    if any(
        row["down_weight_sha256"] != selected_parameter_hashes[0]["down_weight_sha256"]
        or row["up_weight_sha256"] != selected_parameter_hashes[0]["up_weight_sha256"]
        for row in selected_parameter_hashes[1:]
    ):
        raise ContractError("gate-fit parameters differ across distributed ranks")
    if distributed.is_primary:
        write_json(
            report_path,
            {
                "schema_version": 1,
                "status": (
                    "deferred"
                    if defer_chain_acceptance
                    else ("accepted" if improved else "rejected")
                ),
                "layer": layer_index,
                "boundary": boundary,
                "fit_stage": fit_stage,
                "selection_scope": (
                    "dependency-transaction" if defer_chain_acceptance else "component"
                ),
                "world_size": distributed.world_size,
                "fit_rows": fit_rows,
                "ridge": ridge,
                "train_cache_binding": train_reader.manifest.get("binding"),
                "validation_cache_binding": validation_reader.manifest.get("binding"),
                "fit_context_lengths": (
                    list(_attention_context_lengths(train_reader, burn_in_tokens))
                    if not is_gdn
                    else [int(train_reader.manifest["sequence_length"])]
                ),
                "validation_context_lengths": (
                    list(_attention_context_lengths(validation_reader, burn_in_tokens))
                    if not is_gdn
                    else [int(validation_reader.manifest["sequence_length"])]
                ),
                "fit": fit_metrics,
                "functional_fit": functional_fit,
                "validation_before": validation_before,
                "validation_after": validation_after,
                "layer_validation_before": layer_validation_before,
                "layer_validation_after": layer_validation_after,
                "chain_acceptance_baseline": chain_baseline,
                "normal_equations": {
                    "gram_sha256": _tensor_sha256(gram),
                    "rhs_sha256": _tensor_sha256(rhs),
                    "target_squared_sum_sha256": _tensor_sha256(target_squared_sum),
                },
                "original_weight_sha256": _tensor_sha256(original_weight),
                "original_down_weight_sha256": _tensor_sha256(original_down_weight),
                "selected_weight_sha256": _tensor_sha256(gate_up),
                "selected_down_weight_sha256": _tensor_sha256(gate_down),
                "selected_parameter_hashes": list(selected_parameter_hashes),
                "proposed_weight_sha256": _tensor_sha256(
                    fitted_weight.to(gate_up.dtype)
                ),
                "proposed_down_weight_sha256": _tensor_sha256(gate_down),
                "solver_fp32_weight_sha256": _tensor_sha256(fitted_weight),
            },
        )
    distributed.barrier()
    if improved or defer_chain_acceptance:
        return chain_baseline or layer_validation_before
    return chain_baseline


def _retain_gate_fit_candidate(
    *,
    weight,
    original_weight,
    validation_before,
    validation_after,
    layer_validation_before,
    layer_validation_after,
):
    """Keep an installed gate candidate only after held-out functional improvement."""
    improved = (
        math.isfinite(validation_after["normalized_mse"])
        and validation_after["normalized_mse"] < validation_before["normalized_mse"]
        and layer_validation_after["mixer_normalized_mse"]
        <= layer_validation_before["mixer_normalized_mse"]
    )
    if not improved:
        with torch.no_grad():
            weight.copy_(original_weight)
    return improved


def _activation_fit_time_mix(
    *,
    executor,
    train_reader,
    validation_reader,
    mixer,
    loaded_layer,
    layer_index,
    source_layer_type,
    burn_in_tokens,
    fit_rows,
    steps,
    learning_rate,
    micro_batch_size,
    loss_weights,
    chain_baseline,
    run_dir,
    distributed,
    fit_stage="initial",
    defer_chain_acceptance=False,
):
    """Fit the native one-token interpolation vectors that have source support."""
    stage_suffix = "" if fit_stage == "initial" else f"-{fit_stage}"
    report_path = (
        run_dir
        / "activation-fit"
        / f"time-mix{stage_suffix}-layer-{layer_index:03d}.json"
    )
    # Qwen GDN applies its q/k/v depthwise convolution before the recurrence,
    # so only those three paths have a legitimate previous-token component.
    # Decay, erase, and gate consume the current normalized token directly;
    # changing x_w/x_a/x_g would invalidate their already-fitted projections.
    # Both source mixers compute their Q/K/V projections from the current
    # normalized token.  A previous-token component is therefore an
    # architecture-adaptation candidate, not a copied source parameter.  Keep
    # the first ablation limited to the three recurrent projections: changing
    # x_w/x_a/x_g would simultaneously invalidate source-absent recurrent
    # controls and the already-fitted source gate.
    names = ("x_r", "x_k", "x_v")
    parameters = [getattr(mixer, name) for name in names]
    fit_rows = int(fit_rows)
    steps = int(steps)
    if fit_rows > train_reader.row_count or fit_rows < distributed.world_size:
        raise ContractError("time-mix activation-fit rows violate distributed capacity")
    if steps < 1 or learning_rate <= 0:
        raise ContractError("time-mix activation-fit optimizer settings are invalid")
    validation_before = _validate(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        loss_weights=loss_weights,
        distributed=distributed,
    )
    originals = {
        name: parameter.detach().clone()
        for name, parameter in zip(names, parameters, strict=True)
    }
    original_requires_grad = {
        name: parameter.requires_grad for name, parameter in mixer.named_parameters()
    }
    optimizer = None
    loss_history: list[float] = []
    rank_contributions = None
    try:
        for parameter in mixer.parameters():
            parameter.requires_grad_(False)
        for parameter in parameters:
            parameter.requires_grad_(True)
        distributed.validate_trainable_signature(mixer)
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        global_micro_batch_size = micro_batch_size * distributed.world_size
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            local_loss_sum = 0.0
            local_rows_seen = 0
            local_tokens_seen = 0
            start = 0
            while start < fit_rows:
                stop = min(start + global_micro_batch_size, fit_rows)
                if 0 < fit_rows - stop < distributed.world_size:
                    stop = fit_rows
                local_rows = distributed.shard_rows(tuple(range(start, stop)))
                for cached in _iter_activation_fit_batches(
                    train_reader,
                    local_rows,
                    source_layer_type=source_layer_type,
                    burn_in_tokens=burn_in_tokens,
                ):
                    output = executor.forward_cached_layer_local(
                        cached.hidden_states,
                        shared_states=cached.shared_states,
                        active_layer_index=layer_index,
                        active_mixer=mixer,
                        loaded_layer=loaded_layer,
                    )
                    loss, _ = _local_loss(output, burn_in_tokens, loss_weights)
                    batch_rows = int(cached.hidden_states.shape[0])
                    (loss * batch_rows).backward()
                    local_loss_sum += float(loss.detach()) * batch_rows
                    local_rows_seen += batch_rows
                    local_tokens_seen += batch_rows * max(
                        int(cached.hidden_states.shape[1]) - burn_in_tokens, 0
                    )
                start = stop
            if local_rows_seen <= 0:
                raise ContractError("time-mix activation-fit rank saw no rows")
            if step == 0:
                rank_contributions = distributed.all_gather_objects(
                    {
                        "rank": distributed.rank,
                        "rows_per_step": local_rows_seen,
                        "tokens_per_step": local_tokens_seen,
                    }
                )
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(distributed.world_size / fit_rows)
            distributed.synchronize_gradients(mixer)
            optimizer.step()
            with torch.no_grad():
                for parameter in parameters:
                    parameter.clamp_(0.0, 1.0)
            loss_total = torch.tensor(
                [local_loss_sum, float(local_rows_seen)],
                dtype=torch.float64,
                device=parameters[0].device,
            )
            distributed.all_reduce_sum(loss_total)
            loss_history.append(
                float((loss_total[0] / loss_total[1].clamp_min(1)).item())
            )
        proposed_hashes = {
            name: _tensor_sha256(parameter)
            for name, parameter in zip(names, parameters, strict=True)
        }
        validation_after = _validate(
            executor=executor,
            reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            burn_in_tokens=burn_in_tokens,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            distributed=distributed,
        )
        improved = (
            math.isfinite(validation_after["mixer_normalized_mse"])
            and validation_after["mixer_normalized_mse"]
            < validation_before["mixer_normalized_mse"]
            and validation_after["normalized_mse"]
            <= validation_before["normalized_mse"]
        )
        if not improved and not defer_chain_acceptance:
            with torch.no_grad():
                for name, parameter in zip(names, parameters, strict=True):
                    parameter.copy_(originals[name])
        selected_hashes = {
            name: _tensor_sha256(parameter)
            for name, parameter in zip(names, parameters, strict=True)
        }
        rank_parameter_hashes = distributed.all_gather_objects(
            {"rank": distributed.rank, "parameters": selected_hashes}
        )
        if any(
            row["parameters"] != rank_parameter_hashes[0]["parameters"]
            for row in rank_parameter_hashes[1:]
        ):
            raise ContractError("time-mix parameters differ across distributed ranks")
        if distributed.is_primary:
            write_json(
                report_path,
                {
                    "schema_version": 1,
                    "status": (
                        "deferred"
                        if defer_chain_acceptance
                        else ("accepted" if improved else "rejected")
                    ),
                    "layer": layer_index,
                    "fit_stage": fit_stage,
                    "selection_scope": (
                        "dependency-transaction"
                        if defer_chain_acceptance
                        else "component"
                    ),
                    "boundary": "source-mixer-output-to-native-time-mix-v1",
                    "source_layer_type": source_layer_type,
                    "world_size": distributed.world_size,
                    "fit_rows": fit_rows,
                    "steps": steps,
                    "learning_rate": learning_rate,
                    "optimizer": "Adam",
                    "gradient_scope": list(names),
                    "constraint": "elementwise-[0,1]",
                    "train_cache_binding": train_reader.manifest.get("binding"),
                    "validation_cache_binding": validation_reader.manifest.get(
                        "binding"
                    ),
                    "rank_contributions": list(rank_contributions or ()),
                    "fit_loss_history": loss_history,
                    "validation_before": validation_before,
                    "validation_after": validation_after,
                    "chain_acceptance_baseline": chain_baseline,
                    "original_parameter_sha256": {
                        name: _tensor_sha256(originals[name]) for name in names
                    },
                    "proposed_parameter_sha256": proposed_hashes,
                    "selected_parameter_sha256": selected_hashes,
                    "rank_parameter_hashes": list(rank_parameter_hashes),
                },
            )
    except Exception:
        with torch.no_grad():
            for name, parameter in zip(names, parameters, strict=True):
                parameter.copy_(originals[name])
        raise
    finally:
        for name, parameter in mixer.named_parameters():
            parameter.requires_grad_(original_requires_grad[name])
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
    distributed.barrier()
    if improved or defer_chain_acceptance:
        return chain_baseline or validation_before
    return chain_baseline


def _activation_fit_norm_affine(
    *,
    executor,
    train_reader,
    validation_reader,
    mixer,
    loaded_layer,
    layer_index,
    source_layer_type,
    burn_in_tokens,
    fit_rows,
    ridge,
    micro_batch_size,
    loss_weights,
    chain_baseline,
    run_dir,
    distributed,
    fit_stage="initial",
    defer_chain_acceptance=False,
):
    """Fit native GroupNorm affine terms to the source pre-output feature."""
    stage_suffix = "" if fit_stage == "initial" else f"-{fit_stage}"
    report_path = (
        run_dir
        / "activation-fit"
        / f"norm-affine{stage_suffix}-layer-{layer_index:03d}.json"
    )
    source_mixer = getattr(
        loaded_layer.module,
        "linear_attn" if source_layer_type == "linear_attention" else "self_attn",
    )
    source_output_projection = getattr(source_mixer, "out_proj", None)
    if source_output_projection is None:
        source_output_projection = getattr(source_mixer, "o_proj", None)
    if source_output_projection is None:
        raise ContractError("source mixer does not expose an output projection")
    source_width = int(source_output_projection.in_features)
    target_width = int(mixer.g_norm.weight.numel())
    if source_width != target_width:
        raise ContractError(
            "source/target normalization width mismatch is forbidden: "
            f"source={source_width} target={target_width}"
        )
    validation_before = _validate_norm_affine(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        source_layer_type=source_layer_type,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        distributed=distributed,
    )
    layer_validation_before = _validate(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        loss_weights=loss_weights,
        distributed=distributed,
    )
    statistics = torch.zeros(
        6,
        target_width,
        dtype=torch.float64,
        device=mixer.g_norm.weight.device,
    )
    fit_rows = int(fit_rows)
    if fit_rows > train_reader.row_count or fit_rows < distributed.world_size:
        raise ContractError(
            "norm affine activation-fit rows violate distributed capacity"
        )
    global_micro_batch_size = micro_batch_size * distributed.world_size
    local_row_count = 0
    local_token_count = 0
    start = 0
    while start < fit_rows:
        stop = min(start + global_micro_batch_size, fit_rows)
        if 0 < fit_rows - stop < distributed.world_size:
            stop = fit_rows
        local_rows = distributed.shard_rows(tuple(range(start, stop)))
        local_row_count += len(local_rows)
        for cached in _iter_activation_fit_batches(
            train_reader,
            local_rows,
            source_layer_type=source_layer_type,
            burn_in_tokens=burn_in_tokens,
        ):
            with torch.no_grad():
                output = executor.forward_cached_layer_local(
                    cached.hidden_states,
                    shared_states=cached.shared_states,
                    active_layer_index=layer_index,
                    active_mixer=mixer,
                    loaded_layer=loaded_layer,
                )
            if (
                output.teacher_signals is None
                or "pre_output" not in output.teacher_signals
            ):
                raise ContractError("source mixer did not expose pre-output trace")
            gate = (
                output.student_signals["gate"][:, burn_in_tokens:]
                .double()
                .flatten(0, 1)
            )
            norm_base = (
                output.student_signals["norm_base"][:, burn_in_tokens:]
                .double()
                .flatten(0, 1)
            )
            target = (
                output.teacher_signals["pre_output"][:, burn_in_tokens:]
                .double()
                .flatten(0, 1)
            )
            norm_offset = (
                output.student_signals["norm_offset"][:, burn_in_tokens:]
                .double()
                .flatten(0, 1)
            )
            target = target - gate * norm_offset
            first = norm_base * gate
            second = gate
            local_token_count += int(first.shape[0])
            statistics[0] += first.square().sum(0)
            statistics[1] += (first * second).sum(0)
            statistics[2] += second.square().sum(0)
            statistics[3] += (first * target).sum(0)
            statistics[4] += (second * target).sum(0)
            statistics[5] += target.square().sum(0)
        start = stop
    rank_contributions = distributed.all_gather_objects(
        {
            "rank": distributed.rank,
            "rows": local_row_count,
            "tokens": local_token_count,
            "local_statistics_sha256": _tensor_sha256(statistics),
        }
    )
    distributed.all_reduce_sum(statistics)
    original_weight = mixer.g_norm.weight.detach().clone()
    original_bias = mixer.g_norm.bias.detach().clone()
    proposed_weight = torch.empty_like(original_weight, dtype=torch.float32)
    proposed_bias = torch.empty_like(original_bias, dtype=torch.float32)
    fit_metrics = None
    if distributed.is_primary:
        first_first = statistics[0] + ridge
        first_second = statistics[1]
        second_second = statistics[2] + ridge
        determinant = (first_first * second_second - first_second.square()).clamp_min(
            1e-30
        )
        weight = (
            statistics[3] * second_second - first_second * statistics[4]
        ) / determinant
        bias = (
            first_first * statistics[4] - first_second * statistics[3]
        ) / determinant
        proposed_weight.copy_(weight.float())
        proposed_bias.copy_(bias.float())
        prediction_squared = (
            weight.square() * statistics[0]
            + 2 * weight * bias * statistics[1]
            + bias.square() * statistics[2]
        ).sum()
        prediction_target = (weight * statistics[3] + bias * statistics[4]).sum()
        target_squared = statistics[5].sum()
        fit_metrics = {
            "normalized_mse": float(
                (prediction_squared - 2 * prediction_target + target_squared).clamp_min(
                    0
                )
                / target_squared.clamp_min(1e-30)
            ),
            "cosine": float(
                prediction_target
                / torch.sqrt((prediction_squared * target_squared).clamp_min(1e-30))
            ),
        }
    distributed.broadcast_tensor(proposed_weight)
    distributed.broadcast_tensor(proposed_bias)
    fit_metrics = distributed.broadcast_object(fit_metrics)
    with torch.no_grad():
        mixer.g_norm.weight.copy_(proposed_weight.to(mixer.g_norm.weight.dtype))
        mixer.g_norm.bias.copy_(proposed_bias.to(mixer.g_norm.bias.dtype))
    validation_after = _validate_norm_affine(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        source_layer_type=source_layer_type,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        distributed=distributed,
    )
    layer_validation_after = _validate(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        loss_weights=loss_weights,
        distributed=distributed,
    )
    improved = (
        math.isfinite(validation_after["normalized_mse"])
        and validation_after["normalized_mse"] < validation_before["normalized_mse"]
        and layer_validation_after["mixer_normalized_mse"]
        <= layer_validation_before["mixer_normalized_mse"]
    )
    if not improved and not defer_chain_acceptance:
        with torch.no_grad():
            mixer.g_norm.weight.copy_(original_weight)
            mixer.g_norm.bias.copy_(original_bias)
    selected_parameter_hashes = distributed.all_gather_objects(
        {
            "rank": distributed.rank,
            "weight_sha256": _tensor_sha256(mixer.g_norm.weight),
            "bias_sha256": _tensor_sha256(mixer.g_norm.bias),
        }
    )
    if any(
        candidate["weight_sha256"] != selected_parameter_hashes[0]["weight_sha256"]
        or candidate["bias_sha256"] != selected_parameter_hashes[0]["bias_sha256"]
        for candidate in selected_parameter_hashes[1:]
    ):
        raise ContractError("norm affine parameters differ across distributed ranks")
    if distributed.is_primary:
        write_json(
            report_path,
            {
                "schema_version": 1,
                "status": (
                    "deferred"
                    if defer_chain_acceptance
                    else ("accepted" if improved else "rejected")
                ),
                "layer": layer_index,
                "fit_stage": fit_stage,
                "selection_scope": (
                    "dependency-transaction"
                    if fit_stage == "dependency-transaction"
                    else "component"
                ),
                "boundary": "source-pre-output-to-native-groupnorm-affine-v1",
                "world_size": distributed.world_size,
                "fit_rows": fit_rows,
                "ridge": ridge,
                "source_layer_type": source_layer_type,
                "train_cache_binding": train_reader.manifest.get("binding"),
                "validation_cache_binding": validation_reader.manifest.get("binding"),
                "rank_contributions": list(rank_contributions),
                "selected_parameter_hashes": list(selected_parameter_hashes),
                "fit": fit_metrics,
                "validation_before": validation_before,
                "validation_after": validation_after,
                "layer_validation_before": layer_validation_before,
                "layer_validation_after": layer_validation_after,
                "chain_acceptance_baseline": chain_baseline,
                "statistics_sha256": _tensor_sha256(statistics),
                "original_weight_sha256": _tensor_sha256(original_weight),
                "original_bias_sha256": _tensor_sha256(original_bias),
                "selected_weight_sha256": _tensor_sha256(mixer.g_norm.weight),
                "selected_bias_sha256": _tensor_sha256(mixer.g_norm.bias),
                "proposed_weight_sha256": _tensor_sha256(
                    proposed_weight.to(mixer.g_norm.weight.dtype)
                ),
                "proposed_bias_sha256": _tensor_sha256(
                    proposed_bias.to(mixer.g_norm.bias.dtype)
                ),
            },
        )
    distributed.barrier()
    if improved or defer_chain_acceptance:
        return chain_baseline or layer_validation_before
    return chain_baseline


def _validate_norm_affine(
    *,
    executor,
    reader,
    mixer,
    loaded_layer,
    layer_index,
    source_layer_type,
    burn_in_tokens,
    micro_batch_size,
    distributed,
):
    totals = torch.zeros(
        4, dtype=torch.float64, device=next(loaded_layer.module.parameters()).device
    )
    global_micro_batch_size = micro_batch_size * distributed.world_size
    start = 0
    with torch.no_grad():
        while start < reader.row_count:
            stop = min(start + global_micro_batch_size, reader.row_count)
            if 0 < reader.row_count - stop < distributed.world_size:
                stop = reader.row_count
            local_rows = distributed.shard_rows(tuple(range(start, stop)))
            for cached in _iter_activation_fit_batches(
                reader,
                local_rows,
                source_layer_type=source_layer_type,
                burn_in_tokens=burn_in_tokens,
            ):
                output = executor.forward_cached_layer_local(
                    cached.hidden_states,
                    shared_states=cached.shared_states,
                    active_layer_index=layer_index,
                    active_mixer=mixer,
                    loaded_layer=loaded_layer,
                )
                if (
                    output.teacher_signals is None
                    or "pre_output" not in output.teacher_signals
                ):
                    raise ContractError("source mixer did not expose pre-output trace")
                student = output.student_signals["pre_output"][
                    :, burn_in_tokens:
                ].float()
                teacher = output.teacher_signals["pre_output"][
                    :, burn_in_tokens:
                ].float()
                if student.shape != teacher.shape:
                    raise ContractError(
                        "source/target pre-output traces have different shapes"
                    )
                totals[0] += (student - teacher).square().sum().double()
                totals[1] += teacher.square().sum().double()
                totals[2] += (student * teacher).sum().double()
                totals[3] += student.square().sum().double()
            start = stop
    distributed.all_reduce_sum(totals)
    return {
        "normalized_mse": float((totals[0] / totals[1].clamp_min(1e-30)).item()),
        "cosine": float(
            (totals[2] / torch.sqrt((totals[3] * totals[1]).clamp_min(1e-30))).item()
        ),
    }


def _validate_source_erase_rate_projection(
    *,
    executor,
    reader,
    mixer,
    loaded_layer,
    layer_index,
    burn_in_tokens,
    micro_batch_size,
    distributed,
):
    totals = torch.zeros(
        4, dtype=torch.float64, device=next(loaded_layer.module.parameters()).device
    )
    head_errors = torch.zeros(
        mixer.num_heads,
        2,
        dtype=torch.float64,
        device=totals.device,
    )
    global_micro_batch_size = micro_batch_size * distributed.world_size
    start = 0
    with torch.no_grad():
        while start < reader.row_count:
            stop = min(start + global_micro_batch_size, reader.row_count)
            if 0 < reader.row_count - stop < distributed.world_size:
                stop = reader.row_count
            local_rows = distributed.shard_rows(tuple(range(start, stop)))
            for cached in _iter_activation_fit_batches(
                reader,
                local_rows,
                source_layer_type="linear_attention",
                burn_in_tokens=burn_in_tokens,
            ):
                output = executor.forward_cached_layer_local(
                    cached.hidden_states,
                    shared_states=cached.shared_states,
                    active_layer_index=layer_index,
                    active_mixer=mixer,
                    loaded_layer=loaded_layer,
                )
                if (
                    output.teacher_signals is None
                    or "erase_rate" not in output.teacher_signals
                ):
                    raise ContractError(
                        "source GDN did not expose beta*decay erase trace"
                    )
                source = output.teacher_signals["erase_rate"][
                    :, burn_in_tokens:
                ].float()
                target = output.student_signals["erase"][:, burn_in_tokens:].float()
                if source.shape != target.shape:
                    raise ContractError(
                        "source erase rate and native erase traces differ in shape"
                    )
                totals[0] += (target - source).square().sum().double()
                totals[1] += source.square().sum().double()
                totals[2] += (target * source).sum().double()
                totals[3] += target.square().sum().double()
                source_heads = source.view(
                    *source.shape[:2], mixer.num_heads, mixer.head_dim
                )
                target_heads = target.view_as(source_heads)
                head_errors[:, 0] += (
                    (target_heads - source_heads).square().sum(dim=(0, 1, 3)).double()
                )
                head_errors[:, 1] += source_heads.square().sum(dim=(0, 1, 3)).double()
            start = stop
    distributed.all_reduce_sum(totals)
    distributed.all_reduce_sum(head_errors)
    return {
        "normalized_mse": float((totals[0] / totals[1].clamp_min(1e-30)).item()),
        "cosine": float(
            (totals[2] / torch.sqrt((totals[3] * totals[1]).clamp_min(1e-30))).item()
        ),
        "per_head_normalized_mse": [
            float(value)
            for value in (
                head_errors[:, 0] / head_errors[:, 1].clamp_min(1e-30)
            ).tolist()
        ],
    }


def _validate_source_gate_projection(
    *,
    executor,
    reader,
    mixer,
    loaded_layer,
    layer_index,
    burn_in_tokens,
    micro_batch_size,
    source_layer_type,
    distributed,
):
    totals = torch.zeros(
        4, dtype=torch.float64, device=next(loaded_layer.module.parameters()).device
    )
    head_errors = torch.zeros(
        mixer.num_heads, 2, dtype=torch.float64, device=totals.device
    )
    global_micro_batch_size = micro_batch_size * distributed.world_size
    start = 0
    with torch.no_grad():
        while start < reader.row_count:
            stop = min(start + global_micro_batch_size, reader.row_count)
            if 0 < reader.row_count - stop < distributed.world_size:
                stop = reader.row_count
            local_rows = distributed.shard_rows(tuple(range(start, stop)))
            for cached in _iter_activation_fit_batches(
                reader,
                local_rows,
                source_layer_type=source_layer_type,
                burn_in_tokens=burn_in_tokens,
            ):
                output = executor.forward_cached_layer_local(
                    cached.hidden_states,
                    shared_states=cached.shared_states,
                    active_layer_index=layer_index,
                    active_mixer=mixer,
                    loaded_layer=loaded_layer,
                )
                if (
                    output.teacher_signals is None
                    or "gate" not in output.teacher_signals
                ):
                    raise ContractError("source mixer did not expose gate trace")
                student = output.student_signals["gate"][:, burn_in_tokens:].float()
                teacher = output.teacher_signals["gate"][:, burn_in_tokens:].float()
                if student.shape != teacher.shape:
                    raise ContractError(
                        "source/target gate traces have different shapes"
                    )
                totals[0] += (student - teacher).square().sum().double()
                totals[1] += teacher.square().sum().double()
                totals[2] += (student * teacher).sum().double()
                totals[3] += student.square().sum().double()
                student_heads = student.view(
                    *student.shape[:2], mixer.num_heads, mixer.head_dim
                )
                teacher_heads = teacher.view_as(student_heads)
                head_errors[:, 0] += (
                    (student_heads - teacher_heads).square().sum(dim=(0, 1, 3)).double()
                )
                head_errors[:, 1] += teacher_heads.square().sum(dim=(0, 1, 3)).double()
            start = stop
    distributed.all_reduce_sum(totals)
    distributed.all_reduce_sum(head_errors)
    return {
        "normalized_mse": float((totals[0] / totals[1].clamp_min(1e-30)).item()),
        "cosine": float(
            (totals[2] / torch.sqrt((totals[3] * totals[1]).clamp_min(1e-30))).item()
        ),
        "per_target_head_normalized_mse": [
            float(value)
            for value in (
                head_errors[:, 0] / head_errors[:, 1].clamp_min(1e-30)
            ).tolist()
        ],
    }


def _validate_source_qkv_projections(
    *,
    executor,
    reader,
    mixer,
    loaded_layer,
    layer_index,
    burn_in_tokens,
    micro_batch_size,
    source_layer_type,
    distributed,
):
    is_gdn = source_layer_type == "linear_attention"
    roles = (
        (("r", "recurrent_r"), ("k", "write_key"), ("v", "write_value"))
        if is_gdn
        else (("r", "q"), ("k", "k"), ("v", "v"))
    )
    totals = torch.zeros(
        len(roles),
        4,
        dtype=torch.float64,
        device=next(loaded_layer.module.parameters()).device,
    )
    head_errors = torch.zeros(
        len(roles), mixer.num_heads, 2, dtype=torch.float64, device=totals.device
    )
    source_mixer = (
        loaded_layer.module.linear_attn if is_gdn else loaded_layer.module.self_attn
    )
    source_heads = (
        int(source_mixer.num_v_heads)
        if is_gdn
        else int(source_mixer.config.num_attention_heads)
    )
    source_key_dim = (
        int(source_mixer.head_k_dim) if is_gdn else int(source_mixer.head_dim)
    )
    source_value_dim = (
        int(source_mixer.head_v_dim) if is_gdn else int(source_mixer.head_dim)
    )
    source_head_errors = torch.zeros(
        len(roles), source_heads, 2, dtype=torch.float64, device=totals.device
    )
    global_micro_batch_size = micro_batch_size * distributed.world_size
    start = 0
    with torch.no_grad():
        while start < reader.row_count:
            stop = min(start + global_micro_batch_size, reader.row_count)
            if 0 < reader.row_count - stop < distributed.world_size:
                stop = reader.row_count
            local_rows = distributed.shard_rows(tuple(range(start, stop)))
            for cached in _iter_activation_fit_batches(
                reader,
                local_rows,
                source_layer_type=source_layer_type,
                burn_in_tokens=burn_in_tokens,
            ):
                output = executor.forward_cached_layer_local(
                    cached.hidden_states,
                    shared_states=cached.shared_states,
                    active_layer_index=layer_index,
                    active_mixer=mixer,
                    loaded_layer=loaded_layer,
                )
                if output.teacher_signals is None:
                    raise ContractError("source mixer did not expose projection traces")
                for role_index, (role, teacher_name) in enumerate(roles):
                    student = output.student_signals[f"projected_{role}"][
                        :, burn_in_tokens:
                    ].float()
                    teacher = output.teacher_signals[teacher_name][
                        :, burn_in_tokens:
                    ].float()
                    if is_gdn and role == "r":
                        teacher = teacher * float(source_key_dim) ** 0.5
                    if role == "r" and not is_gdn:
                        teacher = teacher / float(source_key_dim) ** 0.5
                    if student.shape != teacher.shape:
                        raise ContractError(
                            "source/target projection traces have different shapes"
                        )
                    totals[role_index, 0] += (student - teacher).square().sum().double()
                    totals[role_index, 1] += teacher.square().sum().double()
                    totals[role_index, 2] += (student * teacher).sum().double()
                    totals[role_index, 3] += student.square().sum().double()
                    student_heads = student.view(
                        *student.shape[:2], mixer.num_heads, mixer.head_dim
                    )
                    teacher_heads = teacher.view_as(student_heads)
                    head_errors[role_index, :, 0] += (
                        (student_heads - teacher_heads)
                        .square()
                        .sum(dim=(0, 1, 3))
                        .double()
                    )
                    head_errors[role_index, :, 1] += (
                        teacher_heads.square().sum(dim=(0, 1, 3)).double()
                    )
                    source_eval_dim = (
                        source_key_dim if role in {"r", "k"} else source_value_dim
                    )
                    student_source_heads = student.view(
                        *student.shape[:2], source_heads, source_eval_dim
                    )
                    teacher_source_heads = teacher.view_as(student_source_heads)
                    source_head_errors[role_index, :, 0] += (
                        (student_source_heads - teacher_source_heads)
                        .square()
                        .sum(dim=(0, 1, 3))
                        .double()
                    )
                    source_head_errors[role_index, :, 1] += (
                        teacher_source_heads.square().sum(dim=(0, 1, 3)).double()
                    )
            start = stop
    distributed.all_reduce_sum(totals)
    distributed.all_reduce_sum(head_errors)
    distributed.all_reduce_sum(source_head_errors)
    result: dict[str, object] = {}
    for role_index, (role, _teacher_name) in enumerate(roles):
        denominator = totals[role_index, 1].clamp_min(1e-30)
        result[role] = {
            "normalized_mse": float((totals[role_index, 0] / denominator).item()),
            "cosine": float(
                (
                    totals[role_index, 2]
                    / torch.sqrt(
                        (totals[role_index, 3] * totals[role_index, 1]).clamp_min(1e-30)
                    )
                ).item()
            ),
            "per_target_head_normalized_mse": [
                float(value)
                for value in (
                    head_errors[role_index, :, 0]
                    / head_errors[role_index, :, 1].clamp_min(1e-30)
                ).tolist()
            ],
        }
        source_head_mse = source_head_errors[role_index, :, 0] / source_head_errors[
            role_index, :, 1
        ].clamp_min(1e-30)
        result[role]["per_source_query_head_normalized_mse"] = [
            float(value) for value in source_head_mse.tolist()
        ]
        if not is_gdn:
            kv_heads = int(source_mixer.config.num_key_value_heads)
            heads_per_group = source_heads // kv_heads
            grouped = (
                source_head_errors[role_index]
                .view(kv_heads, heads_per_group, 2)
                .sum(dim=1)
            )
            result[role]["per_source_kv_group_normalized_mse"] = [
                float(value)
                for value in (grouped[:, 0] / grouped[:, 1].clamp_min(1e-30)).tolist()
            ]
    return result


def _activation_fit_decay_projection(
    *,
    executor,
    train_reader,
    validation_reader,
    mixer,
    loaded_layer,
    layer_index,
    burn_in_tokens,
    fit_rows,
    ridge,
    micro_batch_size,
    loss_weights,
    run_dir,
    distributed,
):
    """Fit the low-rank native decay up-projection from source GDN traces."""
    _require_independent_activation_fit_caches(train_reader, validation_reader)
    fit_rows = int(fit_rows)
    if fit_rows > train_reader.row_count:
        raise ContractError("activation-fit decay rows exceed the train cache")
    if fit_rows < distributed.world_size:
        raise ContractError(
            "activation fitting requires at least one trace row per distributed rank"
        )
    validation_before = _validate_decay_projection(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        distributed=distributed,
    )
    layer_validation_before = _validate(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        loss_weights=loss_weights,
        distributed=distributed,
    )
    gram = rhs = target_squared_sum = None
    local_row_count = 0
    local_row_indices: list[int] = []
    local_tokens = 0
    local_unreachable = 0.0
    local_decay_values = 0
    global_micro_batch_size = micro_batch_size * distributed.world_size
    start = 0
    while start < fit_rows:
        stop = min(start + global_micro_batch_size, fit_rows)
        trailing = fit_rows - stop
        if 0 < trailing < distributed.world_size:
            stop = fit_rows
        local_rows = distributed.shard_rows(tuple(range(start, stop)))
        local_row_count += len(local_rows)
        local_row_indices.extend(local_rows)
        cached = train_reader.read_rows(local_rows)
        with torch.no_grad():
            output = executor.forward_cached_layer_local(
                cached.hidden_states,
                shared_states=cached.shared_states,
                active_layer_index=layer_index,
                active_mixer=mixer,
                loaded_layer=loaded_layer,
            )
        if output.teacher_signals is None or "decay" not in output.teacher_signals:
            raise ContractError("source GDN did not expose a decay trace")
        features = output.student_signals["w_features"][:, burn_in_tokens:].flatten(
            0, 1
        )
        source_decay = output.teacher_signals["decay"][:, burn_in_tokens:]
        targets = native_decay_fit_targets(
            source_decay,
            target_channels=mixer.w_lora.lora[2].out_features,
        )
        logits = targets.logits.flatten(0, 1)
        design = torch.cat(
            (
                features.float(),
                torch.ones(features.shape[0], 1, device=features.device),
            ),
            dim=1,
        )
        statistics = teacher_trace_normal_equations(design, logits)
        gram = statistics.gram if gram is None else gram + statistics.gram
        rhs = statistics.rhs if rhs is None else rhs + statistics.rhs
        target_squared_sum = (
            statistics.target_squared_sum
            if target_squared_sum is None
            else target_squared_sum + statistics.target_squared_sum
        )
        local_tokens += statistics.tokens
        decay_values = int(targets.expanded_source_decay.numel())
        local_unreachable += targets.unreachable_fraction * decay_values
        local_decay_values += decay_values
        start = stop
    if gram is None or rhs is None or target_squared_sum is None:
        raise ContractError(
            "decay activation fitting produced no sufficient statistics"
        )
    rank_contributions = distributed.all_gather_objects(
        {
            "rank": distributed.rank,
            "rows": local_row_count,
            "tokens": local_tokens,
            "row_indices_sha256": _sha256_json(local_row_indices),
            "local_gram_sha256": _tensor_sha256(gram),
            "local_rhs_sha256": _tensor_sha256(rhs),
            "local_target_squared_sum_sha256": _tensor_sha256(target_squared_sum),
        }
    )
    distributed.all_reduce_sum(gram)
    distributed.all_reduce_sum(rhs)
    distributed.all_reduce_sum(target_squared_sum)
    counts = torch.tensor(
        [local_tokens, local_unreachable, local_decay_values],
        dtype=torch.float64,
        device=gram.device,
    )
    distributed.all_reduce_sum(counts)
    original_weight = mixer.w_lora.lora[2].weight.detach().clone()
    original_bias = mixer.w_lora.lora[2].bias.detach().clone()
    fitted_augmented = torch.empty(
        rhs.shape[1], gram.shape[0], dtype=torch.float32, device=gram.device
    )
    fit_metrics = None
    if distributed.is_primary:
        fit = solve_teacher_trace_normal_equations(
            TraceNormalEquations(
                gram,
                rhs,
                target_squared_sum,
                int(counts[0].item()),
            ),
            ridge=ridge,
        )
        fitted_augmented.copy_(fit.weight)
        fit_metrics = {
            "logit_normalized_mse": fit.normalized_mse,
            "logit_cosine": fit.cosine,
        }
    distributed.broadcast_tensor(fitted_augmented)
    fit_metrics = distributed.broadcast_object(fit_metrics)
    with torch.no_grad():
        mixer.w_lora.lora[2].weight.copy_(
            fitted_augmented[:, :-1].to(mixer.w_lora.lora[2].weight.dtype)
        )
        mixer.w_lora.lora[2].bias.copy_(
            fitted_augmented[:, -1].to(mixer.w_lora.lora[2].bias.dtype)
        )
    validation_after = _validate_decay_projection(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        distributed=distributed,
    )
    layer_validation_after = _validate(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        loss_weights=loss_weights,
        distributed=distributed,
    )
    before_per_head = validation_before["per_head_normalized_mse"]
    after_per_head = validation_after["per_head_normalized_mse"]
    finite = all(
        math.isfinite(float(value))
        for value in (
            *fit_metrics.values(),
            validation_after["decay_normalized_mse"],
            validation_after["decay_cosine"],
            *after_per_head,
            layer_validation_after["mixer_normalized_mse"],
        )
    )
    every_head_non_regressed = all(
        float(after) <= float(before)
        for before, after in zip(before_per_head, after_per_head, strict=True)
    )
    improved = (
        finite
        and validation_after["decay_normalized_mse"]
        < validation_before["decay_normalized_mse"]
        and every_head_non_regressed
        and layer_validation_after["mixer_normalized_mse"]
        <= layer_validation_before["mixer_normalized_mse"]
    )
    if not improved:
        with torch.no_grad():
            mixer.w_lora.lora[2].weight.copy_(original_weight)
            mixer.w_lora.lora[2].bias.copy_(original_bias)
    selected_parameter_hashes = distributed.all_gather_objects(
        {
            "rank": distributed.rank,
            "weight_sha256": _tensor_sha256(mixer.w_lora.lora[2].weight),
            "bias_sha256": _tensor_sha256(mixer.w_lora.lora[2].bias),
        }
    )
    if any(
        row["weight_sha256"] != selected_parameter_hashes[0]["weight_sha256"]
        or row["bias_sha256"] != selected_parameter_hashes[0]["bias_sha256"]
        for row in selected_parameter_hashes[1:]
    ):
        raise ContractError("decay-fit parameters differ across distributed ranks")
    report = {
        "schema_version": 1,
        "status": "accepted" if improved else "rejected",
        "layer": layer_index,
        "boundary": "gdn-decay-logit-to-native-w-up-v1",
        "world_size": distributed.world_size,
        "fit_rows": fit_rows,
        "fit_tokens": int(counts[0].item()),
        "unreachable_source_decay_fraction": (
            float(counts[1].item() / counts[2].item())
        ),
        "ridge": ridge,
        "solver": "augmented-bias-normal-equations-ridge-v1",
        "train_cache_binding": train_reader.manifest.get("binding"),
        "validation_cache_binding": validation_reader.manifest.get("binding"),
        "rank_contributions": list(rank_contributions),
        "fit": fit_metrics,
        "validation_before": validation_before,
        "validation_after": validation_after,
        "layer_validation_before": layer_validation_before,
        "layer_validation_after": layer_validation_after,
        "every_head_non_regressed": every_head_non_regressed,
        "original_weight_sha256": _tensor_sha256(original_weight),
        "original_bias_sha256": _tensor_sha256(original_bias),
        "proposed_weight_sha256": _tensor_sha256(
            fitted_augmented[:, :-1].to(mixer.w_lora.lora[2].weight.dtype)
        ),
        "proposed_bias_sha256": _tensor_sha256(
            fitted_augmented[:, -1].to(mixer.w_lora.lora[2].bias.dtype)
        ),
        "selected_parameter_hashes": list(selected_parameter_hashes),
    }
    if distributed.is_primary:
        report["normal_equations"] = {
            "gram_sha256": _tensor_sha256(gram),
            "rhs_sha256": _tensor_sha256(rhs),
            "target_squared_sum_sha256": _tensor_sha256(target_squared_sum),
        }
        report["selected_weight_sha256"] = _tensor_sha256(mixer.w_lora.lora[2].weight)
        report["selected_bias_sha256"] = _tensor_sha256(mixer.w_lora.lora[2].bias)
        write_json(
            run_dir / "activation-fit" / f"decay-layer-{layer_index:03d}.json",
            report,
        )
    distributed.barrier()
    return layer_validation_before if improved else None


def _validate_decay_projection(
    *,
    executor,
    reader,
    mixer,
    loaded_layer,
    layer_index,
    burn_in_tokens,
    micro_batch_size,
    distributed,
):
    totals = torch.zeros(
        5,
        dtype=torch.float64,
        device=next(loaded_layer.module.parameters()).device,
    )
    head_errors = torch.zeros(
        mixer.num_heads,
        2,
        dtype=torch.float64,
        device=totals.device,
    )
    global_micro_batch_size = micro_batch_size * distributed.world_size
    start = 0
    with torch.no_grad():
        while start < reader.row_count:
            stop = min(start + global_micro_batch_size, reader.row_count)
            trailing = reader.row_count - stop
            if 0 < trailing < distributed.world_size:
                stop = reader.row_count
            local_rows = distributed.shard_rows(tuple(range(start, stop)))
            cached = reader.read_rows(local_rows)
            output = executor.forward_cached_layer_local(
                cached.hidden_states,
                shared_states=cached.shared_states,
                active_layer_index=layer_index,
                active_mixer=mixer,
                loaded_layer=loaded_layer,
            )
            if output.teacher_signals is None or "decay" not in output.teacher_signals:
                raise ContractError("source GDN did not expose a decay trace")
            student = output.student_signals["decay"][:, burn_in_tokens:].float()
            targets = native_decay_fit_targets(
                output.teacher_signals["decay"][:, burn_in_tokens:],
                target_channels=student.shape[-1],
            )
            teacher = targets.expanded_source_decay
            totals[0] += (student - teacher).square().sum().double()
            totals[1] += teacher.square().sum().double()
            totals[2] += (student * teacher).sum().double()
            totals[3] += student.square().sum().double()
            totals[4] += teacher.square().sum().double()
            student_heads = student.view(
                *student.shape[:2], mixer.num_heads, mixer.head_dim
            )
            teacher_heads = teacher.view_as(student_heads)
            head_errors[:, 0] += (
                (student_heads - teacher_heads).square().sum(dim=(0, 1, 3)).double()
            )
            head_errors[:, 1] += teacher_heads.square().sum(dim=(0, 1, 3)).double()
            start = stop
    distributed.all_reduce_sum(totals)
    distributed.all_reduce_sum(head_errors)
    if totals[1] <= 0:
        raise ContractError("decay validation target energy is zero")
    return {
        "decay_normalized_mse": float((totals[0] / totals[1]).item()),
        "decay_cosine": float(
            (totals[2] / torch.sqrt((totals[3] * totals[4]).clamp_min(1e-30))).item()
        ),
        "per_head_normalized_mse": [
            float(value)
            for value in (
                head_errors[:, 0] / head_errors[:, 1].clamp_min(1e-30)
            ).tolist()
        ],
    }


def _activation_fit_output_projection(
    *,
    executor,
    train_reader,
    validation_reader,
    mixer,
    loaded_layer,
    layer_index,
    source_layer_type,
    burn_in_tokens,
    fit_rows,
    ridge,
    micro_batch_size,
    loss_weights,
    validation_baseline,
    run_dir,
    distributed,
    fit_stage="initial",
    fail_if_chain_not_improved=True,
):
    """Fit active ``o_proj`` from additive 8-rank teacher-trace statistics."""
    fit_rows = min(int(fit_rows), train_reader.row_count)
    if fit_rows < distributed.world_size:
        raise ContractError(
            "activation fitting requires at least one trace row per distributed rank"
        )
    validation_before = _validate(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        loss_weights=loss_weights,
        distributed=distributed,
    )
    # The output projection is the last candidate in the activation-fit chain.
    # It must improve the parameters that are actually installed immediately
    # before it, not merely an older zero-step baseline from the start of the
    # chain.  Otherwise a regressive final projection could still be accepted.
    acceptance_baseline = validation_before
    gram = rhs = target_squared_sum = None
    local_row_count = 0
    local_row_indices: list[int] = []
    local_tokens = 0
    global_micro_batch_size = micro_batch_size * distributed.world_size
    start = 0
    while start < fit_rows:
        stop = min(start + global_micro_batch_size, fit_rows)
        trailing = fit_rows - stop
        if 0 < trailing < distributed.world_size:
            stop = fit_rows
        global_rows = tuple(range(start, stop))
        local_rows = distributed.shard_rows(global_rows)
        local_row_count += len(local_rows)
        local_row_indices.extend(local_rows)
        cached = train_reader.read_rows(local_rows)
        projection_inputs: list[torch.Tensor] = []

        def capture_projection_input(_module, args):
            projection_inputs.append(args[0].detach())

        hook = mixer.o_proj.register_forward_pre_hook(capture_projection_input)
        try:
            with torch.no_grad():
                output = executor.forward_cached_layer_local(
                    cached.hidden_states,
                    shared_states=cached.shared_states,
                    active_layer_index=layer_index,
                    active_mixer=mixer,
                    loaded_layer=loaded_layer,
                )
        finally:
            hook.remove()
        if len(projection_inputs) == 1 and projection_inputs[0].ndim == 3:
            projection_trace = projection_inputs[0]
        elif projection_inputs and all(value.ndim == 2 for value in projection_inputs):
            projection_trace = torch.stack(projection_inputs, dim=1)
        else:
            raise ContractError(
                "activation fitting captured an unsupported o_proj input trace layout"
            )
        features = projection_trace[:, burn_in_tokens:].flatten(0, 1)
        targets = output.teacher_mixer_output[:, burn_in_tokens:].flatten(0, 1)
        statistics = teacher_trace_normal_equations(features, targets)
        gram = statistics.gram if gram is None else gram + statistics.gram
        rhs = statistics.rhs if rhs is None else rhs + statistics.rhs
        target_squared_sum = (
            statistics.target_squared_sum
            if target_squared_sum is None
            else target_squared_sum + statistics.target_squared_sum
        )
        local_tokens += statistics.tokens
        start = stop
    if gram is None or rhs is None or target_squared_sum is None:
        raise ContractError("activation fitting produced no sufficient statistics")
    rank_contributions = distributed.all_gather_objects(
        {
            "rank": distributed.rank,
            "rows": local_row_count,
            "tokens": local_tokens,
            "row_indices_sha256": _sha256_json(local_row_indices),
            "local_gram_sha256": _tensor_sha256(gram),
            "local_rhs_sha256": _tensor_sha256(rhs),
            "local_target_squared_sum_sha256": _tensor_sha256(target_squared_sum),
        }
    )
    distributed.all_reduce_sum(gram)
    distributed.all_reduce_sum(rhs)
    distributed.all_reduce_sum(target_squared_sum)
    token_count = torch.tensor(local_tokens, dtype=torch.int64, device=gram.device)
    distributed.all_reduce_sum(token_count)
    original_weight = mixer.o_proj.weight.detach().clone()
    fitted_weight = torch.empty_like(mixer.o_proj.weight, dtype=torch.float32)
    fit_metrics = None
    if distributed.is_primary:
        fit = solve_teacher_trace_normal_equations(
            TraceNormalEquations(
                gram,
                rhs,
                target_squared_sum,
                int(token_count.item()),
            ),
            ridge=ridge,
        )
        fitted_weight.copy_(fit.weight)
        fit_metrics = {
            "normalized_mse": fit.normalized_mse,
            "cosine": fit.cosine,
        }
    distributed.broadcast_tensor(fitted_weight)
    fit_metrics = distributed.broadcast_object(fit_metrics)
    with torch.no_grad():
        mixer.o_proj.weight.copy_(fitted_weight.to(mixer.o_proj.weight.dtype))
    validation_after = _validate(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        loss_weights=loss_weights,
        distributed=distributed,
    )
    finite = all(
        torch.isfinite(torch.tensor(value)).item()
        for value in (*fit_metrics.values(), *validation_after.values())
    )
    improved = (
        finite
        and validation_after["mixer_normalized_mse"]
        < acceptance_baseline["mixer_normalized_mse"]
    )
    if not improved:
        with torch.no_grad():
            mixer.o_proj.weight.copy_(original_weight)
    selected_parameter_hashes = distributed.all_gather_objects(
        {
            "rank": distributed.rank,
            "weight_sha256": _tensor_sha256(mixer.o_proj.weight),
        }
    )
    if any(
        row["weight_sha256"] != selected_parameter_hashes[0]["weight_sha256"]
        for row in selected_parameter_hashes[1:]
    ):
        raise ContractError("output-fit parameters differ across distributed ranks")
    report = {
        "schema_version": 1,
        "status": "accepted" if improved else "rejected",
        "layer": layer_index,
        "fit_stage": fit_stage,
        "selection_scope": (
            "dependency-transaction"
            if fit_stage == "dependency-transaction"
            else "component"
        ),
        "boundary": "source-mixer-output-to-native-o-proj-v1",
        "source_layer_type": source_layer_type,
        "world_size": distributed.world_size,
        "fit_rows": fit_rows,
        "fit_tokens": int(token_count.item()),
        "fit_row_indices_sha256": _sha256_json(list(range(fit_rows))),
        "ridge": ridge,
        "solver": "bias-free-normal-equations-ridge-v1",
        "train_cache_binding": train_reader.manifest.get("binding"),
        "validation_cache_binding": validation_reader.manifest.get("binding"),
        "rank_contributions": list(rank_contributions),
        "fit": fit_metrics,
        "validation_before": validation_before,
        "acceptance_baseline": acceptance_baseline,
        "chain_acceptance_baseline": validation_baseline,
        "validation_after": validation_after,
        "original_o_proj_weight_sha256": _tensor_sha256(original_weight),
        "proposed_o_proj_weight_sha256": _tensor_sha256(
            fitted_weight.to(mixer.o_proj.weight.dtype)
        ),
        "selected_o_proj_weight_sha256": _tensor_sha256(mixer.o_proj.weight),
        "selected_parameter_hashes": list(selected_parameter_hashes),
    }
    if distributed.is_primary:
        report["normal_equations"] = {
            "gram_sha256": _tensor_sha256(gram),
            "rhs_sha256": _tensor_sha256(rhs),
            "target_squared_sum_sha256": _tensor_sha256(target_squared_sum),
        }
        stage_suffix = "" if fit_stage == "initial" else f"-{fit_stage}"
        write_json(
            run_dir / "activation-fit" / f"layer{stage_suffix}-{layer_index:03d}.json",
            report,
        )
    distributed.barrier()
    chain_already_improved = (
        validation_baseline is not None
        and validation_before["mixer_normalized_mse"]
        < validation_baseline["mixer_normalized_mse"]
    )
    if fail_if_chain_not_improved and not improved and not chain_already_improved:
        raise ContractError(
            f"activation-fit-held-out-regression: layer {layer_index} did not "
            "leave a strictly improved validation mixer normalized MSE"
        )


def _activation_fit_full_attention(
    *,
    gqa_native_geometry,
    gqa_installation_reader,
    executor,
    train_reader,
    validation_reader,
    mixer,
    loaded_layer,
    layer_index,
    burn_in_tokens,
    fit_rows,
    ridge,
    functional_steps,
    functional_learning_rate,
    micro_batch_size,
    loss_weights,
    max_trace_bytes_per_rank,
    run_time_mix_ablation,
    run_dir,
    distributed,
):
    """Choose exactly one full-attention activation-fit implementation."""
    if gqa_native_geometry is not None:
        if gqa_installation_reader is None:
            raise ContractError(
                "GQA native zero-step lacks its installation reader"
            )
        return _activation_fit_gqa_native_zero_step_transaction(
            executor=executor,
            train_reader=train_reader,
            validation_reader=gqa_installation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            burn_in_tokens=burn_in_tokens,
            fit_rows=fit_rows,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            max_trace_bytes_per_rank=max_trace_bytes_per_rank,
            run_dir=run_dir,
            distributed=distributed,
        )

    attention_projection_baseline = (
        _activation_fit_attention_dependency_transaction(
            executor=executor,
            train_reader=train_reader,
            validation_reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            burn_in_tokens=burn_in_tokens,
            fit_rows=fit_rows,
            ridge=ridge,
            functional_steps=functional_steps,
            functional_learning_rate=functional_learning_rate,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            run_dir=run_dir,
            distributed=distributed,
        )
    )
    if attention_projection_baseline is not None and run_time_mix_ablation:
        _activation_fit_attention_time_mix_transaction(
            executor=executor,
            train_reader=train_reader,
            validation_reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            burn_in_tokens=burn_in_tokens,
            fit_rows=fit_rows,
            ridge=ridge,
            functional_steps=functional_steps,
            functional_learning_rate=functional_learning_rate,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            run_dir=run_dir,
            distributed=distributed,
        )
    return None


def _activation_fit_gqa_native_zero_step_transaction(
    *,
    executor,
    train_reader,
    validation_reader,
    mixer,
    loaded_layer,
    layer_index,
    burn_in_tokens,
    fit_rows,
    micro_batch_size,
    loss_weights,
    max_trace_bytes_per_rank,
    run_dir,
    distributed,
):
    """Fit and atomically gate the native two-state GQA zero-step candidate.

    Calibration and adaptive-development rows come only from ``distill_train``.
    Every context candidate uses the same rows; the longest causal trace is
    selected because it contains every shorter prefix. Adaptive-development
    metrics are diagnostic, not a cross-horizon selection score. The mutually
    exclusive validation cache is first touched only after the selected
    complete parameter set has been materialized in the real BF16 module. That
    complete-forward score solely decides commit versus rollback; post-decision
    component ablations are diagnostic and cannot change it.
    """
    geometry = _gqa_native_zero_step_geometry(
        mixer=mixer,
        loaded_layer=loaded_layer,
    )
    if geometry is None:
        return None
    source_mixer = loaded_layer.module.self_attn
    query_heads = int(geometry["query_heads"])
    key_value_heads = int(geometry["key_value_heads"])
    source_head_dim = int(geometry["source_head_dim"])
    _require_independent_activation_fit_caches(train_reader, validation_reader)
    fit_rows = int(fit_rows)
    if (
        fit_rows > train_reader.row_count
        or fit_rows < max(3, 2 * distributed.world_size)
    ):
        raise ContractError("GQA native zero-step rows violate fit capacity")

    baseline = _validate(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        loss_weights=loss_weights,
        distributed=distributed,
    )
    baseline_state = _snapshot_module_state(mixer)
    baseline_hashes = _module_state_hashes(mixer)
    requested_context_lengths = _attention_context_lengths(
        train_reader,
        burn_in_tokens,
    )
    context_lengths = tuple(
        length
        for length in requested_context_lengths
        if length - int(burn_in_tokens) >= 2
    )
    if not context_lengths:
        raise ContractError("GQA native zero-step has no eligible causal context")
    selected_context_length = max(context_lengths)
    hidden_size = int(train_reader.manifest["hidden_size"])
    trace_width = 2 * hidden_size + 4 * query_heads * source_head_dim
    maximum_local_fit_rows = (
        fit_rows + distributed.world_size - 1
    ) // distributed.world_size
    estimated_primary_trace_bytes = (
        2
        * maximum_local_fit_rows
        * selected_context_length
        * trace_width
    )
    native_head_dim = int(mixer.head_dim)
    solver_row_chunk_size = maximum_local_fit_rows
    solver_peak = estimate_gqa_native_streamed_peak_bytes(
        fit_rows=maximum_local_fit_rows,
        context_length=selected_context_length,
        hidden_size=hidden_size,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        source_head_dim=source_head_dim,
        native_head_dim=native_head_dim,
        row_chunk_size=solver_row_chunk_size,
        decay_rank=int(mixer.w_lora.lora[0].out_features),
        erase_rank=int(mixer.a_lora.lora[0].out_features),
        gate_rank=int(mixer.g_lora.lora[0].out_features),
    )
    estimated_solver_peak_bytes = solver_peak["estimated_peak_bytes"]
    max_trace_bytes_per_rank = int(max_trace_bytes_per_rank)
    if (
        max_trace_bytes_per_rank <= 0
        or estimated_solver_peak_bytes > max_trace_bytes_per_rank
    ):
        raise ContractError(
            "GQA native zero-step solver exceeds the per-rank memory bound: "
            f"estimated={estimated_solver_peak_bytes} "
            f"limit={max_trace_bytes_per_rank}"
        )
    query_weight = source_mixer.q_proj.weight.detach().float()
    key_weight = source_mixer.k_proj.weight.detach().float()
    value_weight = source_mixer.v_proj.weight.detach().float()
    output_weight = source_mixer.o_proj.weight.detach().float()
    output_bias = getattr(source_mixer.o_proj, "bias", None)
    output_bias = (
        None if output_bias is None else output_bias.detach().float()
    )
    group_width = query_heads // key_value_heads
    grouped_indices = torch.arange(
        key_value_heads,
        device=mixer.r_proj.weight.device,
    ) * group_width
    calibration_rows = max(2, fit_rows // 2)
    local_contexts = []
    rank_contributions = []
    for context_length in (selected_context_length,):
        global_rows = tuple(range(fit_rows))
        local_rows = global_rows[distributed.rank :: distributed.world_size]
        local_parts = []
        local_error = None
        try:
            for start in range(0, len(local_rows), micro_batch_size):
                rows = local_rows[start : start + micro_batch_size]
                cached = train_reader.read_rows(rows)
                with torch.no_grad():
                    output = executor.forward_cached_layer_local(
                        cached.hidden_states[:, :context_length],
                        shared_states=(
                            None
                            if cached.shared_states is None
                            else cached.shared_states[:, :context_length]
                        ),
                        active_layer_index=layer_index,
                        active_mixer=mixer,
                        loaded_layer=loaded_layer,
                    )
                required_signals = {
                    "mixer_input",
                    "q_post_rope",
                    "k_post_rope",
                    "v",
                    "gate",
                }
                if (
                    output.teacher_signals is None
                    or not required_signals.issubset(output.teacher_signals)
                ):
                    raise ContractError(
                        "source GQA mixer did not expose the native zero-step trace"
                    )
                local_parts.append(
                    {
                        "row_indices": list(cached.row_indices),
                        "mixer_input": output.teacher_signals["mixer_input"]
                        .detach()
                        .to(device="cpu", dtype=torch.bfloat16),
                        "query": output.teacher_signals["q_post_rope"]
                        .detach()
                        .to(device="cpu", dtype=torch.bfloat16),
                        "key": output.teacher_signals["k_post_rope"]
                        .detach()
                        .to(device="cpu", dtype=torch.bfloat16),
                        "value": output.teacher_signals["v"]
                        .detach()
                        .to(device="cpu", dtype=torch.bfloat16),
                        "gate": output.teacher_signals["gate"]
                        .detach()
                        .to(device="cpu", dtype=torch.bfloat16),
                        "mixer_output": output.teacher_mixer_output.detach()
                        .to(device="cpu", dtype=torch.bfloat16),
                    }
                )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
            local_parts = []
        collection_status = distributed.all_gather_objects(
            {
                "rank": distributed.rank,
                "error": local_error,
            }
        )
        collection_errors = [
            item for item in collection_status if item["error"] is not None
        ]
        if collection_errors:
            _restore_module_state(mixer, baseline_state)
            raise ContractError(
                "GQA native zero-step trace collection failed: "
                f"{collection_errors}"
            )
        assembly_error = None
        local_contribution = None
        try:
            row_indices = torch.tensor(
                [
                    row
                    for part in local_parts
                    for row in part["row_indices"]
                ],
                dtype=torch.long,
            )
            if row_indices.numel() != len(local_rows):
                raise ContractError(
                    "GQA local trace row count differs from its shard"
                )
            order = torch.argsort(row_indices)
            signals = {
                name: torch.cat(
                    [part[name] for part in local_parts],
                    dim=0,
                )
                .index_select(0, order)
                .to(
                    device=mixer.r_proj.weight.device,
                    dtype=torch.float32,
                )
                for name in (
                    "mixer_input",
                    "query",
                    "key",
                    "value",
                    "gate",
                    "mixer_output",
                )
            }
            sorted_rows = tuple(
                int(value)
                for value in row_indices.index_select(0, order).tolist()
            )
            if sorted_rows != local_rows:
                raise ContractError(
                    "GQA local trace row identities differ from their shard"
                )
            local_row_count = len(sorted_rows)
            query = signals["query"].reshape(
                local_row_count,
                context_length,
                query_heads,
                source_head_dim,
            )
            key = signals["key"].reshape_as(query)
            value = signals["value"].reshape_as(query)
            gate = signals["gate"].reshape_as(query)
            positions = torch.arange(
                context_length,
                dtype=torch.long,
                device=query.device,
            ).unsqueeze(0).expand(local_row_count, -1)
            validate_gqa_native_fit_trace(
                mixer,
                GQANativeFitTrace(
                    mixer_input=signals["mixer_input"],
                    query=query,
                    key=key,
                    value=value,
                    grouped_key=key.index_select(2, grouped_indices),
                    grouped_value=value.index_select(2, grouped_indices),
                    gate=gate,
                    mixer_output=signals["mixer_output"],
                    query_weight=query_weight,
                    key_weight=key_weight,
                    value_weight=value_weight,
                    output_weight=output_weight,
                    output_bias=output_bias,
                ),
                GQANativeFitConfig(
                    calibration_rows=calibration_rows,
                    positions=positions,
                    source_head_dim=source_head_dim,
                    rotary_dim=int(mixer.rotary_dim),
                    rope_theta=float(mixer.rope_theta),
                    supervised_token_start=int(burn_in_tokens),
                    row_chunk_size=solver_row_chunk_size,
                    global_row_indices=sorted_rows,
                ),
            )
            local_contexts.append(
                (context_length, sorted_rows, signals)
            )
            local_contribution = {
                "rank": distributed.rank,
                "row_indices": list(sorted_rows),
                "row_indices_sha256": _sha256_json(list(sorted_rows)),
                "signal_sha256": {
                    name: _tensor_sha256(value)
                    for name, value in sorted(signals.items())
                },
            }
        except BaseException as error:
            assembly_error = f"{type(error).__name__}: {error}"
        assembly_statuses = distributed.all_gather_objects(
            {
                "rank": distributed.rank,
                "error": assembly_error,
                "contribution": local_contribution,
            }
        )
        assembly_errors = [
            item for item in assembly_statuses
            if item["error"] is not None
        ]
        if assembly_errors:
            _restore_module_state(mixer, baseline_state)
            raise ContractError(
                "GQA native zero-step trace assembly failed: "
                f"{assembly_errors}"
            )
        rank_contributions.append(
            {
                "context_length": context_length,
                "ranks": [
                    item["contribution"] for item in assembly_statuses
                ],
            }
        )

    selected_result = None
    fit_error = None
    try:
        candidates = []
        for context_length, row_indices, signals in local_contexts:
            local_row_count = len(row_indices)
            query = signals["query"].reshape(
                local_row_count,
                context_length,
                query_heads,
                source_head_dim,
            )
            key = signals["key"].reshape_as(query)
            value = signals["value"].reshape_as(query)
            gate = signals["gate"].reshape_as(query)
            positions = torch.arange(
                context_length,
                dtype=torch.long,
                device=query.device,
            ).unsqueeze(0).expand(local_row_count, -1)
            result = fit_gqa_native_zero_step(
                mixer,
                GQANativeFitTrace(
                    mixer_input=signals["mixer_input"],
                    query=query,
                    key=key,
                    value=value,
                    grouped_key=key.index_select(2, grouped_indices),
                    grouped_value=value.index_select(2, grouped_indices),
                    gate=gate,
                    mixer_output=signals["mixer_output"],
                    query_weight=query_weight,
                    key_weight=key_weight,
                    value_weight=value_weight,
                    output_weight=output_weight,
                    output_bias=output_bias,
                ),
                GQANativeFitConfig(
                    calibration_rows=calibration_rows,
                    positions=positions,
                    source_head_dim=source_head_dim,
                    rotary_dim=int(mixer.rotary_dim),
                    rope_theta=float(mixer.rope_theta),
                    supervised_token_start=int(burn_in_tokens),
                    row_chunk_size=solver_row_chunk_size,
                    global_row_indices=row_indices,
                    reduce_sum=distributed.all_reduce_sum,
                    reduce_max=distributed.all_reduce_max,
                ),
            )
            score = float(
                result.report["native_parameter_projection"][
                    "complete_free_running_mixer"
                ]["nmse"]
            )
            distributed_report = {
                **result.report,
                "trace_hash_scope": "rank-sharded",
                "trace_shards": rank_contributions,
                "trace_aggregate_sha256": _sha256_json(
                    rank_contributions
                ),
            }
            candidates.append(
                {
                    "context_length": context_length,
                    "row_indices": list(range(fit_rows)),
                    "development_mixer_nmse": score,
                    "parameters": result.parameters,
                    "report": distributed_report,
                }
            )
        if not candidates:
            raise ContractError(
                "GQA native zero-step produced no calibration/development split"
            )
        selected_result = max(
            candidates,
            key=lambda item: item["context_length"],
        )
        selected_result["candidate_summaries"] = [
            {
                "context_length": item["context_length"],
                "row_indices": item["row_indices"],
                "development_mixer_nmse": item[
                    "development_mixer_nmse"
                ],
            }
            for item in candidates
        ]
    except BaseException as error:
        fit_error = f"{type(error).__name__}: {error}"
    fit_statuses = distributed.all_gather_objects(
        {
            "rank": distributed.rank,
            "error": fit_error,
            "selected_context_length": (
                None
                if selected_result is None
                else selected_result["context_length"]
            ),
        }
    )
    fit_errors = [
        item for item in fit_statuses if item["error"] is not None
    ]
    if fit_errors:
        _restore_module_state(mixer, baseline_state)
        raise ContractError(
            f"GQA native zero-step fit failed: {fit_errors}"
        )

    candidate_parameters = {
        name: selected_result["parameters"][name].to(
            device=parameter.device,
            dtype=torch.float32,
        )
        for name, parameter in sorted(mixer.named_parameters())
    }
    candidate_digests = distributed.all_gather_objects(
        {
            name: _tensor_sha256(value)
            for name, value in sorted(candidate_parameters.items())
        }
    )
    if any(
        digest != candidate_digests[0]
        for digest in candidate_digests[1:]
    ):
        _restore_module_state(mixer, baseline_state)
        raise ContractError(
            "GQA native zero-step distributed solves produced different parameters"
        )
    materialization = materialize_native_projection(mixer, candidate_parameters)
    candidate_validation = _validate(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        loss_weights=loss_weights,
        distributed=distributed,
    )
    accepted = _gqa_native_validation_improves(
        baseline,
        candidate_validation,
    )
    mapped_component_ablations = (
        _gqa_native_mapped_component_restoration_ablations(
            executor=executor,
            reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            burn_in_tokens=burn_in_tokens,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            distributed=distributed,
            baseline_state=baseline_state,
        )
    )
    proposed_hashes = _module_state_hashes(mixer)
    if not accepted:
        _restore_module_state(mixer, baseline_state)
    selected_hashes = _module_state_hashes(mixer)
    rank_hashes = distributed.all_gather_objects(
        {
            "rank": distributed.rank,
            "materialization_sha256": materialization.aggregate_sha256,
            "selected_module_state_sha256": _sha256_json(selected_hashes),
        }
    )
    if any(
        item["materialization_sha256"]
        != rank_hashes[0]["materialization_sha256"]
        or item["selected_module_state_sha256"]
        != rank_hashes[0]["selected_module_state_sha256"]
        for item in rank_hashes[1:]
    ):
        _restore_module_state(mixer, baseline_state)
        raise ContractError(
            "GQA native zero-step materialization differs across ranks"
        )
    report_path = (
        run_dir
        / "activation-fit"
        / f"gqa-native-zero-step-layer-{layer_index:03d}.json"
    )
    def write_report() -> None:
        write_json(
            report_path,
            {
                "schema_version": 1,
                "status": "accepted" if accepted else "rejected",
                "layer": layer_index,
                "boundary": (
                    "gqa-exact-hazard-bounded-surrogate-observable-two-state-"
                    "native-bf16-validation-v1"
                ),
                "world_size": distributed.world_size,
                "source_geometry": {
                    "query_heads": query_heads,
                    "key_value_heads": key_value_heads,
                    "head_dim": source_head_dim,
                },
                "target_geometry": {
                    "native_heads": int(mixer.num_heads),
                    "head_dim": int(mixer.head_dim),
                },
                "fit_rows": fit_rows,
                "requested_context_lengths": list(requested_context_lengths),
                "eligible_context_lengths": list(context_lengths),
                "skipped_context_lengths": [
                    length
                    for length in requested_context_lengths
                    if length not in context_lengths
                ],
                "selected_context_length": selected_result["context_length"],
                "candidate_summaries": selected_result["candidate_summaries"],
                "rank_contributions": rank_contributions,
                "trace_transport": {
                    "collection": "rank-sharded-teacher-forward",
                    "wire_dtype": "bfloat16",
                    "destination": "rank-local-streamed-solve",
                    "solver_collectives": (
                        "additive-statistic-and-gradient-all-reduce"
                    ),
                    "formal_context_collection": "longest-only",
                    "shorter_contexts": "prefix-diagnostics-only",
                    "estimated_max_local_trace_bytes": (
                        estimated_primary_trace_bytes
                    ),
                    "estimated_solver_peak_bytes": (
                        estimated_solver_peak_bytes
                    ),
                    "solver_peak_breakdown": solver_peak,
                    "max_trace_bytes_per_rank": max_trace_bytes_per_rank,
                },
                "train_cache_binding": train_reader.manifest.get("binding"),
                "validation_cache_binding": validation_reader.manifest.get(
                    "binding"
                ),
                "validation_protocol": (
                    "disjoint-installation-subset; post-decision ablations "
                    "cannot select parameters; epoch selection uses a separate "
                    "row subset"
                ),
                "selection_rule": (
                    "same fit rows at every eligible context; select the "
                    "longest causal trace because it contains every shorter "
                    "prefix; adaptive-development NMSE is diagnostic only"
                ),
                "acceptance_rule": (
                    "finite and strictly lower frozen-validation BF16 native "
                    "free-running mixer normalized MSE"
                ),
                "validation_baseline": baseline,
                "validation_candidate": candidate_validation,
                "mapped_component_restoration_ablations": (
                    mapped_component_ablations
                ),
                "fit_report": selected_result["report"],
                "baseline_parameter_sha256": baseline_hashes,
                "proposed_parameter_sha256": proposed_hashes,
                "selected_parameter_sha256": selected_hashes,
                "materialization": asdict(materialization),
                "rank_parameter_hashes": list(rank_hashes),
            },
        )

    _rank0_filesystem_step(
        distributed,
        "publish GQA native zero-step report",
        write_report,
    )
    return {
        "attempted": True,
        "accepted": accepted,
        "report": str(report_path.relative_to(run_dir)),
        "report_sha256": file_sha256(report_path),
        "selected_module_state_sha256": _sha256_json(selected_hashes),
    }


def _gqa_native_validation_improves(baseline, candidate) -> bool:
    numeric = [
        float(value)
        for metrics in (baseline, candidate)
        for value in metrics.values()
        if isinstance(value, (int, float))
    ]
    return bool(
        numeric
        and all(math.isfinite(value) for value in numeric)
        and float(candidate["mixer_normalized_mse"])
        < float(baseline["mixer_normalized_mse"])
    )


def _gqa_native_mapped_component_restoration_ablations(
    *,
    executor,
    reader,
    mixer,
    loaded_layer,
    layer_index,
    burn_in_tokens,
    micro_batch_size,
    loss_weights,
    distributed,
    baseline_state,
):
    """Restore one mapped component at a time after the commit decision."""
    candidate_state = _snapshot_module_state(mixer)
    parameter_names = set(candidate_state)
    groups = {
        "r_proj": ("x_r", "r_proj.weight"),
        "k_proj": ("x_k", "k_proj.weight", "k_k", "k_a"),
        "v_proj": (
            "x_v",
            "v_proj.weight",
            "v_lora.lora.0.weight",
            "v_lora.lora.2.weight",
            "v_lora.lora.2.bias",
        ),
        "w_a": (
            "x_w",
            "x_a",
            "w_lora.lora.0.weight",
            "w_lora.lora.2.weight",
            "w_lora.lora.2.bias",
            "a_lora.lora.0.weight",
            "a_lora.lora.2.weight",
            "a_lora.lora.2.bias",
        ),
        "gate": (
            "x_g",
            "g_lora.lora.0.weight",
            "g_lora.lora.2.weight",
        ),
        "o_proj": (
            "g_norm.weight",
            "g_norm.bias",
            "r_k",
            "o_proj.weight",
        ),
    }
    reports = {}
    try:
        for group, names in groups.items():
            counterfactual = dict(candidate_state)
            restored_names = [
                name for name in names if name in parameter_names
            ]
            for name in restored_names:
                counterfactual[name] = baseline_state[name]
            _restore_module_state(mixer, counterfactual)
            reports[group] = {
                "operation": "mapped-component-restoration-ablation",
                "parameter_names": restored_names,
                "validation": _validate(
                    executor=executor,
                    reader=reader,
                    mixer=mixer,
                    loaded_layer=loaded_layer,
                    layer_index=layer_index,
                    burn_in_tokens=burn_in_tokens,
                    micro_batch_size=micro_batch_size,
                    loss_weights=loss_weights,
                    distributed=distributed,
                ),
            }
    finally:
        _restore_module_state(mixer, candidate_state)
    return reports


def _activation_fit_attention_dependency_transaction(
    *,
    executor,
    train_reader,
    validation_reader,
    mixer,
    loaded_layer,
    layer_index,
    burn_in_tokens,
    fit_rows,
    ridge,
    functional_steps,
    functional_learning_rate,
    micro_batch_size,
    loss_weights,
    run_dir,
    distributed,
):
    """Fit attention-dependent projections as one held-out generation.

    A zero gate makes every Q/K/V candidate invisible at the mixer output.
    Conversely, opening the gate before GroupNorm and ``o_proj`` are refit can
    make a useful gate candidate look catastrophically regressive.  Keep the
    dependent Q/K/V, gate, normalization, and output candidates alive until
    their complete generation can be accepted or rolled back atomically.

    The first transaction deliberately leaves native time-mix, decay, erase,
    and bonus controls at their source-compatible zero-step values.  Those
    controls have no direct full-attention parameter counterpart and require a
    separate, evidence-backed ablation after this projection transaction.
    """
    fit_stage = "dependency-transaction"
    report_dir = run_dir / "activation-fit"
    report_path = (
        report_dir / f"attention-dependency-transaction-layer-{layer_index:03d}.json"
    )
    baseline = _validate(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        loss_weights=loss_weights,
        distributed=distributed,
    )
    baseline_state = _snapshot_module_state(mixer)
    baseline_parameter_hashes = _module_state_hashes(mixer)
    component_names = (
        f"attention-qkv-{fit_stage}-layer-{layer_index:03d}.json",
        f"attention-gate-{fit_stage}-layer-{layer_index:03d}.json",
        f"norm-affine-{fit_stage}-layer-{layer_index:03d}.json",
        f"layer-{fit_stage}-{layer_index:03d}.json",
    )
    candidate = baseline
    proposed_parameter_hashes = baseline_parameter_hashes
    accepted = False
    try:
        _activation_fit_source_qkv_projections(
            executor=executor,
            train_reader=train_reader,
            validation_reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            source_layer_type="full_attention",
            burn_in_tokens=burn_in_tokens,
            fit_rows=fit_rows,
            ridge=ridge,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            chain_baseline=baseline,
            run_dir=run_dir,
            distributed=distributed,
            fit_stage=fit_stage,
            defer_chain_acceptance=True,
        )
        _activation_fit_source_gate_projection(
            executor=executor,
            train_reader=train_reader,
            validation_reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            source_layer_type="full_attention",
            burn_in_tokens=burn_in_tokens,
            fit_rows=fit_rows,
            ridge=ridge,
            functional_steps=functional_steps,
            functional_learning_rate=functional_learning_rate,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            chain_baseline=baseline,
            run_dir=run_dir,
            distributed=distributed,
            fit_stage=fit_stage,
            defer_chain_acceptance=True,
        )
        _activation_fit_norm_affine(
            executor=executor,
            train_reader=train_reader,
            validation_reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            source_layer_type="full_attention",
            burn_in_tokens=burn_in_tokens,
            fit_rows=fit_rows,
            ridge=ridge,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            chain_baseline=baseline,
            run_dir=run_dir,
            distributed=distributed,
            fit_stage=fit_stage,
            defer_chain_acceptance=True,
        )
        _activation_fit_output_projection(
            executor=executor,
            train_reader=train_reader,
            validation_reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            source_layer_type="full_attention",
            burn_in_tokens=burn_in_tokens,
            fit_rows=fit_rows,
            ridge=ridge,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            validation_baseline=baseline,
            run_dir=run_dir,
            distributed=distributed,
            fit_stage=fit_stage,
            fail_if_chain_not_improved=False,
        )
        candidate = _validate(
            executor=executor,
            reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            burn_in_tokens=burn_in_tokens,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            distributed=distributed,
        )
        proposed_parameter_hashes = _module_state_hashes(mixer)
        accepted = _dependency_transaction_improves(baseline, candidate)
        if not accepted:
            _restore_module_state(mixer, baseline_state)
    except BaseException:
        _restore_module_state(mixer, baseline_state)
        raise
    selected_parameter_hashes = _module_state_hashes(mixer)
    rank_parameter_hashes = distributed.all_gather_objects(
        {
            "rank": distributed.rank,
            "module_state_sha256": _sha256_json(selected_parameter_hashes),
        }
    )
    if any(
        row["module_state_sha256"] != rank_parameter_hashes[0]["module_state_sha256"]
        for row in rank_parameter_hashes[1:]
    ):
        _restore_module_state(mixer, baseline_state)
        raise ContractError(
            "attention dependency-transaction parameters differ across ranks"
        )
    if distributed.is_primary:
        component_reports = {
            name: file_sha256(report_dir / name) for name in component_names
        }
        write_json(
            report_path,
            {
                "schema_version": 1,
                "status": "accepted" if accepted else "rejected",
                "layer": layer_index,
                "boundary": (
                    "attention-qkv-gate-norm-output-activation-fit-generation-v1"
                ),
                "world_size": distributed.world_size,
                "fit_rows": int(fit_rows),
                "acceptance_rule": (
                    "finite-and-strictly-lower-loss-and-normalized-mse-"
                    "with-non-regressed-block-mse"
                ),
                "excluded_native_controls": [
                    "time_mix",
                    "decay",
                    "erase",
                    "r_k_bonus",
                ],
                "validation_baseline": baseline,
                "validation_candidate": candidate,
                "component_report_sha256": component_reports,
                "baseline_parameter_sha256": baseline_parameter_hashes,
                "proposed_parameter_sha256": proposed_parameter_hashes,
                "selected_parameter_sha256": selected_parameter_hashes,
                "rank_parameter_hashes": list(rank_parameter_hashes),
            },
        )
    distributed.barrier()
    return baseline if accepted else None


def _activation_fit_attention_time_mix_transaction(
    *,
    executor,
    train_reader,
    validation_reader,
    mixer,
    loaded_layer,
    layer_index,
    burn_in_tokens,
    fit_rows,
    ridge,
    functional_steps,
    functional_learning_rate,
    micro_batch_size,
    loss_weights,
    run_dir,
    distributed,
):
    """Ablate Q/K/V time-mix only after the attention projections are valid.

    Full attention has no source time-mix tensor.  The candidate is therefore
    fitted against mixer/block outputs.  The transaction evaluates both that
    complete-forward candidate and a second generation that refits every
    projection consuming the changed features.  Among generations that pass
    the independent held-out contract, the lowest-loss candidate is selected;
    otherwise the baseline is restored atomically.  Source-absent decay,
    erase, and current-token bonus controls remain unchanged.
    """
    fit_stage = "attention-time-mix-transaction"
    report_dir = run_dir / "activation-fit"
    report_path = (
        report_dir
        / f"attention-time-mix-dependency-transaction-layer-{layer_index:03d}.json"
    )
    baseline = _validate(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        loss_weights=loss_weights,
        distributed=distributed,
    )
    baseline_state = _snapshot_module_state(mixer)
    baseline_parameter_hashes = _module_state_hashes(mixer)
    component_names = (
        f"time-mix-{fit_stage}-layer-{layer_index:03d}.json",
        f"attention-qkv-{fit_stage}-layer-{layer_index:03d}.json",
        f"attention-gate-{fit_stage}-layer-{layer_index:03d}.json",
        f"norm-affine-{fit_stage}-layer-{layer_index:03d}.json",
        f"layer-{fit_stage}-{layer_index:03d}.json",
    )
    time_mix_candidate = baseline
    time_mix_parameter_hashes = baseline_parameter_hashes
    fully_refit_candidate = baseline
    fully_refit_parameter_hashes = baseline_parameter_hashes
    selected_generation = "baseline"
    selected_validation = baseline
    try:
        _activation_fit_time_mix(
            executor=executor,
            train_reader=train_reader,
            validation_reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            source_layer_type="full_attention",
            burn_in_tokens=burn_in_tokens,
            fit_rows=fit_rows,
            steps=functional_steps,
            learning_rate=functional_learning_rate,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            chain_baseline=baseline,
            run_dir=run_dir,
            distributed=distributed,
            fit_stage=fit_stage,
            defer_chain_acceptance=True,
        )
        time_mix_candidate = _validate(
            executor=executor,
            reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            burn_in_tokens=burn_in_tokens,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            distributed=distributed,
        )
        time_mix_state = _snapshot_module_state(mixer)
        time_mix_parameter_hashes = _module_state_hashes(mixer)
        _activation_fit_source_qkv_projections(
            executor=executor,
            train_reader=train_reader,
            validation_reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            source_layer_type="full_attention",
            burn_in_tokens=burn_in_tokens,
            fit_rows=fit_rows,
            ridge=ridge,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            chain_baseline=baseline,
            run_dir=run_dir,
            distributed=distributed,
            fit_stage=fit_stage,
            defer_chain_acceptance=True,
        )
        _activation_fit_source_gate_projection(
            executor=executor,
            train_reader=train_reader,
            validation_reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            source_layer_type="full_attention",
            burn_in_tokens=burn_in_tokens,
            fit_rows=fit_rows,
            ridge=ridge,
            functional_steps=functional_steps,
            functional_learning_rate=functional_learning_rate,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            chain_baseline=baseline,
            run_dir=run_dir,
            distributed=distributed,
            fit_stage=fit_stage,
            defer_chain_acceptance=True,
        )
        _activation_fit_norm_affine(
            executor=executor,
            train_reader=train_reader,
            validation_reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            source_layer_type="full_attention",
            burn_in_tokens=burn_in_tokens,
            fit_rows=fit_rows,
            ridge=ridge,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            chain_baseline=baseline,
            run_dir=run_dir,
            distributed=distributed,
            fit_stage=fit_stage,
            defer_chain_acceptance=True,
        )
        _activation_fit_output_projection(
            executor=executor,
            train_reader=train_reader,
            validation_reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            source_layer_type="full_attention",
            burn_in_tokens=burn_in_tokens,
            fit_rows=fit_rows,
            ridge=ridge,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            validation_baseline=baseline,
            run_dir=run_dir,
            distributed=distributed,
            fit_stage=fit_stage,
            fail_if_chain_not_improved=False,
        )
        fully_refit_candidate = _validate(
            executor=executor,
            reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            burn_in_tokens=burn_in_tokens,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            distributed=distributed,
        )
        fully_refit_parameter_hashes = _module_state_hashes(mixer)
        eligible = []
        if _dependency_transaction_improves(baseline, time_mix_candidate):
            eligible.append(("time-mix-only", time_mix_candidate, time_mix_state))
        if _dependency_transaction_improves(baseline, fully_refit_candidate):
            eligible.append(("fully-refit", fully_refit_candidate, None))
        if eligible:
            selected_generation, selected_validation, selected_state = min(
                eligible,
                key=lambda item: (
                    float(item[1]["loss"]),
                    float(item[1]["normalized_mse"]),
                    float(item[1]["block_normalized_mse"]),
                    item[0],
                ),
            )
            if selected_state is not None:
                _restore_module_state(mixer, selected_state)
        else:
            _restore_module_state(mixer, baseline_state)
    except BaseException:
        _restore_module_state(mixer, baseline_state)
        raise
    selected_parameter_hashes = _module_state_hashes(mixer)
    rank_parameter_hashes = distributed.all_gather_objects(
        {
            "rank": distributed.rank,
            "module_state_sha256": _sha256_json(selected_parameter_hashes),
        }
    )
    if any(
        row["module_state_sha256"] != rank_parameter_hashes[0]["module_state_sha256"]
        for row in rank_parameter_hashes[1:]
    ):
        _restore_module_state(mixer, baseline_state)
        raise ContractError(
            "attention time-mix transaction parameters differ across ranks"
        )
    if distributed.is_primary:
        component_reports = {
            name: file_sha256(report_dir / name) for name in component_names
        }
        write_json(
            report_path,
            {
                "schema_version": 1,
                "status": (
                    "accepted" if selected_generation != "baseline" else "rejected"
                ),
                "layer": layer_index,
                "boundary": (
                    "attention-qkv-time-mix-dependent-activation-fit-generation-v1"
                ),
                "world_size": distributed.world_size,
                "fit_rows": int(fit_rows),
                "gradient_scope": ["x_r", "x_k", "x_v"],
                "excluded_native_controls": [
                    "decay",
                    "erase",
                    "r_k_bonus",
                    "x_w",
                    "x_a",
                    "x_g",
                ],
                "acceptance_rule": (
                    "finite-and-strictly-lower-loss-and-normalized-mse-"
                    "with-non-regressed-block-mse-then-minimum-loss"
                ),
                "selected_generation": selected_generation,
                "validation_baseline": baseline,
                "validation_candidate": selected_validation,
                "validation_candidates": {
                    "time-mix-only": time_mix_candidate,
                    "fully-refit": fully_refit_candidate,
                },
                "component_report_sha256": component_reports,
                "baseline_parameter_sha256": baseline_parameter_hashes,
                "candidate_parameter_sha256": {
                    "time-mix-only": time_mix_parameter_hashes,
                    "fully-refit": fully_refit_parameter_hashes,
                },
                "selected_parameter_sha256": selected_parameter_hashes,
                "rank_parameter_hashes": list(rank_parameter_hashes),
            },
        )
    distributed.barrier()
    return baseline if selected_generation != "baseline" else None


def _activation_fit_dependency_transaction(
    *,
    executor,
    train_reader,
    validation_reader,
    mixer,
    loaded_layer,
    layer_index,
    burn_in_tokens,
    fit_rows,
    ridge,
    functional_steps,
    functional_learning_rate,
    micro_batch_size,
    loss_weights,
    run_dir,
    distributed,
):
    """Evaluate coupled GDN candidates only after all dependent refits.

    Native time-mix changes the feature matrices consumed by Q/K/V, gate,
    normalization, and the output projection.  Its immediate mixer metric is
    therefore not a valid final acceptance boundary.  This transaction keeps
    that candidate alive through every dependent refit, then either commits the
    complete generation or restores the exact baseline.  Erase is deliberately
    excluded: the v24 coupled experiment showed that its much better signal fit
    still regressed the fully refitted held-out layer.
    """
    fit_stage = "dependency-transaction"
    report_dir = run_dir / "activation-fit"
    report_path = report_dir / f"dependency-transaction-layer-{layer_index:03d}.json"
    baseline = _validate(
        executor=executor,
        reader=validation_reader,
        mixer=mixer,
        loaded_layer=loaded_layer,
        layer_index=layer_index,
        burn_in_tokens=burn_in_tokens,
        micro_batch_size=micro_batch_size,
        loss_weights=loss_weights,
        distributed=distributed,
    )
    baseline_state = _snapshot_module_state(mixer)
    baseline_parameter_hashes = _module_state_hashes(mixer)
    component_names = (
        f"time-mix-{fit_stage}-layer-{layer_index:03d}.json",
        f"gdn-qkv-{fit_stage}-layer-{layer_index:03d}.json",
        f"gdn-gate-{fit_stage}-layer-{layer_index:03d}.json",
        f"norm-affine-{fit_stage}-layer-{layer_index:03d}.json",
        f"layer-{fit_stage}-{layer_index:03d}.json",
    )
    try:
        _activation_fit_time_mix(
            executor=executor,
            train_reader=train_reader,
            validation_reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            source_layer_type="linear_attention",
            burn_in_tokens=burn_in_tokens,
            fit_rows=fit_rows,
            steps=functional_steps,
            learning_rate=functional_learning_rate,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            chain_baseline=baseline,
            run_dir=run_dir,
            distributed=distributed,
            fit_stage=fit_stage,
            defer_chain_acceptance=True,
        )
        _activation_fit_source_qkv_projections(
            executor=executor,
            train_reader=train_reader,
            validation_reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            source_layer_type="linear_attention",
            burn_in_tokens=burn_in_tokens,
            fit_rows=fit_rows,
            ridge=ridge,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            chain_baseline=baseline,
            run_dir=run_dir,
            distributed=distributed,
            fit_stage=fit_stage,
            defer_chain_acceptance=True,
        )
        _activation_fit_source_gate_projection(
            executor=executor,
            train_reader=train_reader,
            validation_reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            source_layer_type="linear_attention",
            burn_in_tokens=burn_in_tokens,
            fit_rows=fit_rows,
            ridge=ridge,
            functional_steps=functional_steps,
            functional_learning_rate=functional_learning_rate,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            chain_baseline=baseline,
            run_dir=run_dir,
            distributed=distributed,
            fit_stage=fit_stage,
            defer_chain_acceptance=True,
        )
        _activation_fit_norm_affine(
            executor=executor,
            train_reader=train_reader,
            validation_reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            source_layer_type="linear_attention",
            burn_in_tokens=burn_in_tokens,
            fit_rows=fit_rows,
            ridge=ridge,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            chain_baseline=baseline,
            run_dir=run_dir,
            distributed=distributed,
            fit_stage=fit_stage,
            defer_chain_acceptance=True,
        )
        _activation_fit_output_projection(
            executor=executor,
            train_reader=train_reader,
            validation_reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            source_layer_type="linear_attention",
            burn_in_tokens=burn_in_tokens,
            fit_rows=fit_rows,
            ridge=ridge,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            validation_baseline=baseline,
            run_dir=run_dir,
            distributed=distributed,
            fit_stage=fit_stage,
            fail_if_chain_not_improved=False,
        )
        candidate = _validate(
            executor=executor,
            reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=layer_index,
            burn_in_tokens=burn_in_tokens,
            micro_batch_size=micro_batch_size,
            loss_weights=loss_weights,
            distributed=distributed,
        )
        proposed_parameter_hashes = _module_state_hashes(mixer)
        accepted = _dependency_transaction_improves(baseline, candidate)
        if not accepted:
            _restore_module_state(mixer, baseline_state)
    except BaseException:
        _restore_module_state(mixer, baseline_state)
        raise
    selected_parameter_hashes = _module_state_hashes(mixer)
    rank_parameter_hashes = distributed.all_gather_objects(
        {
            "rank": distributed.rank,
            "module_state_sha256": _sha256_json(selected_parameter_hashes),
        }
    )
    if any(
        row["module_state_sha256"] != rank_parameter_hashes[0]["module_state_sha256"]
        for row in rank_parameter_hashes[1:]
    ):
        _restore_module_state(mixer, baseline_state)
        raise ContractError(
            "dependency-transaction parameters differ across distributed ranks"
        )
    if distributed.is_primary:
        component_reports = {
            name: file_sha256(report_dir / name) for name in component_names
        }
        write_json(
            report_path,
            {
                "schema_version": 1,
                "status": "accepted" if accepted else "rejected",
                "layer": layer_index,
                "boundary": "gdn-time-mix-dependent-activation-fit-generation-v2",
                "world_size": distributed.world_size,
                "fit_rows": int(fit_rows),
                "acceptance_rule": (
                    "finite-and-strictly-lower-loss-and-normalized-mse-"
                    "with-non-regressed-block-mse"
                ),
                "validation_baseline": baseline,
                "validation_candidate": candidate,
                "component_report_sha256": component_reports,
                "baseline_parameter_sha256": baseline_parameter_hashes,
                "proposed_parameter_sha256": proposed_parameter_hashes,
                "selected_parameter_sha256": selected_parameter_hashes,
                "rank_parameter_hashes": list(rank_parameter_hashes),
            },
        )
    distributed.barrier()
    return baseline if accepted else None


def _dependency_transaction_improves(
    baseline: dict[str, float], candidate: dict[str, float]
) -> bool:
    required = ("loss", "normalized_mse", "block_normalized_mse")
    if not all(
        key in baseline
        and key in candidate
        and math.isfinite(float(baseline[key]))
        and math.isfinite(float(candidate[key]))
        for key in required
    ):
        return False
    return (
        float(candidate["loss"]) < float(baseline["loss"])
        and float(candidate["normalized_mse"]) < float(baseline["normalized_mse"])
        and float(candidate["block_normalized_mse"])
        <= float(baseline["block_normalized_mse"])
    )


def _snapshot_module_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone() for name, tensor in module.state_dict().items()
    }


def _restore_module_state(
    module: torch.nn.Module, snapshot: dict[str, torch.Tensor]
) -> None:
    current = module.state_dict()
    if current.keys() != snapshot.keys():
        raise ContractError(
            "activation-fit module state keys changed during transaction"
        )
    with torch.no_grad():
        for name, tensor in current.items():
            source = snapshot[name]
            if tensor.shape != source.shape or tensor.dtype != source.dtype:
                raise ContractError(
                    f"activation-fit module state changed shape or dtype: {name}"
                )
            tensor.copy_(source)


def _module_state_hashes(module: torch.nn.Module) -> dict[str, str]:
    return {
        name: _tensor_sha256(tensor)
        for name, tensor in sorted(module.state_dict().items())
    }


def _generation_mixer_state_sha256(
    path: Path,
    *,
    layer_index: int,
) -> str:
    if not path.is_file():
        raise ContractError(f"immutable mixer generation is missing: {path}")
    prefix = f"model.layers.{layer_index}.attn."
    hashes = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            if not name.startswith(prefix):
                raise ContractError(
                    "immutable mixer generation contains an unexpected tensor"
                )
            hashes[name.removeprefix(prefix)] = _tensor_sha256(
                handle.get_tensor(name)
            )
    if not hashes:
        raise ContractError("immutable mixer generation contains no tensors")
    return _sha256_json(dict(sorted(hashes.items())))


def _cursor(
    layer_index,
    epoch_index,
    next_train_row,
    permutation_sha,
    *,
    consumed_rows,
    train_cache_manifest_sha256,
    validation_cache_manifest_sha256,
    activation_fit_binding=None,
):
    if len(consumed_rows) != next_train_row or len(set(consumed_rows)) != len(
        consumed_rows
    ):
        raise ContractError("layer-major cursor rows are not a unique consumed prefix")
    cursor = {
        "schedule": "rolling-cache-layer-major-v1",
        "active_layer": layer_index,
        "epoch_index": epoch_index,
        "next_train_row": next_train_row,
        "permutation_sha256": permutation_sha,
        "consumed_row_count": len(consumed_rows),
        "consumed_rows_sha256": _sha256_json(list(consumed_rows)),
        "train_cache_manifest_sha256": train_cache_manifest_sha256,
        "validation_cache_manifest_sha256": validation_cache_manifest_sha256,
    }
    if activation_fit_binding is not None:
        cursor["activation_fit_binding"] = {
            "report_sha256": activation_fit_binding["report_sha256"],
            "selected_module_state_sha256": activation_fit_binding[
                "selected_module_state_sha256"
            ],
        }
    return cursor


def _generation_destination(run_dir, optimizer, cursor) -> Path:
    tag = (
        f"l{cursor['active_layer']:03d}-e{cursor['epoch_index']:03d}-"
        f"r{cursor['next_train_row']:09d}-m{optimizer.micro_step:09d}-"
        f"o{optimizer.optimizer_step:09d}"
    )
    return run_dir / "layer-generations" / tag


def _write_rank_training_state(path, optimizer, *, rank, world_size, cursor) -> None:
    torch.save(
        {
            "rank": rank,
            "world_size": world_size,
            "cursor": cursor,
            "optimizer": optimizer.snapshot(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            ),
        },
        path,
    )
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_canonical_training_state(path, optimizer, *, world_size, cursor) -> None:
    torch.save(
        {
            "world_size": world_size,
            "cursor": cursor,
            "optimizer": optimizer.snapshot(),
        },
        path,
    )
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_rank_rng_state(path, *, rank, world_size, cursor) -> None:
    torch.save(
        {
            "rank": rank,
            "world_size": world_size,
            "cursor": cursor,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            ),
        },
        path,
    )
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _prepare_generation(run_dir, store, mixer, optimizer, cursor) -> tuple[Path, Path]:
    destination = _generation_destination(run_dir, optimizer, cursor)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    if destination.exists():
        shutil.rmtree(destination)
    temporary.mkdir(parents=True)
    store.save_generation(
        temporary / "mixer", int(cursor["active_layer"]), mixer, cursor=cursor
    )
    write_json(temporary / "cursor.json", cursor)
    write_json(
        temporary / "training-state-layout.json",
        {
            "schema_version": 1,
            "layout": (
                "canonical-optimizer-with-rank-rng"
                if optimizer.accumulation_step == 0
                else "rank-local-full-state"
            ),
        },
    )
    return destination, temporary


def _commit_distributed_generation(
    distributed,
    run_dir,
    store,
    mixer,
    optimizer,
    cursor,
) -> Path:
    destination = _generation_destination(run_dir, optimizer, cursor)
    temporary = destination.with_name(destination.name + ".tmp")
    prepare_status = None
    if distributed.is_primary:
        try:
            if destination.is_dir():
                manifest_path = destination / "integrity.json"
                if not manifest_path.is_file():
                    raise ContractError(
                        "existing layer generation has no integrity manifest"
                    )
                _verify_generation_integrity(
                    destination,
                    expected_manifest_sha256=file_sha256(manifest_path),
                )
                existing_cursor = json.loads(
                    (destination / "cursor.json").read_text(encoding="utf-8")
                )
                if existing_cursor != cursor:
                    raise ContractError(
                        "existing layer generation cursor differs from replay"
                    )
                prepare_status = {"status": "existing"}
            else:
                capacity = _require_generation_checkpoint_capacity(
                    run_dir,
                    mixer=mixer,
                    optimizer=optimizer,
                    world_size=distributed.world_size,
                )
                write_json(run_dir / "generation-capacity.json", capacity)
                destination, temporary = _prepare_generation(
                    run_dir, store, mixer, optimizer, cursor
                )
                prepare_status = {"status": "prepared"}
        except BaseException as error:
            prepare_status = {"status": "error", "error": repr(error)}
    prepare_status = distributed.broadcast_object(prepare_status)
    if prepare_status["status"] == "error":
        raise ContractError(
            "distributed generation prepare failed: " + prepare_status["error"]
        )
    if prepare_status["status"] == "existing":
        _restore_distributed_generation(
            distributed=distributed,
            destination=destination,
            store=store,
            mixer=mixer,
            optimizer=optimizer,
            cursor=cursor,
        )
        return destination
    canonical_layout = optimizer.accumulation_step == 0
    try:
        if canonical_layout:
            if distributed.is_primary:
                _write_canonical_training_state(
                    temporary / "training-state-canonical.pt",
                    optimizer,
                    world_size=distributed.world_size,
                    cursor=cursor,
                )
            _write_rank_rng_state(
                temporary / f"rng-state-rank-{distributed.rank:03d}.pt",
                rank=distributed.rank,
                world_size=distributed.world_size,
                cursor=cursor,
            )
        else:
            _write_rank_training_state(
                temporary / f"training-state-rank-{distributed.rank:03d}.pt",
                optimizer,
                rank=distributed.rank,
                world_size=distributed.world_size,
                cursor=cursor,
            )
        local_status = {"status": "ok"}
    except BaseException as error:
        local_status = {
            "status": "error",
            "rank": distributed.rank,
            "error": repr(error),
        }
    rank_statuses = distributed.all_gather_objects(local_status)
    failures = [row for row in rank_statuses if row["status"] != "ok"]
    if failures:
        raise ContractError(
            "distributed generation rank-state write failed: "
            + "; ".join(f"rank={row['rank']} error={row['error']}" for row in failures)
        )
    finalize_status = None
    if distributed.is_primary:
        try:
            _write_generation_integrity(temporary)
            _fsync_tree(temporary)
            temporary.rename(destination)
            _fsync_directory(destination.parent)
            finalize_status = {"status": "ok"}
        except BaseException as error:
            finalize_status = {"status": "error", "error": repr(error)}
    finalize_status = distributed.broadcast_object(finalize_status)
    if finalize_status["status"] != "ok":
        raise ContractError(
            "distributed generation finalize failed: " + finalize_status["error"]
        )
    # The newly written generation already matches the live mixer and
    # optimizer. Replaying an existing generation still restores above.
    return destination


def _restore_distributed_generation(
    *, distributed, destination, store, mixer, optimizer, cursor
) -> None:
    status = None
    if distributed.is_primary:
        try:
            _verify_generation_rank_states(
                destination,
                world_size=distributed.world_size,
                expected_cursor=cursor,
            )
            store.restore_generation(
                destination / "mixer",
                int(cursor["active_layer"]),
                expected_cursor=cursor,
            )
            status = {"status": "ok"}
        except BaseException as error:
            status = {"status": "error", "error": repr(error)}
    status = distributed.broadcast_object(status)
    if status["status"] != "ok":
        raise ContractError(
            "distributed generation canonical restore failed: " + status["error"]
        )
    parameter = next(mixer.parameters())
    canonical = store.load_mixer(
        int(cursor["active_layer"]),
        device=parameter.device,
        dtype=parameter.dtype,
    )
    trainable_names = {
        name for name, value in mixer.named_parameters() if value.requires_grad
    }
    snapshot = _load_generation_state(
        destination,
        device=parameter.device,
        rank=distributed.rank,
        world_size=distributed.world_size,
        expected_cursor=cursor,
    )
    optimizer.release()
    mixer.load_state_dict(canonical.state_dict(), strict=True)
    optimizer.activate(
        int(cursor["active_layer"]),
        mixer,
        snapshot=snapshot,
        trainable_names=trainable_names,
    )


def _verify_generation_rank_states(
    generation: Path, *, world_size: int, expected_cursor: dict[str, object]
) -> None:
    layout_path = generation / "training-state-layout.json"
    layout = (
        json.loads(layout_path.read_text(encoding="utf-8")).get("layout")
        if layout_path.is_file()
        else "rank-local-full-state"
    )
    if layout == "canonical-optimizer-with-rank-rng":
        canonical_path = generation / "training-state-canonical.pt"
        expected_rng = {f"rng-state-rank-{rank:03d}.pt" for rank in range(world_size)}
        actual_rng = {path.name for path in generation.glob("rng-state-rank-*.pt")}
        if not canonical_path.is_file() or actual_rng != expected_rng:
            raise ContractError("canonical generation optimizer/RNG coverage mismatch")
        canonical = torch.load(canonical_path, map_location="cpu", weights_only=False)
        if (
            canonical.get("world_size") != world_size
            or canonical.get("cursor") != expected_cursor
            or not isinstance(canonical.get("optimizer"), ActiveLayerOptimizerSnapshot)
        ):
            raise ContractError("canonical generation optimizer binding mismatch")
        for rank in range(world_size):
            payload = torch.load(
                generation / f"rng-state-rank-{rank:03d}.pt",
                map_location="cpu",
                weights_only=False,
            )
            if (
                payload.get("rank") != rank
                or payload.get("world_size") != world_size
                or payload.get("cursor") != expected_cursor
            ):
                raise ContractError(
                    f"canonical generation rank {rank} RNG binding mismatch"
                )
        return
    if layout != "rank-local-full-state":
        raise ContractError(f"unknown generation training-state layout: {layout}")
    expected = {f"training-state-rank-{rank:03d}.pt" for rank in range(world_size)}
    actual = {path.name for path in generation.glob("training-state-rank-*.pt")}
    if actual != expected:
        raise ContractError(
            "layer generation rank-state coverage mismatch: "
            f"expected={sorted(expected)} actual={sorted(actual)}"
        )
    for rank in range(world_size):
        payload = torch.load(
            generation / f"training-state-rank-{rank:03d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        if (
            payload.get("rank") != rank
            or payload.get("world_size") != world_size
            or payload.get("cursor") != expected_cursor
        ):
            raise ContractError(f"layer generation rank {rank} state binding mismatch")


def _require_generation_checkpoint_capacity(run_dir, *, mixer, optimizer, world_size):
    all_parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in mixer.parameters()
    )
    trainable = tuple(
        parameter for parameter in mixer.parameters() if parameter.requires_grad
    )
    rank_state_bytes = (
        sum(
            parameter.numel() * (8 + parameter.element_size())
            for parameter in trainable
        )
        + 16 * 1024 * 1024
    )
    state_copy_count = 1 if optimizer.accumulation_step == 0 else world_size
    estimated_generation_bytes = int(
        (
            all_parameter_bytes
            + state_copy_count * rank_state_bytes
            + world_size * 1024 * 1024
            + 64 * 1024 * 1024
        )
        * 1.15
    )
    required_free_bytes = estimated_generation_bytes + 256 * 1024 * 1024
    available_free_bytes = shutil.disk_usage(run_dir).free
    if available_free_bytes < required_free_bytes:
        raise ContractError(
            "insufficient disk capacity for immutable layer generation: "
            f"required={required_free_bytes} available={available_free_bytes}"
        )
    return {
        "schema_version": 1,
        "generation_retention_limit": 2,
        "estimated_generation_bytes": estimated_generation_bytes,
        "required_free_bytes_before_commit": required_free_bytes,
        "available_free_bytes": available_free_bytes,
        "world_size": world_size,
        "training_state_layout": (
            "canonical-optimizer-with-rank-rng"
            if state_copy_count == 1
            else "rank-local-full-state"
        ),
        "optimizer_state_copy_count": state_copy_count,
        "all_parameter_bytes": all_parameter_bytes,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
    }


def _load_generation_state(
    generation: Path,
    *,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    expected_cursor: dict[str, object] | None = None,
) -> ActiveLayerOptimizerSnapshot:
    canonical_path = generation / "training-state-canonical.pt"
    if canonical_path.is_file():
        value = torch.load(canonical_path, map_location="cpu", weights_only=False)
        rng_path = generation / f"rng-state-rank-{rank:03d}.pt"
        rng = torch.load(rng_path, map_location="cpu", weights_only=False)
        if (
            value.get("world_size") != world_size
            or rng.get("rank") != rank
            or rng.get("world_size") != world_size
            or (
                expected_cursor is not None
                and (
                    value.get("cursor") != expected_cursor
                    or rng.get("cursor") != expected_cursor
                )
            )
        ):
            raise ContractError(
                "canonical rolling-cache optimizer/RNG binding mismatch"
            )
        snapshot = value.get("optimizer")
        if not isinstance(snapshot, ActiveLayerOptimizerSnapshot):
            raise ContractError("invalid canonical rolling-cache optimizer snapshot")
        torch.set_rng_state(rng["torch_rng"])
        if torch.cuda.is_available() and rng.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(rng["cuda_rng"], device=device)
        return snapshot
    state_path = generation / f"training-state-rank-{rank:03d}.pt"
    if not state_path.is_file() and rank == 0 and world_size == 1:
        state_path = generation / "training-state.pt"
    value = torch.load(state_path, map_location="cpu", weights_only=False)
    if (
        value.get("rank", rank) != rank
        or value.get("world_size", world_size) != world_size
        or (expected_cursor is not None and value.get("cursor") != expected_cursor)
    ):
        raise ContractError(
            "rolling-cache training state rank/world-size/cursor mismatch"
        )
    snapshot = value.get("optimizer") if isinstance(value, dict) else None
    if not isinstance(snapshot, ActiveLayerOptimizerSnapshot):
        raise ContractError("invalid rolling-cache optimizer snapshot")
    torch.set_rng_state(value["torch_rng"])
    if torch.cuda.is_available() and value.get("cuda_rng") is not None:
        torch.cuda.set_rng_state(value["cuda_rng"], device=device)
    return snapshot


def _write_progress(
    path,
    *,
    phase,
    active_layer,
    epoch_index,
    next_train_row,
    prefix_fingerprint,
    generation,
    generation_cursor,
    convergence,
    completed_optimizer_steps,
    active_optimizer_steps,
    history,
    base_binding,
    last_train_metrics,
    best_generation=None,
):
    run_root = generation.parents[1].resolve()
    if best_generation is not None:
        best_generation = best_generation.resolve()
        if not (best_generation / "integrity.json").is_file():
            raise ContractError("best layer generation lacks an integrity manifest")
        best_relative = str(best_generation.relative_to(run_root))
        best_manifest_sha = file_sha256(best_generation / "integrity.json")
    else:
        best_relative = None
        best_manifest_sha = None
    write_json(
        path,
        {
            "schema_version": 4,
            "schedule": "rolling-cache-layer-major-v1",
            "status": (
                "layerwise-local-complete"
                if phase == "local-complete"
                else (
                    "exploratory-layer-calibration-complete"
                    if phase == "calibration-complete"
                    else "running"
                )
            ),
            "phase": phase,
            "active_layer": active_layer,
            "epoch_index": epoch_index,
            "next_train_row": next_train_row,
            "prefix_fingerprint": prefix_fingerprint,
            # Generation directories are always run_dir/generations/<id>. Store
            # the run-relative path so a copied --resume manifest remains valid.
            "generation": str(generation.resolve().relative_to(run_root)),
            "generation_manifest_sha256": file_sha256(generation / "integrity.json"),
            "best_generation": best_relative,
            "best_generation_manifest_sha256": best_manifest_sha,
            "generation_cursor": generation_cursor,
            "convergence": asdict(convergence),
            "completed_optimizer_steps": completed_optimizer_steps,
            "active_optimizer_steps": active_optimizer_steps,
            "optimizer_steps": completed_optimizer_steps + active_optimizer_steps,
            "history": history,
            "binding": base_binding,
            "last_train_metrics": last_train_metrics,
        },
    )


def _read_progress(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 4
        or payload.get("schedule") != "rolling-cache-layer-major-v1"
    ):
        raise ContractError(
            "resume progress does not use the rolling-cache layer-major schema"
        )
    phase = payload.get("phase")
    generation = Path(str(payload.get("generation", "")))
    cursor = payload.get("generation_cursor")
    if (
        phase
        not in {
            "train",
            "epoch-complete",
            "layer-ready",
            "local-complete",
            "calibration-complete",
        }
        or generation.is_absolute()
        or not generation.parts
        or ".." in generation.parts
        or not isinstance(cursor, dict)
        or int(cursor.get("consumed_row_count", -1))
        != int(cursor.get("next_train_row", -2))
        or len(str(cursor.get("consumed_rows_sha256", ""))) != 64
        or len(str(cursor.get("train_cache_manifest_sha256", ""))) != 64
        or len(str(cursor.get("validation_cache_manifest_sha256", ""))) != 64
        or len(str(payload.get("generation_manifest_sha256", ""))) != 64
        or int(payload.get("active_layer", -1)) < 0
        or int(payload.get("epoch_index", -1)) < 0
        or int(payload.get("next_train_row", -1)) < 0
        or not isinstance(payload.get("history"), list)
        or not isinstance(payload.get("convergence"), dict)
        or (
            payload.get("best_generation") is None
            and payload.get("best_generation_manifest_sha256") is not None
        )
        or (
            payload.get("best_generation") is not None
            and (
                Path(str(payload["best_generation"])).is_absolute()
                or ".." in Path(str(payload["best_generation"])).parts
                or len(str(payload.get("best_generation_manifest_sha256", ""))) != 64
            )
        )
    ):
        raise ContractError("resume progress has invalid phase, path, cursor, or state")
    if phase == "train" and (
        int(cursor.get("active_layer", -1)) != int(payload["active_layer"])
        or int(cursor.get("epoch_index", -1)) != int(payload["epoch_index"])
        or int(cursor.get("next_train_row", -1)) != int(payload["next_train_row"])
    ):
        raise ContractError("train resume progress and generation cursor differ")
    if phase == "epoch-complete" and (
        int(cursor.get("active_layer", -1)) != int(payload["active_layer"])
        or int(cursor.get("epoch_index", -1)) + 1 != int(payload["epoch_index"])
        or int(payload["next_train_row"]) != 0
    ):
        raise ContractError("epoch-complete progress and generation cursor differ")
    if phase in {"layer-ready", "local-complete", "calibration-complete"} and (
        int(cursor.get("active_layer", -1)) + 1 != int(payload["active_layer"])
        or int(payload["epoch_index"]) != 0
        or int(payload["next_train_row"]) != 0
    ):
        raise ContractError("layer transition progress and selected generation differ")
    return payload


def _validate_progress_binding(progress, base_binding):
    if progress.get("binding") != base_binding:
        raise ContractError(
            "resume progress binding differs from source/data/training contract"
        )


def _resolve_best_generation(run_dir: Path, progress) -> Path | None:
    relative = progress.get("best_generation")
    if relative is None:
        if _state_from_progress(progress).best_epoch is not None:
            raise ContractError("resume convergence state has no bound best generation")
        return None
    best = (run_dir / str(relative)).resolve()
    try:
        best.relative_to(run_dir.resolve())
    except ValueError as error:
        raise ContractError("best generation escapes the run directory") from error
    manifest = best / "integrity.json"
    expected = str(progress.get("best_generation_manifest_sha256", ""))
    if manifest.is_file() and file_sha256(manifest) == expected:
        _verify_generation_integrity(best, expected_manifest_sha256=expected)
        return best

    candidates = [
        candidate.parent
        for candidate in (run_dir / "layer-generations").glob("*/integrity.json")
        if file_sha256(candidate) == expected
    ]
    if len(candidates) != 1:
        raise ContractError(
            "best generation integrity manifest SHA-256 mismatch and immutable "
            f"recovery candidate count is {len(candidates)}"
        )
    recovered = candidates[0].resolve()
    _verify_generation_integrity(recovered, expected_manifest_sha256=expected)
    progress["best_generation"] = str(recovered.relative_to(run_dir.resolve()))
    return recovered


def _state_from_progress(progress) -> LayerConvergenceState:
    value = progress.get("convergence", {})
    return LayerConvergenceState(
        completed_epochs=int(value.get("completed_epochs", 0)),
        best_metric=value.get("best_metric"),
        best_epoch=value.get("best_epoch"),
        best_training_metric=value.get("best_training_metric"),
        best_training_epoch=value.get("best_training_epoch"),
        bad_epochs=int(value.get("bad_epochs", 0)),
    )


def _notify_progress(callback, phase: str, path: Path) -> None:
    if phase in {
        "epoch-complete",
        "layer-ready",
        "local-complete",
        "calibration-complete",
    }:
        payload = json.loads(path.read_text(encoding="utf-8"))
        convergence = payload.get("convergence", {})
        print(
            json.dumps(
                {
                    "event": "any2rwkv-layer-major-progress",
                    "phase": phase,
                    "active_layer": payload.get("active_layer"),
                    "epoch_index": payload.get("epoch_index"),
                    "best_metric": convergence.get("best_metric"),
                    "best_training_metric": convergence.get("best_training_metric"),
                    "bad_epochs": convergence.get("bad_epochs"),
                    "completed_optimizer_steps": payload.get(
                        "completed_optimizer_steps"
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if callback is not None:
        callback(phase, path)


def _validate_resume_permutation(
    progress, *, permutation, permutation_sha: str
) -> None:
    if progress.get("phase") != "train":
        return
    cursor = progress["generation_cursor"]
    next_row = int(progress["next_train_row"])
    if (
        cursor.get("permutation_sha256") != permutation_sha
        or next_row > len(permutation)
        or int(cursor.get("consumed_row_count", -1)) != next_row
        or cursor.get("consumed_rows_sha256")
        != _sha256_json(list(permutation[:next_row]))
    ):
        raise ContractError("resume row permutation differs from the frozen epoch")


def _validate_optimizer_snapshot(snapshot, progress) -> None:
    if (
        snapshot.layer_index != int(progress["active_layer"])
        or snapshot.optimizer_step != int(progress["active_optimizer_steps"])
        or snapshot.micro_step < snapshot.optimizer_step
        or snapshot.accumulation_step < 0
    ):
        raise ContractError("optimizer snapshot differs from layer-major progress")


def _resolve_progress_generation(run_dir: Path, progress) -> Path:
    relative = Path(str(progress["generation"]))
    destination = (run_dir / relative).resolve()
    try:
        destination.relative_to(run_dir.resolve())
    except ValueError as error:
        raise ContractError("resume generation escapes the run directory") from error
    if not destination.is_dir():
        raise ContractError(f"resume generation is missing: {destination}")
    return destination


def _write_generation_integrity(generation: Path) -> None:
    files = {
        str(path.relative_to(generation)): file_sha256(path)
        for path in sorted(generation.rglob("*"))
        if path.is_file() and path.name != "integrity.json"
    }
    if not files:
        raise ContractError("cannot commit an empty layer generation")
    write_json(
        generation / "integrity.json",
        {"schema_version": 1, "files": files},
    )


def _verify_generation_integrity(
    generation: Path,
    *,
    expected_manifest_sha256: str,
) -> None:
    manifest_path = generation / "integrity.json"
    if (
        not manifest_path.is_file()
        or file_sha256(manifest_path) != expected_manifest_sha256
    ):
        raise ContractError("layer generation integrity manifest SHA-256 mismatch")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if payload.get("schema_version") != 1 or not isinstance(files, dict) or not files:
        raise ContractError("layer generation integrity manifest is invalid")
    for relative, expected in files.items():
        path = generation / str(relative)
        if not path.is_file() or file_sha256(path) != expected:
            raise ContractError(f"layer generation file SHA-256 mismatch: {relative}")


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
