from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Callable

import torch
from torch.utils.checkpoint import checkpoint

from ...artifacts import file_sha256, write_json
from ...distill import DEFAULT_VOCAB_CHUNK_SIZE, SweepController, chunked_token_kl
from ...distributed import DistributedContext
from ...errors import ContractError
from ...layer_schedule import epoch_permutation
from ...mixer_store import RWKV7MixerLayerStore
from ...streamed_teacher import StreamedQwen35HybridExecutor, StreamedQwen35Teacher
from ...streaming_training import ActiveLayerOptimizer, ActiveLayerOptimizerSnapshot


class _FrozenMixerCache:
    """Keep frozen recurrent mixers resident when the complete set fits a declared budget."""

    def __init__(self, store, *, enabled: bool, device, dtype) -> None:
        self.store = store
        self.enabled = enabled
        self.device = device
        self.dtype = dtype
        self._mixers: dict[int, torch.nn.Module] = {}
        self.load_count = 0
        self.hit_count = 0

    def load(self, layer_index: int):
        cached = self._mixers.get(layer_index)
        if cached is not None:
            self.hit_count += 1
            return cached
        mixer = self.store.load_mixer(
            layer_index, device=self.device, dtype=self.dtype
        ).requires_grad_(False)
        self.load_count += 1
        if self.enabled:
            self._mixers[layer_index] = mixer
        return mixer

    def invalidate(self, layer_index: int) -> None:
        self._mixers.pop(layer_index, None)

    def put(self, layer_index: int, mixer: torch.nn.Module) -> None:
        if self.enabled:
            self._mixers[layer_index] = mixer.requires_grad_(False)


def _next_token_prediction_window(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    burn_in_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score every post-burn-in token from its preceding-token logits."""
    if logits.ndim != 3 or input_ids.ndim != 2 or logits.shape[:2] != input_ids.shape:
        raise ContractError("global corrective logits/input_ids shape mismatch")
    if burn_in_tokens < 0 or burn_in_tokens >= input_ids.shape[1]:
        raise ContractError(
            "global corrective burn_in_tokens must leave at least one scored token"
        )
    first_label = max(burn_in_tokens, 1)
    prediction_logits = logits[:, first_label - 1 : -1]
    labels = input_ids[:, first_label:]
    if prediction_logits.shape[:2] != labels.shape:
        raise ContractError("global corrective next-token window is misaligned")
    return prediction_logits, labels


def run_global_corrective(
    *,
    source_manifest,
    run_dir: Path,
    zero_step_dir: Path,
    token_rows: tuple[tuple[int, ...], ...],
    validation_rows: tuple[tuple[int, ...], ...],
    plan,
    training_config: Path,
    dataset_manifest: Path,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    progress_callback: Callable[[str, Path], None] | None = None,
) -> dict[str, object]:
    """Run reverse all-recurrent sweeps while updating one mixer at a time."""
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
            raise ContractError("distributed global corrective training requires CUDA")
    dtype = (torch.bfloat16 if device.type == "cuda" else torch.float32) if dtype is None else dtype
    layer_count = source_manifest.contract.num_hidden_layers
    store = RWKV7MixerLayerStore(zero_step_dir, run_dir / "mixer-overlays")
    teacher_probe = StreamedQwen35Teacher(
        source_manifest,
        device=device,
        dtype=dtype,
        cache_layers=False,
        load_output_head=True,
    )
    teacher_decoder_bytes = teacher_probe.loader.tensor_store.estimated_layer_bytes(dtype)
    mixer_decoder_bytes = sum(store.estimated_mixer_bytes(dtype))
    resident_estimated_bytes = teacher_decoder_bytes + mixer_decoder_bytes
    resident_budget = getattr(plan, "corrective_resident_model_max_bytes", None)
    resident_enabled = (
        resident_budget is not None and resident_estimated_bytes <= resident_budget
    )
    teacher = teacher_probe
    teacher.loader.cache_layers = resident_enabled
    frozen_mixers = _FrozenMixerCache(
        store, enabled=resident_enabled, device=device, dtype=dtype
    )
    executor = StreamedQwen35HybridExecutor(teacher)
    binding = {
        "source_checkpoint_sha256": _sha256_json(source_manifest.file_hashes),
        "training_config_sha256": file_sha256(training_config),
        "dataset_manifest_sha256": file_sha256(dataset_manifest),
        "local_mixer_fingerprint": _local_checkpoint_fingerprint(run_dir),
        "precision": str(dtype),
    }
    progress_path = run_dir / "global-corrective-progress.json"
    pre_sweep = run_dir / "global-snapshots" / "pre-sweep"
    if distributed.is_primary and not progress_path.is_file():
        store.snapshot(pre_sweep)
        write_json(
            progress_path,
            {
                "schema_version": 1,
                "status": "running",
                "binding": binding,
                "sweep_index": 0,
                "next_visit": 0,
                "layer_generations": {},
                "history": [],
                "pre_sweep_snapshot": str(pre_sweep.relative_to(run_dir)),
                "pre_sweep_fingerprint": store.snapshot_fingerprint(pre_sweep),
                "world_size": distributed.world_size,
            },
        )
    distributed.barrier()
    progress = _read_progress(
        progress_path, binding=binding, world_size=distributed.world_size
    )
    pre_sweep = run_dir / str(progress["pre_sweep_snapshot"])
    if store.snapshot_fingerprint(pre_sweep) != progress["pre_sweep_fingerprint"]:
        raise ContractError("global corrective pre-sweep snapshot fingerprint mismatch")
    restore_status = None
    if distributed.is_primary:
        try:
            _restore_committed_generations(store, run_dir, progress)
            _gc_global_generations(run_dir, progress)
            _gc_global_snapshots(run_dir, progress)
            restore_status = {"status": "ok"}
        except BaseException as error:
            restore_status = {"status": "error", "error": repr(error)}
    restore_status = distributed.broadcast_object(restore_status)
    if restore_status["status"] != "ok":
        raise ContractError(
            "global corrective restore/retention failed: "
            + restore_status["error"]
        )
    if progress["status"] == "complete":
        return _completed_result(run_dir, progress)

    if "baseline_validation" not in progress:
        baseline_validation = _validate_global(
            teacher=teacher,
            executor=executor,
            store=store,
            frozen_mixers=frozen_mixers,
            rows=validation_rows,
            burn_in_tokens=plan.burn_in_tokens,
            micro_batch_size=plan.micro_batch_size,
            layer_count=layer_count,
            device=device,
            dtype=dtype,
            distributed=distributed,
        )
        progress["baseline_validation"] = baseline_validation
        _publish_progress_with_retention(
            distributed=distributed,
            run_dir=run_dir,
            progress_path=progress_path,
            progress=progress,
            progress_callback=progress_callback,
            phase="global-baseline-complete",
        )
        progress = _read_progress(
            progress_path, binding=binding, world_size=distributed.world_size
        )

    controller = SweepController(
        min_sweeps=plan.corrective_min_sweeps,
        max_sweeps=plan.corrective_max_sweeps,
        min_delta=plan.corrective_min_delta,
        order=f"{layer_count - 1}..0",
        history=list(progress["history"]),
        baseline_validation_kl=float(
            progress["baseline_validation"]["token_kl"]
        ),
        baseline_checkpoint=str(progress["pre_sweep_snapshot"]),
    )
    order = tuple(reversed(range(layer_count)))
    global_micro_batch_size = plan.micro_batch_size * distributed.world_size
    micro_batches_per_visit = (
        len(token_rows) + global_micro_batch_size - 1
    ) // global_micro_batch_size
    optimizer_steps_per_visit = (
        micro_batches_per_visit + plan.accumulation_steps - 1
    ) // plan.accumulation_steps
    total_optimizer_steps = optimizer_steps_per_visit * plan.corrective_max_sweeps
    configured_warmup_steps = getattr(plan, "learning_rate_warmup_steps", None)
    warmup_steps = (
        int(configured_warmup_steps)
        if configured_warmup_steps is not None
        else int(
            total_optimizer_steps
            * getattr(plan, "learning_rate_warmup_ratio", 0.0)
        )
    )
    while True:
        sweep_index = int(progress["sweep_index"])
        next_visit = int(progress["next_visit"])
        if next_visit == 0:
            start_snapshot = (
                run_dir / "global-snapshots" / f"sweep-{sweep_index:03d}-start"
            )
            snapshot_status = None
            if distributed.is_primary:
                try:
                    store.snapshot(start_snapshot)
                    progress["sweep_start_checkpoint"] = str(
                        start_snapshot.relative_to(run_dir)
                    )
                    progress["sweep_start_fingerprint"] = store.snapshot_fingerprint(
                        start_snapshot
                    )
                    write_json(progress_path, progress)
                    _gc_global_snapshots(run_dir, progress)
                    snapshot_status = {"status": "ok"}
                except BaseException as error:
                    snapshot_status = {"status": "error", "error": repr(error)}
            snapshot_status = distributed.broadcast_object(snapshot_status)
            if snapshot_status["status"] != "ok":
                raise ContractError(
                    "global corrective sweep-start publish failed: "
                    + snapshot_status["error"]
                )
            progress = _read_progress(
                progress_path, binding=binding, world_size=distributed.world_size
            )
        start_checkpoint = run_dir / str(progress["sweep_start_checkpoint"])
        if (
            store.snapshot_fingerprint(start_checkpoint)
            != progress["sweep_start_fingerprint"]
        ):
            raise ContractError("global corrective sweep-start fingerprint mismatch")
        for visit_index in range(next_visit, layer_count):
            layer_index = order[visit_index]
            frozen_mixers.invalidate(layer_index)
            mixer = store.load_mixer(layer_index, device=device, dtype=dtype)
            optimizer = ActiveLayerOptimizer(
                optimizer_name=getattr(plan, "optimizer_name", "adamw"),
                learning_rate=plan.learning_rate,
                final_learning_rate=getattr(
                    plan, "final_learning_rate", plan.learning_rate
                ),
                adam_betas=getattr(plan, "optimizer_betas", (0.9, 0.999)),
                adam_epsilon=getattr(plan, "optimizer_epsilon", 1e-8),
                weight_decay=getattr(plan, "optimizer_weight_decay", 0.0),
                detailed_telemetry_interval_steps=getattr(
                    plan, "optimizer_telemetry_interval_steps", 1
                ),
                learning_rate_schedule=getattr(
                    plan, "learning_rate_schedule", "constant"
                ),
                warmup_steps=warmup_steps,
                total_steps=total_optimizer_steps,
                min_learning_rate_ratio=getattr(
                    plan, "min_learning_rate_ratio", 1.0
                ),
                gradient_clip_norm=getattr(plan, "gradient_clip_norm", None),
                max_parameter_update_relative_l2=getattr(
                    plan, "max_parameter_update_relative_l2", None
                ),
                gradient_sync=distributed.synchronize_gradients,
            )
            snapshot = _load_latest_optimizer_snapshot(
                run_dir,
                progress["layer_generations"],
                layer_index,
                device=device,
                distributed=distributed,
            )
            distributed.synchronize_module_parameters(mixer)
            optimizer.activate(layer_index, mixer, snapshot=snapshot)
            distributed.validate_trainable_signature(mixer)
            permutation, permutation_sha = epoch_permutation(
                row_count=len(token_rows),
                seed=plan.seed,
                layer=layer_index,
                epoch=sweep_index,
            )
            totals = {"token_kl": 0.0, "shifted_ce": 0.0, "loss": 0.0}
            local_rows_seen = 0
            rows_seen = 0
            row_position = 0
            while row_position < len(permutation):
                row_stop = min(row_position + global_micro_batch_size, len(permutation))
                trailing = len(permutation) - row_stop
                if 0 < trailing < distributed.world_size:
                    row_stop = len(permutation)
                global_indices = permutation[row_position:row_stop]
                indices = distributed.shard_rows(global_indices)
                input_ids = torch.tensor(
                    [token_rows[index] for index in indices],
                    dtype=torch.long,
                    device=device,
                )
                with torch.no_grad():
                    teacher_logits = teacher.forward(input_ids).logits.detach().clone()
                student_logits = _student_logits_for_active_layer(
                    teacher=teacher,
                    executor=executor,
                    store=store,
                    frozen_mixers=frozen_mixers,
                    input_ids=input_ids,
                    active_layer=layer_index,
                    active_mixer=mixer,
                    layer_count=layer_count,
                    device=device,
                    dtype=dtype,
                )
                student_window, labels = _next_token_prediction_window(
                    student_logits, input_ids, plan.burn_in_tokens
                )
                teacher_window, teacher_labels = _next_token_prediction_window(
                    teacher_logits, input_ids, plan.burn_in_tokens
                )
                if not torch.equal(labels, teacher_labels):
                    raise ContractError("teacher/student next-token labels differ")
                token_kl = chunked_token_kl(
                    student_window,
                    teacher_window,
                    vocab_chunk_size=DEFAULT_VOCAB_CHUNK_SIZE,
                )
                shifted_ce = torch.nn.functional.cross_entropy(
                    student_window.reshape(-1, student_window.shape[-1]),
                    labels.reshape(-1),
                )
                loss = (
                    plan.global_loss_weights.token_kl * token_kl
                    + plan.global_loss_weights.shifted_ce * shifted_ce
                )
                loss_scale = len(indices) * distributed.world_size / len(global_indices)
                optimizer.backward(
                    loss * loss_scale,
                    accumulation_steps=plan.accumulation_steps,
                    sample_weight=len(global_indices),
                )
                count = len(indices)
                local_metrics = {
                    "token_kl": float(token_kl.detach()),
                    "shifted_ce": float(shifted_ce.detach()),
                    "loss": float(loss.detach()),
                }
                local_rows_seen += count
                for name, value in local_metrics.items():
                    totals[name] += count * value
                global_count = len(global_indices)
                rows_seen += global_count
                row_position = row_stop
            optimizer.flush(accumulation_steps=plan.accumulation_steps)
            metrics = distributed.aggregate_metrics(
                {
                    name: value / local_rows_seen
                    for name, value in totals.items()
                },
                local_rows_seen,
            )
            optimizer_snapshot = optimizer.release()
            cursor = {
                "schedule": "fully-recurrent-global-corrective-v1",
                "sweep_index": sweep_index,
                "visit_index": visit_index,
                "layer_index": layer_index,
                "permutation_sha256": permutation_sha,
                "row_count": rows_seen,
            }
            generation = _commit_distributed_layer_generation(
                distributed=distributed,
                run_dir=run_dir,
                store=store,
                mixer=mixer,
                optimizer_snapshot=optimizer_snapshot,
                cursor=cursor,
            )
            restore_status = None
            if distributed.is_primary:
                try:
                    store.restore_generation(
                        generation / "mixer",
                        layer_index,
                        expected_cursor=cursor,
                    )
                    restore_status = {"status": "ok"}
                except BaseException as error:
                    restore_status = {"status": "error", "error": repr(error)}
            restore_status = distributed.broadcast_object(restore_status)
            if restore_status["status"] != "ok":
                raise ContractError(
                    "global generation mixer restore failed: "
                    + restore_status["error"]
                )
            _load_generation_training_state_collective(
                generation,
                expected_cursor=cursor,
                device=device,
                distributed=distributed,
            )
            mixer = store.load_mixer(layer_index, device=device, dtype=dtype)
            frozen_mixers.put(layer_index, mixer)
            progress["layer_generations"][str(layer_index)] = {
                "path": str(generation.relative_to(run_dir)),
                "integrity_sha256": file_sha256(generation / "integrity.json"),
                "cursor": cursor,
            }
            progress["next_visit"] = visit_index + 1
            progress["last_visit_metrics"] = {
                name: value for name, value in metrics.items()
            }
            _publish_progress_with_retention(
                distributed=distributed,
                run_dir=run_dir,
                progress_path=progress_path,
                progress=progress,
                progress_callback=progress_callback,
                phase="global-layer-committed",
            )

        validation = _validate_global(
            teacher=teacher,
            executor=executor,
            store=store,
            frozen_mixers=frozen_mixers,
            rows=validation_rows,
            burn_in_tokens=plan.burn_in_tokens,
            micro_batch_size=plan.micro_batch_size,
            layer_count=layer_count,
            device=device,
            dtype=dtype,
            distributed=distributed,
        )
        end_snapshot = run_dir / "global-snapshots" / f"sweep-{sweep_index:03d}"
        end_status = None
        if distributed.is_primary:
            try:
                store.snapshot(end_snapshot)
                end_status = {"status": "ok"}
            except BaseException as error:
                end_status = {"status": "error", "error": repr(error)}
        end_status = distributed.broadcast_object(end_status)
        if end_status["status"] != "ok":
            raise ContractError(
                "global corrective sweep-end snapshot failed: "
                + end_status["error"]
            )
        decision = controller.complete(
            start_checkpoint=str(start_checkpoint.relative_to(run_dir)),
            end_checkpoint=str(end_snapshot.relative_to(run_dir)),
            validation_kl=validation["token_kl"],
            token_budget=len(token_rows) * len(token_rows[0]) * layer_count,
        )
        decision["validation"] = validation
        progress["history"] = controller.history
        if decision["stop"]:
            selected = run_dir / str(decision["selected_checkpoint"])
            checkpoint_dir = run_dir / "checkpoint-global-corrective"
            completion_status = None
            if distributed.is_primary:
                try:
                    store.restore_snapshot(selected)
                    if not checkpoint_dir.is_dir():
                        store.materialize_checkpoint(
                            checkpoint_dir,
                            training_stage="fully-recurrent-global-corrective",
                            fitted_evidence_root=run_dir,
                        )
                    progress.update(
                        {
                            "status": "complete",
                            "selected_checkpoint": str(selected.relative_to(run_dir)),
                            "selected_fingerprint": store.fingerprint(),
                            "checkpoint": str(checkpoint_dir),
                        }
                    )
                    write_json(progress_path, progress)
                    write_json(
                        run_dir / "global-corrective.json",
                        {
                            "schema_version": 1,
                            "status": "complete",
                            "world_size": distributed.world_size,
                            "order": f"{layer_count - 1}..0",
                            "history": controller.history,
                            "selected_checkpoint": progress["selected_checkpoint"],
                            "selected_fingerprint": progress["selected_fingerprint"],
                            "hf_checkpoint": str(checkpoint_dir),
                        },
                    )
                    _gc_global_generations(run_dir, progress)
                    _gc_global_snapshots(run_dir, progress)
                    _notify_progress(
                        progress_callback, "global-sweep-complete", progress_path
                    )
                    completion_status = {"status": "ok"}
                except BaseException as error:
                    completion_status = {"status": "error", "error": repr(error)}
            completion_status = distributed.broadcast_object(completion_status)
            if completion_status["status"] != "ok":
                raise ContractError(
                    "global corrective completion publish failed: "
                    + completion_status["error"]
                )
            _write_corrective_residency(
                distributed=distributed,
                run_dir=run_dir,
                teacher=teacher,
                frozen_mixers=frozen_mixers,
                enabled=resident_enabled,
                estimated_bytes=resident_estimated_bytes,
                budget_bytes=resident_budget,
            )
            progress = _read_progress(
                progress_path, binding=binding, world_size=distributed.world_size
            )
            return _completed_result(run_dir, progress)
        progress["sweep_index"] = sweep_index + 1
        progress["next_visit"] = 0
        progress.pop("sweep_start_checkpoint", None)
        progress.pop("sweep_start_fingerprint", None)
        _publish_progress_with_retention(
            distributed=distributed,
            run_dir=run_dir,
            progress_path=progress_path,
            progress=progress,
            progress_callback=progress_callback,
            phase="global-sweep-complete",
        )


def _student_logits_for_active_layer(
    *,
    teacher,
    executor,
    store,
    frozen_mixers,
    input_ids,
    active_layer,
    active_mixer,
    layer_count,
    device,
    dtype,
):
    hidden = teacher.embed_input_ids(input_ids)
    shared = None
    for layer_index in range(active_layer):
        loaded = teacher.loader.load_layer(layer_index, device=device, dtype=dtype)
        mixer = frozen_mixers.load(layer_index)
        hidden, shared = executor.forward_cached_target_block(
            hidden,
            shared_states=shared,
            layer_index=layer_index,
            mixer=mixer,
            loaded_layer=loaded,
        )
    loaded_active = teacher.loader.load_layer(active_layer, device=device, dtype=dtype)
    hidden, shared = executor.forward_cached_target_block_with_grad(
        hidden,
        shared_states=shared,
        layer_index=active_layer,
        mixer=active_mixer,
        loaded_layer=loaded_active,
        mixer_trainable=True,
    )
    if shared is None:
        raise ContractError("active recurrent layer did not produce shared v_first state")
    for layer_index in range(active_layer + 1, layer_count):
        def frozen_suffix(value, shared_value, *, index=layer_index):
            loaded = teacher.loader.load_layer(index, device=device, dtype=dtype)
            mixer = frozen_mixers.load(index)
            output, next_shared = executor.forward_cached_target_block_with_grad(
                value,
                shared_states=shared_value,
                layer_index=index,
                mixer=mixer,
                loaded_layer=loaded,
            )
            if next_shared is None:
                raise ContractError("recurrent suffix lost shared v_first state")
            return output, next_shared

        hidden, shared = checkpoint(
            frozen_suffix,
            hidden,
            shared,
            use_reentrant=False,
        )
    return teacher.project_logits(hidden)


def _student_logits_frozen(
    *, teacher, executor, frozen_mixers, input_ids, layer_count, device, dtype
):
    hidden = teacher.embed_input_ids(input_ids)
    shared = None
    with torch.no_grad():
        for layer_index in range(layer_count):
            loaded = teacher.loader.load_layer(layer_index, device=device, dtype=dtype)
            mixer = frozen_mixers.load(layer_index)
            hidden, shared = executor.forward_cached_target_block(
                hidden,
                shared_states=shared,
                layer_index=layer_index,
                mixer=mixer,
                loaded_layer=loaded,
            )
        return teacher.project_logits(hidden)


def _validate_global(
    *, teacher, executor, store, frozen_mixers, rows, burn_in_tokens, micro_batch_size,
    layer_count, device, dtype, distributed,
):
    token_kl_sum = 0.0
    student_nll_sum = 0.0
    teacher_nll_sum = 0.0
    kl_token_count = 0
    nll_token_count = 0
    global_micro_batch_size = micro_batch_size * distributed.world_size
    row_position = 0
    while row_position < len(rows):
        row_stop = min(row_position + global_micro_batch_size, len(rows))
        trailing = len(rows) - row_stop
        if 0 < trailing < distributed.world_size:
            row_stop = len(rows)
        global_rows = rows[row_position:row_stop]
        batch_rows = distributed.shard_rows(global_rows)
        input_ids = torch.tensor(batch_rows, dtype=torch.long, device=device)
        with torch.no_grad():
            teacher_logits = teacher.forward(input_ids).logits.detach().clone()
            student_logits = _student_logits_frozen(
                teacher=teacher, executor=executor, frozen_mixers=frozen_mixers,
                input_ids=input_ids, layer_count=layer_count,
                device=device, dtype=dtype,
            )
            student_window, labels = _next_token_prediction_window(
                student_logits, input_ids, burn_in_tokens
            )
            teacher_window, teacher_labels = _next_token_prediction_window(
                teacher_logits, input_ids, burn_in_tokens
            )
            if not torch.equal(labels, teacher_labels):
                raise ContractError("teacher/student next-token labels differ")
            kl = chunked_token_kl(
                student_window, teacher_window,
                vocab_chunk_size=DEFAULT_VOCAB_CHUNK_SIZE,
            )
            ce = torch.nn.functional.cross_entropy(
                student_window.reshape(-1, student_window.shape[-1]),
                labels.reshape(-1),
            )
            teacher_ce = torch.nn.functional.cross_entropy(
                teacher_window.reshape(-1, teacher_window.shape[-1]),
                labels.reshape(-1),
            )
        local_kl_tokens = len(batch_rows) * int(student_window.shape[1])
        local_nll_tokens = local_kl_tokens
        token_kl_sum += local_kl_tokens * float(kl)
        student_nll_sum += local_nll_tokens * float(ce)
        teacher_nll_sum += local_nll_tokens * float(teacher_ce)
        kl_token_count += local_kl_tokens
        nll_token_count += local_nll_tokens
        row_position = row_stop
    totals = torch.tensor(
        [
            token_kl_sum,
            student_nll_sum,
            teacher_nll_sum,
            kl_token_count,
            nll_token_count,
        ],
        dtype=torch.float64,
        device=device,
    )
    distributed.all_reduce_sum(totals)
    token_kl_sum = float(totals[0].item())
    student_nll_sum = float(totals[1].item())
    teacher_nll_sum = float(totals[2].item())
    kl_token_count = int(totals[3].item())
    nll_token_count = int(totals[4].item())
    if kl_token_count <= 0 or nll_token_count <= 0:
        raise ContractError("global validation produced no scored tokens")
    token_kl = token_kl_sum / kl_token_count
    shifted_ce = student_nll_sum / nll_token_count
    teacher_shifted_ce = teacher_nll_sum / nll_token_count
    nll_delta_sum = student_nll_sum - teacher_nll_sum
    return {
        "token_kl": token_kl,
        "shifted_ce": shifted_ce,
        "teacher_shifted_ce": teacher_shifted_ce,
        "ppl": math.exp(shifted_ce),
        "ppl_ratio": math.exp(nll_delta_sum / nll_token_count),
        "kl_token_count": kl_token_count,
        "token_kl_sum": token_kl_sum,
        "nll_token_count": nll_token_count,
        "student_nll_sum": student_nll_sum,
        "teacher_nll_sum": teacher_nll_sum,
        "nll_delta_sum": nll_delta_sum,
    }


def _write_corrective_residency(
    *,
    distributed,
    run_dir,
    teacher,
    frozen_mixers,
    enabled,
    estimated_bytes,
    budget_bytes,
):
    per_rank = distributed.gather_scalar_metrics(
        {
            "teacher_layer_loads": teacher.loader.layer_load_count,
            "teacher_cache_hits": teacher.loader.cache_hit_count,
            "student_mixer_loads": frozen_mixers.load_count,
            "student_mixer_cache_hits": frozen_mixers.hit_count,
        }
    )
    if distributed.is_primary:
        write_json(
            run_dir / "corrective-residency.json",
            {
                "schema_version": 1,
                "world_size": distributed.world_size,
                "mode": "resident" if enabled else "streamed",
                "estimated_decoder_bytes_per_rank": estimated_bytes,
                "budget_bytes_per_rank": budget_bytes,
                "per_rank": per_rank,
            },
        )


def _commit_distributed_layer_generation(
    *, distributed, run_dir, store, mixer, optimizer_snapshot, cursor
):
    generation = _global_generation_destination(run_dir, cursor)
    temporary = generation.with_name(generation.name + ".tmp")
    prepare_status = None
    if distributed.is_primary:
        try:
            if generation.is_dir():
                _verify_global_generation_self(
                    generation,
                    expected_cursor=cursor,
                    world_size=distributed.world_size,
                )
                prepare_status = {"status": "existing"}
            else:
                capacity = _require_global_generation_capacity(
                    run_dir,
                    mixer=mixer,
                    optimizer_snapshot=optimizer_snapshot,
                    world_size=distributed.world_size,
                )
                write_json(run_dir / "global-generation-capacity.json", capacity)
                generation, temporary = _prepare_layer_generation(
                    run_dir=run_dir,
                    store=store,
                    mixer=mixer,
                    cursor=cursor,
                )
                prepare_status = {"status": "prepared"}
        except BaseException as error:
            prepare_status = {"status": "error", "error": repr(error)}
    prepare_status = distributed.broadcast_object(prepare_status)
    if prepare_status["status"] == "error":
        raise ContractError(
            "global generation prepare failed: " + prepare_status["error"]
        )
    if prepare_status["status"] == "existing":
        return generation
    try:
        _write_rank_training_state(
            temporary / f"training-state-rank-{distributed.rank:03d}.pt",
            optimizer_snapshot,
            rank=distributed.rank,
            world_size=distributed.world_size,
            cursor=cursor,
        )
        local_status = {"status": "ok", "rank": distributed.rank}
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
            "global generation rank-state write failed: "
            + "; ".join(
                f"rank={row['rank']} error={row['error']}" for row in failures
            )
        )
    finalize_status = None
    if distributed.is_primary:
        try:
            files = {
                str(path.relative_to(temporary)): file_sha256(path)
                for path in sorted(temporary.rglob("*")) if path.is_file()
            }
            write_json(
                temporary / "integrity.json",
                {"schema_version": 1, "files": files},
            )
            _fsync_tree(temporary)
            temporary.rename(generation)
            _fsync_directory(generation.parent)
            finalize_status = {"status": "ok"}
        except BaseException as error:
            finalize_status = {"status": "error", "error": repr(error)}
    finalize_status = distributed.broadcast_object(finalize_status)
    if finalize_status["status"] != "ok":
        raise ContractError(
            "global generation finalize failed: " + finalize_status["error"]
        )
    return generation


def _global_generation_destination(run_dir, cursor):
    return (
        run_dir / "global-generations"
        / f"s{cursor['sweep_index']:03d}-v{cursor['visit_index']:03d}-l{cursor['layer_index']:03d}"
    )


def _prepare_layer_generation(*, run_dir, store, mixer, cursor):
    destination = _global_generation_destination(run_dir, cursor)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    if destination.exists():
        raise ContractError("refusing to replace a published global generation")
    temporary.mkdir(parents=True)
    store.save_generation(
        temporary / "mixer", int(cursor["layer_index"]), mixer, cursor=cursor
    )
    write_json(temporary / "cursor.json", cursor)
    return destination, temporary


def _verify_global_generation_self(
    generation, *, expected_cursor, world_size
):
    manifest = generation / "integrity.json"
    cursor_path = generation / "cursor.json"
    if not manifest.is_file() or not cursor_path.is_file():
        raise ContractError("existing global generation is incomplete")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    files = payload.get("files")
    if payload.get("schema_version") != 1 or not isinstance(files, dict) or not files:
        raise ContractError("existing global generation integrity manifest is invalid")
    if json.loads(cursor_path.read_text(encoding="utf-8")) != expected_cursor:
        raise ContractError("existing global generation cursor differs from replay")
    for relative, expected in files.items():
        path = generation / str(relative)
        if not path.is_file() or file_sha256(path) != expected:
            raise ContractError(
                f"existing global generation file hash mismatch: {relative}"
            )
    expected_states = {
        f"training-state-rank-{rank:03d}.pt" for rank in range(world_size)
    }
    actual_states = {
        path.name for path in generation.glob("training-state-rank-*.pt")
    }
    if actual_states != expected_states:
        raise ContractError(
            "existing global generation rank-state coverage mismatch"
        )
    for rank in range(world_size):
        _read_generation_training_state(
            generation,
            rank=rank,
            world_size=world_size,
            expected_cursor=expected_cursor,
        )


def _require_global_generation_capacity(
    run_dir, *, mixer, optimizer_snapshot, world_size
):
    mixer_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in mixer.state_dict().values()
    )
    optimizer_bytes = _nested_tensor_bytes(
        (optimizer_snapshot.optimizer, optimizer_snapshot.gradients)
    )
    estimated_generation_bytes = int(
        (mixer_bytes + world_size * (optimizer_bytes + 16 * 1024 * 1024)) * 1.15
    )
    required_free_bytes = estimated_generation_bytes + 256 * 1024 * 1024
    available_free_bytes = shutil.disk_usage(run_dir).free
    if available_free_bytes < required_free_bytes:
        raise ContractError(
            "insufficient disk capacity for immutable global generation: "
            f"required={required_free_bytes} available={available_free_bytes}"
        )
    return {
        "schema_version": 1,
        "estimated_generation_bytes": estimated_generation_bytes,
        "required_free_bytes_before_commit": required_free_bytes,
        "available_free_bytes": available_free_bytes,
        "world_size": world_size,
        "mixer_bytes": mixer_bytes,
        "optimizer_bytes_per_rank": optimizer_bytes,
    }


def _nested_tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_nested_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_nested_tensor_bytes(item) for item in value)
    return 0


def _write_rank_training_state(
    path, optimizer_snapshot, *, rank, world_size, cursor
):
    torch.save(
        {
            "rank": rank,
            "world_size": world_size,
            "cursor": cursor,
            "optimizer": optimizer_snapshot,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        },
        path,
    )
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _load_latest_optimizer_snapshot(
    run_dir, layer_generations, layer_index, *, device, distributed
):
    entry = layer_generations.get(str(layer_index))
    if entry is None:
        return None
    generation = run_dir / str(entry["path"])
    _verify_generation(generation, entry)
    payload = _load_generation_training_state_collective(
        generation,
        expected_cursor=entry["cursor"],
        device=device,
        distributed=distributed,
    )
    snapshot = payload.get("optimizer")
    if not isinstance(snapshot, ActiveLayerOptimizerSnapshot):
        raise ContractError("global corrective optimizer snapshot is invalid")
    return snapshot


def _read_generation_training_state(
    generation, *, rank, world_size, expected_cursor
):
    state_path = generation / f"training-state-rank-{rank:03d}.pt"
    if not state_path.is_file() and rank == 0 and world_size == 1:
        state_path = generation / "training-state.pt"
    if not state_path.is_file():
        raise ContractError(
            f"global corrective generation lacks rank {rank} training state"
        )
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    if (
        payload.get("rank", rank) != rank
        or payload.get("world_size") != world_size
        or payload.get("cursor") != expected_cursor
    ):
        raise ContractError(
            "global corrective training state rank/world-size/cursor mismatch"
        )
    return payload


def _load_generation_training_state_collective(
    generation, *, expected_cursor, device, distributed
):
    payload = None
    try:
        payload = _read_generation_training_state(
            generation,
            rank=distributed.rank,
            world_size=distributed.world_size,
            expected_cursor=expected_cursor,
        )
        local_status = {"status": "ok", "rank": distributed.rank}
    except BaseException as error:
        local_status = {
            "status": "error",
            "rank": distributed.rank,
            "error": repr(error),
        }
    statuses = distributed.all_gather_objects(local_status)
    failures = [row for row in statuses if row["status"] != "ok"]
    if failures:
        raise ContractError(
            "global generation rank-state restore failed: "
            + "; ".join(
                f"rank={row['rank']} error={row['error']}" for row in failures
            )
        )
    assert payload is not None
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
        torch.cuda.set_rng_state(payload["cuda_rng"], device=device)
    return payload


def _restore_committed_generations(store, run_dir, progress):
    pre_sweep = run_dir / str(progress["pre_sweep_snapshot"])
    store.restore_snapshot(pre_sweep)
    for layer_text, entry in progress["layer_generations"].items():
        generation = run_dir / str(entry["path"])
        _verify_generation(generation, entry)
        store.restore_generation(
            generation / "mixer", int(layer_text), expected_cursor=entry["cursor"]
        )


def _publish_progress_with_retention(
    *, distributed, run_dir, progress_path, progress, progress_callback, phase
):
    status = None
    if distributed.is_primary:
        try:
            write_json(progress_path, progress)
            _gc_global_generations(run_dir, progress)
            _gc_global_snapshots(run_dir, progress)
            status = {"status": "ok"}
        except BaseException as error:
            status = {"status": "error", "error": repr(error)}
    status = distributed.broadcast_object(status)
    if status["status"] != "ok":
        raise ContractError(
            "global corrective progress/retention failed: " + status["error"]
        )
    if distributed.world_size == 1:
        _notify_progress(progress_callback, phase, progress_path)
        return
    callback_status = None
    if distributed.is_primary:
        try:
            _notify_progress(progress_callback, phase, progress_path)
            callback_status = {"status": "ok"}
        except BaseException as error:
            callback_status = {"status": "error", "error": repr(error)}
    callback_status = distributed.broadcast_object(callback_status)
    if callback_status["status"] != "ok":
        raise ContractError(
            "global corrective progress callback failed: "
            + callback_status["error"]
        )


def _gc_global_generations(run_dir: Path, progress: dict[str, object]) -> None:
    root = run_dir / "global-generations"
    root.mkdir(parents=True, exist_ok=True)
    entries = progress.get("layer_generations", {})
    referenced = set()
    for entry in entries.values():
        generation = (run_dir / str(entry["path"])).resolve()
        if generation.parent != root.resolve():
            raise ContractError(
                "global corrective generation path escapes retention root"
            )
        referenced.add(generation)
    removed: list[str] = []
    for candidate in sorted(root.iterdir()):
        if candidate.resolve() in referenced:
            continue
        if candidate.is_dir():
            shutil.rmtree(candidate)
        else:
            candidate.unlink()
        removed.append(candidate.name)
    write_json(
        run_dir / "global-generation-retention.json",
        {
            "schema_version": 1,
            "policy": "latest-generation-per-layer",
            "retention_limit": len(entries),
            "referenced": sorted(str(path.relative_to(run_dir)) for path in referenced),
            "removed": removed,
        },
    )


def _gc_global_snapshots(run_dir: Path, progress: dict[str, object]) -> None:
    root = run_dir / "global-snapshots"
    root.mkdir(parents=True, exist_ok=True)
    references = set()
    for field in (
        "pre_sweep_snapshot",
        "sweep_start_checkpoint",
        "selected_checkpoint",
    ):
        value = progress.get(field)
        if value:
            references.add(str(value))
    history = progress.get("history", [])
    if history:
        best = min(history, key=lambda row: float(row["validation_kl"]))
        references.add(str(best["end_checkpoint"]))
    referenced = set()
    for value in references:
        snapshot = (run_dir / value).resolve()
        if snapshot.parent != root.resolve():
            raise ContractError(
                "global corrective snapshot path escapes retention root"
            )
        referenced.add(snapshot)
    removed: list[str] = []
    for candidate in sorted(root.iterdir()):
        if candidate.resolve() in referenced:
            continue
        if candidate.is_dir():
            shutil.rmtree(candidate)
        else:
            candidate.unlink()
        removed.append(candidate.name)
    write_json(
        run_dir / "global-snapshot-retention.json",
        {
            "schema_version": 1,
            "policy": "pre-sweep-current-start-best",
            "retention_limit": 3,
            "referenced": sorted(
                str(path.relative_to(run_dir)) for path in referenced
            ),
            "removed": removed,
        },
    )


def _verify_generation(generation, entry):
    manifest = generation / "integrity.json"
    if not manifest.is_file() or file_sha256(manifest) != entry["integrity_sha256"]:
        raise ContractError("global corrective generation integrity SHA mismatch")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for relative, expected in payload.get("files", {}).items():
        path = generation / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ContractError(f"global corrective generation file hash mismatch: {relative}")


def _read_progress(path, *, binding, world_size):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("status") not in {"running", "complete"}
        or payload.get("binding") != binding
        or payload.get("world_size") != world_size
        or not isinstance(payload.get("layer_generations"), dict)
        or not isinstance(payload.get("history"), list)
    ):
        raise ContractError("global corrective progress binding or schema mismatch")
    return payload


def _completed_result(run_dir, progress):
    return {
        "status": "fully-recurrent-global-corrective-complete",
        "checkpoint": str(run_dir / "checkpoint-global-corrective"),
        "selected_checkpoint": progress.get("selected_checkpoint"),
        "selected_fingerprint": progress.get("selected_fingerprint"),
        "history": progress["history"],
    }


def _local_checkpoint_fingerprint(run_dir: Path) -> str:
    config_path = run_dir / "checkpoint-layerwise-local" / "config.json"
    if not config_path.is_file():
        raise ContractError("layerwise-local checkpoint is missing before corrective stage")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    value = str(payload.get("any_to_rwkv", {}).get("mixer_overlay_fingerprint", ""))
    if len(value) != 64:
        raise ContractError("layerwise-local checkpoint lacks mixer fingerprint")
    return value


def _sha256_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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


def _notify_progress(
    callback: Callable[[str, Path], None] | None,
    phase: str,
    progress_path: Path,
) -> None:
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "event": "any2rwkv-global-corrective-progress",
                "phase": phase,
                "status": payload.get("status"),
                "sweep_index": payload.get("sweep_index"),
                "next_visit": payload.get("next_visit"),
                "last_visit_metrics": payload.get("last_visit_metrics"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if callback is not None:
        callback(phase, progress_path)
