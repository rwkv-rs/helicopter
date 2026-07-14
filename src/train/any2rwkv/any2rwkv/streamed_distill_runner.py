from __future__ import annotations

import hashlib
import io
import json
import math
import random
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
    teacher = StreamedQwen35Teacher(
        source_manifest,
        device=device,
        dtype=dtype,
    )
    executor = StreamedQwen35HybridExecutor(teacher)
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
    binding = {
        "source_config_sha256": source_manifest.file_hashes["config.json"],
        "source_shard_sha256": {
            shard.name: source_manifest.file_hashes[shard.name]
            for shard in source_manifest.shards
        },
        "dataset_manifest_sha256": file_sha256(dataset_manifest),
        "training_config_sha256": file_sha256(training_config),
        "execution_mode": "streamed_layer_store",
    }
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
    next_visit = 0
    consumed = 0
    data_cursor = 0
    if resume is not None:
        progress = _load_progress(resume)
        if progress["binding"] != binding:
            raise ContractError("streamed resume binding differs from source/data/plan")
        next_visit = int(progress["next_visit"])
        consumed = int(progress["consumed"])
        data_cursor = int(progress["data_cursor"])
        torch.set_rng_state(progress["torch_rng"])
        torch.cuda.set_rng_state_all(progress["cuda_rng"])
        random.setstate(progress["python_rng"])
    else:
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
    stop = bool(controller.history and controller.history[-1].get("stop", False))
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
        if sweep_index is not None and sweep_index < len(controller.history):
            continue
        mixer = mixer_store.load_mixer(active_layer, device=device, dtype=dtype)
        optimizer = ActiveLayerOptimizer(learning_rate=plan.learning_rate)
        optimizer_snapshot = _load_optimizer_snapshot(
            progress_dir / f"optimizer-layer-{active_layer:03d}.pt"
        )
        signal_names = trainable_names[active_layer] if stage == "signals" else None
        optimizer.activate(
            active_layer,
            mixer,
            snapshot=optimizer_snapshot,
            trainable_names=signal_names,
        )
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
            student_logits = student_output.logits[:, start:stop_token]
            teacher_logits = teacher_output.logits[:, start:stop_token].to(device)
            labels = input_ids[:, start:stop_token]
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
                    student_logits[:, :-1].reshape(-1, student_logits.shape[-1]),
                    labels[:, 1:].reshape(-1),
                ),
                rollout=(
                    normalized_mse(student_logits, teacher_logits)
                    if fully_recurrent
                    else student_logits.new_zeros(())
                ),
            )
            total = losses.weighted(LossWeights.for_stage(stage))
            stepped = optimizer.backward(
                total,
                accumulation_steps=plan.accumulation_steps,
            )
            gradient_norm = math.sqrt(
                sum(
                    float(parameter.grad.detach().float().square().sum())
                    for parameter in mixer.parameters()
                    if parameter.grad is not None
                )
            )
            data_cursor += 1
            consumed += int(attention_mask[:, start:].sum())
            _append_trace(
                trace_path,
                {
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
                    "head_error_space": "target-hidden-channel-partitions-after-source-output-projection",
                    "head_normalized_mse": head_errors,
                    "losses": {
                        name: float(value.detach())
                        for name, value in losses.__dict__.items()
                    },
                },
            )
            snapshot = optimizer.release()
            mixer_store.save_mixer(
                active_layer,
                mixer,
                cursor={
                    "visit_index": visit_index,
                    "consumed": consumed,
                    "data_cursor": data_cursor,
                },
            )
            optimizer_path = progress_dir / f"optimizer-layer-{active_layer:03d}.pt"
            _save_optimizer_snapshot(optimizer_path, snapshot)
            _save_progress(
                progress_dir,
                {
                    "schema_version": 1,
                    "binding": binding,
                    "next_visit": visit_index,
                    "consumed": consumed,
                    "data_cursor": data_cursor,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "python_rng": random.getstate(),
                },
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
        consumed = 0
        next_visit = visit_index + 1
        _save_progress(
            progress_dir,
            {
                "schema_version": 1,
                "binding": binding,
                "next_visit": next_visit,
                "consumed": 0,
                "data_cursor": data_cursor,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(),
            },
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
            snapshot = mixer_store.snapshot(
                run_dir / "streamed-sweeps" / f"sweep-{sweep_index:02d}"
            )
            sweep = controller.complete(
                start_checkpoint=(
                    str(zero_step_dir)
                    if sweep_index == 0
                    else str(run_dir / "streamed-sweeps" / f"sweep-{sweep_index - 1:02d}")
                ),
                end_checkpoint=str(snapshot),
                validation_kl=validation_kl,
                token_budget=plan.stage_tokens_per_layer["global"] * num_layers,
            )
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
            "source_decoder_layers": 1,
            "target_mixers": 1,
            "optimizers": 1,
            "frozen_suffix": "reentrant-reload-on-backward",
        },
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


def _atomic_torch_save(path: Path, payload: object) -> str:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    content = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
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


def _save_progress(root: Path, payload: dict[str, object]) -> None:
    progress_path = root / "progress.pt"
    sha = _atomic_torch_save(progress_path, payload)
    write_json(root / "latest.json", {"schema_version": 1, "path": progress_path.name, "sha256": sha})


def _load_progress(path: Path) -> dict[str, object]:
    root = path if path.is_dir() else path.parent
    pointer_path = root / "latest.json" if path.is_dir() else path
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    progress_path = root / str(pointer["path"])
    if file_sha256(progress_path) != pointer["sha256"]:
        raise ContractError("streamed progress SHA-256 mismatch")
    return torch.load(progress_path, map_location="cpu", weights_only=False)


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
            reference_logits = reference.logits[:, start:].to(device)
            candidate_logits = candidate.logits[:, start:]
            labels = input_ids[:, start:]
            teacher_ce.append(
                torch.nn.functional.cross_entropy(
                    reference_logits[:, :-1].reshape(-1, reference_logits.shape[-1]),
                    labels[:, 1:].reshape(-1),
                )
            )
            student_ce.append(
                torch.nn.functional.cross_entropy(
                    candidate_logits[:, :-1].reshape(-1, candidate_logits.shape[-1]),
                    labels[:, 1:].reshape(-1),
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
    if destination.is_dir():
        _validate_complete_export(destination)
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
    )
    _validate_complete_export(temporary)
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
