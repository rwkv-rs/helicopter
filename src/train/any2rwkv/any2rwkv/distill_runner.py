from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from transformers import AutoModelForCausalLM

from .artifacts import checkpoint_sha256, write_json
from .calibration import file_sha256
from .checkpoint import read_checkpoint
from .distill import (
    DEFAULT_VOCAB_CHUNK_SIZE,
    SweepController,
    chunked_token_kl,
    load_sharded_training_checkpoint,
    normalized_mse,
    save_sharded_training_checkpoint,
)
from .errors import ContractError
from .export import export_hf_checkpoint, export_text_teacher_checkpoint
from .hybrid import HybridModelPatcher
from .configuration_any2rwkv import Any2RWKV7Config, Any2RWKVProxyConfig
from .mixer import ProjectionBoundaryRWKV7Attention
from .migration_init import WarmStartVariant, materialize_warm_start, plan_warm_start
from .mapping import finalize_fitted_mapping
from .target import rwkv7_mixer_specs
from .training import DistillationBatch, LayerwiseDistillationEngine


@dataclass(frozen=True)
class DistillationPlan:
    seed: int
    learning_rate: float
    burn_in_tokens: int
    supervised_tokens: int
    accumulation_steps: int
    stage_tokens_per_layer: dict[str, int]
    corrective_min_sweeps: int
    corrective_max_sweeps: int
    corrective_min_delta: float
    activation_checkpointing: bool
    execution_mode: str
    max_estimated_weight_bytes_moved: int | None
    cache_teacher_layers: bool
    max_teacher_cache_bytes: int | None
    max_cuda_reserved_bytes: int | None


def read_distillation_plan(path: Path) -> DistillationPlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ContractError("distillation plan schema_version must be 1")
    stages = payload.get("stage_tokens_per_layer")
    stage_order = ("signals", "block", "global")
    if not isinstance(stages, dict) or set(stages) != set(stage_order):
        raise ContractError(
            "stage_tokens_per_layer must contain exactly signals, block, global"
        )
    activation_checkpointing = payload.get("activation_checkpointing", True)
    if type(activation_checkpointing) is not bool:
        raise ContractError("activation_checkpointing must be a JSON boolean")
    cache_teacher_layers = payload.get("cache_teacher_layers", False)
    if type(cache_teacher_layers) is not bool:
        raise ContractError("cache_teacher_layers must be a JSON boolean")
    plan = DistillationPlan(
        int(payload.get("seed", 0)),
        float(payload.get("learning_rate", 0)),
        int(payload.get("burn_in_tokens", 0)),
        int(payload.get("supervised_tokens", 0)),
        int(payload.get("accumulation_steps", 0)),
        {name: int(stages[name]) for name in stage_order},
        int(payload.get("corrective_min_sweeps", 0)),
        int(payload.get("corrective_max_sweeps", 0)),
        float(payload.get("corrective_min_delta", -1)),
        activation_checkpointing,
        str(payload.get("execution_mode", "resident")),
        (
            int(payload["max_estimated_weight_bytes_moved"])
            if "max_estimated_weight_bytes_moved" in payload
            else None
        ),
        cache_teacher_layers,
        (
            int(payload["max_teacher_cache_bytes"])
            if "max_teacher_cache_bytes" in payload
            else None
        ),
        (
            int(payload["max_cuda_reserved_bytes"])
            if "max_cuda_reserved_bytes" in payload
            else None
        ),
    )
    if (
        plan.seed < 0
        or plan.learning_rate <= 0
        or plan.burn_in_tokens < 0
        or plan.supervised_tokens < 2
        or plan.accumulation_steps <= 0
        or min(plan.stage_tokens_per_layer.values()) <= 0
        or plan.corrective_min_sweeps < 1
        or plan.corrective_max_sweeps < plan.corrective_min_sweeps
        or plan.corrective_min_delta < 0
        or plan.execution_mode not in {"resident", "streamed_layer_store"}
        or (
            plan.max_estimated_weight_bytes_moved is not None
            and plan.max_estimated_weight_bytes_moved <= 0
        )
        or (
            plan.max_teacher_cache_bytes is not None
            and plan.max_teacher_cache_bytes <= 0
        )
        or (
            plan.cache_teacher_layers
            and plan.max_teacher_cache_bytes is None
        )
        or (
            plan.max_cuda_reserved_bytes is not None
            and plan.max_cuda_reserved_bytes <= 0
        )
        or (
            plan.cache_teacher_layers
            and plan.max_cuda_reserved_bytes is None
        )
    ):
        raise ContractError("distillation plan contains invalid numeric limits")
    return plan


def read_distillation_texts(
    path: Path,
    *,
    expected_split: str = "distill_train",
) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("split") != expected_split:
        raise ContractError(
            f"dataset manifest must use schema_version=1 and split={expected_split}"
        )
    data_file = Path(str(payload.get("data_file", "")))
    if not data_file.is_absolute():
        data_file = (path.parent / data_file).resolve()
    expected = str(payload.get("sha256", ""))
    actual = file_sha256(data_file)
    if actual != expected:
        raise ContractError(
            f"distillation data SHA-256 mismatch: expected {expected}, found {actual}"
        )
    text_field = str(payload.get("text_field", "text"))
    rows: list[str] = []
    with data_file.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            text = row.get(text_field)
            if not isinstance(text, str) or not text.strip():
                raise ContractError("distillation data contains an empty text row")
            rows.append(text)
    if len(rows) != int(payload.get("row_count", 0)) or not rows:
        raise ContractError("distillation data row_count does not match the file")
    return tuple(rows)


def read_packed_token_rows(
    path: Path,
    *,
    split: str,
    burn_in_tokens: int,
    supervised_tokens: int,
) -> tuple[tuple[int, ...], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("status") != "prepared":
        raise ContractError(
            "packed dataset manifest must use schema_version=1 and status=prepared"
        )
    splits = payload.get("splits")
    entry = splits.get(split) if isinstance(splits, dict) else None
    if not isinstance(entry, dict):
        raise ContractError(f"packed dataset manifest has no split: {split}")
    data_file = Path(str(entry.get("path", "")))
    if not data_file.is_absolute():
        data_file = (path.parent / data_file).resolve()
    expected_sha = str(entry.get("sha256", ""))
    actual_sha = file_sha256(data_file)
    if actual_sha != expected_sha:
        raise ContractError(
            f"packed {split} SHA-256 mismatch: expected {expected_sha}, found {actual_sha}"
        )
    expected_length = burn_in_tokens + supervised_tokens
    rows: list[tuple[int, ...]] = []
    row_ids: set[str] = set()
    with data_file.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row_id = str(row.get("row_id", ""))
            values = row.get("input_ids")
            if (
                row.get("split") != split
                or not row_id
                or row_id in row_ids
                or not isinstance(values, list)
                or len(values) != expected_length
                or not all(type(value) is int and value >= 0 for value in values)
                or row.get("burn_in_tokens") != burn_in_tokens
                or row.get("supervised_tokens") != supervised_tokens
            ):
                raise ContractError(f"invalid packed token row in split {split}: {row_id}")
            row_ids.add(row_id)
            rows.append(tuple(values))
    if len(rows) != int(entry.get("row_count", -1)) or not rows:
        raise ContractError(f"packed {split} row_count does not match the file")
    if sum(map(len, rows)) != int(entry.get("token_count", -1)):
        raise ContractError(f"packed {split} token_count does not match the file")
    return tuple(rows)


def _capture_teacher(
    patcher: HybridModelPatcher,
    *,
    layer_index: int,
    input_ids: Tensor,
    attention_mask: Tensor,
    position_ids: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    patcher.restore()
    record = patcher.records[layer_index]
    layer = patcher.layers[layer_index]
    captured: dict[str, Tensor] = {}

    def mixer_hook(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        captured["mixer"] = value.detach()

    def block_hook(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        captured["block"] = value.detach()

    mixer_handle = record.original.register_forward_hook(mixer_hook)
    block_handle = layer.register_forward_hook(block_hook)
    try:
        with torch.inference_mode():
            output = patcher.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )
    finally:
        mixer_handle.remove()
        block_handle.remove()
    if set(captured) != {"mixer", "block"}:
        raise ContractError("teacher hooks did not capture mixer and block outputs")
    teacher_state = None
    cache = getattr(output, "past_key_values", None)
    layers = getattr(cache, "layers", None)
    if record.source_kind == "linear_attention" and isinstance(layers, list):
        teacher_state = getattr(layers[layer_index], "recurrent_states", None)
        if teacher_state is not None:
            teacher_state = teacher_state.detach()
    return (
        captured["mixer"],
        captured["block"],
        output.logits.detach(),
        teacher_state,
    )


def _save_mixer_checkpoint(
    mixers,
    *,
    destination: Path,
    metadata: dict[str, object],
) -> None:
    """Persist only converted mixers; preserved Qwen tensors remain source-bound."""
    # A completed visit can be replayed after a crash between artifact commit
    # and cursor commit. Replace that generated snapshot atomically enough for
    # the single-writer workspace instead of treating recovery as corruption.
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    weight_map: dict[str, str] = {}
    for layer, mixer in enumerate(mixers):
        filename = f"layer-{layer:03d}.safetensors"
        tensors = {
            f"model.layers.{layer}.attn.{name}": value.detach().cpu().contiguous()
            for name, value in mixer.state_dict().items()
        }
        save_file(tensors, destination / filename)
        weight_map.update({name: filename for name in tensors})
    (destination / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": dict(metadata), "weight_map": dict(sorted(weight_map.items()))},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _build_target_mixers(checkpoint: Path):
    payload = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    config_cls = (
        Any2RWKV7Config
        if payload.get("model_type") == Any2RWKV7Config.model_type
        else Any2RWKVProxyConfig
    )
    config = config_cls(**payload)
    source_types = config.any2rwkv["source_layer_types"]
    source = config.any2rwkv["source_text_config"]
    source_head_dim = int(source.get("head_dim", config.head_dim))
    rope = config.rope_parameters
    rotary_dim = min(
        config.head_dim,
        int(source_head_dim * float(rope.get("partial_rotary_factor", 1.0))),
    )
    rotary_dim -= rotary_dim % 2
    mixers = [
        ProjectionBoundaryRWKV7Attention(
            config,
            layer,
            source_used_rope=source_types[layer] == "full_attention",
            rotary_dim=rotary_dim,
            rope_theta=float(rope.get("rope_theta", 10_000.0)),
        ).to(device="cuda", dtype=torch.bfloat16)
        for layer in range(config.num_hidden_layers)
    ]
    _load_mixer_checkpoint(mixers, checkpoint)
    return config, mixers


def _export_trained_checkpoint(
    source_manifest,
    target_config: dict[str, object],
    mixers,
    destination: Path,
) -> None:
    if destination.is_dir():
        required = (
            destination / "config.json",
            destination / "model.safetensors.index.json",
            destination / "roundtrip-manifest.json",
        )
        if all(path.is_file() for path in required):
            return
        raise ContractError(f"incomplete final checkpoint requires repair: {destination}")
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    specs = tuple(
        spec
        for layer in range(len(mixers))
        for spec in rwkv7_mixer_specs(
            layer,
            hidden_size=source_manifest.contract.hidden_size,
        )
    )
    tensors = {
        f"model.layers.{layer}.attn.{name}": value.detach().cpu().contiguous()
        for layer, mixer in enumerate(mixers)
        for name, value in mixer.state_dict().items()
    }
    export_hf_checkpoint(
        source_manifest,
        temporary,
        target_config=target_config,
        target_specs=specs,
        target_tensors=tensors,
    )
    temporary.rename(destination)


def _load_mixer_checkpoint(mixers, checkpoint: Path) -> None:
    index_path = checkpoint / "model.safetensors.index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ContractError(f"invalid selected checkpoint index: {index_path}")
    requested: dict[str, tuple[int, str]] = {}
    for layer, mixer in enumerate(mixers):
        for local_name in mixer.state_dict():
            full_name = f"model.layers.{layer}.attn.{local_name}"
            if full_name not in weight_map:
                raise ContractError(
                    f"selected checkpoint is missing mixer tensor: {full_name}"
                )
            requested[full_name] = (layer, local_name)
    states: list[dict[str, Tensor]] = [dict() for _ in mixers]
    by_shard: dict[str, list[str]] = {}
    for name in requested:
        by_shard.setdefault(str(weight_map[name]), []).append(name)
    for shard_name, names in by_shard.items():
        with safe_open(checkpoint / shard_name, framework="pt", device="cpu") as handle:
            for name in names:
                layer, local_name = requested[name]
                states[layer][local_name] = handle.get_tensor(name)
    for mixer, state in zip(mixers, states, strict=True):
        mixer.load_state_dict(state, strict=True)


def _load_mixer_tensors(mixers, tensors: dict[str, Tensor]) -> None:
    states: list[dict[str, Tensor]] = [dict() for _ in mixers]
    for layer, mixer in enumerate(mixers):
        prefix = f"model.layers.{layer}.attn."
        for local_name, parameter in mixer.state_dict().items():
            full_name = prefix + local_name
            if full_name not in tensors:
                raise ContractError(f"baseline tensor set is missing {full_name}")
            states[layer][local_name] = tensors[full_name].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
    for mixer, state in zip(mixers, states, strict=True):
        mixer.load_state_dict(state, strict=True)


def _score_recurrent_candidate(
    patcher: HybridModelPatcher,
    token_rows: tuple[tuple[int, ...], ...],
    *,
    burn_in_tokens: int,
    supervised_tokens: int,
) -> dict[str, float]:
    total_length = burn_in_tokens + supervised_tokens
    teacher_ce: list[Tensor] = []
    student_ce: list[Tensor] = []
    kl_rows: list[Tensor] = []
    mse_rows: list[Tensor] = []
    cosine_rows: list[Tensor] = []
    for tokens in token_rows:
        input_ids, attention_mask, position_ids = _encode_packed(
            tokens, total_length=total_length
        )
        captured: dict[str, Tensor] = {}

        def final_hook(_module, _args, output):
            captured["block"] = output[0] if isinstance(output, tuple) else output

        patcher.restore()
        handle = patcher.layers[-1].register_forward_hook(final_hook)
        with torch.inference_mode():
            teacher_logits = patcher.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            ).logits[:, burn_in_tokens:]
        teacher_block = captured["block"][:, burn_in_tokens:].detach()
        handle.remove()

        patcher.configure(
            active_layer=0,
            converted_layers=set(range(len(patcher.records))),
        )
        captured.clear()
        handle = patcher.layers[-1].register_forward_hook(final_hook)
        with torch.inference_mode():
            student_logits = patcher.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            ).logits[:, burn_in_tokens:]
        student_block = captured["block"][:, burn_in_tokens:].detach()
        handle.remove()
        labels = input_ids[:, burn_in_tokens:]
        teacher_ce.append(
            torch.nn.functional.cross_entropy(
                teacher_logits[:, :-1].float().reshape(-1, teacher_logits.shape[-1]),
                labels[:, 1:].reshape(-1),
            )
        )
        student_ce.append(
            torch.nn.functional.cross_entropy(
                student_logits[:, :-1].float().reshape(-1, student_logits.shape[-1]),
                labels[:, 1:].reshape(-1),
            )
        )
        kl_rows.append(
            chunked_token_kl(
                student_logits,
                teacher_logits,
                vocab_chunk_size=DEFAULT_VOCAB_CHUNK_SIZE,
            )
            / supervised_tokens
        )
        mse_rows.append(normalized_mse(student_block, teacher_block))
        cosine_rows.append(
            torch.nn.functional.cosine_similarity(
                student_block.flatten(), teacher_block.flatten(), dim=0
            )
        )
    patcher.restore()
    if not teacher_ce:
        raise ContractError("migration baseline split is empty")
    return {
        "teacher_ppl": float(torch.exp(torch.stack(teacher_ce).mean())),
        "student_ppl": float(torch.exp(torch.stack(student_ce).mean())),
        "mean_token_kl": float(torch.stack(kl_rows).mean()),
        "final_block_normalized_mse": float(torch.stack(mse_rows).mean()),
        "final_block_cosine": float(torch.stack(cosine_rows).mean()),
    }


def _write_baseline_result(
    path: Path,
    *,
    name: str,
    metrics: dict[str, float],
    binding: dict[str, object],
    token_budget: int,
) -> None:
    payload = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {"schema_version": 1, "binding": binding, "baselines": {}}
    )
    if payload.get("binding") != binding:
        raise ContractError("migration baseline binding changed within one run")
    payload["baselines"][name] = {
        **metrics,
        "token_budget": token_budget,
    }
    write_json(path, payload)


def _initial_trainable_names(run_dir: Path, layer_count: int) -> list[set[str]]:
    payload = json.loads((run_dir / "warm-start-plan.json").read_text(encoding="utf-8"))
    rows: list[set[str]] = [set() for _ in range(layer_count)]
    for entry in payload.get("entries", []):
        provenance = str(entry.get("provenance", ""))
        if provenance not in {"fitted", "initialized"} and entry.get("is_semantically_lossless") is not False:
            continue
        target = str(entry.get("target", ""))
        parts = target.split(".attn.", 1)
        if len(parts) != 2:
            continue
        layer_parts = parts[0].split(".")
        try:
            layer = int(layer_parts[layer_parts.index("layers") + 1])
        except (ValueError, IndexError):
            continue
        if not 0 <= layer < layer_count:
            raise ContractError(f"warm-start trainable target has invalid layer: {target}")
        # The resident trainer owns QwenRWKV7MixerAdapter modules, whose
        # native mixer is registered under the ``rwkv`` child namespace.
        rows[layer].add(f"rwkv.{parts[1]}")
    if any(not names for names in rows):
        missing = [index for index, names in enumerate(rows) if not names]
        raise ContractError(
            f"warm-start plan leaves no fitted/initialized parameters for layers: {missing}"
        )
    return rows


def _encode_packed(tokens: tuple[int, ...], *, total_length: int) -> tuple[Tensor, Tensor, Tensor]:
    if len(tokens) != total_length:
        raise ContractError(
            f"packed token row has length {len(tokens)}, expected {total_length}"
        )
    input_ids = torch.tensor(tokens, dtype=torch.long, device="cuda").unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return input_ids, attention_mask, position_ids


def _validation_kl(
    patcher: HybridModelPatcher,
    token_rows: tuple[tuple[int, ...], ...],
    *,
    burn_in_tokens: int,
    supervised_tokens: int,
) -> float:
    values: list[float] = []
    total_length = burn_in_tokens + supervised_tokens
    for tokens in token_rows:
        input_ids, attention_mask, position_ids = _encode_packed(
            tokens, total_length=total_length
        )
        patcher.restore()
        with torch.inference_mode():
            teacher_logits = patcher.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            ).logits[:, burn_in_tokens:]
        patcher.configure(
            active_layer=0,
            converted_layers=set(range(len(patcher.records))),
        )
        with torch.inference_mode():
            student_logits = patcher.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            ).logits[:, burn_in_tokens:]
        values.append(
            float(
                chunked_token_kl(
                    student_logits,
                    teacher_logits,
                    vocab_chunk_size=DEFAULT_VOCAB_CHUNK_SIZE,
                ).detach()
            )
        )
    patcher.restore()
    if not values:
        raise ContractError("validation split is empty")
    return sum(values) / len(values)


def run_distillation(
    *,
    source: Path,
    run_dir: Path,
    dataset_manifest: Path,
    training_config: Path,
    allow_proxy_layers: bool,
    resume: Path | None = None,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise ContractError("distillation requires CUDA")
    plan = read_distillation_plan(training_config)
    token_rows = read_packed_token_rows(
        dataset_manifest,
        split="distill_train",
        burn_in_tokens=plan.burn_in_tokens,
        supervised_tokens=plan.supervised_tokens,
    )
    validation_rows = read_packed_token_rows(
        dataset_manifest,
        split="validation",
        burn_in_tokens=plan.burn_in_tokens,
        supervised_tokens=plan.supervised_tokens,
    )
    torch.manual_seed(plan.seed)
    source_manifest = read_checkpoint(
        source, require_final_layers=not allow_proxy_layers
    )
    if plan.execution_mode == "streamed_layer_store":
        from .streamed_distill_runner import run_streamed_distillation

        return run_streamed_distillation(
            source_manifest=source_manifest,
            run_dir=run_dir,
            token_rows=token_rows,
            validation_rows=validation_rows,
            plan=plan,
            dataset_manifest=dataset_manifest,
            training_config=training_config,
            resume=resume,
        )
    teacher_dir = run_dir / "teacher-text"
    if not teacher_dir.exists():
        export_text_teacher_checkpoint(source_manifest, teacher_dir)
    zero_step = run_dir / "checkpoint-zero-step"
    if not zero_step.is_dir():
        raise ContractError("zero-step checkpoint is missing; run convert first")

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    target_config, mixers = _build_target_mixers(zero_step)
    patcher = HybridModelPatcher(teacher, mixers)
    baseline_path = run_dir / "migration-baselines.json"
    baseline_rows = validation_rows[: min(8, len(validation_rows))]
    baseline_binding: dict[str, object] = {
        "schema": "any2rwkv-migration-baseline-v1",
        "source_manifest_sha256": file_sha256(run_dir / "source-manifest.json"),
        "dataset_manifest_sha256": file_sha256(dataset_manifest),
        "training_config_sha256": file_sha256(training_config),
        "seed": plan.seed,
        "burn_in_tokens": plan.burn_in_tokens,
        "supervised_tokens": plan.supervised_tokens,
        "rows": len(baseline_rows),
        "precision": "bf16-io-fp32-state",
    }
    specs = tuple(
        spec
        for layer in range(len(mixers))
        for spec in rwkv7_mixer_specs(
            layer,
            hidden_size=source_manifest.contract.hidden_size,
        )
    )
    variants = (
        ("random", WarmStartVariant.RANDOM),
        ("naive_copy", WarmStartVariant.NAIVE_COPY),
        ("gdn_algebraic", WarmStartVariant.GDN_CONSTRAINED),
        ("kv_repeat", WarmStartVariant.KV_REPEAT),
        ("kv_expand", WarmStartVariant.KV_EXPAND),
        ("mapped", WarmStartVariant.MAPPED),
    )
    existing_payload = (
        json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_path.is_file()
        else None
    )
    if existing_payload is not None and existing_payload.get("binding") != baseline_binding:
        raise ContractError("existing migration baselines use a different frozen protocol")
    existing_baselines = (
        set(existing_payload.get("baselines", {}))
        if existing_payload is not None
        else set()
    )
    for name, variant in variants:
        if name not in existing_baselines:
            candidate_plan = plan_warm_start(source_manifest, specs, variant=variant)
            _load_mixer_tensors(
                mixers,
                materialize_warm_start(source_manifest, specs, candidate_plan),
            )
            _write_baseline_result(
                baseline_path,
                name=name,
                metrics=_score_recurrent_candidate(
                    patcher,
                    baseline_rows,
                    burn_in_tokens=plan.burn_in_tokens,
                    supervised_tokens=plan.supervised_tokens,
                ),
                binding=baseline_binding,
                token_budget=0,
            )
    # Restore the exact mapped checkpoint expected by the first optimizer
    # visit even when a resumed baseline artifact was already complete.
    mapped_plan = plan_warm_start(
        source_manifest, specs, variant=WarmStartVariant.MAPPED
    )
    _load_mixer_tensors(
        mixers,
        materialize_warm_start(source_manifest, specs, mapped_plan),
    )
    trace_path = run_dir / "active-layer-trace.jsonl"
    engine = LayerwiseDistillationEngine(
        patcher,
        lr=plan.learning_rate,
        trace_path=trace_path,
        activation_checkpointing=plan.activation_checkpointing,
        trace_binding={
            "source_manifest_sha256": file_sha256(run_dir / "source-manifest.json"),
            "source_config_sha256": source_manifest.file_hashes["config.json"],
            "dataset_manifest_sha256": file_sha256(dataset_manifest),
            "training_config_sha256": file_sha256(training_config),
        },
    )
    initial_trainable = _initial_trainable_names(run_dir, len(mixers))
    total_length = plan.burn_in_tokens + plan.supervised_tokens
    # Stage-major ordering creates an auditable activation-fitted checkpoint
    # before block/global objectives are introduced.  A layer-major schedule
    # would confound initialization quality with later KL/CE training.
    visits = [
        (False, layer, stage, budget)
        for stage, budget in plan.stage_tokens_per_layer.items()
        for layer in range(len(mixers))
    ]
    visits.extend(
        (True, layer, "rollout", plan.stage_tokens_per_layer["global"])
        for _ in range(plan.corrective_max_sweeps)
        for layer in reversed(range(len(mixers)))
    )
    progressive_visits = len(plan.stage_tokens_per_layer) * len(mixers)
    sweep_path = run_dir / "corrective-sweeps.json"
    sweep_history = []
    if sweep_path.is_file():
        sweep_payload = json.loads(sweep_path.read_text(encoding="utf-8"))
        if sweep_payload.get("schema_version") != 1:
            raise ContractError("corrective sweep history schema_version must be 1")
        sweep_history = list(sweep_payload.get("sweeps", []))
    controller = SweepController(
        min_sweeps=plan.corrective_min_sweeps,
        max_sweeps=plan.corrective_max_sweeps,
        min_delta=plan.corrective_min_delta,
        history=sweep_history,
    )
    next_visit = 0
    resume_consumed = 0
    if resume is not None:
        metadata = load_sharded_training_checkpoint(resume, mixers, engine.trainer)
        next_visit = int(metadata["next_visit"])
        resume_consumed = int(metadata.get("consumed", 0))

    latest = run_dir / "checkpoints"
    for visit_index, (fully_recurrent, layer, stage, token_budget) in enumerate(visits):
        if visit_index < next_visit:
            continue
        sweep_index = (
            None
            if visit_index < progressive_visits
            else (visit_index - progressive_visits) // len(mixers)
        )
        if sweep_index is not None and sweep_index < len(controller.history):
            if controller.history[sweep_index].get("stop"):
                break
            continue
        engine.begin_layer(
            layer,
            # Signal fitting is the isolated teacher-prefix experiment. Later
            # stages deliberately expose the active layer to the student prefix.
            converted_prefix=0 if stage == "signals" else layer,
            loss_stage=stage,
            burn_in_tokens=plan.burn_in_tokens,
            supervised_tokens=plan.supervised_tokens,
            seed=plan.seed + visit_index,
            fully_recurrent=fully_recurrent,
            resume_accumulation=(
                visit_index == next_visit
                and engine.trainer.accumulation_step > 0
            ),
            trainable_names=(initial_trainable[layer] if stage == "signals" else None),
        )
        engine.trainer.visit_cursor = visit_index
        consumed = resume_consumed if visit_index == next_visit else 0
        resume_consumed = 0
        while consumed < token_budget or engine.trainer.accumulation_step:
            tokens = token_rows[engine.trainer.data_cursor % len(token_rows)]
            input_ids, attention_mask, position_ids = _encode_packed(
                tokens,
                total_length=total_length,
            )
            teacher_mixer, teacher_block, teacher_logits, teacher_state = _capture_teacher(
                patcher,
                layer_index=layer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            if fully_recurrent:
                patcher.configure(
                    active_layer=layer,
                    converted_layers=set(range(len(mixers))),
                    reset_gradients=False,
                    checkpoint_suffix=plan.activation_checkpointing,
                )
            else:
                patcher.configure(
                    active_layer=layer,
                    converted_prefix=0 if stage == "signals" else layer,
                    reset_gradients=False,
                    checkpoint_suffix=plan.activation_checkpointing,
                )
            labels = input_ids.masked_fill(attention_mask == 0, -100)
            row = engine.step(
                DistillationBatch(
                    input_ids,
                    attention_mask,
                    labels,
                    teacher_mixer,
                    teacher_block,
                    teacher_logits,
                    teacher_state=teacher_state,
                    rollout_teacher=teacher_logits if fully_recurrent else None,
                ),
                accumulation_steps=plan.accumulation_steps,
            )
            consumed += int(attention_mask[:, plan.burn_in_tokens :].sum())
            save_sharded_training_checkpoint(
                latest,
                mixers,
                engine.trainer,
                metadata={"next_visit": visit_index, "consumed": consumed},
            )
        stop_after_visit = False
        if stage == "signals" and layer == len(mixers) - 1:
            _save_mixer_checkpoint(
                mixers,
                destination=run_dir / "checkpoint-activation-fitted",
                metadata={"stage": "activation-fitted", "source": str(source_manifest.path)},
            )
            _write_baseline_result(
                baseline_path,
                name="activation_fitted",
                metrics=_score_recurrent_candidate(
                    patcher,
                    baseline_rows,
                    burn_in_tokens=plan.burn_in_tokens,
                    supervised_tokens=plan.supervised_tokens,
                ),
                binding=baseline_binding,
                token_budget=plan.stage_tokens_per_layer["signals"] * len(mixers),
            )
        if stage == "block" and layer == len(mixers) - 1:
            _write_baseline_result(
                baseline_path,
                name="hybrid_progressive",
                metrics=_score_recurrent_candidate(
                    patcher,
                    baseline_rows,
                    burn_in_tokens=plan.burn_in_tokens,
                    supervised_tokens=plan.supervised_tokens,
                ),
                binding=baseline_binding,
                token_budget=(
                    plan.stage_tokens_per_layer["signals"]
                    + plan.stage_tokens_per_layer["block"]
                )
                * len(mixers),
            )
        if stage == "global" and layer == len(mixers) - 1:
            _save_mixer_checkpoint(
                mixers,
                destination=run_dir / "checkpoint-first-fully-recurrent",
                metadata={"stage": "first-fully-recurrent", "source": str(source_manifest.path)},
            )
            _write_baseline_result(
                baseline_path,
                name="first_fully_recurrent",
                metrics=_score_recurrent_candidate(
                    patcher,
                    baseline_rows,
                    burn_in_tokens=plan.burn_in_tokens,
                    supervised_tokens=plan.supervised_tokens,
                ),
                binding=baseline_binding,
                token_budget=sum(plan.stage_tokens_per_layer.values()) * len(mixers),
            )
        if fully_recurrent and layer == 0:
            current_sweep = int(sweep_index)
            checkpoint = run_dir / f"checkpoint-sweep-{current_sweep:02d}"
            _save_mixer_checkpoint(
                mixers,
                destination=checkpoint,
                metadata={"stage": "corrective-sweep", "sweep_index": current_sweep},
            )
            validation_kl = _validation_kl(
                patcher,
                validation_rows,
                burn_in_tokens=plan.burn_in_tokens,
                supervised_tokens=plan.supervised_tokens,
            )
            _write_baseline_result(
                baseline_path,
                name=f"corrective_sweep_{current_sweep:02d}",
                metrics=_score_recurrent_candidate(
                    patcher,
                    baseline_rows,
                    burn_in_tokens=plan.burn_in_tokens,
                    supervised_tokens=plan.supervised_tokens,
                ),
                binding=baseline_binding,
                token_budget=(
                    sum(plan.stage_tokens_per_layer.values())
                    + (current_sweep + 1) * token_budget
                )
                * len(mixers),
            )
            start_checkpoint = (
                "checkpoint-first-fully-recurrent"
                if current_sweep == 0
                else f"checkpoint-sweep-{current_sweep - 1:02d}"
            )
            sweep_row = controller.complete(
                start_checkpoint=start_checkpoint,
                end_checkpoint=checkpoint.name,
                validation_kl=validation_kl,
                token_budget=token_budget * len(mixers),
            )
            controller.history[-1] = sweep_row
            engine.trainer.sweep_index = current_sweep + 1
            engine.trainer.validation_history = [
                float(row["validation_kl"]) for row in controller.history
            ]
            engine.trainer.stop_counters = {
                "completed_sweeps": len(controller.history),
                "below_min_delta": int(
                    sweep_row["delta"] is not None
                    and float(sweep_row["delta"]) < plan.corrective_min_delta
                ),
            }
            engine.trainer.selected_checkpoint = str(
                sweep_row["selected_checkpoint"]
            )
            write_json(
                sweep_path,
                {
                    "schema_version": 1,
                    "min_sweeps": plan.corrective_min_sweeps,
                    "max_sweeps": plan.corrective_max_sweeps,
                    "min_delta": plan.corrective_min_delta,
                    "sweeps": controller.history,
                    "selected_checkpoint": sweep_row["selected_checkpoint"],
                },
            )
            if sweep_row["stop"]:
                stop_after_visit = True

        # Advance the durable cursor only after every stage artifact and fixed
        # validation result has committed. This makes interruption at a stage
        # or sweep boundary replayable without silently skipping evidence.
        engine.trainer.visit_cursor = visit_index + 1
        save_sharded_training_checkpoint(
            latest,
            mixers,
            engine.trainer,
            metadata={"next_visit": visit_index + 1, "consumed": 0},
        )
        if stop_after_visit:
            break

    patcher.restore()
    if engine.trainer.selected_checkpoint:
        selected = run_dir / engine.trainer.selected_checkpoint
        current = (
            None
            if not controller.history
            else run_dir / str(controller.history[-1]["end_checkpoint"])
        )
        if current is None or selected != current:
            _load_mixer_checkpoint(mixers, selected)
    _write_baseline_result(
        baseline_path,
        name="layerwise_distilled",
        metrics=_score_recurrent_candidate(
            patcher,
            baseline_rows,
            burn_in_tokens=plan.burn_in_tokens,
            supervised_tokens=plan.supervised_tokens,
        ),
        binding=baseline_binding,
        token_budget=(
            sum(plan.stage_tokens_per_layer.values())
            + len(controller.history) * plan.stage_tokens_per_layer["global"]
        )
        * len(mixers),
    )
    trained = run_dir / "checkpoint-bf16"
    _export_trained_checkpoint(
        source_manifest,
        target_config.to_dict(),
        mixers,
        trained,
    )
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    trained_sha = checkpoint_sha256(trained)
    baseline_payload["student_sha256"] = trained_sha
    write_json(baseline_path, baseline_payload)
    finalize_fitted_mapping(
        run_dir,
        student_sha256=trained_sha,
        trace_sha256=file_sha256(trace_path),
    )
    return {
        "status": "distilled-bf16",
        "checkpoint": str(trained),
        "visits": len(visits),
        "optimizer_steps": engine.trainer.optimizer_step,
        "data_cursor": engine.trainer.data_cursor,
        "plan_sha256": file_sha256(training_config),
        "dataset_manifest_sha256": file_sha256(dataset_manifest),
        "corrective_sweeps": controller.history,
        "selected_checkpoint": engine.trainer.selected_checkpoint,
        "runtime": {
            "checkpoint_dtype": "bfloat16",
            "mixer_parameter_dtype": str(next(mixers[0].parameters()).dtype),
            "recurrent_state_dtype": "torch.float32",
            "recurrent_accumulation_dtype": "torch.float32",
            "gemm_accumulation_policy": "framework-default-correctness-path",
            "RWKV_FLOAT_MODE": os.environ.get("RWKV_FLOAT_MODE"),
            "VLLM_RWKV7_WKV_MODE": os.environ.get("VLLM_RWKV7_WKV_MODE"),
            "cuda_devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": list(
                        torch.cuda.get_device_capability(index)
                    ),
                }
                for index in range(torch.cuda.device_count())
            ],
        },
    }
