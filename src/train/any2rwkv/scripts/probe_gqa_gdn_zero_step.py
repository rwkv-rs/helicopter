#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import Tensor
from transformers import AutoTokenizer
from transformers.masking_utils import create_causal_mask

from any2rwkv.checkpoint import read_checkpoint, sha256_file
from any2rwkv.mixer import apply_partial_rope
from any2rwkv.streamed_teacher import StreamedQwen35Teacher
from any2rwkv.zero_step_probe import (
    LowRankProjection,
    affine_state_rollout,
    causal_attention,
    fit_low_rank_projection,
    four_state_block_output,
    gdn_reference_scan,
    hazard_metrics,
    logit_taylor_hazards,
    native_signal_rollout,
    native_two_state_rollout,
    normalized_mse,
    observable_query_bases,
    operator_input_bases,
    probability_tangent_parameters,
    probability_taylor_hazards,
    query_input_bases,
    qwen35_l2_normalize,
    rope_aligned_two_state_bases,
    rollout_hazards,
    select_bias_free_projection,
    tensor_metrics,
    two_state_projection,
    two_state_outputs,
    verify_gdn_mapping,
)


def parse_dtype(name: str) -> torch.dtype:
    try:
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[name]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {name}") from error


def load_calibration_rows(path: Path, rows: int) -> tuple[list[str], list[str]]:
    sample_ids: list[str] = []
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            sample_id = payload.get("sample_id")
            text = payload.get("text")
            if not isinstance(sample_id, str) or not isinstance(text, str):
                raise ValueError("FineWeb row must contain string sample_id and text")
            sample_ids.append(sample_id)
            texts.append(text)
            if len(texts) == rows:
                break
    if len(texts) != rows:
        raise ValueError(f"requested {rows} rows but dataset only yielded {len(texts)}")
    return sample_ids, texts


def source_positions(
    teacher: StreamedQwen35Teacher,
    hidden_states: Tensor,
    attention_mask: Tensor,
) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
    position_ids = torch.arange(
        hidden_states.shape[1],
        device=hidden_states.device,
        dtype=torch.long,
    ).unsqueeze(0).expand(hidden_states.shape[0], -1)
    position_embeddings = teacher.rotary(hidden_states, position_ids)
    causal_mask = create_causal_mask(
        teacher.loader.config,
        hidden_states,
        attention_mask,
        None,
        position_ids,
    )
    return position_ids, position_embeddings, causal_mask


def trace_gdn(module: torch.nn.Module, hidden_states: Tensor) -> dict[str, Tensor]:
    mixer = module.linear_attn
    normalized = module.input_layernorm(hidden_states)
    projected = mixer.in_proj_qkv(normalized).transpose(1, 2)
    projected = functional.conv1d(
        projected,
        mixer.conv1d.weight,
        mixer.conv1d.bias,
        padding=mixer.conv_kernel_size - 1,
        groups=mixer.conv_dim,
    )[:, :, : hidden_states.shape[1]]
    projected = functional.silu(projected).transpose(1, 2)
    query, key, value = torch.split(
        projected,
        [mixer.key_dim, mixer.key_dim, mixer.value_dim],
        dim=-1,
    )
    query = query.view(
        *query.shape[:2], mixer.num_k_heads, mixer.head_k_dim
    )
    key = key.view(*key.shape[:2], mixer.num_k_heads, mixer.head_k_dim)
    value = value.view(
        *value.shape[:2], mixer.num_v_heads, mixer.head_v_dim
    )
    repeat = mixer.num_v_heads // mixer.num_k_heads
    if repeat > 1:
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)
    beta = torch.sigmoid(mixer.in_proj_b(normalized).float())
    decay = torch.exp(
        -mixer.A_log.float().exp()
        * functional.softplus(
            mixer.in_proj_a(normalized).float() + mixer.dt_bias.float()
        )
    )
    return {
        "query": query.float(),
        "key": key.float(),
        "value": value.float(),
        "beta": beta,
        "decay": decay,
    }


def trace_gqa(
    module: torch.nn.Module,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    causal_mask: Tensor,
) -> dict[str, Tensor]:
    mixer = module.self_attn
    normalized = module.input_layernorm(hidden_states)
    input_shape = normalized.shape[:-1]
    head_dim = int(mixer.head_dim)
    query, gate = torch.chunk(
        mixer.q_proj(normalized).view(
            *input_shape, -1, head_dim * 2
        ),
        2,
        dim=-1,
    )
    query = mixer.q_norm(query).transpose(1, 2)
    grouped_key = mixer.k_norm(
        mixer.k_proj(normalized).view(*input_shape, -1, head_dim)
    ).transpose(1, 2)
    grouped_value = mixer.v_proj(normalized).view(
        *input_shape, -1, head_dim
    ).transpose(1, 2)
    modeling = importlib.import_module(mixer.__class__.__module__)
    query, grouped_key = modeling.apply_rotary_pos_emb(
        query, grouped_key, *position_embeddings
    )
    groups = int(mixer.num_key_value_groups)
    repeated_key = grouped_key.repeat_interleave(groups, dim=1)
    repeated_value = grouped_value.repeat_interleave(groups, dim=1)
    with torch.inference_mode():
        actual_mixer_output = mixer(
            normalized,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            use_cache=False,
        )[0]
    return {
        "query": query.transpose(1, 2).float(),
        "key": repeated_key.transpose(1, 2).float(),
        "value": repeated_value.transpose(1, 2).float(),
        "grouped_key": grouped_key.transpose(1, 2).float(),
        "grouped_value": grouped_value.transpose(1, 2).float(),
        "gate": torch.sigmoid(gate).float(),
        "mixer_input": normalized.float(),
        "query_weight": mixer.q_proj.weight.float(),
        "key_weight": mixer.k_proj.weight.float(),
        "value_weight": mixer.v_proj.weight.float(),
        "output_weight": mixer.o_proj.weight.float(),
        "output_bias": (
            None if mixer.o_proj.bias is None else mixer.o_proj.bias.float()
        ),
        "actual_mixer_output": actual_mixer_output.float(),
    }


def project_gqa_mixer(head_output: Tensor, signals: dict[str, Tensor]) -> Tensor:
    gated = head_output.float() * signals["gate"]
    return functional.linear(
        gated.flatten(2),
        signals["output_weight"],
        signals["output_bias"],
    )


def heldout_metrics(
    prediction: Tensor,
    target: Tensor,
    calibration_rows: int,
) -> dict[str, float]:
    return tensor_metrics(
        prediction[calibration_rows:],
        target[calibration_rows:],
    )


def tensor_sha256(value: Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def rotate_native_read(
    read: Tensor,
    positions: Tensor,
    *,
    source_head_dim: int,
    rotary_dim: int,
    rope_theta: float,
    inverse: bool,
) -> Tensor:
    if read.ndim != 4:
        raise ValueError("native read must be [batch,time,head,feature]")
    batch, tokens, _, _ = read.shape
    flat_width = read.shape[2] * read.shape[3]
    if flat_width % source_head_dim:
        raise ValueError("native read width does not fit source RoPE heads")
    source_heads = flat_width // source_head_dim
    source_view = read.reshape(batch, tokens, source_heads, source_head_dim)
    rotated = apply_partial_rope(
        source_view,
        -positions if inverse else positions,
        rotary_dim=rotary_dim,
        theta=rope_theta,
    )
    return rotated.reshape_as(read)


def fit_native_weight_projection(
    *,
    mixer_input: Tensor,
    target_read: Tensor,
    native_transition,
    positions: Tensor,
    exact_mixer_output: Tensor,
    calibration_rows: int,
    source_head_dim: int,
    rotary_dim: int,
    rope_theta: float,
    native_head_dim: int,
    source_gate: Tensor,
    query_basis: Tensor,
    dc_indices: tuple[int, int],
    source_query_weight: Tensor,
    source_key_weight: Tensor,
    source_value_weight: Tensor,
    source_output_weight: Tensor,
) -> dict[str, object]:
    """Fit native-shaped parameters for FP32 signal-level emulation."""
    ridges = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    target_pre_rope_read = rotate_native_read(
        target_read,
        positions,
        source_head_dim=source_head_dim,
        rotary_dim=rotary_dim,
        rope_theta=rope_theta,
        inverse=True,
    )
    target_pre_rope_key = rotate_native_read(
        native_transition.key,
        positions,
        source_head_dim=source_head_dim,
        rotary_dim=rotary_dim,
        rope_theta=rope_theta,
        inverse=True,
    )
    source_heads = query_basis.shape[0]
    source_hidden = mixer_input.shape[-1]
    if source_query_weight.shape != (
        source_heads * source_head_dim * 2,
        source_hidden,
    ):
        raise ValueError("packed source query/gate weight has unexpected shape")
    query_weight = source_query_weight.reshape(
        source_heads,
        source_head_dim * 2,
        source_hidden,
    )[:, :source_head_dim]
    read_prior = torch.zeros(
        source_heads,
        2,
        native_head_dim,
        source_hidden,
        dtype=torch.float32,
        device=mixer_input.device,
    )
    for head in range(source_heads):
        for state_index, dc_index in enumerate(dc_indices):
            feature_indices = [
                index for index in range(native_head_dim) if index != dc_index
            ]
            read_prior[head, state_index, feature_indices] = (
                query_basis[head, state_index].T @ query_weight[head]
            )
    read_prior = read_prior.flatten(0, 2)
    grouped_width = source_key_weight.shape[0] // source_head_dim
    if grouped_width <= 0 or source_heads % grouped_width:
        raise ValueError("source GQA key heads do not divide query heads")
    group_repeat = source_heads // grouped_width

    def repeated_kv_prior(weight: Tensor) -> Tensor:
        if weight.shape != (grouped_width * source_head_dim, source_hidden):
            raise ValueError("source GQA KV weight has unexpected shape")
        return (
            weight.reshape(grouped_width, source_head_dim, source_hidden)
            .repeat_interleave(group_repeat, dim=0)
            .flatten(0, 1)
        )

    key_prior = repeated_kv_prior(source_key_weight)
    value_prior = repeated_kv_prior(source_value_weight)

    def select_ridge_center(target: Tensor, source_prior: Tensor):
        candidates = {
            "zero": select_bias_free_projection(
                mixer_input,
                target,
                calibration_batches=calibration_rows,
                ridges=ridges,
            ),
            "source-compatible": select_bias_free_projection(
                mixer_input,
                target,
                calibration_batches=calibration_rows,
                ridges=ridges,
                prior_weight=source_prior,
            ),
        }
        selected_name = min(
            candidates,
            key=lambda name: (
                candidates[name].selection_nmse[candidates[name].ridge],
                name,
            ),
        )
        return candidates[selected_name], selected_name, {
            name: candidate.selection_nmse[candidate.ridge]
            for name, candidate in candidates.items()
        }

    read, read_center, read_center_nmse = select_ridge_center(
        target_pre_rope_read,
        read_prior,
    )
    key, key_center, key_center_nmse = select_ridge_center(
        target_pre_rope_key,
        key_prior,
    )
    value, value_center, value_center_nmse = select_ridge_center(
        native_transition.value,
        value_prior,
    )
    calibration_source = mixer_input[:calibration_rows].reshape(
        -1, mixer_input.shape[-1]
    ).float()
    ridge_scale = float(
        calibration_source.square().sum()
        / max(1, calibration_source.shape[-1])
    )
    low_rank_ridge = max(
        ridge_scale * 1e-3,
        torch.finfo(torch.float32).tiny,
    )
    native_heads = native_transition.key.shape[2]
    decay_channels = native_transition.decay.unsqueeze(-1).expand(
        -1,
        -1,
        -1,
        native_head_dim,
    )
    decay_probability = (
        -torch.log(decay_channels.clamp_min(1e-12)) / math.exp(-0.5)
    ).clamp(1e-6, 1 - 1e-6)
    decay_logits = torch.logit(decay_probability)
    erase_logits = torch.logit(
        native_transition.erase.clamp(1e-6, 1 - 1e-6)
    )
    decay = fit_low_rank_projection(
        mixer_input,
        decay_logits,
        calibration_batches=calibration_rows,
        rank=min(64, mixer_input.shape[-1], decay_logits.shape[-1] * native_heads),
        ridge=low_rank_ridge,
        hidden_activation="tanh",
    )
    erase = fit_low_rank_projection(
        mixer_input,
        erase_logits,
        calibration_batches=calibration_rows,
        rank=min(64, mixer_input.shape[-1], erase_logits.shape[-1] * native_heads),
        ridge=low_rank_ridge,
        hidden_activation="identity",
    )
    gate_target = source_gate.reshape(
        *source_gate.shape[:2],
        native_heads,
        native_head_dim,
    )
    source_gate_weight = source_query_weight.reshape(
        source_heads,
        source_head_dim * 2,
        source_hidden,
    )[:, source_head_dim:].flatten(0, 1)
    gate_rank = min(
        128,
        source_gate_weight.shape[0],
        mixer_input.shape[-1],
    )
    gate_basis_indices = torch.linspace(
        0,
        source_gate_weight.shape[0] - 1,
        gate_rank,
        dtype=torch.float64,
        device=mixer_input.device,
    ).round().to(torch.long)
    gate_down_weight = source_gate_weight.index_select(
        0,
        gate_basis_indices,
    )
    gate_features = torch.sigmoid(mixer_input @ gate_down_weight.T)
    gate_up = select_bias_free_projection(
        gate_features,
        gate_target,
        calibration_batches=calibration_rows,
        ridges=ridges,
    )
    gate = LowRankProjection(
        down_weight=gate_down_weight,
        up_weight=gate_up.projection.weight,
        bias=torch.zeros(
            gate_target.shape[-1] * native_heads,
            dtype=torch.float32,
            device=mixer_input.device,
        ),
        prediction=gate_up.projection.prediction,
        hidden_activation="sigmoid",
        output_bias=False,
    )

    predicted_read = rotate_native_read(
        read.projection.prediction,
        positions,
        source_head_dim=source_head_dim,
        rotary_dim=rotary_dim,
        rope_theta=rope_theta,
        inverse=False,
    )
    predicted_key = rotate_native_read(
        key.projection.prediction,
        positions,
        source_head_dim=source_head_dim,
        rotary_dim=rotary_dim,
        rope_theta=rope_theta,
        inverse=False,
    )
    predicted_decay = torch.exp(
        -math.exp(-0.5) * torch.sigmoid(decay.prediction)
    )
    predicted_erase = torch.sigmoid(erase.prediction)
    rollout = native_signal_rollout(
        predicted_read,
        predicted_decay,
        predicted_key,
        value.projection.prediction,
        predicted_erase,
    )
    flat_recurrent = rollout.output.flatten(2)
    normalized = functional.group_norm(
        flat_recurrent.reshape(-1, flat_recurrent.shape[-1]),
        num_groups=native_heads,
        weight=None,
        bias=None,
        eps=native_head_dim * 1e-5,
    ).reshape_as(flat_recurrent)
    pre_output = normalized * gate.prediction.flatten(2)

    def select_output_projection():
        candidates = {
            "zero": select_bias_free_projection(
                pre_output,
                exact_mixer_output,
                calibration_batches=calibration_rows,
                ridges=ridges,
            ),
            "source-compatible": select_bias_free_projection(
                pre_output,
                exact_mixer_output,
                calibration_batches=calibration_rows,
                ridges=ridges,
                prior_weight=source_output_weight,
            ),
        }
        selected_name = min(
            candidates,
            key=lambda name: (
                candidates[name].selection_nmse[candidates[name].ridge],
                name,
            ),
        )
        return candidates[selected_name], selected_name, {
            name: candidate.selection_nmse[candidate.ridge]
            for name, candidate in candidates.items()
        }

    output, output_center, output_center_nmse = select_output_projection()

    def direct_report(
        selected,
        target: Tensor,
        *,
        ridge_center: str,
        center_selection_nmse: dict[str, float],
    ) -> dict[str, object]:
        return {
            "weight_shape": list(selected.projection.weight.shape),
            "weight_sha256": tensor_sha256(selected.projection.weight),
            "selected_ridge_multiplier": selected.ridge,
            "absolute_ridge": selected.absolute_ridge,
            "selected_ridge_center": ridge_center,
            "calibration_center_selection_nmse": center_selection_nmse,
            "heldout_signal": heldout_metrics(
                selected.projection.prediction,
                target,
                calibration_rows,
            ),
        }

    def low_rank_report(projection, target: Tensor) -> dict[str, object]:
        return {
            "hidden_activation": projection.hidden_activation,
            "output_bias": projection.output_bias,
            "down_weight_shape": list(projection.down_weight.shape),
            "up_weight_shape": list(projection.up_weight.shape),
            "bias_shape": (
                list(projection.bias.shape)
                if projection.output_bias
                else None
            ),
            "down_weight_sha256": tensor_sha256(projection.down_weight),
            "up_weight_sha256": tensor_sha256(projection.up_weight),
            "bias_sha256": (
                tensor_sha256(projection.bias)
                if projection.output_bias
                else None
            ),
            "heldout_signal": heldout_metrics(
                projection.prediction,
                target,
                calibration_rows,
            ),
        }

    final_metrics = heldout_metrics(
        output.projection.prediction,
        exact_mixer_output,
        calibration_rows,
    )
    gate_report = low_rank_report(gate, gate_target)
    gate_report.update(
        {
            "basis": "deterministic-source-gate-rows",
            "basis_row_indices": gate_basis_indices.tolist(),
            "selected_ridge_multiplier": gate_up.ridge,
            "absolute_ridge": gate_up.absolute_ridge,
        }
    )
    return {
        "status": "diagnostic-not-installed",
        "scope": "fp32-signal-level-emulation",
        "evaluation_split_role": (
            "adaptive-development-heldout-not-final-generalization-estimate"
        ),
        "required_next_gate": (
            "materialize tensors into ProjectionBoundaryRWKV7Attention, run "
            "BF16 forward_sequence on previously unseen frozen sample IDs, "
            "and compare against the frozen mapped baseline"
        ),
        "installation_rule": (
            "install only when the complete held-out free-running mixer NMSE "
            "is finite and strictly below the frozen mapped baseline"
        ),
        "native_parameterization": {
            "x_r/x_w/x_k/x_v/x_a/x_g": "zero-current-token",
            "k_k": "one",
            "k_a": "zero",
            "r_k": "zero",
            "g_norm.weight": "one",
            "g_norm.bias": "zero",
            "v_lora": "disabled",
        },
        "ridge_center_selection": (
            "choose zero or source-compatible center independently for each "
            "bias-free projection using only the calibration validation split"
        ),
        "r_proj": direct_report(
            read,
            target_pre_rope_read,
            ridge_center=read_center,
            center_selection_nmse=read_center_nmse,
        ),
        "k_proj": direct_report(
            key,
            target_pre_rope_key,
            ridge_center=key_center,
            center_selection_nmse=key_center_nmse,
        ),
        "v_proj": direct_report(
            value,
            native_transition.value,
            ridge_center=value_center,
            center_selection_nmse=value_center_nmse,
        ),
        "w_lora": low_rank_report(decay, decay_logits),
        "a_lora": low_rank_report(erase, erase_logits),
        "g_lora": gate_report,
        "o_proj": direct_report(
            output,
            exact_mixer_output,
            ridge_center=output_center,
            center_selection_nmse=output_center_nmse,
        ),
        "free_running_recurrent_output_vs_dynamic_oracle": heldout_metrics(
            rollout.output,
            native_transition.free_running_output.reshape_as(rollout.output),
            calibration_rows,
        ),
        "complete_free_running_mixer_vs_exact_softmax": final_metrics,
    }


def gqa_diagnostics(
    signals: dict[str, Tensor],
    *,
    calibration_rows: int,
    positions: Tensor,
    source_head_dim: int,
    rotary_dim: int,
    rope_theta: float,
) -> dict[str, object]:
    query = signals["query"]
    key = signals["key"]
    value = signals["value"]
    exact = causal_attention(query, key, value)
    exact_recurrent_output = rollout_hazards(exact.hazards, value)
    exact_mixer_output = project_gqa_mixer(exact.output, signals)
    head_center = query[:calibration_rows].mean(dim=(0, 1))
    zero_center = torch.zeros_like(head_center)

    probability_taylor = probability_taylor_hazards(query, key)
    probability_clipped = probability_taylor.clamp(0, 1)
    probability_clipped[..., 0] = 1
    hazard_candidates = {
        "probability_taylor_at_zero": probability_taylor,
        "probability_taylor_at_zero_clipped": probability_clipped,
        "logit_taylor_at_zero": logit_taylor_hazards(
            query, key, zero_center
        ),
        "logit_taylor_at_calibration_mean": logit_taylor_hazards(
            query, key, head_center
        ),
    }
    hazard_results: dict[str, object] = {}
    for name, predicted_hazards in hazard_candidates.items():
        predicted_output = rollout_hazards(predicted_hazards, value)
        predicted_mixer_output = project_gqa_mixer(predicted_output, signals)
        hazard_results[name] = {
            "hazard": hazard_metrics(
                predicted_hazards[calibration_rows:],
                exact.hazards[calibration_rows:],
                exact.valid_hazards,
            ),
            "attention_output": heldout_metrics(
                predicted_output,
                exact.output,
                calibration_rows,
            ),
            "mixer_output": heldout_metrics(
                predicted_mixer_output,
                exact_mixer_output,
                calibration_rows,
            ),
        }

    grouped_key = signals["grouped_key"]
    grouped_value = signals["grouped_value"]
    groups = grouped_key.shape[2]
    group_width = query.shape[2] // groups
    group_center = torch.stack(
        [
            query[
                :calibration_rows,
                :,
                group * group_width : (group + 1) * group_width,
            ].mean(dim=(0, 1, 2))
            for group in range(groups)
        ]
    )
    affine_without_closure = affine_state_rollout(
        query,
        grouped_key,
        grouped_value,
        group_center,
        calibration_batches=calibration_rows,
        fit_rank_one_closure=False,
    )
    affine_without_closure_output = affine_without_closure.output
    without_closure_metrics = {
        "attention_output": heldout_metrics(
            affine_without_closure_output,
            exact.output,
            calibration_rows,
        ),
        "mixer_output": heldout_metrics(
            project_gqa_mixer(affine_without_closure_output, signals),
            exact_mixer_output,
            calibration_rows,
        ),
    }
    del affine_without_closure

    affine = affine_state_rollout(
        query,
        grouped_key,
        grouped_value,
        group_center,
        calibration_batches=calibration_rows,
        fit_rank_one_closure=True,
    )
    affine_mixer_output = project_gqa_mixer(affine.output, signals)
    tangent_probability, tangent_slope = probability_tangent_parameters(
        grouped_key,
        group_center,
    )
    affine_results: dict[str, object] = {
        "without_rank_one_closure": without_closure_metrics,
        "with_calibrated_rank_one_closure": {
            "attention_output": heldout_metrics(
                affine.output,
                exact.output,
                calibration_rows,
            ),
            "mixer_output": heldout_metrics(
                affine_mixer_output,
                exact_mixer_output,
                calibration_rows,
            ),
        },
        "closure_scale": {
            "minimum": float(affine.closure_scale.min()),
            "median": float(affine.closure_scale.median()),
            "maximum": float(affine.closure_scale.max()),
        },
    }

    query_subspace_rank = query.shape[-1] // 2 - 1
    operator_basis_by_group = operator_input_bases(
        affine.states,
        calibration_batches=calibration_rows,
        rank=query_subspace_rank,
    )
    head_to_group = torch.arange(query.shape[2], device=query.device) // group_width
    operator_basis = operator_basis_by_group[head_to_group]
    query_basis = query_input_bases(
        affine.centered_query,
        calibration_batches=calibration_rows,
        rank=query_subspace_rank,
    )
    observable_basis = observable_query_bases(
        affine.states,
        affine.centered_query,
        calibration_batches=calibration_rows,
        rank=query_subspace_rank,
    )
    coordinate_basis = torch.eye(
        query.shape[-1], device=query.device, dtype=torch.float32
    )[:, :query_subspace_rank].expand(query.shape[2], -1, -1)
    bias_by_head = affine.bias[:, :, head_to_group]
    uncentered_bias = affine.bias - torch.einsum(
        "btgod,gd->btgo",
        affine.states,
        group_center,
    )
    native_dim = query.shape[-1] // 2
    disjoint_basis = torch.zeros(
        query.shape[2],
        2,
        query.shape[-1],
        native_dim - 1,
        device=query.device,
        dtype=torch.float32,
    )
    feature_identity = torch.eye(
        native_dim - 1,
        device=query.device,
        dtype=torch.float32,
    )
    disjoint_basis[:, 0, : native_dim - 1] = feature_identity
    disjoint_basis[
        :,
        1,
        native_dim : 2 * native_dim - 1,
    ] = feature_identity
    rope_observable_basis = rope_aligned_two_state_bases(
        affine.states,
        affine.centered_query,
        calibration_batches=calibration_rows,
        native_dim=native_dim,
        rotary_dim=rotary_dim,
    )
    full_matrix_outputs = []
    for head in range(query.shape[2]):
        group = head // group_width
        full_matrix_outputs.append(
            torch.einsum(
                "btod,btd->bto",
                affine.states[:, :, group],
                affine.centered_query[:, :, head],
            )
        )
    full_matrix_output = torch.stack(full_matrix_outputs, dim=2)
    full_affine = full_matrix_output + bias_by_head
    native_candidate_name = "rope_aligned_observable_127x2_plus_two_dc"
    sketch_results: dict[str, object] = {}
    for name, (basis, projection_bias, projection_query, dc_indices) in {
        "first_127_coordinates_plus_dc": (
            coordinate_basis,
            affine.bias,
            affine.centered_query,
            (0, 0),
        ),
        "query_pca_127_plus_dc": (
            query_basis,
            affine.bias,
            affine.centered_query,
            (0, 0),
        ),
        "future_read_observable_127_plus_dc": (
            observable_basis,
            affine.bias,
            affine.centered_query,
            (0, 0),
        ),
        "operator_svd_127_plus_dc": (
            operator_basis,
            affine.bias,
            affine.centered_query,
            (0, 0),
        ),
        "rope_aligned_disjoint_127x2_plus_two_dc": (
            disjoint_basis,
            uncentered_bias,
            query,
            (native_dim - 1, native_dim - 1),
        ),
        "rope_aligned_observable_127x2_plus_two_dc": (
            rope_observable_basis,
            uncentered_bias,
            query,
            (native_dim - 1, native_dim - 1),
        ),
    }.items():
        materialized = two_state_projection(
            affine.states,
            projection_bias,
            projection_query,
            basis,
            dc_indices=dc_indices,
        )
        sketch_affine = materialized.output
        sketch_matrix = sketch_affine - bias_by_head
        target_native_read = materialized.read.flatten(2, 3)
        target_pre_rope_read = rotate_native_read(
            target_native_read,
            positions,
            source_head_dim=source_head_dim,
            rotary_dim=rotary_dim,
            rope_theta=rope_theta,
            inverse=True,
        )
        read_selection = select_bias_free_projection(
            signals["mixer_input"],
            target_pre_rope_read,
            calibration_batches=calibration_rows,
            ridges=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
        )
        read_projection = read_selection.projection
        projected_native_read = rotate_native_read(
            read_projection.prediction,
            positions,
            source_head_dim=source_head_dim,
            rotary_dim=rotary_dim,
            rope_theta=rope_theta,
            inverse=False,
        )
        native_states = materialized.states.flatten(2, 3)
        native_read_output = torch.einsum(
            "bthod,bthd->btho",
            native_states,
            projected_native_read,
        ).reshape_as(sketch_affine)
        native_transition = native_two_state_rollout(
            materialized,
            tangent_probability,
            tangent_slope,
        )
        native_weight_projection = None
        if name == native_candidate_name:
            native_weight_projection = fit_native_weight_projection(
                mixer_input=signals["mixer_input"],
                target_read=target_native_read,
                native_transition=native_transition,
                positions=positions,
                exact_mixer_output=exact_mixer_output,
                calibration_rows=calibration_rows,
                source_head_dim=source_head_dim,
                rotary_dim=rotary_dim,
                rope_theta=rope_theta,
                native_head_dim=native_dim,
                source_gate=signals["gate"],
                query_basis=materialized.query_basis,
                dc_indices=materialized.dc_indices,
                source_query_weight=signals["query_weight"],
                source_key_weight=signals["key_weight"],
                source_value_weight=signals["value_weight"],
                source_output_weight=signals["output_weight"],
            )
        teacher_forced_native_states = (
            native_transition.teacher_forced_states.flatten(2, 3)
        )
        free_running_native_states = (
            native_transition.free_running_states.flatten(2, 3)
        )
        teacher_forced_native_output = torch.einsum(
            "bthod,bthd->btho",
            teacher_forced_native_states,
            target_native_read,
        ).reshape_as(sketch_affine)
        free_running_projected_read_output = torch.einsum(
            "bthod,bthd->btho",
            free_running_native_states,
            projected_native_read,
        ).reshape_as(sketch_affine)
        requested_decay = native_transition.requested_decay
        realized_decay = native_transition.decay
        native_dc_indices = torch.tensor(
            materialized.dc_indices,
            device=query.device,
            dtype=torch.long,
        ).repeat(query.shape[2]).view(1, 1, -1, 1)
        native_dc_indices = native_dc_indices.expand(
            target_native_read.shape[0],
            target_native_read.shape[1],
            -1,
            -1,
        )
        projected_dc = torch.gather(
            projected_native_read,
            -1,
            native_dc_indices,
        )
        target_dc = torch.gather(
            target_native_read,
            -1,
            native_dc_indices,
        )
        sketch_results[name] = {
            "materialized_state_shape": list(materialized.states.shape),
            "materialized_read_shape": list(materialized.read.shape),
            "dc_indices": list(materialized.dc_indices),
            "native_bias_free_read_projection": {
                "weight_shape": list(read_projection.weight.shape),
                "weight_sha256": tensor_sha256(read_projection.weight),
                "selected_ridge_multiplier": read_selection.ridge,
                "absolute_ridge": read_selection.absolute_ridge,
                "ridge_scale": read_selection.ridge_scale,
                "calibration_selection_nmse_by_multiplier": {
                    str(ridge): nmse
                    for ridge, nmse in read_selection.selection_nmse.items()
                },
                "read_signal": heldout_metrics(
                    projected_native_read,
                    target_native_read,
                    calibration_rows,
                ),
                "dc_channel": heldout_metrics(
                    projected_dc,
                    target_dc,
                    calibration_rows,
                ),
                "attention_output_vs_materialized_two_state": heldout_metrics(
                    native_read_output,
                    sketch_affine,
                    calibration_rows,
                ),
                "attention_output_vs_exact_softmax": heldout_metrics(
                    native_read_output,
                    exact.output,
                    calibration_rows,
                ),
                "mixer_output_vs_exact_softmax": heldout_metrics(
                    project_gqa_mixer(native_read_output, signals),
                    exact_mixer_output,
                    calibration_rows,
                ),
            },
            "native_dynamic_signal_oracle": {
                "scope": (
                    "native-compatible k_k=1, k_a=0, scalar decay and "
                    "channel-wise erase; source-to-weight fitting is not included"
                ),
                "signal_shapes": {
                    "decay": list(native_transition.decay.shape),
                    "erase": list(native_transition.erase.shape),
                    "key": list(native_transition.key.shape),
                    "value": list(native_transition.value.shape),
                },
                "requested_decay": {
                    "minimum": float(requested_decay.min()),
                    "median": float(requested_decay.median()),
                    "maximum": float(requested_decay.max()),
                    "clamped_fraction": float(
                        (realized_decay != requested_decay).float().mean()
                    ),
                },
                "realized_decay": {
                    "minimum": float(realized_decay.min()),
                    "median": float(realized_decay.median()),
                    "maximum": float(realized_decay.max()),
                },
                "erase": {
                    "minimum": float(native_transition.erase.min()),
                    "median": float(native_transition.erase.median()),
                    "maximum": float(native_transition.erase.max()),
                    "zero_fraction": float(
                        (native_transition.erase == 0).float().mean()
                    ),
                    "one_fraction": float(
                        (native_transition.erase == 1).float().mean()
                    ),
                },
                "teacher_forced_attention_output_vs_materialized_two_state": (
                    heldout_metrics(
                        teacher_forced_native_output,
                        sketch_affine,
                        calibration_rows,
                    )
                ),
                "free_running_attention_output_vs_materialized_two_state": (
                    heldout_metrics(
                        native_transition.free_running_output,
                        sketch_affine,
                        calibration_rows,
                    )
                ),
                "free_running_attention_output_vs_exact_softmax": (
                    heldout_metrics(
                        native_transition.free_running_output,
                        exact.output,
                        calibration_rows,
                    )
                ),
                "free_running_mixer_output_vs_exact_softmax": heldout_metrics(
                    project_gqa_mixer(
                        native_transition.free_running_output,
                        signals,
                    ),
                    exact_mixer_output,
                    calibration_rows,
                ),
                "free_running_with_bias_free_read_vs_materialized_two_state": (
                    heldout_metrics(
                        free_running_projected_read_output,
                        sketch_affine,
                        calibration_rows,
                    )
                ),
                "free_running_with_bias_free_read_vs_exact_softmax": (
                    heldout_metrics(
                        free_running_projected_read_output,
                        exact.output,
                        calibration_rows,
                    )
                ),
                "free_running_with_bias_free_read_mixer_vs_exact_softmax": (
                    heldout_metrics(
                        project_gqa_mixer(
                            free_running_projected_read_output,
                            signals,
                        ),
                        exact_mixer_output,
                        calibration_rows,
                    )
                ),
            },
            "source_to_native_weight_projection": native_weight_projection,
            "matrix_observable_loss_vs_full_affine_state": heldout_metrics(
                sketch_matrix,
                full_matrix_output,
                calibration_rows,
            ),
            "total_loss_vs_full_affine_state": heldout_metrics(
                sketch_affine,
                full_affine,
                calibration_rows,
            ),
            "attention_output_vs_exact_softmax": heldout_metrics(
                sketch_affine,
                exact.output,
                calibration_rows,
            ),
            "mixer_output_vs_exact_softmax": heldout_metrics(
                project_gqa_mixer(sketch_affine, signals),
                exact_mixer_output,
                calibration_rows,
            ),
        }
    baseline_name = "query_pca_127_plus_dc"
    proposed_name = "future_read_observable_127_plus_dc"
    baseline_metrics = sketch_results[baseline_name]
    proposed_metrics = sketch_results[proposed_name]
    observable_improved = (
        proposed_metrics["total_loss_vs_full_affine_state"]["nmse"]
        < baseline_metrics["total_loss_vs_full_affine_state"]["nmse"]
    )
    mixer_non_regressed = (
        proposed_metrics["mixer_output_vs_exact_softmax"]["nmse"]
        <= baseline_metrics["mixer_output_vs_exact_softmax"]["nmse"]
    )
    selected_name = (
        proposed_name
        if observable_improved and mixer_non_regressed
        else baseline_name
    )
    native_baseline = sketch_results[baseline_name]
    native_candidate = sketch_results[native_candidate_name]
    native_candidate_compression_non_regressed = (
        native_candidate["total_loss_vs_full_affine_state"]["nmse"]
        <= native_baseline["total_loss_vs_full_affine_state"]["nmse"]
    )
    native_candidate_transition_non_regressed = (
        native_candidate["native_dynamic_signal_oracle"][
            "free_running_mixer_output_vs_exact_softmax"
        ]["nmse"]
        <= native_baseline["native_dynamic_signal_oracle"][
            "free_running_mixer_output_vs_exact_softmax"
        ]["nmse"]
    )
    native_candidate_pipeline_nmse = native_candidate[
        "native_dynamic_signal_oracle"
    ]["free_running_with_bias_free_read_mixer_vs_exact_softmax"]["nmse"]
    native_baseline_pipeline_nmse = native_baseline[
        "native_dynamic_signal_oracle"
    ]["free_running_with_bias_free_read_mixer_vs_exact_softmax"]["nmse"]
    native_candidate_pipeline_improved = (
        math.isfinite(native_candidate_pipeline_nmse)
        and native_candidate_pipeline_nmse
        < native_baseline_pipeline_nmse
    )
    native_selected_name = (
        native_candidate_name
        if native_candidate_pipeline_improved
        else baseline_name
    )

    four_state_outputs = []
    for head in range(query.shape[2]):
        group = head // group_width
        four_state_outputs.append(
            four_state_block_output(
                affine.states[:, :, group],
                affine.centered_query[:, :, head],
            )
        )
    four_state_matrix = torch.stack(four_state_outputs, dim=2)
    compression_results = {
        "two_state_128x128": {
            "feature_budget": {
                "per_native_state": {
                    "constant_channels": 1,
                    "query_feature_channels": query_subspace_rank,
                },
                "two_state_total": {
                    "constant_channels": 2,
                    "query_feature_slots": query_subspace_rank * 2,
                },
                "constraint": (
                    "each output-row block can read only its own native state; "
                    "feature slots are not a shared arbitrary 254-dimensional "
                    "input subspace"
                ),
            },
            "selection": {
                "baseline": baseline_name,
                "proposed": proposed_name,
                "selected": selected_name,
                "status": (
                    "accepted"
                    if selected_name == proposed_name
                    else "rejected"
                ),
                "rule": (
                    "strictly lower held-out total affine-state NMSE and "
                    "non-regressed held-out mixer NMSE"
                ),
                "observable_improved": observable_improved,
                "mixer_non_regressed": mixer_non_regressed,
            },
            "native_pipeline_selection": {
                "baseline": baseline_name,
                "proposed": native_candidate_name,
                "selected": native_selected_name,
                "status": (
                    "accepted"
                    if native_selected_name == native_candidate_name
                    else "rejected"
                ),
                "rule": (
                    "strictly lower finite held-out combined native-transition/"
                    "read mixer NMSE; intermediate stages remain diagnostics"
                ),
                "compression_non_regressed": (
                    native_candidate_compression_non_regressed
                ),
                "transition_non_regressed": (
                    native_candidate_transition_non_regressed
                ),
                "combined_pipeline_improved": native_candidate_pipeline_improved,
            },
            "bases": sketch_results,
        },
        "four_state_128x128_exact_block_control_matrix_only": (
            heldout_metrics(
                four_state_matrix,
                full_matrix_output,
                calibration_rows,
            )
        ),
    }

    return {
        "exactness_checks": {
            "hazard_recurrence_vs_softmax_attention": heldout_metrics(
                exact_recurrent_output,
                exact.output,
                calibration_rows,
            ),
            "manual_attention_vs_qwen_mixer": heldout_metrics(
                exact_mixer_output,
                signals["actual_mixer_output"],
                calibration_rows,
            ),
        },
        "hazard_linearization": hazard_results,
        "affine_state": affine_results,
        "state_compression": compression_results,
    }


def gdn_diagnostics(
    signals: dict[str, Tensor],
    *,
    calibration_rows: int,
) -> dict[str, object]:
    query = signals["query"][calibration_rows:]
    key = qwen35_l2_normalize(signals["key"][calibration_rows:])
    value = signals["value"][calibration_rows:]
    beta = signals["beta"][calibration_rows:].unsqueeze(-1)
    decay = signals["decay"][calibration_rows:].unsqueeze(-1)
    exact_mapping = verify_gdn_mapping(
        decay.double(),
        beta.double(),
        query.double(),
        key.double(),
        value.double(),
    )
    native_minimum = float(torch.exp(-torch.exp(torch.tensor(-0.5))))
    clipped_decay = decay.clamp_min(native_minimum)
    state = torch.zeros(
        query.shape[0],
        query.shape[2],
        value.shape[-1],
        key.shape[-1],
        dtype=torch.float32,
        device=query.device,
    )
    normalized_query = qwen35_l2_normalize(query)
    exact_output, _ = gdn_reference_scan(
        state,
        decay,
        beta,
        normalized_query,
        key,
        value,
    )
    clipped_output, _ = gdn_reference_scan(
        state,
        clipped_decay,
        beta,
        normalized_query,
        key,
        value,
    )
    all_decay = signals["decay"].float().flatten()
    quantile_levels = torch.tensor(
        [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1],
        device=all_decay.device,
    )
    quantiles = torch.quantile(all_decay, quantile_levels)
    return {
        "canonical_recurrence_mapping": exact_mapping,
        "native_decay_link": {
            "minimum_reachable_decay": native_minimum,
            "fraction_below_minimum": float(
                (all_decay < native_minimum).float().mean()
            ),
            "decay_quantiles": {
                f"q{int(level * 100):02d}": float(value)
                for level, value in zip(quantile_levels, quantiles, strict=True)
            },
            "clipped_core_output_vs_exact": tensor_metrics(
                clipped_output,
                exact_output,
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe Qwen3.5 GQA/GDN zero-step assumptions on real layers."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--calibration-rows", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--gdn-layer", type=int, default=0)
    parser.add_argument("--gqa-layer", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the exact JSON payload printed by this probe.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    args = parser.parse_args()
    if args.rows < 2 or not 0 < args.calibration_rows < args.rows:
        raise SystemExit("rows must exceed a positive calibration-row prefix")
    if args.sequence_length < 2:
        raise SystemExit("sequence length must be at least two")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    dtype = parse_dtype(args.dtype)

    checkpoint = read_checkpoint(args.source, require_final_layers=False)
    sample_ids, texts = load_calibration_rows(args.dataset, args.rows)
    tokenizer = AutoTokenizer.from_pretrained(
        args.source,
        local_files_only=True,
    )
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        padding="max_length",
        max_length=args.sequence_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    if not bool(attention_mask.all()):
        raise SystemExit("selected FineWeb rows must fill the fixed sequence length")

    teacher = StreamedQwen35Teacher(
        checkpoint,
        device=device,
        dtype=dtype,
        cache_layers=False,
    )
    hidden_states = functional.embedding(input_ids, teacher.embedding_weight)
    position_ids, position_embeddings, causal_mask = source_positions(
        teacher,
        hidden_states,
        attention_mask,
    )
    gdn_signals: dict[str, Tensor] | None = None
    loaded_layer_bytes: dict[str, int] = {}
    with torch.inference_mode():
        for layer_index in range(args.gqa_layer):
            with teacher.loader.layer_lease(layer_index):
                loaded = teacher.loader.load_layer(
                    layer_index,
                    device=device,
                    dtype=dtype,
                )
                loaded_layer_bytes[str(layer_index)] = loaded.source_tensor_bytes
                if layer_index == args.gdn_layer:
                    gdn_signals = trace_gdn(loaded.module, hidden_states)
                hidden_states = loaded.module(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                )
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
            del loaded
            if device.type == "cuda":
                torch.cuda.empty_cache()
        with teacher.loader.layer_lease(args.gqa_layer):
            loaded = teacher.loader.load_layer(
                args.gqa_layer,
                device=device,
                dtype=dtype,
            )
            loaded_layer_bytes[str(args.gqa_layer)] = loaded.source_tensor_bytes
            gqa_signals = trace_gqa(
                loaded.module,
                hidden_states,
                position_embeddings,
                causal_mask,
            )
    if gdn_signals is None:
        raise SystemExit("requested GDN layer was not traversed before the GQA layer")
    text_config = checkpoint.config.get("text_config", checkpoint.config)
    if not isinstance(text_config, dict):
        raise SystemExit("source text_config must be a JSON object")
    source_head_dim = int(text_config["head_dim"])

    result = {
        "schema_version": 1,
        "experiment": "qwen35-2b-real-layer-gqa-gdn-zero-step-probe",
        "provenance": {
            "source": str(args.source.resolve()),
            "source_shards": {
                shard.name: checkpoint.file_hashes[shard.name]
                for shard in checkpoint.shards
            },
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": sha256_file(args.dataset),
            "sample_ids": sample_ids,
            "calibration_rows": args.calibration_rows,
            "heldout_rows": args.rows - args.calibration_rows,
            "sequence_length": args.sequence_length,
            "gdn_layer": args.gdn_layer,
            "gqa_layer": args.gqa_layer,
            "loaded_layer_source_bytes": loaded_layer_bytes,
            "torch_version": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu"
            ),
            "activation_dtype": str(dtype),
            "metric_dtype": "torch.float32",
            "mapping_check_dtype": "torch.float64",
        },
        "gqa": gqa_diagnostics(
            gqa_signals,
            calibration_rows=args.calibration_rows,
            positions=position_ids,
            source_head_dim=source_head_dim,
            rotary_dim=int(
                source_head_dim
                * checkpoint.contract.partial_rotary_factor
            ),
            rope_theta=checkpoint.contract.rope_theta,
        ),
        "gdn": gdn_diagnostics(
            gdn_signals,
            calibration_rows=args.calibration_rows,
        ),
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{payload}\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
