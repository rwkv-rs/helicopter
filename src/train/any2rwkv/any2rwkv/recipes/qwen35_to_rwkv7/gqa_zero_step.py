from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as functional
from torch import Tensor

from ...mixer import ProjectionBoundaryRWKV7Attention, apply_partial_rope
from ...zero_step_probe import (
    LowRankProjection,
    SelectedBiasFreeProjection,
    affine_state_rollout,
    causal_attention,
    fit_low_rank_projection,
    hazard_metrics,
    logit_taylor_hazards,
    native_signal_rollout,
    native_two_state_rollout,
    probability_tangent_parameters,
    rope_aligned_two_state_bases,
    rollout_hazards,
    select_bias_free_projection,
    tensor_metrics,
    tensor_sha256,
    two_state_projection,
)


@dataclass(frozen=True)
class GQANativeFitTrace:
    """Post-RMSNorm source signals for one GQA layer.

    Row-aligned tensors contain a calibration prefix followed by an adaptive
    development suffix.  They never contain the frozen installation split.
    """

    mixer_input: Tensor
    query: Tensor
    key: Tensor
    value: Tensor
    grouped_key: Tensor
    grouped_value: Tensor
    gate: Tensor
    mixer_output: Tensor
    query_weight: Tensor
    key_weight: Tensor
    value_weight: Tensor
    output_weight: Tensor
    output_bias: Tensor | None


@dataclass(frozen=True)
class GQANativeFitConfig:
    calibration_rows: int
    positions: Tensor
    source_head_dim: int
    rotary_dim: int
    rope_theta: float
    supervised_token_start: int = 0
    observable_fit_steps: int = 64
    observable_fit_learning_rate: float = 0.05


@dataclass(frozen=True)
class GQANativeFitResult:
    """A complete native-shaped candidate and its observable error ledger."""

    parameters: Mapping[str, Tensor]
    report: Mapping[str, object]


@dataclass(frozen=True)
class _NativeProjectionFit:
    read: SelectedBiasFreeProjection
    key: SelectedBiasFreeProjection
    value: SelectedBiasFreeProjection
    decay: LowRankProjection
    erase: LowRankProjection
    gate: LowRankProjection
    output: SelectedBiasFreeProjection
    report: Mapping[str, object]


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _per_head_group_metrics(
    prediction: Tensor,
    target: Tensor,
    *,
    key_value_heads: int,
) -> dict[str, list[dict[str, float]]]:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(
            "per-head GQA metrics require aligned [batch,time,head,feature] tensors"
        )
    query_heads = prediction.shape[2]
    if key_value_heads <= 0 or query_heads % key_value_heads:
        raise ValueError("per-head GQA metrics do not align with KV groups")
    group_width = query_heads // key_value_heads
    per_query_head = [
        tensor_metrics(prediction[:, :, head], target[:, :, head])
        for head in range(query_heads)
    ]
    per_key_value_group = [
        tensor_metrics(
            prediction[
                :,
                :,
                group * group_width : (group + 1) * group_width,
            ],
            target[
                :,
                :,
                group * group_width : (group + 1) * group_width,
            ],
        )
        for group in range(key_value_heads)
    ]
    return {
        "per_query_head": per_query_head,
        "per_key_value_group": per_key_value_group,
    }


def fit_gqa_native_zero_step(
    module: ProjectionBoundaryRWKV7Attention,
    trace: GQANativeFitTrace,
    config: GQANativeFitConfig,
) -> GQANativeFitResult:
    """Fit the exact-hazard/observable-compression/native-projection chain.

    Only ``trace[:calibration_rows]`` participates in solving.  The remaining
    rows are used for adaptive-development diagnostics.  Installation on the
    separately frozen validation cache is intentionally owned by the caller.
    """

    _validate_trace(module, trace, config)
    calibration_rows = config.calibration_rows
    exact = causal_attention(trace.query, trace.key, trace.value)
    exact_hazard_output = rollout_hazards(exact.hazards, trace.value)
    head_center = trace.query[:calibration_rows].mean(dim=(0, 1))
    bounded_hazards = logit_taylor_hazards(
        trace.query,
        trace.key,
        head_center,
    )
    bounded_output = rollout_hazards(bounded_hazards, trace.value)

    groups = trace.grouped_key.shape[2]
    query_heads = trace.query.shape[2]
    group_width = query_heads // groups
    group_center = torch.stack(
        [
            trace.query[
                :calibration_rows,
                :,
                group * group_width : (group + 1) * group_width,
            ].mean(dim=(0, 1, 2))
            for group in range(groups)
        ]
    )
    affine = affine_state_rollout(
        trace.query,
        trace.grouped_key,
        trace.grouped_value,
        group_center,
        calibration_batches=calibration_rows,
        fit_rank_one_closure=True,
    )
    basis = rope_aligned_two_state_bases(
        affine.states,
        affine.centered_query,
        calibration_batches=calibration_rows,
        native_dim=module.head_dim,
        rotary_dim=config.rotary_dim,
        steps=config.observable_fit_steps,
        learning_rate=config.observable_fit_learning_rate,
    )
    dc_indices = (module.head_dim - 1, module.head_dim - 1)
    head_to_group = torch.arange(
        query_heads,
        device=trace.query.device,
    ) // group_width
    compressed = two_state_projection(
        affine.states,
        affine.bias,
        trace.query,
        basis,
        dc_indices=dc_indices,
        query_center=group_center.index_select(0, head_to_group),
    )
    tangent_probability, tangent_slope = probability_tangent_parameters(
        trace.grouped_key,
        group_center,
    )
    native_transition = native_two_state_rollout(
        compressed,
        tangent_probability,
        tangent_slope,
    )
    projection_fit = _fit_native_projection(
        module=module,
        trace=trace,
        target_read=compressed.read.flatten(2, 3),
        native_transition=native_transition,
        config=config,
        native_head_dim=module.head_dim,
        query_basis=compressed.query_basis,
        dc_indices=compressed.dc_indices,
    )
    parameters = _native_parameter_set(
        module,
        projection_fit,
        source_value_weight=trace.value_weight,
    )

    development = slice(calibration_rows, None)
    supervised = slice(config.supervised_token_start, None)
    solver_config = {
        "calibration_rows": calibration_rows,
        "supervised_token_start": config.supervised_token_start,
        "source_head_dim": config.source_head_dim,
        "rotary_dim": config.rotary_dim,
        "rope_theta": config.rope_theta,
        "observable_fit_steps": config.observable_fit_steps,
        "observable_fit_learning_rate": config.observable_fit_learning_rate,
        "positions_sha256": tensor_sha256(config.positions),
    }
    trace_sha256 = {
        name: tensor_sha256(value)
        for name, value in {
            "mixer_input": trace.mixer_input,
            "query": trace.query,
            "key": trace.key,
            "value": trace.value,
            "grouped_key": trace.grouped_key,
            "grouped_value": trace.grouped_value,
            "gate": trace.gate,
            "mixer_output": trace.mixer_output,
        }.items()
    }
    source_weight_sha256 = {
        name: tensor_sha256(value)
        for name, value in {
            "query_weight": trace.query_weight,
            "key_weight": trace.key_weight,
            "value_weight": trace.value_weight,
            "output_weight": trace.output_weight,
        }.items()
    }
    exact_development = exact.output[development, supervised]
    bounded_development = bounded_output[development, supervised]
    compressed_development = compressed.output[development, supervised]
    native_development = native_transition.free_running_output[
        development,
        supervised,
    ]
    report = {
        "schema_version": 1,
        "boundary": (
            "gqa-exact-hazard-bounded-surrogate-observable-two-state-"
            "native-projection-v1"
        ),
        "fit_rows": calibration_rows,
        "development_rows": int(trace.query.shape[0] - calibration_rows),
        "split_contract": (
            "calibration prefix solves/selects; adaptive-development suffix "
            "reports decomposition; frozen validation is caller-owned"
        ),
        "source_geometry": {
            "query_heads": query_heads,
            "key_value_heads": groups,
            "source_head_dim": config.source_head_dim,
        },
        "target_geometry": {
            "native_heads": module.num_heads,
            "native_head_dim": module.head_dim,
        },
        "solver_config": solver_config,
        "solver_config_sha256": _json_sha256(solver_config),
        "trace_sha256": trace_sha256,
        "trace_aggregate_sha256": _json_sha256(trace_sha256),
        "source_weight_sha256": source_weight_sha256,
        "source_weight_aggregate_sha256": _json_sha256(
            source_weight_sha256
        ),
        "exact_prefix_hazard_oracle": {
            "attention_output": tensor_metrics(
                exact_hazard_output[development, supervised],
                exact_development,
            ),
            **_per_head_group_metrics(
                exact_hazard_output[development, supervised],
                exact_development,
                key_value_heads=groups,
            ),
        },
        "bounded_hazard_surrogate": {
            "link": "sigmoid",
            "hazard": hazard_metrics(
                bounded_hazards[
                    development,
                    :,
                    supervised,
                    :,
                ],
                exact.hazards[
                    development,
                    :,
                    supervised,
                    :,
                ],
                exact.valid_hazards[supervised, :],
            ),
            "attention_output": tensor_metrics(
                bounded_development,
                exact_development,
            ),
            **_per_head_group_metrics(
                bounded_development,
                exact_development,
                key_value_heads=groups,
            ),
        },
        "observable_state_compression": {
            "budget": {
                "states_per_source_head": 2,
                "dc_per_state": 1,
                "query_features_per_state": module.head_dim - 1,
                "shared_input_subspace": False,
            },
            "partial_rope": {
                "rotary_coordinates_in_first_state": config.rotary_dim,
                "first_state_invariant_features": (
                    module.head_dim - 1 - config.rotary_dim
                ),
                "second_state_invariant_features": module.head_dim - 1,
            },
            "affine_output": tensor_metrics(
                compressed.output[development, supervised],
                affine.output[development, supervised],
            ),
            "exact_attention_output": tensor_metrics(
                compressed_development,
                exact_development,
            ),
            **_per_head_group_metrics(
                compressed_development,
                exact_development,
                key_value_heads=groups,
            ),
        },
        "native_transition": {
            "free_running_vs_compressed": tensor_metrics(
                native_transition.free_running_output[development, supervised],
                compressed.output[development, supervised],
            ),
            "free_running_vs_exact_attention": tensor_metrics(
                native_development,
                exact_development,
            ),
            **_per_head_group_metrics(
                native_development,
                exact_development,
                key_value_heads=groups,
            ),
            "requested_decay_clamped_fraction": float(
                (
                    native_transition.requested_decay
                    != native_transition.decay
                )
                .float()
                .mean()
            ),
        },
        "native_parameter_projection": projection_fit.report,
        "parameter_sha256": {
            name: tensor_sha256(value) for name, value in sorted(parameters.items())
        },
    }
    return GQANativeFitResult(parameters=parameters, report=report)


def _validate_trace(
    module: ProjectionBoundaryRWKV7Attention,
    trace: GQANativeFitTrace,
    config: GQANativeFitConfig,
) -> None:
    row_aligned = (
        trace.mixer_input,
        trace.query,
        trace.key,
        trace.value,
        trace.grouped_key,
        trace.grouped_value,
        trace.gate,
        trace.mixer_output,
    )
    batch, tokens = trace.mixer_input.shape[:2]
    if any(value.shape[:2] != (batch, tokens) for value in row_aligned):
        raise ValueError("GQA fit trace batch/time dimensions do not align")
    if not 1 < config.calibration_rows < batch:
        raise ValueError(
            "GQA fit requires non-empty calibration and development rows"
        )
    if trace.query.shape != trace.key.shape or trace.query.shape != trace.value.shape:
        raise ValueError("repeated GQA query/key/value traces must align")
    query_heads, source_head_dim = trace.query.shape[2:]
    if source_head_dim != config.source_head_dim:
        raise ValueError("GQA source head dimension differs from fit config")
    if trace.grouped_key.shape != trace.grouped_value.shape:
        raise ValueError("grouped GQA key/value traces must align")
    if query_heads % trace.grouped_key.shape[2]:
        raise ValueError("GQA KV groups must divide query heads")
    if source_head_dim != 2 * module.head_dim:
        raise ValueError("GQA two-state fit requires source_dim=2*native_dim")
    if module.num_heads != query_heads * 2:
        raise ValueError("GQA two-state fit requires two native heads per query head")
    if config.positions.shape != (batch, tokens):
        raise ValueError("GQA position ids do not align with fit rows")
    if not 0 <= config.supervised_token_start < tokens:
        raise ValueError("GQA supervised token start is outside the trace")
    if config.observable_fit_steps < 0:
        raise ValueError("GQA observable fit steps must be non-negative")
    if config.observable_fit_learning_rate <= 0:
        raise ValueError("GQA observable fit learning rate must be positive")
    if trace.gate.shape != trace.query.shape:
        raise ValueError("GQA gate must align with repeated query heads")
    if trace.mixer_output.shape[-1] != module.hidden_size:
        raise ValueError("GQA mixer output width differs from target residual width")
    source_hidden = trace.mixer_input.shape[-1]
    groups = trace.grouped_key.shape[2]
    expected_weights = {
        "query_weight": (
            query_heads * source_head_dim * 2,
            source_hidden,
        ),
        "key_weight": (groups * source_head_dim, source_hidden),
        "value_weight": (groups * source_head_dim, source_hidden),
        "output_weight": (module.hidden_size, query_heads * source_head_dim),
    }
    for name, expected in expected_weights.items():
        if tuple(getattr(trace, name).shape) != expected:
            raise ValueError(
                f"GQA {name} shape differs from source geometry: "
                f"got={tuple(getattr(trace, name).shape)} expected={expected}"
            )
    if trace.output_bias is not None and tuple(trace.output_bias.shape) != (
        module.hidden_size,
    ):
        raise ValueError("GQA output bias shape differs from target residual width")
    if trace.output_bias is not None and bool(trace.output_bias.ne(0).any()):
        raise ValueError(
            "GQA source output bias is not representable by bias-free o_proj"
        )
    for value in (*row_aligned, config.positions):
        if value.is_floating_point() and not bool(torch.isfinite(value.float()).all()):
            raise ValueError("GQA fit trace contains non-finite values")


def _rotate_native(
    value: Tensor,
    positions: Tensor,
    *,
    source_head_dim: int,
    rotary_dim: int,
    rope_theta: float,
    inverse: bool,
) -> Tensor:
    batch, tokens = value.shape[:2]
    flat_width = value.shape[2] * value.shape[3]
    if flat_width % source_head_dim:
        raise ValueError("native projection width does not fit source RoPE heads")
    source_view = value.reshape(
        batch,
        tokens,
        flat_width // source_head_dim,
        source_head_dim,
    )
    rotated = apply_partial_rope(
        source_view,
        -positions if inverse else positions,
        rotary_dim=rotary_dim,
        theta=rope_theta,
    )
    return rotated.reshape_as(value)


def _fit_native_projection(
    *,
    module: ProjectionBoundaryRWKV7Attention,
    trace: GQANativeFitTrace,
    target_read: Tensor,
    native_transition,
    config: GQANativeFitConfig,
    native_head_dim: int,
    query_basis: Tensor,
    dc_indices: tuple[int, int],
) -> _NativeProjectionFit:
    calibration_rows = config.calibration_rows
    ridges = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    target_pre_rope_read = _rotate_native(
        target_read,
        config.positions,
        source_head_dim=config.source_head_dim,
        rotary_dim=config.rotary_dim,
        rope_theta=config.rope_theta,
        inverse=True,
    )
    target_pre_rope_key = _rotate_native(
        native_transition.key,
        config.positions,
        source_head_dim=config.source_head_dim,
        rotary_dim=config.rotary_dim,
        rope_theta=config.rope_theta,
        inverse=True,
    )
    source_heads = query_basis.shape[0]
    source_hidden = trace.mixer_input.shape[-1]
    query_weight = trace.query_weight.reshape(
        source_heads,
        config.source_head_dim * 2,
        source_hidden,
    )[:, : config.source_head_dim]
    read_prior = torch.zeros(
        source_heads,
        2,
        native_head_dim,
        source_hidden,
        dtype=torch.float32,
        device=trace.mixer_input.device,
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
    kv_heads = trace.key_weight.shape[0] // config.source_head_dim
    group_repeat = source_heads // kv_heads

    def repeated_kv_prior(weight: Tensor) -> Tensor:
        return (
            weight.reshape(kv_heads, config.source_head_dim, source_hidden)
            .repeat_interleave(group_repeat, dim=0)
            .flatten(0, 1)
        )

    def select_direct(target: Tensor, prior: Tensor):
        candidates = {
            "zero": select_bias_free_projection(
                trace.mixer_input,
                target,
                calibration_batches=calibration_rows,
                ridges=ridges,
            ),
            "source-compatible": select_bias_free_projection(
                trace.mixer_input,
                target,
                calibration_batches=calibration_rows,
                ridges=ridges,
                prior_weight=prior,
            ),
        }
        selected_name = min(
            candidates,
            key=lambda name: (
                candidates[name].selection_nmse[candidates[name].ridge],
                name,
            ),
        )
        return candidates[selected_name], selected_name

    read, read_center = select_direct(target_pre_rope_read, read_prior)
    key, key_center = select_direct(
        target_pre_rope_key,
        repeated_kv_prior(trace.key_weight),
    )
    value, value_center = select_direct(
        native_transition.value,
        repeated_kv_prior(trace.value_weight),
    )
    calibration_source = trace.mixer_input[:calibration_rows].reshape(
        -1, source_hidden
    ).float()
    ridge_scale = float(
        calibration_source.square().sum() / max(1, calibration_source.shape[-1])
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
    decay_rank = int(module.w_lora.lora[0].out_features)
    erase_rank = int(module.a_lora.lora[0].out_features)
    gate_rank = int(module.g_lora.lora[0].out_features)
    decay = _fit_native_low_rank_projection(
        trace.mixer_input,
        decay_logits,
        calibration_batches=calibration_rows,
        native_rank=decay_rank,
        ridge=low_rank_ridge,
        hidden_activation="tanh",
    )
    erase = _fit_native_low_rank_projection(
        trace.mixer_input,
        erase_logits,
        calibration_batches=calibration_rows,
        native_rank=erase_rank,
        ridge=low_rank_ridge,
        hidden_activation="identity",
    )
    gate_target = trace.gate.reshape(
        *trace.gate.shape[:2],
        native_heads,
        native_head_dim,
    )
    source_gate_weight = trace.query_weight.reshape(
        source_heads,
        config.source_head_dim * 2,
        source_hidden,
    )[:, config.source_head_dim :].flatten(0, 1)
    effective_gate_rank = min(
        gate_rank,
        source_gate_weight.shape[0],
        source_hidden,
    )
    gate_indices = torch.linspace(
        0,
        source_gate_weight.shape[0] - 1,
        effective_gate_rank,
        dtype=torch.float64,
        device=trace.mixer_input.device,
    ).round().to(torch.long)
    gate_down = source_gate_weight.index_select(0, gate_indices)
    gate_features = torch.sigmoid(trace.mixer_input @ gate_down.T)
    gate_up = select_bias_free_projection(
        gate_features,
        gate_target,
        calibration_batches=calibration_rows,
        ridges=ridges,
    )
    gate = _pad_low_rank_projection(
        LowRankProjection(
            down_weight=gate_down,
            up_weight=gate_up.projection.weight,
            bias=torch.zeros(
                native_heads * native_head_dim,
                dtype=torch.float32,
                device=trace.mixer_input.device,
            ),
            prediction=gate_up.projection.prediction,
            hidden_activation="sigmoid",
            output_bias=False,
        ),
        native_rank=gate_rank,
    )

    predicted_read = _rotate_native(
        read.projection.prediction,
        config.positions,
        source_head_dim=config.source_head_dim,
        rotary_dim=config.rotary_dim,
        rope_theta=config.rope_theta,
        inverse=False,
    )
    predicted_key = _rotate_native(
        key.projection.prediction,
        config.positions,
        source_head_dim=config.source_head_dim,
        rotary_dim=config.rotary_dim,
        rope_theta=config.rope_theta,
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
    output_candidates = {
        "zero": select_bias_free_projection(
            pre_output,
            trace.mixer_output,
            calibration_batches=calibration_rows,
            ridges=ridges,
        ),
        "source-compatible": select_bias_free_projection(
            pre_output,
            trace.mixer_output,
            calibration_batches=calibration_rows,
            ridges=ridges,
            prior_weight=trace.output_weight,
        ),
    }
    output_center = min(
        output_candidates,
        key=lambda name: (
            output_candidates[name].selection_nmse[
                output_candidates[name].ridge
            ],
            name,
        ),
    )
    output = output_candidates[output_center]
    development = slice(calibration_rows, None)
    supervised = slice(config.supervised_token_start, None)
    report = {
        "ridge_center": {
            "r_proj": read_center,
            "k_proj": key_center,
            "v_proj": value_center,
            "o_proj": output_center,
        },
        "native_rank": {
            "w_lora": decay_rank,
            "a_lora": erase_rank,
            "g_lora": gate_rank,
        },
        "effective_rank": {
            "w_lora": min(
                decay_rank,
                source_hidden,
                math.prod(decay_logits.shape[2:]),
            ),
            "a_lora": min(
                erase_rank,
                source_hidden,
                math.prod(erase_logits.shape[2:]),
            ),
            "g_lora": effective_gate_rank,
        },
        "pre_post_rope_read_projection": tensor_metrics(
            predicted_read[development, supervised],
            target_read[development, supervised],
        ),
        "decay_signal": tensor_metrics(
            predicted_decay[development, supervised],
            native_transition.decay[development, supervised]
            .unsqueeze(-1)
            .expand_as(predicted_decay[development, supervised]),
        ),
        "erase_signal": tensor_metrics(
            predicted_erase[development, supervised],
            native_transition.erase[development, supervised],
        ),
        "write_key_signal": tensor_metrics(
            predicted_key[development, supervised],
            native_transition.key[development, supervised],
        ),
        "write_value_signal": tensor_metrics(
            value.projection.prediction[development, supervised],
            native_transition.value[development, supervised],
        ),
        "gate_signal": tensor_metrics(
            gate.prediction[development, supervised],
            gate_target[development, supervised],
        ),
        "complete_free_running_mixer": tensor_metrics(
            output.projection.prediction[development, supervised],
            trace.mixer_output[development, supervised],
        ),
    }
    return _NativeProjectionFit(
        read=read,
        key=key,
        value=value,
        decay=decay,
        erase=erase,
        gate=gate,
        output=output,
        report=report,
    )


def _fit_native_low_rank_projection(
    source: Tensor,
    target: Tensor,
    *,
    calibration_batches: int,
    native_rank: int,
    ridge: float,
    hidden_activation: str,
) -> LowRankProjection:
    effective_rank = min(
        native_rank,
        source.shape[-1],
        math.prod(target.shape[2:]),
    )
    fitted = fit_low_rank_projection(
        source,
        target,
        calibration_batches=calibration_batches,
        rank=effective_rank,
        ridge=ridge,
        hidden_activation=hidden_activation,
    )
    return _pad_low_rank_projection(fitted, native_rank=native_rank)


def _pad_low_rank_projection(
    projection: LowRankProjection,
    *,
    native_rank: int,
) -> LowRankProjection:
    fitted_rank = projection.down_weight.shape[0]
    if native_rank < fitted_rank:
        raise ValueError("native low-rank width is smaller than fitted rank")
    if native_rank == fitted_rank:
        return projection
    down_weight = functional.pad(
        projection.down_weight,
        (0, 0, 0, native_rank - fitted_rank),
    )
    up_weight = functional.pad(
        projection.up_weight,
        (0, native_rank - fitted_rank),
    )
    return LowRankProjection(
        down_weight=down_weight,
        up_weight=up_weight,
        bias=projection.bias,
        prediction=projection.prediction,
        hidden_activation=projection.hidden_activation,
        output_bias=projection.output_bias,
    )


def _native_parameter_set(
    module: ProjectionBoundaryRWKV7Attention,
    fit: _NativeProjectionFit,
    *,
    source_value_weight: Tensor,
) -> dict[str, Tensor]:
    parameters = {
        name: torch.zeros_like(parameter, dtype=torch.float32)
        for name, parameter in module.named_parameters()
    }
    parameters["k_k"].fill_(1)
    parameters["g_norm.weight"].fill_(1)
    parameters["r_proj.weight"].copy_(fit.read.projection.weight)
    parameters["k_proj.weight"].copy_(fit.key.projection.weight)
    parameters["v_proj.weight"].copy_(fit.value.projection.weight)
    parameters["o_proj.weight"].copy_(fit.output.projection.weight)
    parameters["w_lora.lora.0.weight"].copy_(fit.decay.down_weight)
    parameters["w_lora.lora.2.weight"].copy_(fit.decay.up_weight)
    parameters["w_lora.lora.2.bias"].copy_(fit.decay.bias)
    parameters["a_lora.lora.0.weight"].copy_(fit.erase.down_weight)
    parameters["a_lora.lora.2.weight"].copy_(fit.erase.up_weight)
    parameters["a_lora.lora.2.bias"].copy_(fit.erase.bias)
    parameters["g_lora.lora.0.weight"].copy_(fit.gate.down_weight)
    parameters["g_lora.lora.2.weight"].copy_(fit.gate.up_weight)
    if module.layer_idx:
        value_rank = parameters["v_lora.lora.0.weight"].shape[0]
        basis_indices = torch.linspace(
            0,
            source_value_weight.shape[0] - 1,
            value_rank,
            dtype=torch.float64,
            device=source_value_weight.device,
        ).round().to(torch.long)
        parameters["v_lora.lora.0.weight"].copy_(
            source_value_weight.index_select(0, basis_indices)
        )
        parameters["v_lora.lora.2.bias"].fill_(
            torch.logit(
                torch.tensor(
                    torch.finfo(torch.bfloat16).eps**2,
                    device=source_value_weight.device,
                )
            )
        )
    return parameters
