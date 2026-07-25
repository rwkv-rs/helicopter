from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from .recurrent import rwkv7_scan, rwkv7_step


RWKV7_MINIMUM_DECAY = math.exp(-math.exp(-0.5))


def normalized_mse(prediction: Tensor, target: Tensor) -> float:
    prediction = prediction.float()
    target = target.float()
    denominator = target.square().sum().clamp_min(1e-30)
    return float((prediction - target).square().sum() / denominator)


def relative_l2(prediction: Tensor, target: Tensor) -> float:
    prediction = prediction.float()
    target = target.float()
    denominator = torch.linalg.vector_norm(target).clamp_min(1e-30)
    return float(torch.linalg.vector_norm(prediction - target) / denominator)


def cosine(prediction: Tensor, target: Tensor) -> float:
    prediction = prediction.float().flatten()
    target = target.float().flatten()
    denominator = (
        torch.linalg.vector_norm(prediction)
        * torch.linalg.vector_norm(target)
    ).clamp_min(1e-30)
    return float(torch.dot(prediction, target) / denominator)


def tensor_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    return {
        "nmse": normalized_mse(prediction, target),
        "relative_l2": relative_l2(prediction, target),
        "cosine": cosine(prediction, target),
    }


def qwen35_l2_normalize(
    value: Tensor,
    *,
    squared_norm_epsilon: float = 1e-6,
) -> Tensor:
    """Apply Qwen3.5's additive squared-norm L2 normalization."""
    if squared_norm_epsilon <= 0:
        raise ValueError("squared_norm_epsilon must be positive")
    return value * torch.rsqrt(
        value.square().sum(dim=-1, keepdim=True) + squared_norm_epsilon
    )


def gdn_reference_scan(
    state: Tensor,
    decay: Tensor,
    beta: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run Qwen3.5's GDN recurrence in canonical [value,key] coordinates."""
    outputs: list[Tensor] = []
    current = state
    for index in range(query.shape[1]):
        current_decay = decay[:, index].unsqueeze(-2)
        current_key = key[:, index]
        current_beta = beta[:, index]
        current_value = value[:, index]
        decayed = current * current_decay
        state_key = torch.einsum("bhvk,bhk->bhv", decayed, current_key)
        update = torch.einsum(
            "bhv,bhk->bhvk",
            current_beta * (current_value - state_key),
            current_key,
        )
        current = decayed + update
        outputs.append(
            torch.einsum(
                "bhvk,bhk->bhv",
                current,
                query[:, index],
            )
            / query.shape[-1] ** 0.5
        )
    return torch.stack(outputs, dim=1), current


def verify_gdn_mapping(
    decay: Tensor,
    beta: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> dict[str, float]:
    """Compare the exact GDN recurrence with its canonical RWKV7 signals."""
    state_shape = (
        query.shape[0],
        query.shape[2],
        value.shape[-1],
        key.shape[-1],
    )
    source_state = torch.zeros(
        state_shape,
        dtype=query.dtype,
        device=query.device,
    )
    target_state = torch.zeros_like(source_state)
    normalized_query = qwen35_l2_normalize(query)
    source_output, _ = gdn_reference_scan(
        source_state,
        decay,
        beta,
        normalized_query,
        key,
        value,
    )
    target_output, _ = rwkv7_scan(
        target_state,
        normalized_query / query.shape[-1] ** 0.5,
        decay.expand_as(key),
        key,
        value * beta,
        -key,
        key * beta * decay,
    )
    difference = source_output - target_output
    return {
        "output_max_abs": float(difference.abs().max()),
        "output_relative_l2": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(source_output).clamp_min(1e-30)
        ),
    }


@dataclass(frozen=True)
class CausalAttention:
    output: Tensor
    weights: Tensor
    hazards: Tensor
    valid_hazards: Tensor


def causal_attention(query: Tensor, key: Tensor, value: Tensor) -> CausalAttention:
    """Compute attention and its exact prefix-update hazards.

    All inputs use ``[batch, time, head, feature]``. ``hazards[..., q, i]`` is
    the probability assigned to update ``i`` when the fixed future query at
    position ``q`` sees only keys ``0..i``. It is therefore not the final
    attention weight, whose denominator contains keys ``0..q``.
    """
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have identical shapes")
    if query.ndim != 4:
        raise ValueError("attention inputs must be [batch,time,head,feature]")
    _, sequence_length, _, head_dim = query.shape
    query_by_head = query.transpose(1, 2).float()
    key_by_head = key.transpose(1, 2).float()
    value_by_head = value.transpose(1, 2).float()
    scores = torch.einsum(
        "bhqd,bhid->bhqi", query_by_head, key_by_head
    ) * (head_dim**-0.5)
    valid = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=query.device,
    ).tril()
    masked_scores = scores.masked_fill(~valid, float("-inf"))
    weights = torch.softmax(masked_scores, dim=-1)
    output = torch.einsum("bhqi,bhid->bhqd", weights, value_by_head).transpose(
        1, 2
    )
    prefix_log_normalizer = torch.logcumsumexp(scores, dim=-1)
    hazards = torch.exp(scores - prefix_log_normalizer).masked_fill(~valid, 0)
    return CausalAttention(output, weights, hazards, valid)


def probability_taylor_hazards(query: Tensor, key: Tensor) -> Tensor:
    """First-order Taylor expansion of the probability at ``query=0``."""
    if query.shape != key.shape or query.ndim != 4:
        raise ValueError("query and key must share [batch,time,head,feature]")
    _, sequence_length, _, head_dim = query.shape
    query_by_head = query.transpose(1, 2).float()
    key_by_head = key.transpose(1, 2).float()
    counts = torch.arange(
        1,
        sequence_length + 1,
        dtype=query_by_head.dtype,
        device=query.device,
    )
    prefix_mean = key_by_head.cumsum(dim=2) / counts.view(1, 1, -1, 1)
    gradient = (
        (key_by_head - prefix_mean)
        * (head_dim**-0.5)
        / counts.view(1, 1, -1, 1)
    )
    prediction = torch.einsum(
        "bhqd,bhid->bhqi", query_by_head, gradient
    ) + counts.reciprocal().view(1, 1, 1, -1)
    valid = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=query.device,
    ).tril()
    return prediction.masked_fill(~valid, 0)


def logit_taylor_hazards(
    query: Tensor,
    key: Tensor,
    center: Tensor,
) -> Tensor:
    """Linearize prefix log-sum-exp, then retain the exact sigmoid link."""
    if query.shape != key.shape or query.ndim != 4:
        raise ValueError("query and key must share [batch,time,head,feature]")
    batch, sequence_length, heads, head_dim = query.shape
    if center.shape != (heads, head_dim):
        raise ValueError("center must be [head,feature]")
    query_by_head = query.transpose(1, 2).float()
    key_by_head = key.transpose(1, 2).float()
    center = center.float()
    centered_query = query_by_head - center.view(1, heads, 1, head_dim)
    prediction = query.new_zeros(
        (batch, heads, sequence_length, sequence_length),
        dtype=torch.float32,
    )
    prediction[..., 0] = 1
    scale = head_dim**-0.5
    for index in range(1, sequence_length):
        previous_key = key_by_head[:, :, :index]
        center_scores = torch.einsum(
            "hd,bhjd->bhj", center, previous_key
        ) * scale
        center_weights = torch.softmax(center_scores, dim=-1)
        center_key = torch.einsum(
            "bhj,bhjd->bhd", center_weights, previous_key
        )
        current_key = key_by_head[:, :, index]
        center_logit = (
            torch.einsum("hd,bhd->bh", center, current_key) * scale
            - torch.logsumexp(center_scores, dim=-1)
        )
        slope = (current_key - center_key) * scale
        query_logit = center_logit.unsqueeze(-1) + torch.einsum(
            "bhqd,bhd->bhq",
            centered_query[:, :, index:],
            slope,
        )
        prediction[:, :, index:, index] = torch.sigmoid(query_logit)
    return prediction


def rollout_hazards(hazards: Tensor, value: Tensor) -> Tensor:
    """Evaluate prefix hazards on every future query and return diagonal reads."""
    if hazards.ndim != 4 or value.ndim != 4:
        raise ValueError("hazards and value must be rank four")
    batch, heads, query_length, update_length = hazards.shape
    if query_length != update_length:
        raise ValueError("hazards must contain a square query/update grid")
    if value.shape[:3] != (batch, update_length, heads):
        raise ValueError("value does not align with hazards")
    output = value.new_zeros(
        (batch, heads, query_length, value.shape[-1]), dtype=torch.float32
    )
    value_by_head = value.transpose(1, 2).float()
    for index in range(update_length):
        probability = hazards[:, :, index:, index].unsqueeze(-1).float()
        output[:, :, index:] = (
            (1 - probability) * output[:, :, index:]
            + probability * value_by_head[:, :, index].unsqueeze(2)
        )
    return output.transpose(1, 2)


def hazard_metrics(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
) -> dict[str, float]:
    if prediction.shape != target.shape:
        raise ValueError("hazard prediction and target must align")
    mask = valid.view(1, 1, *valid.shape).expand_as(target)
    predicted_valid = prediction[mask].float()
    target_valid = target[mask].float()
    metrics = tensor_metrics(predicted_valid, target_valid)
    metrics.update(
        {
            "mae": float((predicted_valid - target_valid).abs().mean()),
            "outside_unit_interval_fraction": float(
                ((predicted_valid < 0) | (predicted_valid > 1)).float().mean()
            ),
        }
    )
    return metrics


def probability_tangent_parameters(
    key: Tensor,
    center: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return ``p(center)`` and ``grad p(center)`` for every prefix update."""
    if key.ndim != 4:
        raise ValueError("key must be [batch,time,group,feature]")
    batch, sequence_length, groups, head_dim = key.shape
    if center.shape != (groups, head_dim):
        raise ValueError("center must be [group,feature]")
    probability = key.new_ones((batch, sequence_length, groups)).float()
    slope = key.new_zeros((batch, sequence_length, groups, head_dim)).float()
    key = key.float()
    center = center.float()
    scale = head_dim**-0.5
    for index in range(1, sequence_length):
        previous_key = key[:, :index]
        center_scores = torch.einsum(
            "gd,bjgd->bgj", center, previous_key
        ) * scale
        center_weights = torch.softmax(center_scores, dim=-1)
        center_key = torch.einsum(
            "bgj,bjgd->bgd", center_weights, previous_key
        )
        current_key = key[:, index]
        center_logit = (
            torch.einsum("gd,bgd->bg", center, current_key) * scale
            - torch.logsumexp(center_scores, dim=-1)
        )
        current_probability = torch.sigmoid(center_logit)
        probability[:, index] = current_probability
        slope[:, index] = (
            current_probability * (1 - current_probability)
        ).unsqueeze(-1) * (current_key - center_key) * scale
    return probability, slope


@dataclass(frozen=True)
class AffineRollout:
    output: Tensor
    states: Tensor
    bias: Tensor
    closure_scale: Tensor
    centered_query: Tensor


def affine_state_rollout(
    query: Tensor,
    grouped_key: Tensor,
    grouped_value: Tensor,
    center: Tensor,
    *,
    calibration_batches: int,
    fit_rank_one_closure: bool,
    fixed_closure_scale: Tensor | None = None,
) -> AffineRollout:
    """Roll out the centered affine state and an optional calibrated closure.

    ``fixed_closure_scale`` replays parameters selected on an earlier split.
    It is mutually exclusive with fitting so a frozen final split cannot
    silently influence the rank-one closure.
    """
    if query.ndim != 4 or grouped_key.ndim != 4 or grouped_value.ndim != 4:
        raise ValueError("query, key, and value must be rank four")
    if grouped_key.shape != grouped_value.shape:
        raise ValueError("grouped key and value must align")
    batch, sequence_length, heads, head_dim = query.shape
    if grouped_key.shape[:2] != (batch, sequence_length):
        raise ValueError("grouped key does not align with query")
    groups = grouped_key.shape[2]
    if heads % groups:
        raise ValueError("query heads must divide evenly into KV groups")
    if grouped_key.shape[-1] != head_dim or center.shape != (groups, head_dim):
        raise ValueError("GQA head dimensions or center do not align")
    if not 0 < calibration_batches <= batch:
        raise ValueError("calibration_batches must select a non-empty prefix")
    if fit_rank_one_closure and fixed_closure_scale is not None:
        raise ValueError(
            "cannot fit and replay a fixed rank-one closure simultaneously"
        )
    if fixed_closure_scale is not None:
        if fixed_closure_scale.shape != (groups, sequence_length):
            raise ValueError(
                "fixed closure scale must align with group and sequence dimensions"
            )
        if not bool(torch.isfinite(fixed_closure_scale).all()):
            raise ValueError("fixed closure scale must be finite")

    group_width = heads // groups
    head_to_group = torch.arange(heads, device=query.device) // group_width
    centered_query = query.float() - center[head_to_group].view(
        1, 1, heads, head_dim
    ).float()
    probability, slope = probability_tangent_parameters(grouped_key, center)
    state = query.new_zeros(
        (batch, groups, head_dim, head_dim), dtype=torch.float32
    )
    bias = query.new_zeros((batch, groups, head_dim), dtype=torch.float32)
    outputs = query.new_empty(
        (batch, sequence_length, heads, head_dim), dtype=torch.float32
    )
    states = query.new_empty(
        (batch, sequence_length, groups, head_dim, head_dim),
        dtype=torch.float32,
    )
    biases = query.new_empty(
        (batch, sequence_length, groups, head_dim), dtype=torch.float32
    )
    closure_scale = (
        query.new_zeros((groups, sequence_length), dtype=torch.float32)
        if fixed_closure_scale is None
        else fixed_closure_scale.detach().to(
            device=query.device,
            dtype=torch.float32,
        ).clone()
    )

    for index in range(sequence_length):
        current_slope = slope[:, index]
        slope_norm = torch.linalg.vector_norm(
            current_slope, dim=-1
        ).clamp_min(1e-30)
        direction = current_slope / slope_norm.unsqueeze(-1)
        if fit_rank_one_closure and index:
            for group in range(groups):
                head_start = group * group_width
                head_stop = head_start + group_width
                future_query = centered_query[
                    :calibration_batches,
                    index:,
                    head_start:head_stop,
                ].reshape(calibration_batches, -1, head_dim)
                calibration_state = state[:calibration_batches, group]
                calibration_slope = current_slope[
                    :calibration_batches, group
                ]
                calibration_direction = direction[
                    :calibration_batches, group
                ]
                calibration_norm = slope_norm[
                    :calibration_batches, group
                ]
                state_query = torch.einsum(
                    "bod,bsd->bso", calibration_state, future_query
                )
                slope_query = torch.einsum(
                    "bd,bsd->bs", calibration_slope, future_query
                )
                target = slope_query.unsqueeze(-1) * state_query
                state_direction = torch.einsum(
                    "bod,bd->bo",
                    calibration_state,
                    calibration_direction,
                )
                direction_query = torch.einsum(
                    "bd,bsd->bs",
                    calibration_direction,
                    future_query,
                )
                basis = (
                    calibration_norm.unsqueeze(-1) * direction_query
                ).unsqueeze(-1) * state_direction.unsqueeze(1)
                denominator = basis.square().sum()
                if float(denominator) > 1e-30:
                    closure_scale[group, index] = (
                        (basis * target).sum() / denominator
                    )

        old_bias = bias
        decayed_state = (
            (1 - probability[:, index]).unsqueeze(-1).unsqueeze(-1) * state
        )
        write = (
            grouped_value[:, index].float() - old_bias
        ).unsqueeze(-1) * current_slope.unsqueeze(-2)
        if fit_rank_one_closure or fixed_closure_scale is not None:
            state_direction = torch.einsum(
                "bgod,bgd->bgo", state, direction
            )
            correction = (
                closure_scale[:, index].view(1, groups, 1, 1)
                * slope_norm.unsqueeze(-1).unsqueeze(-1)
                * state_direction.unsqueeze(-1)
                * direction.unsqueeze(-2)
            )
        else:
            correction = torch.zeros_like(state)
        state = decayed_state + write - correction
        bias = (
            (1 - probability[:, index]).unsqueeze(-1) * old_bias
            + probability[:, index].unsqueeze(-1)
            * grouped_value[:, index].float()
        )
        states[:, index] = state
        biases[:, index] = bias
        current_query = centered_query[:, index].reshape(
            batch, groups, group_width, head_dim
        )
        current_output = torch.einsum(
            "bgod,bgwd->bgwo", state, current_query
        ) + bias.unsqueeze(2)
        outputs[:, index] = current_output.reshape(batch, heads, head_dim)

    return AffineRollout(
        outputs,
        states,
        biases,
        closure_scale,
        centered_query,
    )


def top_eigenvectors(matrix: Tensor, rank: int) -> Tensor:
    matrix = matrix.float()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if not 0 < rank <= matrix.shape[0]:
        raise ValueError("rank must fit the matrix")
    eigenvalues, eigenvectors = torch.linalg.eigh(
        (matrix + matrix.T) * 0.5
    )
    return eigenvectors[:, -rank:]


def operator_input_bases(
    states: Tensor,
    *,
    calibration_batches: int,
    rank: int,
) -> Tensor:
    """Best common input subspace for operator Frobenius error."""
    if states.ndim != 5:
        raise ValueError("states must be [batch,time,group,output,input]")
    groups = states.shape[2]
    bases = []
    for group in range(groups):
        calibration = states[:calibration_batches, :, group].reshape(
            -1, states.shape[-2], states.shape[-1]
        )
        gram = torch.einsum("nod,noe->de", calibration, calibration)
        bases.append(top_eigenvectors(gram, rank))
    return torch.stack(bases)


def query_input_bases(
    centered_query: Tensor,
    *,
    calibration_batches: int,
    rank: int,
) -> Tensor:
    """PCA input subspace for each Query head."""
    if centered_query.ndim != 4:
        raise ValueError("query must be [batch,time,head,feature]")
    bases = []
    for head in range(centered_query.shape[2]):
        calibration = centered_query[
            :calibration_batches, :, head
        ].reshape(-1, centered_query.shape[-1])
        bases.append(top_eigenvectors(calibration.T @ calibration, rank))
    return torch.stack(bases)


def _observable_projection_loss(
    state: Tensor,
    query: Tensor,
    basis: Tensor,
    fixed_error: Tensor | None = None,
) -> Tensor:
    projected = (query @ basis) @ basis.T
    residual = query - projected
    observable_error = torch.einsum("nod,nd->no", state, residual)
    if fixed_error is not None:
        observable_error = observable_error + fixed_error
    return observable_error.square().sum()


def _fit_observable_basis(
    state: Tensor,
    query: Tensor,
    *,
    rank: int,
    steps: int,
    learning_rate: float,
    fixed_error: Tensor | None = None,
) -> Tensor:
    initial = top_eigenvectors(query.T @ query, rank)
    sensitivity = top_eigenvectors(
        torch.einsum("nod,noe->de", state, state),
        rank,
    )
    candidates = (initial.float(), sensitivity)
    candidate_losses = tuple(
        _observable_projection_loss(
            state,
            query,
            candidate,
            fixed_error,
        )
        for candidate in candidates
    )
    selected_index = min(
        range(len(candidates)),
        key=lambda index: float(candidate_losses[index]),
    )
    selected = candidates[selected_index]
    selected_loss = candidate_losses[selected_index]
    parameter = selected.clone().requires_grad_(True)
    optimizer = torch.optim.Adam((parameter,), lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        orthogonal, _ = torch.linalg.qr(parameter, mode="reduced")
        loss = _observable_projection_loss(
            state,
            query,
            orthogonal,
            fixed_error,
        )
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        candidate, _ = torch.linalg.qr(parameter, mode="reduced")
        candidate_loss = _observable_projection_loss(
            state,
            query,
            candidate,
            fixed_error,
        )
        if torch.isfinite(candidate_loss) and candidate_loss < selected_loss:
            selected = candidate
    return selected


def observable_query_bases(
    states: Tensor,
    centered_query: Tensor,
    *,
    calibration_batches: int,
    rank: int,
    steps: int = 64,
    learning_rate: float = 0.05,
) -> Tensor:
    """Fit orthogonal input bases against future-read observable error.

    The objective is ``sum ||S(q - PP^T q)||²`` on calibration rows.  Each
    Query head is fitted against the state of its KV group.  Query PCA is the
    deterministic starting point and remains selected unless projected-gradient
    refinement strictly improves this observable objective.
    """
    if states.ndim != 5:
        raise ValueError("states must be [batch,time,group,output,input]")
    if centered_query.ndim != 4:
        raise ValueError("query must be [batch,time,head,feature]")
    batch, sequence_length, groups, output_dim, input_dim = states.shape
    if centered_query.shape[:2] != (batch, sequence_length):
        raise ValueError("state and query batch/time dimensions must align")
    heads = centered_query.shape[2]
    if heads % groups:
        raise ValueError("query heads must divide evenly into state groups")
    if output_dim != input_dim or centered_query.shape[-1] != input_dim:
        raise ValueError("state and query feature dimensions must align")
    if not 0 < calibration_batches <= batch:
        raise ValueError("calibration_batches must select a non-empty prefix")
    if not 0 < rank <= input_dim:
        raise ValueError("rank must fit the input dimension")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    group_width = heads // groups
    fitted: list[Tensor] = []
    for head in range(heads):
        group = head // group_width
        state = states[:calibration_batches, :, group].reshape(
            -1, output_dim, input_dim
        ).float()
        query = centered_query[:calibration_batches, :, head].reshape(
            -1, input_dim
        ).float()
        fitted.append(
            _fit_observable_basis(
                state,
                query,
                rank=rank,
                steps=steps,
                learning_rate=learning_rate,
            )
        )
    return torch.stack(fitted)


def rope_aligned_two_state_bases(
    states: Tensor,
    centered_query: Tensor,
    *,
    calibration_batches: int,
    native_dim: int,
    rotary_dim: int,
    steps: int = 64,
    learning_rate: float = 0.05,
) -> Tensor:
    """Fit independent two-state bases inside the source-RoPE commutant.

    The first native state retains all rotary coordinates and fits its remaining
    invariant features.  The second state can only use invariant coordinates
    under the preserved flattened source-head RoPE layout, so its unavoidable
    rotary contribution is included as a fixed observable residual.
    """
    if states.ndim != 5 or centered_query.ndim != 4:
        raise ValueError("states and query must be rank five/four")
    batch, sequence_length, groups, output_dim, input_dim = states.shape
    if centered_query.shape[:2] != (batch, sequence_length):
        raise ValueError("states and query batch/time dimensions must align")
    heads = centered_query.shape[2]
    if heads % groups:
        raise ValueError("query heads must divide evenly into state groups")
    if input_dim != output_dim or output_dim != native_dim * 2:
        raise ValueError("two-state RoPE basis requires source_dim=2*native_dim")
    if not 0 <= rotary_dim < native_dim:
        raise ValueError("rotary_dim must leave room for an invariant DC channel")
    if not 0 < calibration_batches <= batch:
        raise ValueError("calibration_batches must select a non-empty prefix")
    invariant_dim = input_dim - rotary_dim
    first_invariant_rank = native_dim - 1 - rotary_dim
    second_invariant_rank = native_dim - 1
    if first_invariant_rank <= 0 or second_invariant_rank > invariant_dim:
        raise ValueError("native feature budget does not fit RoPE partition")

    group_width = heads // groups
    rotary_identity = torch.eye(
        input_dim,
        rotary_dim,
        dtype=torch.float32,
        device=states.device,
    )
    fitted: list[Tensor] = []
    for head in range(heads):
        group = head // group_width
        query = centered_query[:calibration_batches, :, head].reshape(
            -1, input_dim
        ).float()
        invariant_query = query[:, rotary_dim:]
        head_bases = []
        for state_index, invariant_rank in enumerate(
            (first_invariant_rank, second_invariant_rank)
        ):
            start = state_index * native_dim
            stop = start + native_dim
            state = states[
                :calibration_batches,
                :,
                group,
                start:stop,
            ].reshape(-1, native_dim, input_dim).float()
            invariant_state = state[..., rotary_dim:]
            if state_index == 0:
                fixed_error = None
            else:
                fixed_error = torch.einsum(
                    "nor,nr->no",
                    state[..., :rotary_dim],
                    query[:, :rotary_dim],
                )
            invariant_basis = _fit_observable_basis(
                invariant_state,
                invariant_query,
                rank=invariant_rank,
                steps=steps,
                learning_rate=learning_rate,
                fixed_error=fixed_error,
            )
            lifted_invariant = torch.zeros(
                input_dim,
                invariant_rank,
                dtype=torch.float32,
                device=states.device,
            )
            lifted_invariant[rotary_dim:] = invariant_basis
            if state_index == 0:
                basis = torch.cat((rotary_identity, lifted_invariant), dim=-1)
            else:
                basis = lifted_invariant
            head_bases.append(basis)
        fitted.append(torch.stack(head_bases))
    return torch.stack(fitted)


@dataclass(frozen=True)
class TwoStateProjection:
    states: Tensor
    read: Tensor
    output: Tensor
    query_basis: Tensor
    dc_indices: tuple[int, int]


@dataclass(frozen=True)
class NativeTwoStateRollout:
    teacher_forced_states: Tensor
    free_running_states: Tensor
    free_running_output: Tensor
    requested_decay: Tensor
    decay: Tensor
    erase: Tensor
    key: Tensor
    value: Tensor


@dataclass(frozen=True)
class BiasFreeProjection:
    weight: Tensor
    prediction: Tensor


@dataclass(frozen=True)
class SelectedBiasFreeProjection:
    projection: BiasFreeProjection
    ridge: float
    absolute_ridge: float
    ridge_scale: float
    selection_nmse: dict[float, float]


@dataclass(frozen=True)
class LowRankProjection:
    down_weight: Tensor
    up_weight: Tensor
    bias: Tensor
    prediction: Tensor
    hidden_activation: str
    output_bias: bool


@dataclass(frozen=True)
class NativeSignalRollout:
    states: Tensor
    output: Tensor


@dataclass(frozen=True)
class MaterializedTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str


@dataclass(frozen=True)
class NativeProjectionMaterialization:
    """Hash-bound record of one atomic native attention installation."""

    aggregate_sha256: str
    tensors: tuple[MaterializedTensor, ...]


def tensor_bytes(value: Tensor) -> bytes:
    """Return exact storage bytes, including dtypes unsupported by NumPy."""
    contiguous = value.detach().cpu().contiguous()
    return contiguous.view(torch.uint8).numpy().tobytes()


def tensor_sha256(value: Tensor) -> str:
    return hashlib.sha256(tensor_bytes(value)).hexdigest()


def _materialization_digest(
    tensors: tuple[tuple[str, Tensor], ...],
) -> str:
    digest = hashlib.sha256()
    for name, value in tensors:
        encoded_name = name.encode("utf-8")
        encoded_dtype = str(value.dtype).encode("ascii")
        digest.update(struct.pack(">I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack(">I", len(encoded_dtype)))
        digest.update(encoded_dtype)
        digest.update(struct.pack(">I", value.ndim))
        for dimension in value.shape:
            digest.update(struct.pack(">Q", int(dimension)))
        payload = tensor_bytes(value)
        digest.update(struct.pack(">Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def materialize_native_projection(
    module: torch.nn.Module,
    tensors: Mapping[str, Tensor],
) -> NativeProjectionMaterialization:
    """Atomically install a complete native attention parameter set.

    The mapping must cover every parameter exactly.  Values are first cast to
    the destination dtype/device and validated without mutating ``module``;
    only then are all parameters copied.  The returned digest is computed from
    the installed tensors, so evidence cannot accidentally bind the FP32 fit
    rather than the BF16 module that was actually evaluated.
    """
    parameters = dict(module.named_parameters())
    expected = set(parameters)
    provided = set(tensors)
    if provided != expected:
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        raise ValueError(
            "native projection must cover every module parameter exactly; "
            f"missing={missing} unexpected={unexpected}"
        )
    prepared: list[tuple[str, Tensor]] = []
    for name in sorted(parameters):
        parameter = parameters[name]
        source = tensors[name]
        if tuple(source.shape) != tuple(parameter.shape):
            raise ValueError(
                f"native projection shape mismatch for {name}: "
                f"got={tuple(source.shape)} expected={tuple(parameter.shape)}"
            )
        converted = source.detach().to(
            device=parameter.device,
            dtype=parameter.dtype,
        ).contiguous()
        if not bool(torch.isfinite(converted.float()).all()):
            raise ValueError(
                f"native projection contains non-finite values after cast: {name}"
            )
        prepared.append((name, converted))
    with torch.no_grad():
        for name, converted in prepared:
            parameters[name].copy_(converted)
    installed = tuple(
        (name, parameters[name].detach().cpu().contiguous())
        for name in sorted(parameters)
    )
    return NativeProjectionMaterialization(
        aggregate_sha256=_materialization_digest(installed),
        tensors=tuple(
            MaterializedTensor(
                name=name,
                shape=tuple(value.shape),
                dtype=str(value.dtype),
                sha256=tensor_sha256(value),
            )
            for name, value in installed
        ),
    )


def _ridge_affine_projection(
    source: Tensor,
    target: Tensor,
    *,
    ridge: float,
) -> tuple[Tensor, Tensor]:
    if source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]:
        raise ValueError("affine projection inputs must be aligned matrices")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    source = source.float()
    target = target.float()
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    if ridge == 0:
        weight = torch.linalg.lstsq(
            centered_source,
            centered_target,
        ).solution.T
    elif centered_source.shape[0] >= centered_source.shape[1]:
        gram = centered_source.T @ centered_source
        gram = gram + torch.eye(
            centered_source.shape[1],
            dtype=gram.dtype,
            device=gram.device,
        ) * ridge
        weight = torch.linalg.solve(
            gram,
            centered_source.T @ centered_target,
        ).T
    else:
        gram = centered_source @ centered_source.T
        gram = gram + torch.eye(
            centered_source.shape[0],
            dtype=gram.dtype,
            device=gram.device,
        ) * ridge
        weight = (
            centered_source.T @ torch.linalg.solve(gram, centered_target)
        ).T
    bias = target_mean - source_mean @ weight.T
    return weight, bias


def fit_low_rank_projection(
    source: Tensor,
    target: Tensor,
    *,
    calibration_batches: int,
    rank: int,
    ridge: float = 1e-3,
    hidden_activation: str = "identity",
    output_bias: bool = True,
) -> LowRankProjection:
    """Fit a native low-rank affine projection on calibration rows.

    The source-to-target affine map supplies deterministic right-singular
    feature directions.  The native hidden activation is then applied and the
    up projection plus bias are re-solved, so the returned tensors correspond
    to the actual ``down -> activation -> up+bias`` parameterization.
    """
    if source.ndim != 3 or target.ndim < 3:
        raise ValueError("source and target must include batch and time")
    if source.shape[:2] != target.shape[:2]:
        raise ValueError("source and target batch/time dimensions must align")
    if not 0 < calibration_batches <= source.shape[0]:
        raise ValueError("calibration_batches must select a non-empty prefix")
    if hidden_activation not in {"identity", "tanh", "sigmoid"}:
        raise ValueError(f"unsupported hidden activation: {hidden_activation}")
    source_width = source.shape[-1]
    target_shape = target.shape[2:]
    target_width = math.prod(target_shape)
    maximum_rank = min(source_width, target_width)
    if not 0 < rank <= maximum_rank:
        raise ValueError("rank must fit source and target widths")
    calibration_source = source[:calibration_batches].reshape(
        -1, source_width
    ).float()
    calibration_target = target[:calibration_batches].reshape(
        -1, target_width
    ).float()
    affine_weight, _ = _ridge_affine_projection(
        calibration_source,
        calibration_target,
        ridge=ridge,
    )
    if not bool(torch.isfinite(affine_weight).all()):
        raise ValueError("low-rank affine solve produced non-finite directions")
    # The affine solve can be strongly rank deficient for nearly constant
    # decay/erase targets.  LAPACK SVD can terminate the process on that exact
    # regime, so diagonalize the finite FP64 right Gram matrix instead.
    right_gram = affine_weight.double().T @ affine_weight.double()
    _, eigenvectors = torch.linalg.eigh(right_gram)
    down_weight = (
        eigenvectors.flip(-1).T[:rank].float().contiguous()
    )
    if hidden_activation == "tanh":
        calibration_linear = calibration_source @ down_weight.T
        maximum = calibration_linear.abs().max().clamp_min(1e-6)
        down_weight = down_weight * (0.25 / maximum)

    def hidden(value: Tensor) -> Tensor:
        linear = value @ down_weight.T
        if hidden_activation == "tanh":
            return torch.tanh(linear)
        if hidden_activation == "sigmoid":
            return torch.sigmoid(linear)
        return linear

    calibration_features = hidden(calibration_source)
    if output_bias:
        up_weight, bias = _ridge_affine_projection(
            calibration_features,
            calibration_target,
            ridge=ridge,
        )
    else:
        fitted = fit_bias_free_projection(
            calibration_features.unsqueeze(0),
            calibration_target.unsqueeze(0),
            calibration_batches=1,
            ridge=ridge,
        )
        up_weight = fitted.weight
        bias = torch.zeros(
            target_width,
            dtype=torch.float32,
            device=source.device,
        )
    prediction = (
        hidden(source.float().reshape(-1, source_width)) @ up_weight.T
        + (bias if output_bias else 0)
    ).reshape(*source.shape[:2], *target_shape)
    return LowRankProjection(
        down_weight=down_weight,
        up_weight=up_weight,
        bias=bias,
        prediction=prediction,
        hidden_activation=hidden_activation,
        output_bias=output_bias,
    )


def native_signal_rollout(
    read: Tensor,
    decay: Tensor,
    key: Tensor,
    value: Tensor,
    erase: Tensor,
) -> NativeSignalRollout:
    """Replay fitted native RWKV7 signals from a zero state."""
    if read.ndim != 4:
        raise ValueError("read must be [batch,time,head,feature]")
    if key.shape != read.shape or value.shape != read.shape or erase.shape != read.shape:
        raise ValueError("key, value, and erase must align with read")
    if decay.shape != read.shape:
        raise ValueError("decay must be expanded to every state input channel")
    batch, sequence_length, heads, head_dim = read.shape
    state = torch.zeros(
        batch,
        heads,
        head_dim,
        head_dim,
        dtype=torch.float32,
        device=read.device,
    )
    states = torch.empty(
        batch,
        sequence_length,
        heads,
        head_dim,
        head_dim,
        dtype=torch.float32,
        device=read.device,
    )
    outputs = torch.empty_like(read, dtype=torch.float32)
    for index in range(sequence_length):
        normalized_key = torch.nn.functional.normalize(
            key[:, index].float(),
            dim=-1,
        )
        output, state = rwkv7_step(
            state,
            read[:, index].float(),
            decay[:, index].float(),
            key[:, index].float(),
            value[:, index].float(),
            -normalized_key,
            normalized_key * erase[:, index].float(),
        )
        states[:, index] = state
        outputs[:, index] = output
    return NativeSignalRollout(states=states, output=outputs)


def fit_bias_free_projection(
    source: Tensor,
    target: Tensor,
    *,
    calibration_batches: int,
    ridge: float = 1e-3,
    prior_weight: Tensor | None = None,
) -> BiasFreeProjection:
    """Fit a native bias-free linear projection on calibration batches.

    When ``prior_weight`` is provided, ridge is centered on that source-
    compatible map rather than zero.  This preserves directions not identified
    by a calibration matrix with fewer independent rows than hidden features.
    """
    if source.ndim != 3 or target.ndim < 3:
        raise ValueError("source and target must include batch and time")
    if source.shape[:2] != target.shape[:2]:
        raise ValueError("source and target batch/time dimensions must align")
    if not 0 < calibration_batches <= source.shape[0]:
        raise ValueError("calibration_batches must select a non-empty prefix")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    source_width = source.shape[-1]
    target_shape = target.shape[2:]
    target_width = math.prod(target_shape)
    calibration_source = source[:calibration_batches].reshape(
        -1, source_width
    ).float()
    calibration_target = target[:calibration_batches].reshape(
        -1, target_width
    ).float()
    if prior_weight is None:
        prior = torch.zeros(
            target_width,
            source_width,
            dtype=torch.float32,
            device=source.device,
        )
    else:
        if prior_weight.shape != (target_width, source_width):
            raise ValueError("prior weight does not match projection geometry")
        prior = prior_weight.float()
    calibration_residual = calibration_target - calibration_source @ prior.T
    if calibration_source.shape[0] >= source_width:
        gram = calibration_source.T @ calibration_source
        if ridge:
            gram = gram + torch.eye(
                source_width,
                dtype=gram.dtype,
                device=gram.device,
            ) * ridge
        rhs = calibration_source.T @ calibration_residual
        weight = prior + torch.linalg.solve(gram, rhs).T
    else:
        gram = calibration_source @ calibration_source.T
        if ridge:
            gram = gram + torch.eye(
                calibration_source.shape[0],
                dtype=gram.dtype,
                device=gram.device,
            ) * ridge
        coefficients = torch.linalg.solve(gram, calibration_residual)
        weight = prior + (calibration_source.T @ coefficients).T
    prediction = (source.float() @ weight.T).reshape(
        *source.shape[:2],
        *target_shape,
    )
    return BiasFreeProjection(weight=weight, prediction=prediction)


def select_bias_free_projection(
    source: Tensor,
    target: Tensor,
    *,
    calibration_batches: int,
    ridges: tuple[float, ...],
    prior_weight: Tensor | None = None,
) -> SelectedBiasFreeProjection:
    """Select a scale-relative ridge, then refit all calibration rows.

    ``ridges`` contains dimensionless multipliers.  Their absolute values are
    scaled by the mean diagonal of the fit-split feature Gram matrix, making
    the candidate set invariant to the number of calibration tokens and a
    common rescaling of the source activations.
    """
    if calibration_batches < 2:
        raise ValueError("ridge selection requires at least two calibration batches")
    if not ridges or any(ridge <= 0 for ridge in ridges):
        raise ValueError("ridge candidates must be non-empty and positive")
    fit_batches = max(1, calibration_batches // 2)
    fit_source = source[:fit_batches].reshape(-1, source.shape[-1]).float()
    ridge_scale = float(
        fit_source.square().sum() / max(1, fit_source.shape[-1])
    )
    ridge_scale = max(ridge_scale, torch.finfo(torch.float32).tiny)
    selection_nmse: dict[float, float] = {}
    for ridge_multiplier in ridges:
        candidate = fit_bias_free_projection(
            source[:calibration_batches],
            target[:calibration_batches],
            calibration_batches=fit_batches,
            ridge=ridge_multiplier * ridge_scale,
            prior_weight=prior_weight,
        )
        selection_nmse[ridge_multiplier] = normalized_mse(
            candidate.prediction[fit_batches:calibration_batches],
            target[fit_batches:calibration_batches],
        )
    selected_multiplier = min(
        ridges,
        key=lambda ridge: (selection_nmse[ridge], ridge),
    )
    selected_ridge = selected_multiplier * ridge_scale
    projection = fit_bias_free_projection(
        source,
        target,
        calibration_batches=calibration_batches,
        ridge=selected_ridge,
        prior_weight=prior_weight,
    )
    return SelectedBiasFreeProjection(
        projection=projection,
        ridge=selected_multiplier,
        absolute_ridge=selected_ridge,
        ridge_scale=ridge_scale,
        selection_nmse=selection_nmse,
    )


def two_state_projection(
    states: Tensor,
    bias: Tensor,
    query: Tensor,
    bases: Tensor,
    *,
    dc_indices: tuple[int, int] = (0, 0),
    query_center: Tensor | None = None,
) -> TwoStateProjection:
    """Materialize two independent DC-plus-query native states.

    Each half of the source value/output dimension owns one native state, one
    DC channel, and ``native_dim-1`` query features.  A rank-three ``bases``
    tensor is accepted as the legacy shared-basis special case; rank four
    provides the full independent two-state budget. When bases were fitted
    against centered queries, ``query_center`` supplies the per-head center and
    the DC channel stores ``b - S P Pᵀ c`` so raw-query materialization remains
    consistent with the fitted observable objective.
    """
    if states.ndim != 5 or bias.ndim != 4:
        raise ValueError("states and bias have invalid ranks")
    batch, sequence_length, groups, output_dim, input_dim = states.shape
    if query.ndim != 4 or query.shape[:2] != (batch, sequence_length):
        raise ValueError("query does not align with states")
    heads = query.shape[2]
    if bias.shape != (batch, sequence_length, groups, output_dim):
        raise ValueError("bias does not align with states")
    if heads % groups:
        raise ValueError("query heads must divide evenly into state groups")
    native_dim = output_dim // 2
    if output_dim % 2 or input_dim != output_dim:
        raise ValueError("two-state projection requires a square even operator")
    expected_rank = native_dim - 1
    if bases.shape == (heads, input_dim, expected_rank):
        bases = bases.unsqueeze(1).expand(-1, 2, -1, -1)
    if bases.shape != (heads, 2, input_dim, expected_rank):
        raise ValueError(
            "each native state requires one DC plus native_dim-1 query features"
        )
    if query_center is not None and query_center.shape != (heads, input_dim):
        raise ValueError("query center must align with every query head")
    if len(dc_indices) != 2 or any(
        not 0 <= index < native_dim for index in dc_indices
    ):
        raise ValueError("each DC index must fit the native head dimension")
    group_width = heads // groups
    projected_states = []
    reads = []
    outputs = []
    for head in range(heads):
        group = head // group_width
        head_query = query[:, :, head].float()
        head_states = []
        head_reads = []
        head_outputs = []
        for state_index in range(2):
            start = state_index * native_dim
            stop = start + native_dim
            basis = bases[head, state_index].float()
            feature_indices = [
                index
                for index in range(native_dim)
                if index != dc_indices[state_index]
            ]
            read = head_query.new_empty(
                (batch, sequence_length, native_dim)
            )
            read[..., dc_indices[state_index]] = 1
            read[..., feature_indices] = torch.einsum(
                "dr,btd->btr",
                basis,
                head_query,
            )
            compressed = states[
                :, :, group, start:stop
            ].float() @ basis
            dc_bias = bias[:, :, group, start:stop].float()
            if query_center is not None:
                projected_center = torch.einsum(
                    "dr,d->r",
                    basis,
                    query_center[head].float(),
                )
                dc_bias = dc_bias - torch.einsum(
                    "btor,r->bto",
                    compressed,
                    projected_center,
                )
            native_state = states.new_empty(
                (batch, sequence_length, native_dim, native_dim),
                dtype=torch.float32,
            )
            native_state[..., dc_indices[state_index]] = dc_bias
            native_state[..., feature_indices] = compressed
            head_states.append(native_state)
            head_reads.append(read)
            head_outputs.append(
                torch.einsum("btod,btd->bto", native_state, read)
            )
        projected_states.append(torch.stack(head_states, dim=2))
        reads.append(torch.stack(head_reads, dim=2))
        outputs.append(torch.cat(head_outputs, dim=-1))
    return TwoStateProjection(
        states=torch.stack(projected_states, dim=2),
        read=torch.stack(reads, dim=2),
        output=torch.stack(outputs, dim=2),
        query_basis=bases.float(),
        dc_indices=dc_indices,
    )


def native_two_state_rollout(
    projection: TwoStateProjection,
    probability: Tensor,
    slope: Tensor,
    *,
    minimum_decay: float = RWKV7_MINIMUM_DECAY,
) -> NativeTwoStateRollout:
    """Project affine two-state transitions onto native RWKV7 dynamic signals.

    For each token, the bounded hazard fixes a scalar decay and the compressed
    affine slope fixes the native key.  With ``k_k=1`` and ``k_a=0``, the
    channel-wise erase and write value are fitted by bounded coordinate least
    squares under the exact native recurrence.  Signals are fitted
    teacher-forced, then replayed from zero to expose free-running drift.  This
    is a dynamic-signal oracle: fitting source activations to native projection
    weights is a separate stage.
    """
    target = projection.states.float()
    if target.ndim != 6:
        raise ValueError("projection states must be [batch,time,head,2,d,d]")
    batch, sequence_length, heads, state_count, head_dim, input_dim = target.shape
    if state_count != 2 or head_dim != input_dim:
        raise ValueError("native projection requires two square states per head")
    if probability.ndim != 3 or slope.ndim != 4:
        raise ValueError("probability and slope must be grouped token signals")
    groups = probability.shape[2]
    if probability.shape[:2] != (batch, sequence_length):
        raise ValueError("probability does not align with projection")
    if slope.shape[:3] != (batch, sequence_length, groups):
        raise ValueError("slope does not align with probability")
    if heads % groups or slope.shape[-1] != projection.query_basis.shape[2]:
        raise ValueError("grouped hazard signals do not align with Query heads")
    expected_basis_shape = (heads, 2, slope.shape[-1], head_dim - 1)
    if projection.query_basis.shape != expected_basis_shape:
        raise ValueError(
            "query basis must provide native head_dim-1 compressed features"
        )
    if not 0 < minimum_decay <= 1:
        raise ValueError("minimum_decay must be in (0,1]")

    group_width = heads // groups
    head_to_group = torch.arange(heads, device=target.device) // group_width
    grouped_probability = probability[:, :, head_to_group].float()
    grouped_slope = slope[:, :, head_to_group].float()
    compressed_slope = torch.einsum(
        "hsfr,bthf->bthsr",
        projection.query_basis.float(),
        grouped_slope,
    )
    shared_key = target.new_empty(
        (batch, sequence_length, heads, 2, head_dim)
    )
    for state_index, dc_index in enumerate(projection.dc_indices):
        feature_indices = [
            index for index in range(head_dim) if index != dc_index
        ]
        shared_key[:, :, :, state_index, dc_index] = grouped_probability
        shared_key[:, :, :, state_index, feature_indices] = (
            compressed_slope[:, :, :, state_index]
        )
    if shared_key.shape[-1] != head_dim:
        raise ValueError("DC plus compressed slope must fill native head_dim")
    key = shared_key.flatten(2, 3)
    requested_decay = 1 - grouped_probability
    desired_decay = requested_decay.clamp(
        min=minimum_decay,
        max=1,
    )
    requested_decay = requested_decay.unsqueeze(-1).expand(
        -1, -1, -1, 2
    ).flatten(2, 3)
    decay = desired_decay.unsqueeze(-1).expand(
        -1, -1, -1, 2
    ).flatten(2, 3)

    native_target = target.flatten(2, 3)
    native_heads = native_target.shape[2]
    teacher_states = torch.empty_like(native_target)
    free_states = torch.empty_like(native_target)
    erase_rows = torch.empty(
        batch,
        sequence_length,
        native_heads,
        head_dim,
        dtype=torch.float32,
        device=target.device,
    )
    values = torch.empty(
        batch,
        sequence_length,
        native_heads,
        head_dim,
        dtype=torch.float32,
        device=target.device,
    )
    native_read = projection.read.flatten(2, 3).float()
    free_outputs = torch.empty(
        batch,
        sequence_length,
        native_heads,
        head_dim,
        dtype=torch.float32,
        device=target.device,
    )
    target_previous = torch.zeros_like(native_target[:, 0])
    free_previous = torch.zeros_like(native_target[:, 0])
    for index in range(sequence_length):
        current_target = native_target[:, index]
        current_key = key[:, index]
        normalized_key = torch.nn.functional.normalize(current_key, dim=-1)
        current_decay = decay[:, index]
        decayed = target_previous * current_decay[:, :, None, None]
        residual = current_target - decayed

        key_norm = current_key.square().sum(dim=-1, keepdim=True).clamp_min(
            1e-30
        )
        state_key = torch.einsum(
            "bhod,bhd->bho",
            target_previous,
            normalized_key,
        )
        erase = torch.zeros_like(current_key)
        value = torch.zeros(
            batch,
            native_heads,
            head_dim,
            dtype=torch.float32,
            device=target.device,
        )
        for _ in range(3):
            erase_update = -torch.einsum(
                "bho,bhd->bhod",
                state_key,
                normalized_key * erase,
            )
            after_erase = residual - erase_update
            value = torch.einsum(
                "bhod,bhd->bho",
                after_erase,
                current_key,
            ) / key_norm
            after_write = residual - torch.einsum(
                "bho,bhd->bhod",
                value,
                current_key,
            )
            erase_coefficient = -torch.einsum(
                "bho,bhd->bhod",
                state_key,
                normalized_key,
            )
            erase_denominator = erase_coefficient.square().sum(
                dim=-2
            ).clamp_min(1e-30)
            erase = (
                (erase_coefficient * after_write).sum(dim=-2)
                / erase_denominator
            ).clamp(0, 1)
        erase_update = -torch.einsum(
            "bho,bhd->bhod",
            state_key,
            normalized_key * erase,
        )
        value = torch.einsum(
            "bhod,bhd->bho",
            residual - erase_update,
            current_key,
        ) / key_norm
        erase_left = -normalized_key
        erase_right = normalized_key * erase
        _, teacher_current = rwkv7_step(
            target_previous,
            native_read[:, index],
            current_decay.unsqueeze(-1).expand_as(current_key),
            current_key,
            value,
            erase_left,
            erase_right,
        )

        free_output, free_current = rwkv7_step(
            free_previous,
            native_read[:, index],
            current_decay.unsqueeze(-1).expand_as(current_key),
            current_key,
            value,
            erase_left,
            erase_right,
        )
        teacher_states[:, index] = teacher_current
        free_states[:, index] = free_current
        free_outputs[:, index] = free_output
        erase_rows[:, index] = erase
        values[:, index] = value
        target_previous = current_target
        free_previous = free_current

    native_output = free_outputs.reshape(
        batch,
        sequence_length,
        heads,
        head_dim * 2,
    )
    return NativeTwoStateRollout(
        teacher_forced_states=teacher_states.unflatten(2, (heads, 2)),
        free_running_states=free_states.unflatten(2, (heads, 2)),
        free_running_output=native_output,
        requested_decay=requested_decay,
        decay=decay,
        erase=erase_rows,
        key=key,
        value=values,
    )


def two_state_outputs(
    states: Tensor,
    centered_query: Tensor,
    bases: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply a two-128-state input sketch to every 256-dimensional head."""
    if states.ndim != 5 or centered_query.ndim != 4 or bases.ndim != 3:
        raise ValueError("states, query, and bases have invalid ranks")
    batch, sequence_length, groups, output_dim, input_dim = states.shape
    heads = centered_query.shape[2]
    if heads % groups:
        raise ValueError("query heads must divide evenly into state groups")
    if output_dim != input_dim or bases.shape[:2] != (heads, input_dim):
        raise ValueError("state/query/basis dimensions do not align")
    group_width = heads // groups
    full_outputs = []
    bias = states.new_zeros(
        (batch, sequence_length, groups, output_dim)
    )
    for head in range(heads):
        group = head // group_width
        state = states[:, :, group]
        query = centered_query[:, :, head]
        full_outputs.append(
            torch.einsum("btod,btd->bto", state, query)
        )
    projection = two_state_projection(states, bias, centered_query, bases)
    return torch.stack(full_outputs, dim=2), projection.output


def four_state_block_output(state: Tensor, query: Tensor) -> Tensor:
    """Exact 2x2 block evaluation of a 256x256 operator."""
    if state.ndim < 2 or state.shape[-1] != state.shape[-2]:
        raise ValueError("state must end in a square matrix")
    dimension = state.shape[-1]
    if dimension % 2:
        raise ValueError("state dimension must be even")
    if query.shape != state.shape[:-2] + (dimension,):
        raise ValueError("query does not align with state")
    half = dimension // 2
    query_left, query_right = query.split(half, dim=-1)
    top = (
        torch.einsum("...od,...d->...o", state[..., :half, :half], query_left)
        + torch.einsum(
            "...od,...d->...o", state[..., :half, half:], query_right
        )
    )
    bottom = (
        torch.einsum(
            "...od,...d->...o", state[..., half:, :half], query_left
        )
        + torch.einsum(
            "...od,...d->...o", state[..., half:, half:], query_right
        )
    )
    return torch.cat((top, bottom), dim=-1)
