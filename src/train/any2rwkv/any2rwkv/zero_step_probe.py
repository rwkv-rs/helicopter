from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .recurrent import rwkv7_scan


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
) -> AffineRollout:
    """Roll out the centered affine state and an optional calibrated closure."""
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
    closure_scale = query.new_zeros(
        (groups, sequence_length), dtype=torch.float32
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
        if fit_rank_one_closure:
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
    sketch_outputs = []
    for head in range(heads):
        group = head // group_width
        state = states[:, :, group]
        query = centered_query[:, :, head]
        basis = bases[head]
        compressed_query = torch.einsum(
            "dr,btd->btr", basis, query
        )
        projected_query = torch.einsum(
            "dr,btr->btd", basis, compressed_query
        )
        full_outputs.append(
            torch.einsum("btod,btd->bto", state, query)
        )
        sketch_outputs.append(
            torch.einsum("btod,btd->bto", state, projected_query)
        )
    return torch.stack(full_outputs, dim=2), torch.stack(sketch_outputs, dim=2)


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
