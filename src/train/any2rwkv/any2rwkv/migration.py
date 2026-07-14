from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .errors import ContractError
from .recurrent import native_decay_logit, rwkv7_scan


@dataclass(frozen=True)
class GroupMap:
    query_to_kv: tuple[int, ...]
    num_query_heads: int
    num_kv_heads: int


def build_group_map(num_query_heads: int, num_kv_heads: int) -> GroupMap:
    if num_query_heads <= 0 or num_kv_heads <= 0 or num_query_heads % num_kv_heads:
        raise ContractError(
            f"ambiguous GQA layout: query_heads={num_query_heads} kv_heads={num_kv_heads}"
        )
    width = num_query_heads // num_kv_heads
    return GroupMap(tuple(index // width for index in range(num_query_heads)), num_query_heads, num_kv_heads)


def kv_repeat(weight: Tensor, *, num_query_heads: int, num_kv_heads: int) -> Tensor:
    mapping = build_group_map(num_query_heads, num_kv_heads)
    if weight.shape[0] % num_kv_heads:
        raise ContractError("KV projection first dimension is not divisible by num_kv_heads")
    return weight.reshape(num_kv_heads, -1, *weight.shape[1:])[list(mapping.query_to_kv)].flatten(0, 1)


def kv_expand(weight: Tensor, *, num_query_heads: int, num_kv_heads: int) -> Tensor:
    """Explicit ablation: repeat plus deterministic per-head scale separation."""
    repeated = kv_repeat(weight, num_query_heads=num_query_heads, num_kv_heads=num_kv_heads)
    head_dim = repeated.shape[0] // num_query_heads
    scale = torch.linspace(0.99, 1.01, num_query_heads, dtype=repeated.dtype, device=repeated.device)
    return (repeated.reshape(num_query_heads, head_dim, *repeated.shape[1:]) * scale.view(-1, 1, *([1] * (repeated.ndim - 1)))).flatten(0, 1)


def gdn_to_rwkv7_dynamics(
    decay: Tensor,
    beta: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Map post-activation normalized GDN delta-rule signals into RWKV7 signals.

    Source contract: S' = decay*S + beta*(v-(decay*S)@k)k^T; y=S'@q/sqrt(Dk).
    The mapping is conditional: decay must be strictly in (0, 1]. Peripheral
    projections, conv1d, normalization, and activation remain fitted.
    """
    if torch.any(decay <= 0) or torch.any(decay > 1):
        raise ContractError("GDN algebraic mapping requires post-activation decay in (0,1]")
    if decay.shape != key.shape[:-1] + (1,) or beta.shape != key.shape[:-1] + (1,):
        raise ContractError(
            "GDN decay/beta must be head-scalar dynamic signals compatible with native RWKV7"
        )
    native_decay_logit(decay)
    r = query / query.shape[-1] ** 0.5
    mapped_decay = decay.expand_as(key)
    # Qwen3.5 emits one scalar decay per value head and token.  With normalized
    # k, native RWKV7's constrained erase signals ``a=-k`` and
    # ``b=(decay*beta)k`` exactly recover the transposed GDN state update.
    # A per-key decay would require a more general a/b geometry and is rejected
    # above instead of being mislabeled as native-RWKV7 algebraic transfer.
    return r, mapped_decay, key, value * beta, -key, key * beta * decay


def gdn_reference_scan(
    state: Tensor,
    decay: Tensor,
    beta: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> tuple[Tensor, Tensor]:
    outputs: list[Tensor] = []
    current = state
    for index in range(query.shape[1]):
        d = decay[:, index].unsqueeze(-2)
        k = key[:, index]
        b = beta[:, index]
        v = value[:, index]
        decayed = current * d
        state_key = torch.einsum("bhvk,bhk->bhv", decayed, k)
        update = torch.einsum("bhv,bhk->bhvk", b * (v - state_key), k)
        current = decayed + update
        outputs.append(torch.einsum("bhvk,bhk->bhv", current, query[:, index]) / query.shape[-1] ** 0.5)
    return torch.stack(outputs, dim=1), current


def verify_gdn_mapping(
    state: Tensor,
    decay: Tensor,
    beta: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> dict[str, float]:
    source_output, source_state = gdn_reference_scan(state, decay, beta, query, key, value)
    r, target_decay, k, v, a, b = gdn_to_rwkv7_dynamics(decay, beta, query, key, value)
    target_output, target_state = rwkv7_scan(state, r, target_decay, k, v, a, b)
    return {
        "output_max_abs": float((source_output - target_output).abs().max()),
        "state_max_abs": float((source_state - target_state).abs().max()),
        "output_relative_l2": float(torch.linalg.vector_norm(source_output - target_output) / torch.clamp(torch.linalg.vector_norm(source_output), min=1e-30)),
        "state_relative_l2": float(torch.linalg.vector_norm(source_state - target_state) / torch.clamp(torch.linalg.vector_norm(source_state), min=1e-30)),
    }


@dataclass(frozen=True)
class FitResult:
    weight: Tensor
    bias: Tensor
    normalized_mse: float
    cosine: float


@dataclass(frozen=True)
class PartitionFit:
    index: int
    group: int | None
    normalized_mse: float
    cosine: float
    tokens: int


def fit_teacher_trace(inputs: Tensor, outputs: Tensor, *, ridge: float = 1e-5) -> FitResult:
    """Deterministic ridge solver used for peripheral and attention initialization."""
    if inputs.ndim != 2 or outputs.ndim != 2 or inputs.shape[0] != outputs.shape[0]:
        raise ContractError("teacher trace solver expects aligned [tokens,features] matrices")
    ones = torch.ones(inputs.shape[0], 1, dtype=inputs.dtype, device=inputs.device)
    design = torch.cat((inputs, ones), dim=1)
    if design.shape[1] <= design.shape[0]:
        gram = design.T @ design
        eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        solved = torch.linalg.solve(gram + ridge * eye, design.T @ outputs)
    else:
        # Teacher traces normally have fewer tokens than hidden features. The
        # dual ridge form avoids an O(hidden^3) solve while producing the same
        # deterministic minimum-norm solution.
        gram = design @ design.T
        eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        solved = design.T @ torch.linalg.solve(gram + ridge * eye, outputs)
    prediction = design @ solved
    residual = prediction - outputs
    mse = torch.mean(residual.square()) / torch.clamp(torch.mean(outputs.square()), min=1e-30)
    cosine = torch.nn.functional.cosine_similarity(prediction.flatten(), outputs.flatten(), dim=0)
    return FitResult(solved[:-1].T.contiguous(), solved[-1].contiguous(), float(mse), float(cosine))


def fit_headwise_teacher_trace(
    inputs: Tensor,
    outputs: Tensor,
    *,
    num_kv_heads: int | None = None,
    ridge: float = 1e-5,
) -> tuple[tuple[FitResult, ...], tuple[PartitionFit, ...]]:
    """Fit and report every query head instead of hiding failures in a layer mean."""
    if inputs.ndim != 3 or outputs.ndim != 3 or inputs.shape[:2] != outputs.shape[:2]:
        raise ContractError("headwise trace expects aligned [tokens,heads,features] tensors")
    heads = int(inputs.shape[1])
    group_map = None if num_kv_heads is None else build_group_map(heads, num_kv_heads)
    fits: list[FitResult] = []
    rows: list[PartitionFit] = []
    for head in range(heads):
        fit = fit_teacher_trace(inputs[:, head], outputs[:, head], ridge=ridge)
        fits.append(fit)
        rows.append(
            PartitionFit(
                index=head,
                group=None if group_map is None else group_map.query_to_kv[head],
                normalized_mse=fit.normalized_mse,
                cosine=fit.cosine,
                tokens=int(inputs.shape[0]),
            )
        )
    return tuple(fits), tuple(rows)


def validate_attention_trace_contract(
    *,
    context_lengths: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
) -> dict[str, object]:
    if context_lengths.ndim != 1 or context_lengths.numel() < 3:
        raise ContractError("attention fitting requires at least three context samples")
    unique = sorted({int(value) for value in context_lengths.tolist()})
    if len(unique) < 3:
        raise ContractError("attention fitting requires at least three distinct context lengths")
    if position_ids.shape != attention_mask.shape or position_ids.ndim != 2:
        raise ContractError("position_ids and attention_mask must be aligned [batch,tokens]")
    if torch.any(context_lengths <= 0) or torch.any(context_lengths > attention_mask.shape[-1]):
        raise ContractError("context length falls outside the trace tensor boundary")
    return {
        "contexts": unique,
        "samples": int(context_lengths.numel()),
        "masked_tokens": int((~attention_mask.to(torch.bool)).sum()),
        "position_min": int(position_ids.min()),
        "position_max": int(position_ids.max()),
    }
