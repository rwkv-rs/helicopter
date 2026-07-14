from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CanonicalStateContract:
    equation: str = "S_t = S_{t-1} A_t + B_t"
    state_orientation: str = "batch,head,value,key"
    update: str = "S_t = S_{t-1} diag(decay_t) + (S_{t-1} a_t) b_t^T + v_t k_t^T"
    readout: str = "S_t r_t"
    reset: str = "zero state at document boundary"
    rope_boundary: str = "source projection boundary before native RWKV7 mixer"
    native_kernel: str = "rwkv-lm/rwkv7"


def rwkv7_step(
    state: Tensor,
    r: Tensor,
    decay: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
) -> tuple[Tensor, Tensor]:
    """FP64-friendly native recurrence with state [B,H,V,K]."""
    if state.ndim != 4 or any(value.ndim != 3 for value in (r, decay, k, v, a, b)):
        raise ValueError("state must be [B,H,V,K] and token vectors must be [B,H,N]")
    state_a = torch.einsum("bhvk,bhk->bhv", state, a)
    rank_one = torch.einsum("bhv,bhk->bhvk", state_a, b)
    write = torch.einsum("bhv,bhk->bhvk", v, k)
    next_state = state * decay.unsqueeze(-2) + rank_one + write
    output = torch.einsum("bhvk,bhk->bhv", next_state, r)
    return output, next_state


def rwkv7_scan(
    state: Tensor,
    r: Tensor,
    decay: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
) -> tuple[Tensor, Tensor]:
    """Sequential reference over vectors shaped [B,T,H,N]."""
    outputs: list[Tensor] = []
    current = state
    for index in range(r.shape[1]):
        output, current = rwkv7_step(
            current,
            r[:, index],
            decay[:, index],
            k[:, index],
            v[:, index],
            a[:, index],
            b[:, index],
        )
        outputs.append(output)
    empty = r.new_empty((r.shape[0], 0, r.shape[2], r.shape[3]))
    return (torch.stack(outputs, dim=1) if outputs else empty), current


def chunked_rwkv7_scan(
    state: Tensor,
    r: Tensor,
    decay: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    *,
    chunk_size: int,
) -> tuple[Tensor, Tensor]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    outputs: list[Tensor] = []
    current = state
    for start in range(0, r.shape[1], chunk_size):
        stop = start + chunk_size
        output, current = rwkv7_scan(
            current,
            r[:, start:stop],
            decay[:, start:stop],
            k[:, start:stop],
            v[:, start:stop],
            a[:, start:stop],
            b[:, start:stop],
        )
        outputs.append(output)
    return (torch.cat(outputs, dim=1) if outputs else r.new_empty(r.shape)), current


def reset_state(batch: int, heads: int, head_size: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.zeros(batch, heads, head_size, head_size, dtype=dtype)


def native_decay_from_logit(logit: Tensor) -> Tensor:
    """Decay parameterization used by rwkv-lm's clamp-w state-passing kernel."""
    return torch.exp(-torch.exp(logit.new_tensor(-0.5)) * torch.sigmoid(logit))


def native_decay_logit(decay: Tensor) -> Tensor:
    """Inverse of the native decay parameterization on its reachable interval."""
    scale = torch.exp(decay.new_tensor(-0.5))
    probability = -torch.log(decay) / scale
    if torch.any(probability <= 0) or torch.any(probability >= 1):
        low = float(torch.exp(-scale))
        raise ValueError(f"native RWKV7 decay must be strictly in ({low},1)")
    return torch.logit(probability)
