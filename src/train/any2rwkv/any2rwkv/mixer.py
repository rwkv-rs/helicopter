from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from rwkv7_hf.native_model import NativeRWKV7Attention

from .kernel import NativeRwkv7Kernel


EXP_HALF = 0.606531


def apply_partial_rope(x: Tensor, positions: Tensor, *, rotary_dim: int, theta: float) -> Tensor:
    """Apply source-compatible text RoPE to the leading per-head channels."""
    if rotary_dim == 0:
        return x
    if rotary_dim < 0 or rotary_dim > x.shape[-1] or rotary_dim % 2:
        raise ValueError(f"invalid rotary_dim={rotary_dim} for head_dim={x.shape[-1]}")
    frequencies = 1.0 / (
        theta
        ** (torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32) / rotary_dim)
    )
    angles = positions.to(torch.float32).unsqueeze(-1) * frequencies
    embedding = torch.cat((angles, angles), dim=-1)
    cos = embedding.cos().to(x.dtype).unsqueeze(-2)
    sin = embedding.sin().to(x.dtype).unsqueeze(-2)
    rotary = x[..., :rotary_dim]
    left, right = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-right, left), dim=-1)
    return torch.cat((rotary * cos + rotated * sin, x[..., rotary_dim:]), dim=-1)


class ProjectionBoundaryRWKV7Attention(NativeRWKV7Attention):
    """Native RWKV7 recurrence with optional source RoPE before state access."""

    def __init__(self, config, layer_idx: int, *, source_used_rope: bool, rotary_dim: int, rope_theta: float):
        super().__init__(config, layer_idx)
        self.source_used_rope = bool(source_used_rope)
        self.rotary_dim = int(rotary_dim)
        self.rope_theta = float(rope_theta)

    def forward(
        self,
        x: Tensor,
        x_prev: Tensor,
        v_first: Tensor,
        state: Tensor,
        *,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        batch = int(x.shape[0])
        heads, head_dim = self.num_heads, self.head_dim
        hidden = heads * head_dim
        delta = x_prev - x
        mixed = {
            name: x + delta * getattr(self, f"x_{name}").reshape(1, hidden)
            for name in ("r", "w", "k", "v", "a", "g")
        }
        r = self.r_proj(mixed["r"])
        w = self.w_lora.lora[2](torch.tanh(self.w_lora.lora[0](mixed["w"])))
        k = self.k_proj(mixed["k"])
        v = self.v_proj(mixed["v"])
        a = torch.sigmoid(self.a_lora.lora[2](self.a_lora.lora[0](mixed["a"])))
        g = self.g_lora.lora[2](torch.sigmoid(self.g_lora.lora[0](mixed["g"])))
        if self.source_used_rope:
            r = apply_partial_rope(
                r.view(batch, heads, head_dim), positions, rotary_dim=self.rotary_dim, theta=self.rope_theta
            ).reshape(batch, hidden)
            k = apply_partial_rope(
                k.view(batch, heads, head_dim), positions, rotary_dim=self.rotary_dim, theta=self.rope_theta
            ).reshape(batch, hidden)

        normalized_key = F.normalize(
            (k * self.k_k.reshape(1, hidden)).view(batch, heads, head_dim), dim=-1, p=2
        ).view(batch, hidden)
        k = k * (1 + (a - 1) * self.k_a.reshape(1, hidden))
        if self.layer_idx == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(
                self.v_lora.lora[2](self.v_lora.lora[0](mixed["v"]))
            )
        decay = torch.exp(-EXP_HALF * torch.sigmoid(w.float()))
        write = v.view(batch, heads, head_dim, 1) @ k.view(batch, heads, 1, head_dim)
        erase = (-normalized_key).view(batch, heads, head_dim, 1) @ (
            normalized_key * a
        ).view(batch, heads, 1, head_dim)
        state = state * decay.view(batch, heads, 1, head_dim) + state @ erase.float() + write.float()
        output = (state.to(x.dtype) @ r.view(batch, heads, head_dim, 1)).view(batch, hidden)
        output = F.group_norm(
            output,
            num_groups=heads,
            weight=self.g_norm.weight,
            bias=self.g_norm.bias,
            eps=head_dim * 1e-5,
        )
        bonus = (
            r.view(batch, heads, head_dim)
            * k.view(batch, heads, head_dim)
            * self.r_k.reshape(1, heads, head_dim)
        ).sum(dim=-1, keepdim=True)
        output = output + (bonus * v.view(batch, heads, head_dim)).view(batch, hidden)
        output = self.o_proj(output * g)
        signals = {"r": r, "decay": decay, "k": k, "v": v, "a": normalized_key, "erase": a}
        return output, x, state, v_first, signals

    def project_v_first_sequence(self, x: Tensor) -> Tensor:
        """Project the frozen layer-0 value stream without running recurrence."""
        if x.ndim != 3:
            raise ValueError("v_first shadow projection expects x=[B,T,C]")
        previous = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
        mixed_v = x + (previous - x) * self.x_v.reshape(1, 1, x.shape[-1])
        return self.v_proj(mixed_v)

    def forward_sequence(
        self,
        x: Tensor,
        *,
        positions: Tensor,
        kernel: NativeRwkv7Kernel,
        v_first: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Run one full training sequence through rwkv-lm's state-passing kernel."""
        if x.ndim != 3 or positions.shape != x.shape[:2]:
            raise ValueError("sequence mixer expects x=[B,T,C] and aligned positions=[B,T]")
        batch, tokens, hidden = x.shape
        previous = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
        delta = previous - x
        mixed = {
            name: x + delta * getattr(self, f"x_{name}").reshape(1, 1, hidden)
            for name in ("r", "w", "k", "v", "a", "g")
        }
        r = self.r_proj(mixed["r"])
        w = self.w_lora.lora[2](torch.tanh(self.w_lora.lora[0](mixed["w"])))
        k = self.k_proj(mixed["k"])
        v = self.v_proj(mixed["v"])
        erase = torch.sigmoid(
            self.a_lora.lora[2](self.a_lora.lora[0](mixed["a"]))
        )
        gate = self.g_lora.lora[2](
            torch.sigmoid(self.g_lora.lora[0](mixed["g"]))
        )
        if self.source_used_rope:
            r = apply_partial_rope(
                r.view(batch, tokens, self.num_heads, self.head_dim),
                positions,
                rotary_dim=self.rotary_dim,
                theta=self.rope_theta,
            ).reshape(batch, tokens, hidden)
            k = apply_partial_rope(
                k.view(batch, tokens, self.num_heads, self.head_dim),
                positions,
                rotary_dim=self.rotary_dim,
                theta=self.rope_theta,
            ).reshape(batch, tokens, hidden)
        normalized_key = F.normalize(
            (k * self.k_k.reshape(1, 1, hidden)).view(
                batch, tokens, self.num_heads, self.head_dim
            ),
            dim=-1,
            p=2,
        ).view(batch, tokens, hidden)
        k = k * (1 + (erase - 1) * self.k_a.reshape(1, 1, hidden))
        if self.layer_idx == 0:
            v_first = v
        else:
            if v_first is None or v_first.shape != v.shape:
                raise ValueError("nonzero RWKV7 layers require aligned layer-0 v_first")
            value_mix = torch.sigmoid(
                self.v_lora.lora[2](self.v_lora.lora[0](mixed["v"]))
            )
            v = v + (v_first - v) * value_mix
        padding = (-tokens) % kernel.chunk_size
        vectors = (r, w, k, v, -normalized_key, normalized_key * erase)
        if padding:
            vectors = tuple(
                F.pad(value, (0, 0, 0, padding)) for value in vectors
            )
        state = torch.zeros(
            batch,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=x.device,
            dtype=torch.float32,
        )
        recurrent, final_state = kernel(state, *(value.to(torch.bfloat16) for value in vectors))
        recurrent = recurrent[:, :tokens].to(x.dtype)
        recurrent = F.group_norm(
            recurrent.reshape(batch * tokens, hidden),
            num_groups=self.num_heads,
            weight=self.g_norm.weight,
            bias=self.g_norm.bias,
            eps=self.head_dim * 1e-5,
        ).view(batch, tokens, hidden)
        bonus = (
            r.view(batch, tokens, self.num_heads, self.head_dim)
            * k.view(batch, tokens, self.num_heads, self.head_dim)
            * self.r_k.reshape(1, 1, self.num_heads, self.head_dim)
        ).sum(dim=-1, keepdim=True)
        recurrent = recurrent + (
            bonus * v.view(batch, tokens, self.num_heads, self.head_dim)
        ).reshape(batch, tokens, hidden)
        output = self.o_proj(recurrent * gate)
        signals = {
            "r": r,
            "w": w,
            "k": k,
            "v": v,
            "a": normalized_key,
            "erase": erase,
        }
        return output, v_first, final_state, signals
