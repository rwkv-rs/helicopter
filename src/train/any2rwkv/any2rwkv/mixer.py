from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .kernel import Rwkv7OperatorAdapter

EXP_HALF = 0.606531


def apply_partial_rope(
    x: Tensor, positions: Tensor, *, rotary_dim: int, theta: float
) -> Tensor:
    """Apply source-compatible text RoPE to the leading per-head channels."""
    if rotary_dim == 0:
        return x
    if rotary_dim < 0 or rotary_dim > x.shape[-1] or rotary_dim % 2:
        raise ValueError(f"invalid rotary_dim={rotary_dim} for head_dim={x.shape[-1]}")
    frequencies = 1.0 / (
        theta
        ** (
            torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32)
            / rotary_dim
        )
    )
    angles = positions.to(torch.float32).unsqueeze(-1) * frequencies
    embedding = torch.cat((angles, angles), dim=-1)
    cos = embedding.cos().to(x.dtype).unsqueeze(-2)
    sin = embedding.sin().to(x.dtype).unsqueeze(-2)
    rotary = x[..., :rotary_dim]
    left, right = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-right, left), dim=-1)
    return torch.cat((rotary * cos + rotated * sin, x[..., rotary_dim:]), dim=-1)


class _LowRankProjection(nn.Module):
    """Parameter layout owned by the Any-to-RWKV mapping ledger."""

    def __init__(
        self,
        input_size: int,
        low_rank: int,
        output_size: int,
        *,
        bias: bool,
    ) -> None:
        super().__init__()
        self.lora = nn.Sequential(
            nn.Linear(input_size, low_rank, bias=False),
            nn.Identity(),
            nn.Linear(low_rank, output_size, bias=bias),
        )


class ProjectionBoundaryRWKV7Attention(nn.Module):
    """Any-to-RWKV projections around the public RWKV7 state contract."""

    def __init__(
        self,
        config,
        layer_idx: int,
        *,
        source_used_rope: bool,
        rotary_dim: int,
        rope_theta: float,
        rope_num_heads: int | None = None,
        rope_head_dim: int | None = None,
    ):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.num_heads = int(config.num_heads)
        self.head_dim = int(config.head_dim)
        self.hidden_size = int(config.hidden_size)
        self.attention_hidden_size = int(config.attention_hidden_size)
        hidden = self.hidden_size
        recurrent_width = self.attention_hidden_size
        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            setattr(self, name, nn.Parameter(torch.zeros(1, 1, hidden)))
        self.k_k = nn.Parameter(torch.zeros(recurrent_width))
        self.k_a = nn.Parameter(torch.zeros(recurrent_width))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.r_proj = nn.Linear(hidden, recurrent_width, bias=False)
        self.k_proj = nn.Linear(hidden, recurrent_width, bias=False)
        self.v_proj = nn.Linear(hidden, recurrent_width, bias=False)
        self.o_proj = nn.Linear(recurrent_width, hidden, bias=False)
        self.w_lora = _LowRankProjection(
            hidden,
            int(config.decay_low_rank_dim),
            recurrent_width,
            bias=True,
        )
        self.a_lora = _LowRankProjection(
            hidden,
            int(config.a_low_rank_dim),
            recurrent_width,
            bias=True,
        )
        self.g_lora = _LowRankProjection(
            hidden,
            int(config.gate_low_rank_dim),
            recurrent_width,
            bias=False,
        )
        if self.layer_idx:
            self.v_lora = _LowRankProjection(
                hidden,
                int(config.v_low_rank_dim),
                recurrent_width,
                bias=True,
            )
        self.g_norm = nn.GroupNorm(
            self.num_heads,
            recurrent_width,
            eps=self.head_dim * 1e-5,
        )
        self.source_used_rope = bool(source_used_rope)
        self.rotary_dim = int(rotary_dim)
        self.rope_theta = float(rope_theta)
        self.rope_num_heads = int(
            self.num_heads if rope_num_heads is None else rope_num_heads
        )
        self.rope_head_dim = int(
            self.head_dim if rope_head_dim is None else rope_head_dim
        )
        if self.rope_num_heads * self.rope_head_dim != self.attention_hidden_size:
            raise ValueError(
                "source RoPE head geometry must exactly cover the recurrent width"
            )
        if self.rotary_dim < 0 or self.rotary_dim > self.rope_head_dim:
            raise ValueError("source rotary_dim must fit the source RoPE head geometry")

    def _apply_source_rope(self, value: Tensor, positions: Tensor) -> Tensor:
        shape = value.shape
        return apply_partial_rope(
            value.view(*shape[:-1], self.rope_num_heads, self.rope_head_dim),
            positions,
            rotary_dim=self.rotary_dim,
            theta=self.rope_theta,
        ).reshape(shape)

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
        hidden = self.hidden_size
        recurrent_width = heads * head_dim
        delta = x_prev - x
        mixed = {
            name: x + delta * getattr(self, f"x_{name}").reshape(1, hidden)
            for name in ("r", "w", "k", "v", "a", "g")
        }
        r = self.r_proj(mixed["r"])
        w_features = torch.tanh(self.w_lora.lora[0](mixed["w"]))
        w = self.w_lora.lora[2](w_features)
        k = self.k_proj(mixed["k"])
        v = self.v_proj(mixed["v"])
        projected_r, projected_k, projected_v = r, k, v
        a_features = self.a_lora.lora[0](mixed["a"])
        a = torch.sigmoid(self.a_lora.lora[2](a_features))
        g_features = torch.sigmoid(self.g_lora.lora[0](mixed["g"]))
        g = self.g_lora.lora[2](g_features)
        if self.source_used_rope:
            r = self._apply_source_rope(r, positions)
            k = self._apply_source_rope(k, positions)

        write_key_base = k
        normalized_key = F.normalize(
            (k * self.k_k.reshape(1, recurrent_width)).view(batch, heads, head_dim),
            dim=-1,
            p=2,
        ).view(batch, recurrent_width)
        k = k * (1 + (a - 1) * self.k_a.reshape(1, recurrent_width))
        if self.layer_idx == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(
                self.v_lora.lora[2](self.v_lora.lora[0](mixed["v"]))
            )
        decay = torch.exp(-EXP_HALF * torch.sigmoid(w.float()))
        r_heads = r.view(batch, heads, head_dim)
        k_heads = k.view(batch, heads, head_dim)
        v_heads = v.view(batch, heads, head_dim)
        erase_read = -normalized_key.view(batch, heads, head_dim)
        erase_write = (normalized_key * a).view(batch, heads, head_dim)
        state_projection = torch.einsum(
            "bhk,bhkv->bhv", erase_read.float(), state.float()
        )
        state = (
            decay.view(batch, heads, head_dim, 1) * state.float()
            + erase_write.unsqueeze(-1) * state_projection.unsqueeze(-2)
            + k_heads.float().unsqueeze(-1) * v_heads.float().unsqueeze(-2)
        )
        output = torch.einsum("bhk,bhkv->bhv", r_heads.float(), state)
        output = output.to(x.dtype).reshape(batch, recurrent_width)
        norm_base = F.group_norm(
            output,
            num_groups=heads,
            weight=None,
            bias=None,
            eps=head_dim * 1e-5,
        )
        output = norm_base * self.g_norm.weight + self.g_norm.bias
        bonus = (
            r.view(batch, heads, head_dim)
            * k.view(batch, heads, head_dim)
            * self.r_k.reshape(1, heads, head_dim)
        ).sum(dim=-1, keepdim=True)
        norm_offset = (bonus * v.view(batch, heads, head_dim)).view(
            batch, recurrent_width
        )
        output = output + norm_offset
        pre_output = output * g
        output = self.o_proj(pre_output)
        signals = {
            "r": r,
            "projected_r": projected_r,
            "mixed_r": mixed["r"],
            "w_features": w_features,
            "decay": decay,
            "k": k,
            "projected_k": projected_k,
            "write_key_base": write_key_base,
            "mixed_k": mixed["k"],
            "v": v,
            "projected_v": projected_v,
            "mixed_v": mixed["v"],
            "a": normalized_key,
            "erase": a,
            "a_features": a_features,
            "mixed_a": mixed["a"],
            "gate": g,
            "norm_base": norm_base,
            "norm_offset": norm_offset,
            "pre_output": pre_output,
            "g_features": g_features,
            "mixed_g": mixed["g"],
        }
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
        kernel: Rwkv7OperatorAdapter,
        v_first: Tensor | None = None,
        initial_state: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        cu_seqlens_cpu: Tensor | None = None,
        state_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Run fixed or packed training through rwkv-rs FLA's public operator."""
        if x.ndim != 3 or positions.shape != x.shape[:2]:
            raise ValueError(
                "sequence mixer expects x=[B,T,C] and aligned positions=[B,T]"
            )
        batch, tokens, hidden = x.shape
        recurrent_width = self.num_heads * self.head_dim
        previous = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
        if cu_seqlens is not None:
            if batch != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
                raise ValueError(
                    "packed RWKV7 requires x=[1,total,C] and cu_seqlens=[N+1]"
                )
            if int(cu_seqlens[0]) != 0 or int(cu_seqlens[-1]) != tokens:
                raise ValueError("packed RWKV7 cu_seqlens must span every input token")
            previous[:, cu_seqlens[1:-1].to(device=x.device, dtype=torch.long)] = 0
        delta = previous - x
        mixed = {
            name: x + delta * getattr(self, f"x_{name}").reshape(1, 1, hidden)
            for name in ("r", "w", "k", "v", "a", "g")
        }
        r = self.r_proj(mixed["r"])
        w_features = torch.tanh(self.w_lora.lora[0](mixed["w"]))
        w = self.w_lora.lora[2](w_features)
        k = self.k_proj(mixed["k"])
        v = self.v_proj(mixed["v"])
        projected_r, projected_k, projected_v = r, k, v
        a_features = self.a_lora.lora[0](mixed["a"])
        erase = torch.sigmoid(self.a_lora.lora[2](a_features))
        g_features = torch.sigmoid(self.g_lora.lora[0](mixed["g"]))
        gate = self.g_lora.lora[2](g_features)
        if self.source_used_rope:
            r = self._apply_source_rope(r, positions)
            k = self._apply_source_rope(k, positions)
        write_key_base = k
        normalized_key = F.normalize(
            (k * self.k_k.reshape(1, 1, recurrent_width)).view(
                batch, tokens, self.num_heads, self.head_dim
            ),
            dim=-1,
            p=2,
        ).view(batch, tokens, recurrent_width)
        k = k * (1 + (erase - 1) * self.k_a.reshape(1, 1, recurrent_width))
        if self.layer_idx == 0:
            v_first = v
        else:
            if v_first is None or v_first.shape != v.shape:
                raise ValueError("nonzero RWKV7 layers require aligned layer-0 v_first")
            value_mix = torch.sigmoid(
                self.v_lora.lora[2](self.v_lora.lora[0](mixed["v"]))
            )
            v = v + (v_first - v) * value_mix
        vectors = tuple(
            value.view(batch, tokens, self.num_heads, self.head_dim)
            for value in (
                r,
                -EXP_HALF * torch.sigmoid(w),
                k,
                v,
                -normalized_key,
                normalized_key * erase,
            )
        )
        if initial_state is None:
            if state_indices is not None:
                raise ValueError("state-indexed RWKV7 requires an explicit state pool")
            state_rows = batch if cu_seqlens is None else int(cu_seqlens.numel() - 1)
            initial_state = torch.zeros(
                state_rows,
                self.num_heads,
                self.head_dim,
                self.head_dim,
                device=x.device,
                dtype=torch.float32,
            )
        recurrent, final_state = kernel(
            *vectors,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            state_indices=state_indices,
        )
        recurrent = recurrent.to(x.dtype).reshape(batch, tokens, recurrent_width)
        norm_base = F.group_norm(
            recurrent.reshape(batch * tokens, recurrent_width),
            num_groups=self.num_heads,
            weight=None,
            bias=None,
            eps=self.head_dim * 1e-5,
        ).view(batch, tokens, recurrent_width)
        recurrent = norm_base * self.g_norm.weight.reshape(
            1, 1, recurrent_width
        ) + self.g_norm.bias.reshape(1, 1, recurrent_width)
        bonus = (
            r.view(batch, tokens, self.num_heads, self.head_dim)
            * k.view(batch, tokens, self.num_heads, self.head_dim)
            * self.r_k.reshape(1, 1, self.num_heads, self.head_dim)
        ).sum(dim=-1, keepdim=True)
        norm_offset = (
            bonus * v.view(batch, tokens, self.num_heads, self.head_dim)
        ).reshape(batch, tokens, recurrent_width)
        recurrent = recurrent + norm_offset
        pre_output = recurrent * gate
        output = self.o_proj(pre_output)
        signals = {
            "r": r,
            "projected_r": projected_r,
            "mixed_r": mixed["r"],
            "w_features": w_features,
            "w": w,
            "decay": torch.exp(-EXP_HALF * torch.sigmoid(w.float())),
            "k": k,
            "projected_k": projected_k,
            "write_key_base": write_key_base,
            "mixed_k": mixed["k"],
            "v": v,
            "projected_v": projected_v,
            "mixed_v": mixed["v"],
            "a": normalized_key,
            "erase": erase,
            "a_features": a_features,
            "mixed_a": mixed["a"],
            "gate": gate,
            "norm_base": norm_base,
            "norm_offset": norm_offset,
            "pre_output": pre_output,
            "g_features": g_features,
            "mixed_g": mixed["g"],
        }
        return output, v_first, final_state, signals
