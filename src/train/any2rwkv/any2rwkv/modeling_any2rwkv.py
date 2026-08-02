from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_any2rwkv import (
    AnyToRWKVConfig,
    AnyToRWKVHybridConfig,
    AnyToRWKVProxyConfig,
)
from .kernel import load_rwkv7_operator_adapter
from .mixer import ProjectionBoundaryRWKV7Attention, apply_partial_rope


class AnyToRWKVRMSNorm(nn.Module):
    """RMSNorm owned by the independent Any-to-RWKV model family."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, value: Tensor) -> Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return (normalized * (1.0 + self.weight.float())).type_as(value)


class AnyToRWKVPreservedRMSNormGated(nn.Module):
    """Preserve Qwen3.5's gated RMSNorm parameter and operation semantics."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value: Tensor, gate: Tensor) -> Tensor:
        input_dtype = value.dtype
        normalized = value.float()
        variance = normalized.pow(2).mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        normalized = self.weight * normalized.to(input_dtype)
        normalized = normalized * nn.functional.silu(gate.float())
        return normalized.to(input_dtype)


class AnyToRWKVDenseMLP(nn.Module):
    def __init__(self, config: AnyToRWKVConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, value: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(value)) * self.up_proj(value))


class AnyToRWKVExperts(nn.Module):
    """Independent expert weights with the source-compatible tensor layout."""

    def __init__(self, config: AnyToRWKVConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                2 * self.intermediate_size,
                self.hidden_size,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.hidden_size,
                self.intermediate_size,
            )
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: Tensor,
        selected_experts: Tensor,
        routing_weights: Tensor,
    ) -> Tensor:
        output = torch.zeros_like(hidden_states)
        expert_mask = nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)
        for expert_index in range(self.num_experts):
            top_k_position, token_index = torch.where(expert_mask[expert_index])
            if token_index.numel() == 0:
                continue
            current = hidden_states[token_index]
            gate, up = nn.functional.linear(
                current, self.gate_up_proj[expert_index]
            ).chunk(2, dim=-1)
            current = self.act_fn(gate) * up
            current = nn.functional.linear(current, self.down_proj[expert_index])
            current = current * routing_weights[token_index, top_k_position, None]
            output.index_add_(0, token_index, current.to(output.dtype))
        return output


class AnyToRWKVTopKRouter(nn.Module):
    def __init__(self, config: AnyToRWKVConfig):
        super().__init__()
        if config.num_experts <= 0 or config.num_experts_per_tok <= 0:
            raise ValueError("Any-to-RWKV MoE routing requires positive expert counts")
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_size))

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        flattened = hidden_states.reshape(-1, self.hidden_size)
        probabilities = nn.functional.softmax(
            nn.functional.linear(flattened, self.weight), dtype=torch.float, dim=-1
        )
        routing_weights, selected_experts = torch.topk(
            probabilities, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        return routing_weights.to(flattened.dtype), selected_experts


class AnyToRWKVSparseMoEBlock(nn.Module):
    def __init__(self, config: AnyToRWKVConfig):
        super().__init__()
        self.gate = AnyToRWKVTopKRouter(config)
        self.experts = AnyToRWKVExperts(config)
        self.shared_expert = AnyToRWKVDenseMLP(
            _feed_forward_config(
                config,
                intermediate_size=config.shared_expert_intermediate_size,
            )
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        flattened = hidden_states.reshape(-1, hidden_size)
        shared = self.shared_expert(flattened)
        routing_weights, selected_experts = self.gate(flattened)
        expert = self.experts(flattened, selected_experts, routing_weights)
        shared = torch.sigmoid(self.shared_expert_gate(flattened)) * shared
        return (expert + shared).reshape(batch_size, sequence_length, hidden_size)


def _feed_forward_config(
    config: AnyToRWKVConfig,
    *,
    intermediate_size: int,
) -> AnyToRWKVConfig:
    payload = config.to_dict()
    payload["intermediate_size"] = intermediate_size
    payload.pop("auto_map", None)
    return type(config)(**payload)


def _feed_forward(config: AnyToRWKVConfig) -> nn.Module:
    if config.num_experts:
        return AnyToRWKVSparseMoEBlock(config)
    return AnyToRWKVDenseMLP(config)


class AnyToRWKVPreservedAttention(nn.Module):
    """Own the preserved MTP tensor layout without importing a Qwen model."""

    def __init__(self, config: AnyToRWKVConfig):
        super().__init__()
        source = config.any_to_rwkv.get("source_text_config", {})
        source_heads = int(source.get("num_attention_heads", config.num_heads))
        source_kv_heads = int(source.get("num_key_value_heads", source_heads))
        source_head_dim = int(source.get("head_dim", config.head_dim))
        hidden_size = config.hidden_size
        self.q_proj = nn.Linear(
            hidden_size, source_heads * source_head_dim * 2, bias=False
        )
        self.k_proj = nn.Linear(
            hidden_size, source_kv_heads * source_head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, source_kv_heads * source_head_dim, bias=False
        )
        self.o_proj = nn.Linear(source_heads * source_head_dim, hidden_size, bias=False)
        self.q_norm = AnyToRWKVRMSNorm(source_head_dim, eps=config.rms_norm_eps)
        self.k_norm = AnyToRWKVRMSNorm(source_head_dim, eps=config.rms_norm_eps)
        self.num_heads = source_heads
        self.num_key_value_heads = source_kv_heads
        self.head_dim = source_head_dim
        self.rotary_dim = int(
            source_head_dim
            * float(config.rope_parameters.get("partial_rotary_factor", 1.0))
        )
        self.rotary_dim -= self.rotary_dim % 2
        self.rope_theta = float(
            config.rope_parameters.get("rope_theta", source.get("rope_theta", 10_000.0))
        )

    def forward_sequence(self, hidden: Tensor, positions: Tensor) -> Tensor:
        """Run package-owned causal GQA for a preserved source layer."""
        batch, tokens, _hidden = hidden.shape
        query, gate = (
            self.q_proj(hidden)
            .view(batch, tokens, self.num_heads, self.head_dim * 2)
            .chunk(2, dim=-1)
        )
        key = self.k_proj(hidden).view(
            batch, tokens, self.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(hidden).view(
            batch, tokens, self.num_key_value_heads, self.head_dim
        )
        query = self.q_norm(query)
        key = self.k_norm(key)
        query = apply_partial_rope(
            query,
            positions,
            rotary_dim=self.rotary_dim,
            theta=self.rope_theta,
        )
        key = apply_partial_rope(
            key,
            positions,
            rotary_dim=self.rotary_dim,
            theta=self.rope_theta,
        )
        groups = self.num_heads // self.num_key_value_heads
        if groups * self.num_key_value_heads != self.num_heads:
            raise ValueError("preserved GQA heads are not evenly grouped")
        key = key.repeat_interleave(groups, dim=2)
        value = value.repeat_interleave(groups, dim=2)
        scores = torch.einsum("bthd,bshd->bhts", query.float(), key.float())
        scores = scores / self.head_dim**0.5
        causal = torch.ones(
            tokens, tokens, device=hidden.device, dtype=torch.bool
        ).tril()
        scores.masked_fill_(~causal, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1).to(value.dtype)
        output = torch.einsum("bhts,bshd->bthd", probabilities, value)
        output = output * torch.sigmoid(gate)
        return self.o_proj(output.reshape(batch, tokens, -1))


class AnyToRWKVPreservedGDN(nn.Module):
    """Package-owned preserved Gated DeltaNet tensor layout and recurrence."""

    def __init__(self, config: AnyToRWKVConfig):
        super().__init__()
        source = config.any_to_rwkv.get("source_text_config", {})
        hidden = config.hidden_size
        self.num_key_heads = int(source.get("linear_num_key_heads", config.num_heads))
        self.num_value_heads = int(
            source.get("linear_num_value_heads", self.num_key_heads)
        )
        self.key_head_dim = int(source.get("linear_key_head_dim", config.head_dim))
        self.value_head_dim = int(source.get("linear_value_head_dim", config.head_dim))
        self.key_dim = self.num_key_heads * self.key_head_dim
        self.value_dim = self.num_value_heads * self.value_head_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = int(source.get("linear_conv_kernel_dim", 4))
        self.activation = ACT2FN[config.hidden_act]
        self.in_proj_qkv = nn.Linear(hidden, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden, self.num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden, self.num_value_heads, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.zeros(self.num_value_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_value_heads))
        self.norm = AnyToRWKVPreservedRMSNormGated(
            self.value_head_dim, eps=config.rms_norm_eps
        )
        self.out_proj = nn.Linear(self.value_dim, hidden, bias=False)

    def forward_sequence(self, hidden: Tensor) -> Tensor:
        batch, tokens, _hidden = hidden.shape
        projected = self.in_proj_qkv(hidden).transpose(1, 2)
        projected = nn.functional.conv1d(
            projected,
            self.conv1d.weight,
            padding=self.conv_kernel_size - 1,
            groups=self.conv_dim,
        )[:, :, :tokens]
        query, key, value = torch.split(
            self.activation(projected.transpose(1, 2)),
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.view(batch, tokens, self.num_key_heads, self.key_head_dim)
        key = key.view(batch, tokens, self.num_key_heads, self.key_head_dim)
        value = value.view(batch, tokens, self.num_value_heads, self.value_head_dim)
        groups = self.num_value_heads // self.num_key_heads
        if groups * self.num_key_heads != self.num_value_heads:
            raise ValueError("preserved GDN key heads do not divide value heads")
        query = query.repeat_interleave(groups, dim=2)
        key = key.repeat_interleave(groups, dim=2)
        query = query.float()
        query = query * torch.rsqrt(query.square().sum(dim=-1, keepdim=True) + 1e-6)
        query = query * self.key_head_dim**-0.5
        key = key.float()
        key = key * torch.rsqrt(key.square().sum(dim=-1, keepdim=True) + 1e-6)
        beta = torch.sigmoid(self.in_proj_b(hidden))
        decay = torch.exp(
            -self.A_log.float().exp()
            * nn.functional.softplus(self.in_proj_a(hidden).float() + self.dt_bias)
        )
        state = torch.zeros(
            batch,
            self.num_value_heads,
            self.key_head_dim,
            self.value_head_dim,
            device=hidden.device,
            dtype=torch.float32,
        )
        rows: list[Tensor] = []
        for token in range(tokens):
            current_key = key[:, token]
            current_value = value[:, token].float()
            state = decay[:, token, :, None, None] * state
            prediction = torch.einsum("bhk,bhkv->bhv", current_key, state)
            correction = beta[:, token, :, None].float() * (current_value - prediction)
            state = state + torch.einsum("bhk,bhv->bhkv", current_key, correction)
            rows.append(torch.einsum("bhk,bhkv->bhv", query[:, token], state))
        output = torch.stack(rows, dim=1).to(hidden.dtype)
        gate = self.in_proj_z(hidden).view_as(output)
        output = self.norm(output, gate)
        return self.out_proj(output.reshape(batch, tokens, self.value_dim))


@dataclass
class AnyToRWKVCache(Cache):
    states: list[Tensor]
    previous: list[Tensor]
    histories: list[list[Tensor]]
    history_positions: list[list[Tensor]]
    seen_tokens: int = 0

    def get_seq_length(self, layer_idx: int | None = 0, cache_position=None) -> int:
        return self.seen_tokens

    @property
    def is_initialized(self) -> bool:
        return bool(self.states)

    @property
    def max_batch_size(self) -> int | None:
        return int(self.states[0].shape[0]) if self.states else None

    @property
    def max_cache_len(self) -> int:
        return -1

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def get_batch_size(self) -> int:
        return int(self.states[0].shape[0]) if self.states else 0

    def get_mask_sizes(
        self, cache_position: Tensor | int | None, layer_idx: int = 0
    ) -> tuple[int, int]:
        query_length = (
            int(cache_position.numel())
            if isinstance(cache_position, Tensor)
            else int(cache_position or 0)
        )
        return self.seen_tokens + query_length, 0

    def reset(self) -> None:
        for value in (*self.states, *self.previous):
            value.zero_()
        self.histories = [
            [row[:, :0] for row in layer_rows] for layer_rows in self.histories
        ]
        self.history_positions = [
            [row[:, :0] for row in layer_rows] for layer_rows in self.history_positions
        ]
        self.seen_tokens = 0

    def batch_repeat_interleave(self, repeats: int) -> AnyToRWKVCache:
        if repeats <= 0:
            raise ValueError("cache repeat count must be positive")
        self.states = [value.repeat_interleave(repeats, dim=0) for value in self.states]
        self.previous = [
            value.repeat_interleave(repeats, dim=0) for value in self.previous
        ]
        self.histories = [
            [row for row in rows for _ in range(repeats)] for rows in self.histories
        ]
        self.history_positions = [
            [row for row in rows for _ in range(repeats)]
            for rows in self.history_positions
        ]
        return self

    def batch_select_indices(self, indices: Tensor) -> AnyToRWKVCache:
        self.states = [
            value.index_select(0, indices.to(value.device)) for value in self.states
        ]
        self.previous = [
            value.index_select(0, indices.to(value.device)) for value in self.previous
        ]
        selected = indices.detach().cpu().tolist()
        self.histories = [
            [rows[index] for index in selected] for rows in self.histories
        ]
        self.history_positions = [
            [rows[index] for index in selected] for rows in self.history_positions
        ]
        return self

    def crop(self, max_length: int) -> None:
        target = self.seen_tokens + max_length if max_length < 0 else max_length
        if target >= self.seen_tokens:
            return
        if target <= 0:
            self.reset()
            return
        raise NotImplementedError(
            "Any-to-RWKV recurrent state cannot be cropped to an earlier positive "
            "length; assisted/speculative rollback requires recomputation"
        )

    def update(self, *args, **kwargs):
        raise NotImplementedError(
            "update Any-to-RWKV recurrent state through model.forward"
        )

    def reorder(self, beam_index: Tensor) -> AnyToRWKVCache:
        return self.batch_select_indices(beam_index)

    def reorder_cache(self, beam_index: Tensor) -> AnyToRWKVCache:
        return self.batch_select_indices(beam_index)


class AnyToRWKVDecoderLayer(nn.Module):
    def __init__(self, config: AnyToRWKVConfig, layer_index: int):
        super().__init__()
        self.layer_index = layer_index
        self.mixer_type = config.mixer_types[layer_index]
        self.input_layernorm = AnyToRWKVRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = AnyToRWKVRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = _feed_forward(config)
        if self.mixer_type == "rwkv7":
            source_types = config.any_to_rwkv["source_layer_types"]
            source_text = config.any_to_rwkv["source_text_config"]
            source_used_rope = source_types[layer_index] == "full_attention"
            source_head_dim = int(source_text.get("head_dim", config.head_dim))
            source_num_heads = int(
                source_text.get("num_attention_heads", config.num_heads)
            )
            rotary_dim = int(
                source_head_dim
                * float(config.rope_parameters.get("partial_rotary_factor", 1.0))
            )
            rotary_dim -= rotary_dim % 2
            self.attn = ProjectionBoundaryRWKV7Attention(
                config,
                layer_index,
                source_used_rope=source_used_rope,
                rotary_dim=rotary_dim,
                rope_theta=float(config.rope_parameters.get("rope_theta", 10_000.0)),
                rope_num_heads=source_num_heads,
                rope_head_dim=source_head_dim,
            )
        elif self.mixer_type == "full_attention":
            self.self_attn = AnyToRWKVPreservedAttention(config)
        elif self.mixer_type == "linear_attention":
            self.linear_attn = AnyToRWKVPreservedGDN(config)
        else:
            raise ValueError(f"unsupported Any-to-RWKV mixer type: {self.mixer_type}")

    def forward_sequence(
        self,
        hidden: Tensor,
        *,
        positions: Tensor,
        valid: Tensor,
        cache: AnyToRWKVCache,
        v_first: Tensor | None,
        kernel=None,
    ) -> tuple[Tensor, Tensor | None]:
        residual = hidden
        normalized = self.input_layernorm(hidden)
        mixed = torch.zeros_like(hidden)
        next_v_first = v_first
        if self.mixer_type == "rwkv7":
            if kernel is None:
                raise RuntimeError(
                    "RWKV7 product forward requires a recurrent operator"
                )
            if self.layer_index and v_first is None:
                raise RuntimeError("nonzero RWKV7 layer requires layer-0 v_first")
            if self.layer_index == 0:
                next_v_first = torch.zeros(
                    *hidden.shape[:2],
                    self.attn.attention_hidden_size,
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
            next_state = cache.states[self.layer_index].detach().clone()
            next_previous = cache.previous[self.layer_index].detach().clone()
            for row in range(hidden.shape[0]):
                indices = torch.nonzero(valid[row], as_tuple=False).flatten()
                if not indices.numel():
                    continue
                row_input = normalized[row : row + 1].index_select(1, indices)
                row_positions = positions[row : row + 1].index_select(1, indices)
                row_v_first = (
                    None
                    if self.layer_index == 0
                    else v_first[row : row + 1].index_select(1, indices)
                )
                output, candidate_v_first, final_state, _signals = (
                    self.attn.forward_sequence(
                        row_input,
                        positions=row_positions,
                        kernel=kernel,
                        v_first=row_v_first,
                        initial_state=cache.states[self.layer_index][row : row + 1],
                        previous=cache.previous[self.layer_index][row : row + 1],
                    )
                )
                mixed[row].index_copy_(0, indices, output[0])
                if self.layer_index == 0:
                    next_v_first[row].index_copy_(0, indices, candidate_v_first[0])
                next_state[row : row + 1].copy_(final_state.detach())
                next_previous[row].copy_(row_input[0, -1].detach())
            cache.states[self.layer_index] = next_state
            cache.previous[self.layer_index] = next_previous
        else:
            for row in range(hidden.shape[0]):
                indices = torch.nonzero(valid[row], as_tuple=False).flatten()
                if not indices.numel():
                    continue
                current = normalized[row : row + 1].index_select(1, indices)
                current_positions = positions[row : row + 1].index_select(1, indices)
                history = cache.histories[self.layer_index][row]
                history_positions = cache.history_positions[self.layer_index][row]
                complete = torch.cat((history, current), dim=1)
                complete_positions = torch.cat(
                    (history_positions, current_positions), dim=1
                )
                if self.mixer_type == "full_attention":
                    complete_output = self.self_attn.forward_sequence(
                        complete, complete_positions
                    )
                else:
                    complete_output = self.linear_attn.forward_sequence(complete)
                mixed[row].index_copy_(
                    0, indices, complete_output[0, -indices.numel() :]
                )
                cache.histories[self.layer_index][row] = complete.detach()
                cache.history_positions[self.layer_index][row] = (
                    complete_positions.detach()
                )
        hidden = residual + mixed
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden, next_v_first


class AnyToRWKVMTPDecoderLayer(nn.Module):
    def __init__(self, config: AnyToRWKVConfig):
        super().__init__()
        self.input_layernorm = AnyToRWKVRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = AnyToRWKVRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = AnyToRWKVPreservedAttention(config)
        self.mlp = _feed_forward(config)


class AnyToRWKVMTP(nn.Module):
    """Preserve optional MTP tensors under an Any-to-RWKV-owned module tree."""

    def __init__(self, config: AnyToRWKVConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.layers = nn.ModuleList(
            AnyToRWKVMTPDecoderLayer(config)
            for _ in range(config.mtp_num_hidden_layers)
        )
        self.norm = AnyToRWKVRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = AnyToRWKVRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_embedding = AnyToRWKVRMSNorm(
            hidden_size, eps=config.rms_norm_eps
        )
        if config.mtp_use_dedicated_embeddings:
            self.embed_tokens = nn.Embedding(config.vocab_size, hidden_size)


class AnyToRWKVModel(nn.Module):
    def __init__(self, config: AnyToRWKVConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            AnyToRWKVDecoderLayer(config, layer_index)
            for layer_index in range(config.num_hidden_layers)
        )
        self.norm = AnyToRWKVRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class AnyToRWKVForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = AnyToRWKVConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules: ClassVar[list[str]] = ["AnyToRWKVDecoderLayer"]
    supports_gradient_checkpointing = False
    accepts_loss_kwargs = False
    _tied_weights_keys: ClassVar[dict[str, str]] = {
        "lm_head.weight": "model.embed_tokens.weight"
    }

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False

    def __init__(self, config: AnyToRWKVConfig):
        super().__init__(config)
        self.model = AnyToRWKVModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.mtp_num_hidden_layers:
            self.mtp = AnyToRWKVMTP(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def _new_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> AnyToRWKVCache:
        states = [
            torch.zeros(
                batch_size,
                self.config.num_heads,
                self.config.head_dim,
                self.config.head_dim,
                device=device,
                dtype=torch.float32,
            )
            for _layer in self.model.layers
        ]
        previous = [
            torch.zeros(
                batch_size,
                self.config.hidden_size,
                device=device,
                dtype=dtype,
            )
            for _ in self.model.layers
        ]
        histories = [
            [
                torch.empty(
                    1,
                    0,
                    self.config.hidden_size,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(batch_size)
            ]
            for _ in self.model.layers
        ]
        history_positions = [
            [
                torch.empty(1, 0, device=device, dtype=torch.long)
                for _ in range(batch_size)
            ]
            for _ in self.model.layers
        ]
        return AnyToRWKVCache(states, previous, histories, history_positions)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: AnyToRWKVCache | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[Tensor, ...]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        hidden_sequence = (
            self.model.embed_tokens(input_ids)
            if inputs_embeds is None
            else inputs_embeds
        )
        batch_size, sequence_length, _ = hidden_sequence.shape
        cache = (
            past_key_values
            if past_key_values is not None
            else self._new_cache(
                batch_size, hidden_sequence.device, hidden_sequence.dtype
            )
        )
        if (
            len(cache.states) != len(self.model.layers)
            or cache.states[0].shape[0] != batch_size
        ):
            raise ValueError(
                "Any-to-RWKV cache does not match model layer count or batch size"
            )
        return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        output_attentions = (
            self.config.output_attentions
            if output_attentions is None
            else output_attentions
        )
        if output_hidden_states or output_attentions:
            raise NotImplementedError(
                "Any-to-RWKV exposes logits and recurrent cache only"
            )
        full_attention_mask = attention_mask
        if position_ids is None and full_attention_mask is not None:
            position_ids = full_attention_mask.to(torch.long).cumsum(-1) - 1
            position_ids.masked_fill_(full_attention_mask == 0, 0)
            position_ids = position_ids[:, -sequence_length:]
        elif position_ids is None:
            position_ids = (
                torch.arange(
                    cache.seen_tokens,
                    cache.seen_tokens + sequence_length,
                    device=hidden_sequence.device,
                )
                .view(1, sequence_length)
                .expand(batch_size, -1)
            )
        else:
            position_ids = position_ids[:, -sequence_length:]
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
                device=hidden_sequence.device,
            )
        else:
            attention_mask = attention_mask[:, -sequence_length:].to(torch.bool)

        kernel = None
        if any(layer.mixer_type == "rwkv7" for layer in self.model.layers):
            kernel = load_rwkv7_operator_adapter(self.config.head_dim)
        hidden_states = hidden_sequence
        v_first = None
        for layer in self.model.layers:
            hidden_states, v_first = layer.forward_sequence(
                hidden_states,
                positions=position_ids,
                valid=attention_mask,
                cache=cache,
                v_first=v_first,
                kernel=kernel,
            )
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
            )
        cache.seen_tokens += sequence_length
        use_cache = self.config.use_cache if use_cache is None else use_cache
        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=cache if use_cache else None,
        )
        if not return_dict:
            return tuple(
                value
                for value in (loss, logits, result.past_key_values)
                if value is not None
            )
        return result

    def prepare_inputs_for_generation(
        self,
        input_ids: Tensor | None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds: Tensor | None = None,
        position_ids: Tensor | None = None,
        next_sequence_length: int | None = None,
        is_first_iteration: bool | None = None,
        **kwargs,
    ):
        model_inputs: dict[str, Any] = {}
        if past_key_values is not None:
            take = int(next_sequence_length or 1)
            if take <= 0:
                raise ValueError("next_sequence_length must be positive")
            if input_ids is not None:
                model_inputs["input_ids"] = input_ids[:, -take:]
            elif inputs_embeds is not None:
                model_inputs["inputs_embeds"] = inputs_embeds[:, -take:]
        elif inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs["input_ids"] = input_ids
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "use_cache": kwargs.get("use_cache", True),
            }
        )
        if position_ids is not None:
            values = model_inputs.get("input_ids", model_inputs.get("inputs_embeds"))
            model_inputs["position_ids"] = position_ids[:, -values.shape[1] :]
        for name in ("return_dict", "output_hidden_states", "output_attentions"):
            if name in kwargs:
                model_inputs[name] = kwargs[name]
        return model_inputs

    def _reorder_cache(
        self, past_key_values: AnyToRWKVCache, beam_index: Tensor
    ) -> AnyToRWKVCache:
        return past_key_values.reorder(beam_index)


class AnyToRWKVProxyForCausalLM(AnyToRWKVForCausalLM):
    config_class = AnyToRWKVProxyConfig


class AnyToRWKVHybridForCausalLM(AnyToRWKVForCausalLM):
    config_class = AnyToRWKVHybridConfig


__all__ = [
    "AnyToRWKVCache",
    "AnyToRWKVForCausalLM",
    "AnyToRWKVHybridForCausalLM",
    "AnyToRWKVModel",
    "AnyToRWKVProxyForCausalLM",
]
