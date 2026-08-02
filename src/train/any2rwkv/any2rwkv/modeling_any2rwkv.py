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
from .mixer import ProjectionBoundaryRWKV7Attention


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


@dataclass
class AnyToRWKVCache(Cache):
    states: list[Tensor]
    previous: list[Tensor]
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
        self.seen_tokens = 0

    def batch_repeat_interleave(self, repeats: int) -> AnyToRWKVCache:
        if repeats <= 0:
            raise ValueError("cache repeat count must be positive")
        self.states = [value.repeat_interleave(repeats, dim=0) for value in self.states]
        self.previous = [
            value.repeat_interleave(repeats, dim=0) for value in self.previous
        ]
        return self

    def batch_select_indices(self, indices: Tensor) -> AnyToRWKVCache:
        self.states = [
            value.index_select(0, indices.to(value.device)) for value in self.states
        ]
        self.previous = [
            value.index_select(0, indices.to(value.device)) for value in self.previous
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
        self.input_layernorm = AnyToRWKVRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = AnyToRWKVRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = _feed_forward(config)
        source_types = config.any_to_rwkv["source_layer_types"]
        source_text = config.any_to_rwkv["source_text_config"]
        source_used_rope = source_types[layer_index] == "full_attention"
        source_head_dim = int(source_text.get("head_dim", config.head_dim))
        source_num_heads = int(source_text.get("num_attention_heads", config.num_heads))
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

    def step(
        self,
        hidden: Tensor,
        previous: Tensor,
        state: Tensor,
        v_first: Tensor,
        positions: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        residual = hidden
        old_state = state
        old_previous = previous
        mixed, candidate_previous, candidate_state, candidate_v_first, signals = (
            self.attn(
                self.input_layernorm(hidden),
                previous,
                v_first,
                state,
                positions=positions,
            )
        )
        state_mask = valid[:, None, None, None]
        vector_mask = valid[:, None]
        state = torch.where(state_mask, candidate_state, old_state)
        previous = torch.where(vector_mask, candidate_previous, old_previous)
        v_first = torch.where(vector_mask, candidate_v_first, v_first)
        mixed = torch.where(vector_mask, mixed, torch.zeros_like(mixed))
        hidden = residual + mixed
        hidden = hidden + self.mlp(
            self.post_attention_layernorm(hidden).unsqueeze(1)
        ).squeeze(1)
        return hidden, previous, state, v_first, signals


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
                layer.attn.num_heads,
                layer.attn.head_dim,
                layer.attn.head_dim,
                device=device,
                dtype=torch.float32,
            )
            for layer in self.model.layers
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
        return AnyToRWKVCache(states, previous)

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

        outputs: list[Tensor] = []
        for token_index in range(sequence_length):
            hidden = hidden_sequence[:, token_index]
            v_first = torch.zeros(
                batch_size,
                self.config.attention_hidden_size,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            valid = attention_mask[:, token_index]
            for layer_index, layer in enumerate(self.model.layers):
                (
                    hidden,
                    cache.previous[layer_index],
                    cache.states[layer_index],
                    v_first,
                    _,
                ) = layer.step(
                    hidden,
                    cache.previous[layer_index],
                    cache.states[layer_index],
                    v_first,
                    position_ids[:, token_index],
                    valid,
                )
            outputs.append(self.model.norm(hidden))
        hidden_states = torch.stack(outputs, dim=1)
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

    def __init__(self, config: AnyToRWKVHybridConfig):
        if any(mixer_type != "rwkv7" for mixer_type in config.mixer_types):
            raise ValueError(
                "hybrid Any-to-RWKV checkpoints are training intermediates; resume "
                "conversion instead of loading them as final models"
            )
        super().__init__(config)


__all__ = [
    "AnyToRWKVCache",
    "AnyToRWKVForCausalLM",
    "AnyToRWKVHybridForCausalLM",
    "AnyToRWKVModel",
    "AnyToRWKVProxyForCausalLM",
]
