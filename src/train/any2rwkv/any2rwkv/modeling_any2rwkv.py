from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_any2rwkv import (
    Any2RWKV7Config,
    Any2RWKVHybridConfig,
    Any2RWKVProxyConfig,
)
from .mixer import ProjectionBoundaryRWKV7Attention


def _source_classes(config: Any2RWKV7Config):
    source = config.any2rwkv.get("source_text_config")
    if not isinstance(source, dict):
        raise ValueError("any2rwkv.source_text_config is required for the preserved Qwen3.5 shell")
    model_type = str(source.get("model_type", ""))
    if model_type == "qwen3_5_moe_text":
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeDecoderLayer,
            Qwen3_5MoeRMSNorm,
        )

        return Qwen3_5MoeTextConfig, Qwen3_5MoeDecoderLayer, Qwen3_5MoeRMSNorm
    if model_type == "qwen3_5_text":
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm

        return Qwen3_5TextConfig, Qwen3_5DecoderLayer, Qwen3_5RMSNorm
    raise ValueError(f"unsupported preserved Qwen3.5 text shell: {model_type!r}")


@dataclass
class Any2RWKV7Cache(Cache):
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

    def get_mask_sizes(self, cache_position: Tensor | int | None, layer_idx: int = 0) -> tuple[int, int]:
        query_len = int(cache_position.numel()) if isinstance(cache_position, Tensor) else int(cache_position or 0)
        return self.seen_tokens + query_len, 0

    def reset(self) -> None:
        for value in (*self.states, *self.previous):
            value.zero_()
        self.seen_tokens = 0

    def batch_repeat_interleave(self, repeats: int) -> "Any2RWKV7Cache":
        if repeats <= 0:
            raise ValueError("cache repeat count must be positive")
        self.states = [value.repeat_interleave(repeats, dim=0) for value in self.states]
        self.previous = [value.repeat_interleave(repeats, dim=0) for value in self.previous]
        return self

    def batch_select_indices(self, indices: Tensor) -> "Any2RWKV7Cache":
        self.states = [value.index_select(0, indices.to(value.device)) for value in self.states]
        self.previous = [value.index_select(0, indices.to(value.device)) for value in self.previous]
        return self

    def crop(self, max_length: int) -> None:
        target = self.seen_tokens + max_length if max_length < 0 else max_length
        if target >= self.seen_tokens:
            return self
        if target <= 0:
            self.reset()
            return self
        raise NotImplementedError(
            "RWKV7 recurrent state cannot be cropped to an earlier positive length; "
            "assisted/speculative rollback requires recomputation"
        )

    def update(self, *args, **kwargs):
        raise NotImplementedError("update Any2RWKV recurrent state through model.forward")

    def reorder(self, beam_idx: Tensor) -> "Any2RWKV7Cache":
        return self.batch_select_indices(beam_idx)

    def reorder_cache(self, beam_idx: Tensor) -> "Any2RWKV7Cache":
        return self.batch_select_indices(beam_idx)


class Any2RWKV7DecoderLayer(nn.Module):
    def __init__(self, config: Any2RWKV7Config, source_config, decoder_cls, layer_idx: int):
        super().__init__()
        shell = decoder_cls(source_config, layer_idx)
        self.input_layernorm = shell.input_layernorm
        self.post_attention_layernorm = shell.post_attention_layernorm
        self.mlp = shell.mlp
        source_types = config.any2rwkv["source_layer_types"]
        source_used_rope = source_types[layer_idx] == "full_attention"
        source_head_dim = int(config.any2rwkv["source_text_config"].get("head_dim", config.head_dim))
        rope = config.rope_parameters
        rotary_dim = min(config.head_dim, int(source_head_dim * float(rope.get("partial_rotary_factor", 1.0))))
        rotary_dim -= rotary_dim % 2
        self.attn = ProjectionBoundaryRWKV7Attention(
            config,
            layer_idx,
            source_used_rope=source_used_rope,
            rotary_dim=rotary_dim,
            rope_theta=float(rope.get("rope_theta", 10_000.0)),
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
        normalized = self.input_layernorm(hidden)
        old_state = state
        old_previous = previous
        mixed, candidate_previous, candidate_state, candidate_v_first, signals = self.attn(
            normalized, previous, v_first, state, positions=positions
        )
        state_mask = valid[:, None, None, None]
        vector_mask = valid[:, None]
        state = torch.where(state_mask, candidate_state, old_state)
        previous = torch.where(vector_mask, candidate_previous, old_previous)
        v_first = torch.where(vector_mask, candidate_v_first, v_first)
        mixed = torch.where(vector_mask, mixed, torch.zeros_like(mixed))
        hidden = residual + mixed
        # Qwen3.5 MoE dispatchers require an explicit sequence dimension even
        # though the recurrent backbone advances one token at a time.
        mlp_input = self.post_attention_layernorm(hidden).unsqueeze(1)
        hidden = hidden + self.mlp(mlp_input).squeeze(1)
        return hidden, previous, state, v_first, signals


class Any2RWKV7MTP(nn.Module):
    """Preserve Qwen3.5 MTP parameters without changing the 60-layer backbone."""

    def __init__(self, config: Any2RWKV7Config, source_config, decoder_cls, norm_cls):
        super().__init__()
        hidden = config.hidden_size
        self.fc = nn.Linear(hidden * 2, hidden, bias=False)
        mtp_dict = source_config.to_dict()
        mtp_dict["num_hidden_layers"] = max(config.mtp_num_hidden_layers, 1)
        mtp_dict["layer_types"] = ["full_attention"] * mtp_dict["num_hidden_layers"]
        mtp_config = type(source_config)(**mtp_dict)
        self.layers = nn.ModuleList(
            decoder_cls(mtp_config, index) for index in range(config.mtp_num_hidden_layers)
        )
        self.norm = norm_cls(hidden, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = norm_cls(hidden, eps=config.rms_norm_eps)
        self.pre_fc_norm_embedding = norm_cls(hidden, eps=config.rms_norm_eps)
        if config.mtp_use_dedicated_embeddings:
            self.embed_tokens = nn.Embedding(config.vocab_size, hidden)


class Any2RWKV7Model(nn.Module):
    def __init__(self, config: Any2RWKV7Config):
        super().__init__()
        config_cls, decoder_cls, norm_cls = _source_classes(config)
        source_config = config_cls(**config.any2rwkv["source_text_config"])
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Any2RWKV7DecoderLayer(config, source_config, decoder_cls, index)
            for index in range(config.num_hidden_layers)
        )
        self.norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.source_config = source_config
        self.decoder_cls = decoder_cls
        self.norm_cls = norm_cls


class Any2RWKV7ForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = Any2RWKV7Config
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = ["Any2RWKV7DecoderLayer"]
    supports_gradient_checkpointing = False
    accepts_loss_kwargs = False
    # Qwen3.5 checkpoints may preserve the language-model head by tying it to
    # the token embedding and therefore omit ``lm_head.weight`` from the
    # serialized state dict.  Advertise the exact HF tying relation so strict
    # loading restores that semantic instead of reporting a missing weight.
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False

    def __init__(self, config: Any2RWKV7Config):
        super().__init__(config)
        self.model = Any2RWKV7Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.mtp_num_hidden_layers:
            self.mtp = Any2RWKV7MTP(
                config,
                self.model.source_config,
                self.model.decoder_cls,
                self.model.norm_cls,
            )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def _new_cache(self, batch: int, device: torch.device, dtype: torch.dtype) -> Any2RWKV7Cache:
        states = [
            torch.zeros(batch, self.config.num_heads, self.config.head_dim, self.config.head_dim, device=device, dtype=torch.float32)
            for _ in self.model.layers
        ]
        previous = [
            torch.zeros(batch, self.config.hidden_size, device=device, dtype=dtype)
            for _ in self.model.layers
        ]
        return Any2RWKV7Cache(states, previous)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Any2RWKV7Cache | None = None,
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
        hidden_sequence = self.model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch, length, _ = hidden_sequence.shape
        cache = (
            past_key_values
            if past_key_values is not None
            else self._new_cache(batch, hidden_sequence.device, hidden_sequence.dtype)
        )
        if len(cache.states) != len(self.model.layers) or cache.states[0].shape[0] != batch:
            raise ValueError("RWKV7 cache does not match model layer count or batch size")
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        if output_hidden_states or output_attentions:
            raise NotImplementedError("Any2RWKV7 currently exposes logits/cache only; hidden states and attentions are unsupported")
        full_attention_mask = attention_mask
        if position_ids is None and full_attention_mask is not None:
            position_ids = full_attention_mask.to(torch.long).cumsum(-1) - 1
            position_ids.masked_fill_(full_attention_mask == 0, 0)
            position_ids = position_ids[:, -length:]
        elif position_ids is None:
            position_ids = torch.arange(
                cache.seen_tokens, cache.seen_tokens + length, device=hidden_sequence.device
            ).view(1, length).expand(batch, -1)
        else:
            position_ids = position_ids[:, -length:]
        if attention_mask is None:
            attention_mask = torch.ones(batch, length, dtype=torch.bool, device=hidden_sequence.device)
        else:
            attention_mask = attention_mask[:, -length:].to(torch.bool)

        outputs: list[Tensor] = []
        for token_index in range(length):
            hidden = hidden_sequence[:, token_index]
            v_first = torch.zeros_like(hidden)
            valid = attention_mask[:, token_index]
            for layer_index, layer in enumerate(self.model.layers):
                hidden, cache.previous[layer_index], cache.states[layer_index], v_first, _ = layer.step(
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
                logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1)
            )
        cache.seen_tokens += length
        use_cache = self.config.use_cache if use_cache is None else use_cache
        result = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=cache if use_cache else None)
        if not return_dict:
            return tuple(value for value in (loss, logits, result.past_key_values) if value is not None)
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
        model_inputs.update({
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        })
        if position_ids is not None:
            take = model_inputs.get("input_ids", model_inputs.get("inputs_embeds")).shape[1]
            model_inputs["position_ids"] = position_ids[:, -take:]
        for name in ("return_dict", "output_hidden_states", "output_attentions"):
            if name in kwargs:
                model_inputs[name] = kwargs[name]
        return model_inputs

    def _reorder_cache(self, past_key_values: Any2RWKV7Cache, beam_idx: Tensor):
        return past_key_values.reorder(beam_idx)


class Any2RWKVProxyForCausalLM(Any2RWKV7ForCausalLM):
    """Loadable identity for a fully recurrent, non-60-layer pilot model."""

    config_class = Any2RWKVProxyConfig


class Any2RWKVHybridForCausalLM(Any2RWKV7ForCausalLM):
    """Guard rail for a progressive checkpoint that still contains source mixers."""

    config_class = Any2RWKVHybridConfig

    def __init__(self, config: Any2RWKVHybridConfig):
        if any(layer_type != "rwkv7" for layer_type in config.layer_types):
            raise ValueError(
                "hybrid Any2RWKV checkpoints are training intermediates; resume "
                "distillation instead of loading them as fully recurrent models"
            )
        super().__init__(config)
