from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import threading
from typing import Callable

import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

from .checkpoint import CheckpointManifest
from .errors import ContractError
from .layer_store import LayerTensorStore
from .hybrid import HybridRecurrentContext, QwenRWKV7MixerAdapter
from .mixer import ProjectionBoundaryRWKV7Attention


@dataclass(frozen=True)
class LoadedTeacherLayer:
    layer_index: int
    module: nn.Module
    source_tensor_bytes: int


@dataclass(frozen=True)
class StreamedTeacherOutput:
    logits: torch.Tensor
    active_layer_input: torch.Tensor | None
    active_mixer_output: torch.Tensor | None
    active_block_output: torch.Tensor | None
    active_recurrent_state: torch.Tensor | None


@dataclass(frozen=True)
class StreamedHybridOutput:
    logits: torch.Tensor
    active_layer_input: torch.Tensor
    active_mixer_output: torch.Tensor
    active_block_output: torch.Tensor
    active_state: torch.Tensor
    active_signals: dict[str, torch.Tensor]


class Qwen35TeacherLayerLoader:
    """Construct and strictly load one frozen Qwen3.5 decoder layer on demand."""

    def __init__(
        self,
        checkpoint: CheckpointManifest,
        *,
        cache_layers: bool = False,
    ) -> None:
        self.checkpoint = checkpoint
        self.tensor_store = LayerTensorStore(checkpoint)
        self.cache_layers = cache_layers
        self._layer_cache: dict[
            tuple[int, torch.device, torch.dtype], LoadedTeacherLayer
        ] = {}
        self._layer_leases = {
            index: threading.Lock()
            for index in range(checkpoint.contract.num_hidden_layers)
        }
        text_config = checkpoint.config.get("text_config", checkpoint.config)
        if not isinstance(text_config, dict):
            raise ContractError("Qwen3.5 text_config must be an object")
        if checkpoint.contract.has_moe:
            from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
                Qwen3_5MoeTextConfig,
            )
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                Qwen3_5MoeDecoderLayer,
            )

            self.config = Qwen3_5MoeTextConfig.from_dict(text_config)
            self.layer_class = Qwen3_5MoeDecoderLayer
        else:
            from transformers.models.qwen3_5.configuration_qwen3_5 import (
                Qwen3_5TextConfig,
            )
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5DecoderLayer,
            )

            self.config = Qwen3_5TextConfig.from_dict(text_config)
            self.layer_class = Qwen3_5DecoderLayer
        self.config._attn_implementation = "sdpa"
        if checkpoint.contract.has_moe:
            self.config._experts_implementation = "grouped_mm"

    def load_layer(
        self,
        layer_index: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> LoadedTeacherLayer:
        target_device = torch.device(device)
        cache_key = (layer_index, target_device, dtype)
        cached = self._layer_cache.get(cache_key)
        if cached is not None:
            return cached
        construction = torch.device("meta") if target_device.type == "cuda" else nullcontext()
        with construction:
            module = self.layer_class(self.config, layer_index)
        if target_device.type == "cuda":
            module.to_empty(device=target_device)
        state = {}
        prefix = f"model.layers.{layer_index}."
        source_tensor_bytes = 0
        for name, tensor in self.tensor_store.load_layer(layer_index).items():
            if not name.startswith(prefix):
                raise ContractError(
                    f"layer store returned a tensor outside layer {layer_index}: {name}"
                )
            local_name = name.removeprefix(prefix)
            source_tensor_bytes += tensor.numel() * tensor.element_size()
            state[local_name] = tensor.to(device=target_device, dtype=dtype)
        incompatible = module.load_state_dict(state, strict=False, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ContractError(
                f"strict streamed layer load failed at layer {layer_index}: "
                f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
            )
        module.eval().requires_grad_(False)
        loaded = LoadedTeacherLayer(layer_index, module, source_tensor_bytes)
        if self.cache_layers:
            self._layer_cache[cache_key] = loaded
        return loaded

    @contextmanager
    def layer_lease(self, layer_index: int):
        try:
            lease = self._layer_leases[layer_index]
        except KeyError as error:
            raise ContractError(
                f"decoder layer index out of range: {layer_index}"
            ) from error
        if not lease.acquire(blocking=False):
            raise ContractError(
                f"cached teacher layer {layer_index} does not permit overlapping execution"
            )
        try:
            yield
        finally:
            lease.release()

    @property
    def cached_layer_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for loaded in self._layer_cache.values()
            for tensor in (*loaded.module.parameters(), *loaded.module.buffers())
        )

    @property
    def cached_layer_count(self) -> int:
        return len(self._layer_cache)


class StreamedQwen35Teacher:
    """Execute a Qwen3.5 teacher while keeping at most one decoder layer loaded."""

    def __init__(
        self,
        checkpoint: CheckpointManifest,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        cache_layers: bool = False,
    ) -> None:
        self.loader = Qwen35TeacherLayerLoader(
            checkpoint,
            cache_layers=cache_layers,
        )
        self.device = torch.device(device)
        self.dtype = dtype
        global_names = ["model.embed_tokens.weight", "model.norm.weight"]
        if self.loader.tensor_store.has_tensor("lm_head.weight"):
            global_names.append("lm_head.weight")
        tensors = self.loader.tensor_store.load_named_tensors(tuple(global_names))
        self.embedding_weight = tensors["model.embed_tokens.weight"].to(
            device=self.device, dtype=dtype
        )
        self.norm_weight = tensors["model.norm.weight"].to(
            device=self.device, dtype=dtype
        )
        self.lm_head_weight = tensors.get(
            "lm_head.weight", tensors["model.embed_tokens.weight"]
        ).to(device=self.device, dtype=dtype)
        self.rms_norm_eps = float(getattr(self.loader.config, "rms_norm_eps"))
        if checkpoint.contract.has_moe:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                Qwen3_5MoeTextRotaryEmbedding,
            )

            rotary_class = Qwen3_5MoeTextRotaryEmbedding
        else:
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5TextRotaryEmbedding,
            )

            rotary_class = Qwen3_5TextRotaryEmbedding
        self.rotary = rotary_class(self.loader.config, device=self.device)

    @property
    def resident_global_bytes(self) -> int:
        tensors = (
            self.embedding_weight,
            self.norm_weight,
            self.lm_head_weight,
            *self.rotary.parameters(),
            *self.rotary.buffers(),
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        capture_layer_index: int | None = None,
    ) -> StreamedTeacherOutput:
        input_ids = input_ids.to(self.device)
        hidden_states = functional.embedding(input_ids, self.embedding_weight)
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.shape[-1], device=self.device, dtype=torch.long
            ).unsqueeze(0).expand(input_ids.shape[0], -1)
        else:
            position_ids = position_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        position_embeddings = self.rotary(hidden_states, position_ids)
        causal_mask = create_causal_mask(
            self.loader.config,
            hidden_states,
            attention_mask,
            None,
            position_ids,
        )
        captured_input = None
        captured_mixer = None
        captured_block = None
        captured_state = None
        # Cached teacher modules are reused by the hybrid executor, whose
        # frozen suffix must preserve input gradients back to the active
        # RWKV7 layer. inference_mode would taint reused tensors for autograd.
        with torch.no_grad():
            for layer_index in range(self.loader.tensor_store.num_layers):
                with self.loader.layer_lease(layer_index):
                    loaded = self.loader.load_layer(
                        layer_index,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    hook = None
                    if layer_index == capture_layer_index:
                        captured_input = hidden_states.detach().cpu()
                        mixer = getattr(loaded.module, "linear_attn", None) or getattr(
                            loaded.module, "self_attn", None
                        )

                        def capture_mixer(module, args, output):
                            nonlocal captured_mixer
                            value = output[0] if isinstance(output, tuple) else output
                            captured_mixer = value.detach().cpu()

                        hook = mixer.register_forward_hook(capture_mixer)
                    try:
                        capture_cache = (
                            DynamicCache(config=self.loader.config)
                            if layer_index == capture_layer_index
                            and self.loader.config.layer_types[layer_index] == "linear_attention"
                            else None
                        )
                        hidden_states = loaded.module(
                            hidden_states,
                            position_embeddings=position_embeddings,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_values=capture_cache,
                        )
                        if isinstance(hidden_states, tuple):
                            hidden_states = hidden_states[0]
                    finally:
                        if hook is not None:
                            hook.remove()
                if layer_index == capture_layer_index:
                    captured_block = hidden_states.detach().cpu()
                    if capture_cache is not None:
                        source_state = capture_cache.layers[layer_index].recurrent_states
                        if source_state is None:
                            raise ContractError(
                                f"GDN teacher layer {layer_index} did not expose its recurrent state"
                            )
                        captured_state = source_state.detach().cpu()
                del loaded
            variance = hidden_states.float().square().mean(-1, keepdim=True)
            hidden_states = (
                hidden_states.float() * torch.rsqrt(variance + self.rms_norm_eps)
            ) * (1.0 + self.norm_weight.float())
            hidden_states = hidden_states.to(self.dtype)
            logits = functional.linear(hidden_states, self.lm_head_weight)
        return StreamedTeacherOutput(
            logits,
            captured_input,
            captured_mixer,
            captured_block,
            captured_state,
        )


class StreamedQwen35HybridExecutor:
    """Run a progressive hybrid with reloadable frozen Qwen suffix layers."""

    def __init__(self, teacher: StreamedQwen35Teacher) -> None:
        self.teacher = teacher

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        active_layer_index: int,
        active_mixer: ProjectionBoundaryRWKV7Attention,
        converted_layer_indices: set[int],
        frozen_mixer_provider: Callable[[int], ProjectionBoundaryRWKV7Attention],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> StreamedHybridOutput:
        num_layers = self.teacher.loader.tensor_store.num_layers
        if not 0 <= active_layer_index < num_layers:
            raise ContractError(f"active streamed layer out of range: {active_layer_index}")
        if active_layer_index in converted_layer_indices:
            raise ContractError("active layer must not also be listed as a frozen converted layer")
        if any(not 0 <= index < num_layers for index in converted_layer_indices):
            raise ContractError("streamed converted layer index is out of range")
        if attention_mask is not None and not torch.all(attention_mask.to(torch.bool)):
            raise ContractError(
                "RWKV7 distillation uses fixed packed rows and does not permit padding"
            )
        hidden_states, position_ids, position_embeddings, causal_mask = self._prepare_inputs(
            input_ids, attention_mask, position_ids
        )
        context = HybridRecurrentContext()
        with torch.no_grad():
            if active_layer_index > 0 and 0 not in converted_layer_indices:
                # The Qwen source prefix does not expose RWKV7's cross-layer
                # v_first stream. Run only a frozen layer-0 RWKV7 shadow block
                # on the aligned layer-0 input; discard its block output and
                # retain the detached v_first stream for the active mixer.
                _, shadow_adapter = self._run_loaded_layer(
                    0,
                    hidden_states.detach(),
                    position_ids,
                    position_embeddings,
                    causal_mask,
                    mixer=frozen_mixer_provider(0),
                    context=context,
                    is_active=False,
                )
                if shadow_adapter is None or context.v_first is None:
                    raise ContractError(
                        "frozen RWKV7 layer-0 shadow did not produce v_first"
                    )
                context.v_first = context.v_first.detach()
            for layer_index in range(active_layer_index):
                mixer = (
                    frozen_mixer_provider(layer_index)
                    if layer_index in converted_layer_indices
                    else None
                )
                hidden_states, _ = self._run_loaded_layer(
                    layer_index,
                    hidden_states,
                    position_ids,
                    position_embeddings,
                    causal_mask,
                    mixer=mixer,
                    context=context,
                    is_active=False,
                )
        active_input = hidden_states.detach()
        hidden_states, active_adapter = self._run_loaded_layer(
            active_layer_index,
            active_input,
            position_ids,
            position_embeddings,
            causal_mask,
            mixer=active_mixer,
            context=context,
            is_active=True,
        )
        assert active_adapter is not None
        active_block_output = hidden_states
        for layer_index in range(active_layer_index + 1, num_layers):
            if layer_index in converted_layer_indices:
                if context.v_first is None:
                    raise ContractError(
                        "converted RWKV7 suffix requires the layer-0 v_first stream"
                    )
                hidden_states = checkpoint(
                    lambda value, v_first, index=layer_index: self._run_reloadable_converted_layer(
                        index,
                        value,
                        v_first,
                        position_ids,
                        position_embeddings,
                        causal_mask,
                        frozen_mixer_provider,
                    ),
                    hidden_states,
                    context.v_first,
                    use_reentrant=True,
                )
            else:
                hidden_states = checkpoint(
                    lambda value, index=layer_index: self._run_reloadable_source_layer(
                        index,
                        value,
                        position_ids,
                        position_embeddings,
                        causal_mask,
                    ),
                    hidden_states,
                    use_reentrant=True,
                )
        logits = self._finalize_logits(hidden_states)
        if (
            active_adapter.last_output is None
            or active_adapter.last_state is None
            or active_adapter.last_signals is None
        ):
            raise ContractError("active streamed RWKV7 adapter did not expose training signals")
        return StreamedHybridOutput(
            logits,
            active_input,
            active_adapter.last_output,
            active_block_output,
            active_adapter.last_state,
            active_adapter.last_signals,
        )

    def _prepare_inputs(self, input_ids, attention_mask, position_ids):
        input_ids = input_ids.to(self.teacher.device)
        hidden_states = functional.embedding(input_ids, self.teacher.embedding_weight)
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.shape[-1], device=self.teacher.device, dtype=torch.long
            ).unsqueeze(0).expand(input_ids.shape[0], -1)
        else:
            position_ids = position_ids.to(self.teacher.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.teacher.device)
        position_embeddings = self.teacher.rotary(hidden_states, position_ids)
        causal_mask = create_causal_mask(
            self.teacher.loader.config,
            hidden_states,
            attention_mask,
            None,
            position_ids,
        )
        return hidden_states, position_ids, position_embeddings, causal_mask

    def _run_loaded_layer(
        self,
        layer_index,
        hidden_states,
        position_ids,
        position_embeddings,
        causal_mask,
        *,
        mixer,
        context,
        is_active,
    ):
        with self.teacher.loader.layer_lease(layer_index):
            loaded = self.teacher.loader.load_layer(
                layer_index,
                device=self.teacher.device,
                dtype=self.teacher.dtype,
            )
            adapter = None
            original_mixer = None
            attribute = None
            if mixer is not None:
                if hasattr(loaded.module, "linear_attn"):
                    attribute, returns_tuple = "linear_attn", False
                elif hasattr(loaded.module, "self_attn"):
                    attribute, returns_tuple = "self_attn", True
                else:
                    raise ContractError(f"streamed Qwen layer {layer_index} has no sequence mixer")
                mixer.to(device=self.teacher.device, dtype=self.teacher.dtype)
                mixer.eval()
                if not is_active:
                    mixer.requires_grad_(False)
                adapter = QwenRWKV7MixerAdapter(
                    mixer,
                    returns_attention_tuple=returns_tuple,
                    context=context,
                )
                adapter.eval()
                if not is_active:
                    adapter.requires_grad_(False)
                original_mixer = getattr(loaded.module, attribute)
                setattr(loaded.module, attribute, adapter)
            try:
                output = loaded.module(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
            finally:
                if attribute is not None:
                    setattr(loaded.module, attribute, original_mixer)
        if isinstance(output, tuple):
            output = output[0]
        return output, adapter

    def _run_reloadable_source_layer(
        self,
        layer_index,
        hidden_states,
        position_ids,
        position_embeddings,
        causal_mask,
    ):
        output, _ = self._run_loaded_layer(
            layer_index,
            hidden_states,
            position_ids,
            position_embeddings,
            causal_mask,
            mixer=None,
            context=HybridRecurrentContext(),
            is_active=False,
        )
        return output

    def _run_reloadable_converted_layer(
        self,
        layer_index,
        hidden_states,
        v_first,
        position_ids,
        position_embeddings,
        causal_mask,
        frozen_mixer_provider,
    ):
        output, _ = self._run_loaded_layer(
            layer_index,
            hidden_states,
            position_ids,
            position_embeddings,
            causal_mask,
            mixer=frozen_mixer_provider(layer_index),
            context=HybridRecurrentContext(v_first=v_first),
            is_active=False,
        )
        return output

    def _finalize_logits(self, hidden_states):
        variance = hidden_states.float().square().mean(-1, keepdim=True)
        hidden_states = hidden_states.float() * torch.rsqrt(
            variance + self.teacher.rms_norm_eps
        )
        hidden_states = hidden_states * (1.0 + self.teacher.norm_weight.float())
        return functional.linear(
            hidden_states.to(self.teacher.dtype), self.teacher.lm_head_weight
        )
