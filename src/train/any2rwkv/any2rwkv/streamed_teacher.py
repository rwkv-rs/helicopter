from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import importlib
import threading

import torch
import torch.nn.functional as functional
from torch import nn
from transformers.masking_utils import create_causal_mask

from .checkpoint import CheckpointManifest
from .errors import ContractError
from .layer_store import LayerTensorStore
from .hybrid import HybridRecurrentContext, QwenRWKV7MixerAdapter
from .mixer import ProjectionBoundaryRWKV7Attention
from .migration import qwen35_l2_normalize


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


@dataclass(frozen=True)
class StreamedLayerLocalOutput:
    active_layer_input: torch.Tensor
    student_mixer_output: torch.Tensor
    student_block_output: torch.Tensor
    student_signals: dict[str, torch.Tensor]
    teacher_signals: dict[str, torch.Tensor] | None
    teacher_mixer_output: torch.Tensor
    teacher_block_output: torch.Tensor
    shared_states: torch.Tensor | None


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
        self._layer_load_count = 0
        self._cache_hit_count = 0
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
            self._cache_hit_count += 1
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
        if target_device.type != "cuda":
            linear_attention = getattr(module, "linear_attn", None)
            if linear_attention is not None:
                modeling = importlib.import_module(type(module).__module__)
                linear_attention.causal_conv1d_fn = None
                linear_attention.causal_conv1d_update = getattr(
                    modeling, "torch_causal_conv1d_update"
                )
                linear_attention.chunk_gated_delta_rule = getattr(
                    modeling, "torch_chunk_gated_delta_rule"
                )
                linear_attention.recurrent_gated_delta_rule = getattr(
                    modeling, "torch_recurrent_gated_delta_rule"
                )
                norm_classes = [
                    getattr(modeling, name)
                    for name in dir(modeling)
                    if name.endswith("RMSNormGated") and not name.startswith("Fused")
                ]
                if len(norm_classes) != 1:
                    raise ContractError(
                        "could not resolve the unique Qwen3.5 torch RMSNormGated fallback"
                    )
                fallback_norm = norm_classes[0](
                    linear_attention.head_v_dim,
                    eps=linear_attention.layer_norm_epsilon,
                ).to(device=target_device, dtype=dtype)
                fallback_norm.load_state_dict(linear_attention.norm.state_dict())
                linear_attention.norm = fallback_norm
        module.eval().requires_grad_(False)
        loaded = LoadedTeacherLayer(layer_index, module, source_tensor_bytes)
        self._layer_load_count += 1
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

    @property
    def layer_load_count(self) -> int:
        return self._layer_load_count

    @property
    def cache_hit_count(self) -> int:
        return self._cache_hit_count


class StreamedQwen35Teacher:
    """Execute a Qwen3.5 teacher while keeping at most one decoder layer loaded."""

    def __init__(
        self,
        checkpoint: CheckpointManifest,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        cache_layers: bool = False,
        load_output_head: bool = True,
    ) -> None:
        self.loader = Qwen35TeacherLayerLoader(
            checkpoint,
            cache_layers=cache_layers,
        )
        self.device = torch.device(device)
        self.dtype = dtype
        global_names = ["model.embed_tokens.weight"]
        if load_output_head:
            global_names.append("model.norm.weight")
            if self.loader.tensor_store.has_tensor("lm_head.weight"):
                global_names.append("lm_head.weight")
        tensors = self.loader.tensor_store.load_named_tensors(tuple(global_names))
        self.embedding_weight = tensors["model.embed_tokens.weight"].to(
            device=self.device, dtype=dtype
        )
        self.norm_weight = (
            tensors["model.norm.weight"].to(device=self.device, dtype=dtype)
            if load_output_head
            else None
        )
        self.lm_head_weight = (
            tensors.get("lm_head.weight", tensors["model.embed_tokens.weight"]).to(
                device=self.device, dtype=dtype
            )
            if load_output_head
            else None
        )
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
            *(value for value in (self.norm_weight, self.lm_head_weight) if value is not None),
            *self.rotary.parameters(),
            *self.rotary.buffers(),
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return functional.embedding(input_ids.to(self.device), self.embedding_weight)

    def project_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.norm_weight is None or self.lm_head_weight is None:
            raise ContractError("logit projection requires load_output_head=True")
        variance = hidden_states.float().square().mean(-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(
            variance + self.rms_norm_eps
        )
        normalized = normalized * (1.0 + self.norm_weight.float())
        return functional.linear(normalized.to(self.dtype), self.lm_head_weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        capture_layer_index: int | None = None,
    ) -> StreamedTeacherOutput:
        if self.norm_weight is None or self.lm_head_weight is None:
            raise ContractError("full teacher forward requires load_output_head=True")
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
        # Cached teacher modules can be reused by local training. Avoid
        # inference-mode tensors because the same parameters later participate
        # in a student block forward whose input must retain autograd.
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
                        hidden_states = loaded.module(
                            hidden_states,
                            position_embeddings=position_embeddings,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_values=None,
                            use_cache=False,
                        )
                        if isinstance(hidden_states, tuple):
                            hidden_states = hidden_states[0]
                    finally:
                        if hook is not None:
                            hook.remove()
                if layer_index == capture_layer_index:
                    captured_block = hidden_states.detach().cpu()
                del loaded
            logits = self.project_logits(hidden_states)
        return StreamedTeacherOutput(
            logits,
            captured_input,
            captured_mixer,
            captured_block,
        )


class StreamedQwen35HybridExecutor:
    """Run a suffix-free local teacher/student comparison for one layer."""

    def __init__(self, teacher: StreamedQwen35Teacher) -> None:
        self.teacher = teacher

    def forward_cached_layer_local(
        self,
        hidden_states: torch.Tensor,
        *,
        active_layer_index: int,
        active_mixer: ProjectionBoundaryRWKV7Attention,
        loaded_layer: LoadedTeacherLayer,
        shared_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> StreamedLayerLocalOutput:
        """Compare one resident source layer and target mixer on cached inputs."""
        if loaded_layer.layer_index != active_layer_index:
            raise ContractError("resident Qwen layer index differs from active layer")
        hidden_states, position_ids, position_embeddings, causal_mask = (
            self._prepare_cached_inputs(hidden_states, attention_mask, position_ids)
        )
        active_input = hidden_states.detach()
        context = HybridRecurrentContext(
            v_first=(None if shared_states is None else shared_states.to(self.teacher.device))
        )
        teacher_mixer: torch.Tensor | None = None
        teacher_signals: dict[str, torch.Tensor] | None = None
        source_decay_input: torch.Tensor | None = None
        source_qkv_input: torch.Tensor | None = None
        source_gate_output: torch.Tensor | None = None
        source_beta_logits: torch.Tensor | None = None
        source_pre_output: torch.Tensor | None = None
        source_mixer_input: torch.Tensor | None = None
        source_mixer = getattr(loaded_layer.module, "linear_attn", None) or getattr(
            loaded_layer.module, "self_attn", None
        )

        def capture_source(_module, _args, output):
            nonlocal teacher_mixer
            teacher_mixer = (output[0] if isinstance(output, tuple) else output).detach()

        def capture_source_input(_module, args, kwargs):
            nonlocal source_mixer_input
            value = kwargs.get("hidden_states")
            if value is None and args:
                value = args[0]
            if value is None:
                raise ContractError(
                    "source mixer pre-hook did not receive normalized hidden states"
                )
            source_mixer_input = value.detach()

        def capture_source_decay_input(_module, _args, output):
            nonlocal source_decay_input
            source_decay_input = output.detach()

        def capture_source_qkv_input(_module, _args, output):
            nonlocal source_qkv_input
            source_qkv_input = output.detach()

        def capture_source_gate_output(_module, _args, output):
            nonlocal source_gate_output
            source_gate_output = output.detach()

        def capture_source_beta_logits(_module, _args, output):
            nonlocal source_beta_logits
            source_beta_logits = output.detach()

        def capture_source_pre_output(_module, args):
            nonlocal source_pre_output
            source_pre_output = args[0].detach()

        with self.teacher.loader.layer_lease(active_layer_index):
            input_hook = source_mixer.register_forward_pre_hook(
                capture_source_input,
                with_kwargs=True,
            )
            hook = source_mixer.register_forward_hook(capture_source)
            decay_hook = (
                source_mixer.in_proj_a.register_forward_hook(
                    capture_source_decay_input
                )
                if hasattr(source_mixer, "in_proj_a")
                else None
            )
            conv_hook = (
                source_mixer.in_proj_qkv.register_forward_hook(capture_source_qkv_input)
                if hasattr(source_mixer, "in_proj_qkv")
                else None
            )
            gate_hook = (
                source_mixer.in_proj_z.register_forward_hook(capture_source_gate_output)
                if hasattr(source_mixer, "in_proj_z")
                else None
            )
            beta_hook = (
                source_mixer.in_proj_b.register_forward_hook(
                    capture_source_beta_logits
                )
                if hasattr(source_mixer, "in_proj_b")
                else None
            )
            source_output_projection = getattr(source_mixer, "out_proj", None)
            if source_output_projection is None:
                source_output_projection = getattr(source_mixer, "o_proj", None)
            if source_output_projection is None:
                raise ContractError("source mixer does not expose an output projection")
            output_hook = source_output_projection.register_forward_pre_hook(
                capture_source_pre_output
            )
            try:
                with torch.inference_mode():
                    teacher_block = loaded_layer.module(
                        active_input,
                        position_embeddings=position_embeddings,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_values=None,
                        use_cache=False,
                    )
                if isinstance(teacher_block, tuple):
                    teacher_block = teacher_block[0]
            finally:
                input_hook.remove()
                hook.remove()
                if decay_hook is not None:
                    decay_hook.remove()
                if conv_hook is not None:
                    conv_hook.remove()
                if gate_hook is not None:
                    gate_hook.remove()
                if beta_hook is not None:
                    beta_hook.remove()
                output_hook.remove()
            if source_decay_input is not None:
                source_decay = torch.exp(
                    -source_mixer.A_log.float().exp()
                    * torch.nn.functional.softplus(
                        source_decay_input.float() + source_mixer.dt_bias.float()
                    )
                )
                teacher_signals = {"decay": source_decay.detach()}
                if (
                    source_qkv_input is None
                    or source_gate_output is None
                    or source_beta_logits is None
                ):
                    raise ContractError(
                        "source GDN did not expose conv1d/QKV, gate, and beta traces"
                    )
                sequence_length = active_input.shape[1]
                projected = torch.nn.functional.conv1d(
                    source_qkv_input.transpose(1, 2),
                    source_mixer.conv1d.weight,
                    source_mixer.conv1d.bias,
                    padding=source_mixer.conv_kernel_size - 1,
                    groups=source_mixer.conv_dim,
                )[:, :, :sequence_length]
                projected = torch.nn.functional.silu(projected).transpose(1, 2)
                query, key, value = torch.split(
                    projected,
                    [source_mixer.key_dim, source_mixer.key_dim, source_mixer.value_dim],
                    dim=-1,
                )
                query = query.view(
                    *query.shape[:2], source_mixer.num_k_heads, source_mixer.head_k_dim
                )
                key = key.view(
                    *key.shape[:2], source_mixer.num_k_heads, source_mixer.head_k_dim
                )
                value = value.view(
                    *value.shape[:2], source_mixer.num_v_heads, source_mixer.head_v_dim
                )
                repeat = source_mixer.num_v_heads // source_mixer.num_k_heads
                if repeat > 1:
                    query = query.repeat_interleave(repeat, dim=2)
                    key = key.repeat_interleave(repeat, dim=2)
                beta = torch.sigmoid(source_beta_logits)
                recurrent_r = qwen35_l2_normalize(query.float()).to(
                    query.dtype
                ) * (source_mixer.head_k_dim ** -0.5)
                write_key = qwen35_l2_normalize(key.float()).to(key.dtype)
                write_value = value * beta.unsqueeze(-1)
                erase_rate = (beta * source_decay).repeat_interleave(
                    source_mixer.head_v_dim, dim=-1
                )
                teacher_signals.update(
                    {
                        "q": query.flatten(2).detach(),
                        "k": key.flatten(2).detach(),
                        "v": value.flatten(2).detach(),
                        "recurrent_r": recurrent_r.flatten(2).detach(),
                        "write_key": write_key.flatten(2).detach(),
                        "write_value": write_value.flatten(2).detach(),
                        "erase_rate": erase_rate.detach(),
                        "gate": torch.nn.functional.silu(source_gate_output).detach(),
                        "beta_logits": source_beta_logits,
                        "beta": beta,
                    }
                )
            elif hasattr(source_mixer, "q_proj"):
                if source_mixer_input is None:
                    raise ContractError(
                        "source full attention did not expose its normalized input"
                    )
                input_shape = source_mixer_input.shape[:-1]
                head_dim = int(source_mixer.head_dim)
                hidden_shape = (*input_shape, -1, head_dim)
                query, gate = torch.chunk(
                    source_mixer.q_proj(source_mixer_input).view(
                        *input_shape, -1, head_dim * 2
                    ),
                    2,
                    dim=-1,
                )
                query = source_mixer.q_norm(query.view(hidden_shape)).transpose(1, 2)
                key = source_mixer.k_norm(
                    source_mixer.k_proj(source_mixer_input).view(hidden_shape)
                ).transpose(1, 2)
                value = source_mixer.v_proj(source_mixer_input).view(
                    hidden_shape
                ).transpose(1, 2)
                query_pre_rope, key_pre_rope = query, key
                modeling = importlib.import_module(source_mixer.__class__.__module__)
                query, key = modeling.apply_rotary_pos_emb(
                    query, key, *position_embeddings
                )
                groups = int(source_mixer.num_key_value_groups)
                if groups > 1:
                    key = key.repeat_interleave(groups, dim=1)
                    key_pre_rope = key_pre_rope.repeat_interleave(groups, dim=1)
                    value = value.repeat_interleave(groups, dim=1)
                teacher_signals = {
                    "mixer_input": source_mixer_input.detach(),
                    "q": query_pre_rope.transpose(1, 2).flatten(2).detach(),
                    "k": key_pre_rope.transpose(1, 2).flatten(2).detach(),
                    "v": value.transpose(1, 2).flatten(2).detach(),
                    "q_post_rope": query.transpose(1, 2).flatten(2).detach(),
                    "k_post_rope": key.transpose(1, 2).flatten(2).detach(),
                    "gate": torch.sigmoid(gate.flatten(2)).detach(),
                }
            if teacher_signals is not None:
                if source_pre_output is None:
                    raise ContractError("source mixer did not expose pre-output trace")
                teacher_signals["pre_output"] = source_pre_output
            student_block, adapter = self._run_loaded_module(
                loaded_layer,
                active_input,
                position_ids,
                position_embeddings,
                causal_mask,
                mixer=active_mixer,
                context=context,
                is_active=True,
            )
        if teacher_mixer is None:
            raise ContractError("resident source layer did not expose its mixer output")
        if (
            adapter is None
            or adapter.last_output is None
            or adapter.last_signals is None
        ):
            raise ContractError("resident active layer did not expose local training signals")
        return StreamedLayerLocalOutput(
            active_input,
            adapter.last_output,
            student_block,
            adapter.last_signals,
            teacher_signals,
            teacher_mixer,
            teacher_block,
            context.v_first,
        )

    def forward_cached_target_block(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_index: int,
        mixer: ProjectionBoundaryRWKV7Attention,
        loaded_layer: LoadedTeacherLayer,
        shared_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Advance one finalized target block while preserving shared target state."""
        with torch.no_grad():
            return self.forward_cached_target_block_with_grad(
                hidden_states,
                layer_index=layer_index,
                mixer=mixer,
                loaded_layer=loaded_layer,
                shared_states=shared_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

    def forward_cached_target_block_with_grad(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_index: int,
        mixer: ProjectionBoundaryRWKV7Attention,
        loaded_layer: LoadedTeacherLayer,
        mixer_trainable: bool = False,
        shared_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run one target block while retaining gradient to its hidden input."""
        if loaded_layer.layer_index != layer_index:
            raise ContractError("resident Qwen layer index differs from cache-advance layer")
        hidden_states, position_ids, position_embeddings, causal_mask = (
            self._prepare_cached_inputs(hidden_states, attention_mask, position_ids)
        )
        context = HybridRecurrentContext(
            v_first=(None if shared_states is None else shared_states.to(self.teacher.device))
        )
        with self.teacher.loader.layer_lease(layer_index):
            output, adapter = self._run_loaded_module(
                loaded_layer,
                hidden_states,
                position_ids,
                position_embeddings,
                causal_mask,
                mixer=mixer,
                context=context,
                is_active=mixer_trainable,
            )
        if adapter is None:
            raise ContractError("cache-advance target block did not install its mixer")
        return output, context.v_first

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

    def _prepare_cached_inputs(self, hidden_states, attention_mask, position_ids):
        hidden_states = hidden_states.to(
            device=self.teacher.device,
            dtype=self.teacher.dtype,
            non_blocking=hidden_states.is_pinned(),
        )
        batch_size, sequence_length = hidden_states.shape[:2]
        if position_ids is None:
            position_ids = torch.arange(
                sequence_length, device=self.teacher.device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)
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
            return self._run_loaded_module(
                loaded,
                hidden_states,
                position_ids,
                position_embeddings,
                causal_mask,
                mixer=mixer,
                context=context,
                is_active=is_active,
            )

    def _run_loaded_module(
        self,
        loaded,
        hidden_states,
        position_ids,
        position_embeddings,
        causal_mask,
        *,
        mixer,
        context,
        is_active,
    ):
        adapter = None
        original_mixer = None
        attribute = None
        if mixer is not None:
            if hasattr(loaded.module, "linear_attn"):
                attribute, returns_tuple = "linear_attn", False
            elif hasattr(loaded.module, "self_attn"):
                attribute, returns_tuple = "self_attn", True
            else:
                raise ContractError(
                    f"streamed Qwen layer {loaded.layer_index} has no sequence mixer"
                )
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
