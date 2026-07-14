from __future__ import annotations

from dataclasses import dataclass
from types import MethodType

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .errors import ContractError
from .kernel import load_rwkv_lm_kernel
from .mixer import ProjectionBoundaryRWKV7Attention


def _valid_tokens(attention_mask: Tensor | None, hidden: Tensor) -> Tensor:
    batch, tokens = hidden.shape[:2]
    if attention_mask is None:
        return torch.ones(batch, tokens, dtype=torch.bool, device=hidden.device)
    if attention_mask.ndim == 2:
        return attention_mask[:, -tokens:].to(torch.bool)
    if attention_mask.ndim == 4:
        query_rows = attention_mask[:, 0, -tokens:, -tokens:]
        return (torch.diagonal(query_rows, dim1=-2, dim2=-1) >= 0).to(torch.bool)
    raise ContractError(f"unsupported hybrid attention mask rank: {attention_mask.ndim}")


class QwenRWKV7MixerAdapter(nn.Module):
    """Make one native RWKV7 mixer obey either Qwen GDN or attention API."""

    def __init__(
        self,
        rwkv: ProjectionBoundaryRWKV7Attention,
        *,
        returns_attention_tuple: bool,
        context: "HybridRecurrentContext",
    ):
        super().__init__()
        self.rwkv = rwkv
        self.returns_attention_tuple = returns_attention_tuple
        self.context = context
        self.last_state: Tensor | None = None
        self.last_signals: dict[str, Tensor] | None = None
        self.last_output: Tensor | None = None

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        cache_params=None,
        past_key_values=None,
        **kwargs,
    ):
        if cache_params is not None or past_key_values is not None:
            raise ContractError("hybrid distillation adapter requires use_cache=False")
        batch, tokens, hidden = hidden_states.shape
        if position_ids is None:
            position_ids = torch.arange(tokens, device=hidden_states.device).view(1, -1).expand(batch, -1)
        valid = _valid_tokens(attention_mask, hidden_states)
        if not torch.all(valid):
            raise ContractError(
                "RWKV7 distillation uses fixed packed rows and does not permit padding"
            )
        if hidden_states.is_cuda and hidden_states.dtype == torch.bfloat16:
            if torch.any(valid[:, 1:].to(torch.int8) > valid[:, :-1].to(torch.int8)):
                raise ContractError(
                    "native RWKV7 training kernel requires right-padded contiguous sequences"
                )
            output, candidate_v_first, state, signals = self.rwkv.forward_sequence(
                hidden_states,
                positions=position_ids,
                kernel=load_rwkv_lm_kernel(),
                v_first=self.context.v_first,
            )
            output = torch.where(valid[..., None], output, torch.zeros_like(output))
            self.last_state = state
            self.last_signals = signals
            if self.rwkv.layer_idx == 0:
                self.context.v_first = candidate_v_first
            self.last_output = output
            return (output, None) if self.returns_attention_tuple else output
        state = torch.zeros(
            batch,
            self.rwkv.num_heads,
            self.rwkv.head_dim,
            self.rwkv.head_dim,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        previous = torch.zeros(batch, hidden, device=hidden_states.device, dtype=hidden_states.dtype)
        outputs: list[Tensor] = []
        signal_rows: dict[str, list[Tensor]] = {}
        v_first_rows: list[Tensor] = []
        if self.rwkv.layer_idx and self.context.v_first is None:
            raise ContractError(
                "nonzero RWKV7 layers require the aligned frozen layer-0 v_first stream"
            )
        for token in range(tokens):
            if self.rwkv.layer_idx:
                v_first = self.context.v_first[:, token]
            else:
                v_first = torch.zeros(batch, hidden, device=hidden_states.device, dtype=hidden_states.dtype)
            old_state, old_previous = state, previous
            output, candidate_previous, candidate_state, candidate_v_first, signals = self.rwkv(
                hidden_states[:, token],
                previous,
                v_first,
                state,
                positions=position_ids[:, token],
            )
            vector_mask = valid[:, token, None]
            state_mask = valid[:, token, None, None, None]
            state = torch.where(state_mask, candidate_state, old_state)
            previous = torch.where(vector_mask, candidate_previous, old_previous)
            outputs.append(torch.where(vector_mask, output, torch.zeros_like(output)))
            v_first_rows.append(torch.where(vector_mask, candidate_v_first, v_first))
            for name, value in signals.items():
                signal_rows.setdefault(name, []).append(value)
        self.last_state = state
        self.last_signals = {name: torch.stack(values, dim=1) for name, values in signal_rows.items()}
        if self.rwkv.layer_idx == 0:
            self.context.v_first = torch.stack(v_first_rows, dim=1)
        result = torch.stack(outputs, dim=1)
        self.last_output = result
        return (result, None) if self.returns_attention_tuple else result


@dataclass
class PatchedLayer:
    index: int
    source_kind: str
    attribute: str
    original: nn.Module
    adapter: QwenRWKV7MixerAdapter


@dataclass
class HybridRecurrentContext:
    v_first: Tensor | None = None


class HybridModelPatcher:
    """Patch teacher mixers in-place while retaining the frozen suffix graph."""

    def __init__(self, teacher: nn.Module, mixers: list[ProjectionBoundaryRWKV7Attention]):
        self.teacher = teacher.eval().requires_grad_(False)
        self.layers = self._layers(teacher)
        if len(self.layers) != len(mixers):
            raise ContractError("teacher/student mixer layer counts differ")
        self.records: list[PatchedLayer] = []
        self.context = HybridRecurrentContext()
        self._v_first_shadow_handle = None
        self._original_layer_forwards = [layer.forward for layer in self.layers]
        for index, (layer, mixer) in enumerate(zip(self.layers, mixers, strict=True)):
            if hasattr(layer, "linear_attn"):
                attribute, source_kind, returns_tuple = "linear_attn", "linear_attention", False
            elif hasattr(layer, "self_attn"):
                attribute, source_kind, returns_tuple = "self_attn", "full_attention", True
            else:
                raise ContractError(f"teacher layer {index} has no recognized mixer")
            original = getattr(layer, attribute)
            adapter = QwenRWKV7MixerAdapter(
                mixer, returns_attention_tuple=returns_tuple, context=self.context
            )
            adapter.eval().requires_grad_(False)
            self.records.append(PatchedLayer(index, source_kind, attribute, original, adapter))

    @staticmethod
    def _layers(model: nn.Module) -> list[nn.Module]:
        base = getattr(model, "model", None)
        value = getattr(base, "layers", None)
        if isinstance(value, nn.ModuleList):
            return list(value)
        value = getattr(getattr(base, "language_model", None), "layers", None)
        if isinstance(value, nn.ModuleList):
            return list(value)
        raise ContractError("teacher model does not expose Qwen3.5 text layers")

    def configure(
        self,
        *,
        active_layer: int,
        converted_prefix: int | None = None,
        converted_layers: set[int] | None = None,
        reset_gradients: bool = True,
        checkpoint_suffix: bool = False,
    ) -> QwenRWKV7MixerAdapter:
        if not 0 <= active_layer < len(self.records):
            raise ContractError(f"active layer out of range: {active_layer}")
        if (converted_prefix is None) == (converted_layers is None):
            raise ContractError(
                "configure requires exactly one of converted_prefix or converted_layers"
            )
        if converted_prefix is not None:
            if not 0 <= converted_prefix <= active_layer:
                raise ContractError("converted_prefix must end at or before active_layer")
            frozen_students = set(range(converted_prefix))
        else:
            frozen_students = set(converted_layers or ())
            if any(not 0 <= index < len(self.records) for index in frozen_students):
                raise ContractError("converted_layers contains an out-of-range layer")
            frozen_students.discard(active_layer)
        self.context.v_first = None
        self._remove_v_first_shadow()
        self._restore_layer_forwards()
        for record in self.records:
            use_student = record.index in frozen_students or record.index == active_layer
            setattr(self.layers[record.index], record.attribute, record.adapter if use_student else record.original)
            record.adapter.requires_grad_(record.index == active_layer)
            if reset_gradients:
                for parameter in record.adapter.parameters():
                    parameter.grad = None
        if checkpoint_suffix:
            self._checkpoint_frozen_suffix(active_layer)
        if active_layer > 0 and 0 not in frozen_students:
            shadow = self.records[0].adapter.rwkv

            def populate_v_first(_module, args, kwargs):
                hidden = kwargs.get("hidden_states")
                if hidden is None and args:
                    hidden = args[0]
                if hidden is None:
                    raise ContractError("teacher layer-0 shadow could not resolve hidden_states")
                with torch.no_grad():
                    self.context.v_first = shadow.project_v_first_sequence(hidden).detach()

            self._v_first_shadow_handle = self.layers[0].register_forward_pre_hook(
                populate_v_first,
                with_kwargs=True,
            )
        return self.records[active_layer].adapter

    def _remove_v_first_shadow(self) -> None:
        if self._v_first_shadow_handle is not None:
            self._v_first_shadow_handle.remove()
            self._v_first_shadow_handle = None

    def _restore_layer_forwards(self) -> None:
        for layer, original in zip(
            self.layers, self._original_layer_forwards, strict=True
        ):
            layer.forward = original

    def _checkpoint_frozen_suffix(self, active_layer: int) -> None:
        """Recompute frozen suffix activations while every teacher module stays eval."""
        for index in range(active_layer + 1, len(self.layers)):
            layer = self.layers[index]
            original = self._original_layer_forwards[index]

            def checkpointed(_module, *args, _original=original, **kwargs):
                if not torch.is_grad_enabled():
                    return _original(*args, **kwargs)
                return checkpoint(
                    _original,
                    *args,
                    use_reentrant=False,
                    **kwargs,
                )

            layer.forward = MethodType(checkpointed, layer)

    def restore(self) -> None:
        self._remove_v_first_shadow()
        self._restore_layer_forwards()
        for record in self.records:
            setattr(self.layers[record.index], record.attribute, record.original)
            record.adapter.requires_grad_(False)

    def layout(
        self,
        *,
        active_layer: int,
        converted_prefix: int | None = None,
        converted_layers: set[int] | None = None,
    ) -> list[str]:
        if converted_prefix is not None:
            frozen_students = set(range(converted_prefix))
        elif converted_layers is not None:
            frozen_students = set(converted_layers)
        else:
            raise ContractError("layout requires converted_prefix or converted_layers")
        return [
            "rwkv7-active"
            if index == active_layer
            else "rwkv7-frozen"
            if index in frozen_students
            else record.source_kind
            for index, record in enumerate(self.records)
        ]
