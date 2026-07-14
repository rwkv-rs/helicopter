from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TeacherTrace:
    layer_index: int
    layer_type: str
    layer_input: Tensor
    mixer_output: Tensor
    block_output: Tensor
    logits: Tensor | None
    position_ids: Tensor
    causal_mask: Tensor | None
    metadata: dict[str, object]


def trace_hash(tensors: dict[str, Tensor], metadata: dict[str, object]) -> str:
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, default=str).encode())
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class TeacherRunner:
    """Frozen/eval teacher that yields detached layer traces one at a time."""

    def __init__(self, model: nn.Module, *, source_hash: str, config_hash: str, data_hash: str) -> None:
        self.model = model.eval().requires_grad_(False)
        self.hashes = {"source_hash": source_hash, "config_hash": config_hash, "data_hash": data_hash}

    @staticmethod
    def _text_layers(model: nn.Module) -> list[nn.Module]:
        candidates = (
            getattr(getattr(model, "model", None), "layers", None),
            getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
        )
        for value in candidates:
            if isinstance(value, nn.ModuleList):
                return list(value)
        raise ValueError("teacher model does not expose Qwen3.5 text decoder layers")

    def capture_to_disk(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        layer_types: list[str],
        trace_dir: Path,
        include_logits: bool = True,
    ) -> tuple[Path, ...]:
        """Run the real teacher once and spill each completed layer trace.

        Hooks retain only the active layer's input/mixer output until its block
        output arrives. Completed traces are detached to CPU and written
        immediately, so all 60 layer activations are never resident together.
        """
        layers = self._text_layers(self.model)
        if len(layers) != len(layer_types):
            raise ValueError("teacher layer count and layer_types differ")
        trace_dir.mkdir(parents=True, exist_ok=True)
        pending: dict[int, dict[str, Tensor | None]] = {}
        paths: list[Path] = []
        hooks = []

        def pre_hook(index: int):
            def capture(module, args, kwargs):
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                if not isinstance(hidden, Tensor):
                    raise ValueError(f"teacher layer {index} did not receive hidden_states")
                pending[index] = {
                    "layer_input": hidden.detach().to("cpu"),
                    "causal_mask": kwargs.get("attention_mask").detach().to("cpu")
                    if isinstance(kwargs.get("attention_mask"), Tensor)
                    else None,
                }
            return capture

        def mixer_hook(index: int):
            def capture(module, args, output):
                value = output[0] if isinstance(output, tuple) else output
                pending[index]["mixer_output"] = value.detach().to("cpu")
            return capture

        def block_hook(index: int):
            def capture(module, args, kwargs, output):
                value = output[0] if isinstance(output, tuple) else output
                row = pending.pop(index)
                metadata = {
                    **self.hashes,
                    "layer_index": index,
                    "layer_type": layer_types[index],
                    "context_length": int(input_ids.shape[-1]),
                    "attention_mask_sha256": trace_hash(
                        {"attention_mask": attention_mask}, {"kind": "attention-mask"}
                    ),
                }
                trace = TeacherTrace(
                    index,
                    layer_types[index],
                    row["layer_input"],
                    row["mixer_output"],
                    value.detach().to("cpu"),
                    None,
                    position_ids.detach().to("cpu"),
                    row["causal_mask"],
                    metadata,
                )
                path = trace_dir / f"layer-{index:02d}.pt"
                digest = self.save(trace, path)
                metadata_path = trace_dir / f"layer-{index:02d}.sha256"
                metadata_path.write_text(digest + "\n", encoding="utf-8")
                paths.append(path)
            return capture

        for index, layer in enumerate(layers):
            hooks.append(layer.register_forward_pre_hook(pre_hook(index), with_kwargs=True))
            mixer = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
            if mixer is None:
                raise ValueError(f"teacher layer {index} has no recognized sequence mixer")
            hooks.append(mixer.register_forward_hook(mixer_hook(index)))
            hooks.append(layer.register_forward_hook(block_hook(index), with_kwargs=True))
        try:
            with torch.inference_mode():
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
            if include_logits:
                logits = output.logits.detach().to("cpu")
                torch.save(
                    {
                        "logits": logits,
                        "metadata": {**self.hashes, "context_length": int(input_ids.shape[-1])},
                    },
                    trace_dir / "logits.pt",
                )
        finally:
            for hook in hooks:
                hook.remove()
        if pending or len(paths) != len(layers):
            raise RuntimeError("teacher trace capture did not complete every layer")
        return tuple(paths)

    @staticmethod
    def save(trace: TeacherTrace, path: Path) -> str:
        payload = {
            "layer_input": trace.layer_input,
            "mixer_output": trace.mixer_output,
            "block_output": trace.block_output,
            "logits": trace.logits,
            "position_ids": trace.position_ids,
            "causal_mask": trace.causal_mask,
            "metadata": trace.metadata,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        tensors = {key: value for key, value in payload.items() if isinstance(value, Tensor)}
        return trace_hash(tensors, trace.metadata)
