from __future__ import annotations

from pathlib import Path

from lighteval.models.model_input import GenerationParameters
from lighteval.models.vllm.vllm_model import VLLMModel, VLLMModelConfig
from vllm import LLM
from vllm.transformers_utils.configs.rwkv7 import build_rwkv7_config_from_pth


class RWKVGenerationParameters(GenerationParameters):
    penalty_decay: float = 0.988

    def to_vllm_dict(self) -> dict[str, object]:
        values = super().to_vllm_dict()
        values.pop("stop", None)
        values.update(
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=1.0,
            penalty_decay=self.penalty_decay,
            stop_token_ids=[0],
            ignore_eos=False,
        )
        return values


class RWKVModelConfig(VLLMModelConfig):
    generation_parameters: RWKVGenerationParameters
    rwkv_prompt_template: str
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None


class PromptRenderer:
    def __init__(self, tokenizer, raw_prompt_template: str) -> None:
        self._tokenizer = tokenizer
        self._raw_prompt_template = raw_prompt_template

    def apply_chat_template(self, *args, **kwargs):
        kwargs["rwkv_prompt_template"] = self._raw_prompt_template
        return self._tokenizer.apply_chat_template(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


class RWKVModel(VLLMModel):
    def __init__(self, model_config) -> None:
        super().__init__(model_config)
        self.prompt_manager.tokenizer = PromptRenderer(
            self.tokenizer,
            model_config.rwkv_prompt_template,
        )

    def _create_auto_model(self, model_config):
        self.model_args = {
            "model": model_config.model_name,
            "gpu_memory_utilization": model_config.gpu_memory_utilization,
            "enable_prefix_caching": False,
            "revision": model_config.revision
            + (
                f"/{model_config.subfolder}"
                if model_config.subfolder is not None
                else ""
            ),
            "dtype": model_config.dtype,
            "trust_remote_code": model_config.trust_remote_code,
            "tensor_parallel_size": model_config.tensor_parallel_size,
            "pipeline_parallel_size": model_config.pipeline_parallel_size,
            "max_model_len": self._max_length,
            "hf_overrides": {"model_max_length": self._max_length},
            "swap_space": model_config.swap_space,
            "seed": int(model_config.seed),
            "enforce_eager": True,
        }
        if model_config.data_parallel_size > 1:
            raise ValueError("RWKV evaluation does not use data parallel models")
        if model_config.quantization is not None:
            self.model_args["quantization"] = model_config.quantization
        if model_config.load_format is not None:
            self.model_args["load_format"] = model_config.load_format
        return LLM(**self.model_args)

    def _greedy_until(self, docs):
        uses_chat_template = self.use_chat_template
        self.use_chat_template = False
        try:
            return super()._greedy_until(docs)
        finally:
            self.use_chat_template = uses_chat_template


def create_local_model(
    *,
    weight: Path,
    output_dir: Path,
    raw_prompt_template: str,
    stop: str,
    max_new_tokens: int,
) -> RWKVModel:
    checkpoint = build_rwkv7_config_from_pth(str(weight))
    if checkpoint is None:
        raise ValueError("weight is not a supported RWKV7 checkpoint")
    model_config = RWKVModelConfig(
        model_name=weight.as_uri(),
        cache_dir=str(output_dir / "cache"),
        dtype="float16",
        max_model_length=checkpoint.max_position_embeddings + max_new_tokens,
        max_num_seqs=None,
        max_num_batched_tokens=None,
        enable_prefix_caching=False,
        override_chat_template=True,
        rwkv_prompt_template=raw_prompt_template,
        generation_parameters=RWKVGenerationParameters(
            temperature=0.96,
            top_p=0.76,
            top_k=32,
            presence_penalty=1.0,
            frequency_penalty=0.1,
            penalty_decay=0.988,
            stop_tokens=[stop],
            max_new_tokens=max_new_tokens,
        ),
    )
    return RWKVModel(model_config)
