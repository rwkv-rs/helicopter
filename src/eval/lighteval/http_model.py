from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.utils.cache_management import SampleCache

from .http_pool import Completion, PoolManifest, VLLMHttpPool


@dataclass(frozen=True)
class _Job:
    document_index: int
    sample_index: int
    messages: list[dict[str, str]]
    parameters: dict[str, object]


class VLLMHttpModel(LightevalModel):
    def __init__(
        self,
        *,
        manifest: PoolManifest,
        cache_dir: Path,
        raw_prompt_template: str,
        generation_parameters: dict[str, object],
    ) -> None:
        self.pool = VLLMHttpPool(manifest)
        model_id = self.pool.preflight()
        cache_identity = hashlib.sha256(
            f"{model_id}\0{manifest.global_step}".encode()
        ).hexdigest()[:16]
        self.config = ModelConfig(
            model_name=f"vllm-http-{cache_identity}",
            cache_dir=str(cache_dir),
        )
        self._cache = SampleCache(self.config)
        self.prompt_manager = PromptManager(use_chat_template=True, tokenizer=None)
        self._raw_prompt_template = raw_prompt_template
        self._generation_parameters = generation_parameters

    @property
    def tokenizer(self):
        return None

    @property
    def add_special_tokens(self) -> bool:
        return False

    @property
    def max_length(self) -> int:
        return self.pool.manifest.max_model_len

    def greedy_until(self, docs) -> list[ModelResponse]:
        jobs: list[_Job] = []
        response_slots: list[list[Completion | None]] = []
        prompts: list[list[dict[str, str]]] = []
        for document_index, doc in enumerate(docs):
            if doc.use_logits:
                raise ValueError(
                    "vLLM HTTP evaluation does not support generation logits"
                )
            if (
                not isinstance(doc.num_samples, int)
                or isinstance(doc.num_samples, bool)
                or doc.num_samples <= 0
            ):
                raise ValueError("evaluation num_samples must be positive")
            messages = self.prompt_manager.prepare_prompt_api(doc)
            prompts.append(messages)
            response_slots.append([None] * doc.num_samples)
            parameters = dict(self._generation_parameters)
            parameters.update(
                max_completion_tokens=doc.generation_size,
                stop=doc.stop_sequences or None,
                chat_template_kwargs={
                    "rwkv_prompt_template": self._raw_prompt_template
                },
                return_token_ids=True,
            )
            for sample_index in range(doc.num_samples):
                jobs.append(
                    _Job(
                        document_index=document_index,
                        sample_index=sample_index,
                        messages=messages,
                        parameters=parameters,
                    )
                )

        def execute(job: _Job) -> tuple[_Job, Completion]:
            return job, self.pool.complete(job.messages, job.parameters)

        if jobs:
            workers = min(len(jobs), self.pool.total_capacity)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for job, completion in executor.map(execute, jobs):
                    response_slots[job.document_index][job.sample_index] = completion

        responses: list[ModelResponse] = []
        for messages, slots in zip(prompts, response_slots, strict=True):
            if any(completion is None for completion in slots):
                raise RuntimeError("vLLM HTTP evaluation returned incomplete samples")
            completions = [completion for completion in slots if completion is not None]
            responses.append(
                ModelResponse(
                    input=messages,
                    input_tokens=list(completions[0].prompt_token_ids),
                    text=[completion.text for completion in completions],
                    reasonings=[completion.reasoning for completion in completions],
                    output_tokens=[
                        list(completion.output_token_ids) for completion in completions
                    ],
                )
            )
        return responses

    def loglikelihood(self, docs) -> list[ModelResponse]:
        raise NotImplementedError("vLLM HTTP evaluation is generative only")

    def loglikelihood_rolling(self, docs) -> list[ModelResponse]:
        raise NotImplementedError("vLLM HTTP evaluation is generative only")

    def cleanup(self) -> None:
        self.pool.close()
