from __future__ import annotations

import copy
import gc
import hashlib
import json
import importlib.metadata
import math
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

from .config import (
    EvaluationConfigurationError,
    EvaluationEnvironment,
    EvaluationPlan,
    EvaluationShard,
    EvaluationUnit,
    PROMPT_TEMPLATE_STOPS,
    PromptTemplate,
    build_plan,
    load_configured_registry,
    load_evaluation_config,
    load_evaluation_environment,
    public_environment,
    public_plan,
    resolve_weights,
    verify_weight_identity,
)
from .publish import (
    MANIFEST_VERSION,
    CampaignManifest,
    ManifestError,
    ManifestStore,
    ScoreboardClient,
    ScoreboardError,
    campaign_directory,
    create_campaign_directory,
    ensure_private_staging_root,
    publications_from_shard,
    remove_acknowledged_shard,
    remove_campaign_child_directory,
    remove_empty_campaign,
    run_preflight,
    validate_campaign_child,
    write_control_metadata,
)


MAX_NEW_TOKENS = 8192
PERPLEXITY_WINDOW_BATCH_SIZE = 32
_MARKUP = re.compile(r"\*\*|__|`+")
_BOXED = re.compile(
    r"\\boxed\{\s*(?:([A-Z])|\\(?:text|mathrm)\{\s*([A-Z])\s*\})\s*\}",
    re.I,
)
_EXPLICIT = re.compile(
    r"^\s*(?:(?:thus|therefore|hence|so)[,:]?\s+)?(?:the\s+)?"
    r"(?:(?:final|correct)\s+)?(?:answer|choice|option)\s*"
    r"(?:is|:|=)\s*([A-Z])\b",
    re.I | re.M,
)
_BARE = re.compile(
    r"^\s*(?:([A-Z])\.?|\(([A-Z])\)|\[([A-Z])\])\s*$",
    re.I,
)


def evaluation_max_model_length(checkpoint_context_length: int) -> int:
    """Reserve the checkpoint context window plus the fixed output budget."""
    if (
        isinstance(checkpoint_context_length, bool)
        or not isinstance(checkpoint_context_length, int)
        or checkpoint_context_length <= 0
    ):
        raise ValueError("RWKV checkpoint context length must be positive")
    return checkpoint_context_length + MAX_NEW_TOKENS


def _rolling_token_windows(
    token_ids: list[int],
    *,
    prefix_token_id: int,
    checkpoint_context_length: int,
) -> tuple[tuple[list[int], list[int]], ...]:
    """Partition a document so every token is scored exactly once."""
    if checkpoint_context_length < 3:
        raise ValueError(
            "RWKV checkpoint context length must be at least 3 for perplexity"
        )
    if not token_ids:
        return ()
    # vLLM logprob requests append one dummy generation token. The remaining
    # capacity follows the standard rolling-loglikelihood contract: one token
    # is context and every continuation token is scored exactly once.
    max_scored_tokens = checkpoint_context_length - 2
    windows: list[tuple[list[int], list[int]]] = []
    first_length = min(max_scored_tokens, len(token_ids))
    windows.append(([prefix_token_id], token_ids[:first_length]))
    scored = first_length
    while scored < len(token_ids):
        window_length = min(len(token_ids) - scored, max_scored_tokens)
        window_end = scored + window_length
        context_start = max(0, window_end - max_scored_tokens - 1)
        context = token_ids[context_start:scored]
        continuation = token_ids[scored:window_end]
        if not context:
            raise RuntimeError("rolling perplexity window lost its context")
        windows.append((context, continuation))
        scored = window_end
    return tuple(windows)


@dataclass(frozen=True)
class ShardEvaluation:
    shard: EvaluationShard
    path: Path
    model_execution: dict[str, object]

    def __iter__(self):
        yield self.shard
        yield self.path
        yield self.model_execution


@dataclass(frozen=True)
class ShardFailure:
    shard: EvaluationShard
    path: Path
    error_type: str
    error_phase: str
    error_site: str
    message: str


class UnsafeModelCleanupError(RuntimeError):
    """The process must stop because a model lifecycle was not closed safely."""


def _official_prompt_template(
    prompt_template: PromptTemplate,
) -> tuple[str, str]:
    from vllm.tokenizers.rwkv_defaults import RWKV_PROMPT_TEMPLATES

    matches = [
        template
        for template in RWKV_PROMPT_TEMPLATES.values()
        if template.style == prompt_template
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"vLLM-RWKV does not expose exactly one {prompt_template} prompt template"
        )
    template = matches[0]
    expected_stop = PROMPT_TEMPLATE_STOPS[prompt_template]
    if template.stop != expected_stop:
        raise RuntimeError(
            f"vLLM-RWKV {prompt_template} prompt stop does not match "
            "the Helicopter evaluation contract"
        )
    return template.name, template.stop


def _exception_type(error: BaseException) -> str:
    error_type = type(error)
    return f"{error_type.__module__}.{error_type.__qualname__}"


def _exception_site(error: BaseException) -> str:
    traceback = error.__traceback__
    if traceback is None:
        return "unknown"
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    frame = traceback.tb_frame
    module = frame.f_globals.get("__name__", "unknown")
    return f"{module}.{frame.f_code.co_qualname}"


@contextmanager
def _temporary_environment(
    values: Mapping[str, str],
) -> Iterator[None]:
    missing = object()
    previous: dict[str, str | object] = {
        key: os.environ.get(key, missing) for key in values
    }
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is missing:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _runtime_types():
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.metrics.metrics_sample import ExactMatches
    from lighteval.metrics.utils.metric_utils import SampleLevelMetric
    from lighteval.models.model_input import GenerationParameters
    from lighteval.models.vllm.vllm_model import VLLMModel, VLLMModelConfig
    from lighteval.pipeline import (
        ParallelismManager,
        Pipeline,
        PipelineParameters,
    )
    from lighteval.tasks.requests import SamplingMethod

    return {
        "EvaluationTracker": EvaluationTracker,
        "ExactMatches": ExactMatches,
        "SampleLevelMetric": SampleLevelMetric,
        "GenerationParameters": GenerationParameters,
        "VLLMModel": VLLMModel,
        "VLLMModelConfig": VLLMModelConfig,
        "ParallelismManager": ParallelismManager,
        "Pipeline": Pipeline,
        "PipelineParameters": PipelineParameters,
        "SamplingMethod": SamplingMethod,
    }


def _pipeline_parameters(types: dict[str, Any]):
    return types["PipelineParameters"](
        launcher_type=types["ParallelismManager"].VLLM,
        max_samples=None,
        remove_reasoning_tags=False,
        load_tasks_multilingual=True,
    )


def _build_runtime_classes(types: dict[str, Any]):
    GenerationParameters = types["GenerationParameters"]
    VLLMModelConfig = types["VLLMModelConfig"]
    VLLMModel = types["VLLMModel"]
    Pipeline = types["Pipeline"]
    SamplingMethod = types["SamplingMethod"]
    ExactMatches = types["ExactMatches"]
    SampleLevelMetric = types["SampleLevelMetric"]

    class RWKVGenerationParameters(GenerationParameters):
        penalty_decay: float = 0.988

        def to_vllm_dict(self) -> dict:
            backend = super().to_vllm_dict()
            # LightEval applies each generative document's stop sequences after
            # constructing SamplingParams. Keeping the config-level stop here
            # leaks it into logprob requests, which later disable detokenization
            # and are rejected by vLLM when the request is deserialized.
            backend.pop("stop", None)
            backend.update(
                repetition_penalty=1.0,
                frequency_penalty=self.frequency_penalty,
                penalty_decay=self.penalty_decay,
                stop_token_ids=[0],
                ignore_eos=False,
            )
            return backend

    class RWKVVLLMModelConfig(VLLMModelConfig):
        generation_parameters: RWKVGenerationParameters
        wkv_mode: str
        checkpoint_context_length: int
        rwkv_prompt_template: str
        rwkv_stop_sequence: str
        max_num_seqs: int | None = None
        max_num_batched_tokens: int | None = None

    class RWKVPromptRenderer:
        def __init__(self, tokenizer, prompt_template: str):
            self._tokenizer = tokenizer
            self._prompt_template = prompt_template

        def apply_chat_template(self, *args, **kwargs):
            requested = kwargs.get("rwkv_prompt_template")
            if requested is not None and requested != self._prompt_template:
                raise RuntimeError(
                    "LightEval attempted to override the campaign prompt template"
                )
            kwargs["rwkv_prompt_template"] = self._prompt_template
            return self._tokenizer.apply_chat_template(*args, **kwargs)

    class RWKVVLLMModel(VLLMModel):
        def __init__(self, config):
            self._unit_closed = False
            if config.checkpoint_context_length < 3:
                raise ValueError("RWKV checkpoint context length must be at least 3")
            self._checkpoint_context_length = config.checkpoint_context_length
            super().__init__(config)
            self.prompt_manager.tokenizer = RWKVPromptRenderer(
                self.tokenizer,
                config.rwkv_prompt_template,
            )
            self._cache = None

        def _create_auto_model(self, config):
            from vllm import LLM

            self.model_args = {
                "model": config.model_name,
                "gpu_memory_utilization": config.gpu_memory_utilization,
                "enable_prefix_caching": False,
                "revision": config.revision
                + (f"/{config.subfolder}" if config.subfolder is not None else ""),
                "dtype": config.dtype,
                "trust_remote_code": config.trust_remote_code,
                "tensor_parallel_size": config.tensor_parallel_size,
                "pipeline_parallel_size": config.pipeline_parallel_size,
                "max_model_len": self._max_length,
                # RWKV is recurrent and has no positional embedding ceiling.
                # LightEval interprets max_model_len as prompt + output, so
                # advertise the same total through the HF override while
                # retaining the filename-derived prompt context separately.
                "hf_overrides": {"model_max_length": self._max_length},
                "swap_space": config.swap_space,
                "seed": int(config.seed),
                "enforce_eager": True,
            }
            if config.quantization is not None:
                self.model_args["quantization"] = config.quantization
            if config.load_format is not None:
                self.model_args["load_format"] = config.load_format
            if config.data_parallel_size > 1:
                self.model_args["distributed_executor_backend"] = "ray"
                self._batch_size = "auto"
                return None
            model = LLM(**self.model_args)
            if self._max_length is None:
                self._max_length = model.llm_engine.model_config.max_seq_len_to_capture
            return model

        def cleanup(self):
            # LightEval Pipeline calls cleanup after every shard. The campaign
            # owner alone closes the shared model once the weight/mode unit
            # finishes.
            return None

        def close_unit(self):
            if self._unit_closed:
                return
            super().cleanup()
            self._unit_closed = True

        def reset_shard_state(self):
            self._cache = None
            gc.collect()

        def _greedy_until(self, docs):
            uses_chat_template = self.use_chat_template
            if not uses_chat_template:
                raise RuntimeError("RWKV evaluation requires its chat template")
            self.use_chat_template = False
            try:
                return super()._greedy_until(docs)
            finally:
                self.use_chat_template = uses_chat_template

        def loglikelihood_rolling(self, docs):
            from lighteval.models.model_output import ModelResponse

            documents = list(docs)
            document_tokens: list[list[int]] = []
            document_logprobs: list[list[float]] = []
            document_argmax: list[list[bool]] = []
            pending_inputs: list[list[int]] = []
            pending_windows: list[tuple[int, list[int]]] = []

            def flush_windows() -> None:
                if not pending_inputs:
                    return
                outputs = self._generate(
                    list(pending_inputs),
                    generate=False,
                )
                if len(outputs) != len(pending_windows):
                    raise RuntimeError(
                        "vLLM returned an unexpected number of perplexity windows"
                    )
                for output, (document_index, continuation) in zip(
                    outputs,
                    pending_windows,
                    strict=True,
                ):
                    prompt_token_ids = list(output.prompt_token_ids)
                    prompt_logprobs = output.prompt_logprobs
                    if len(prompt_token_ids) != len(prompt_logprobs):
                        raise RuntimeError(
                            "vLLM perplexity token and logprob counts differ"
                        )
                    continuation_logprobs = prompt_logprobs[
                        len(prompt_token_ids) - len(continuation) :
                    ]
                    if len(continuation_logprobs) != len(continuation):
                        raise RuntimeError(
                            "vLLM omitted perplexity continuation logprobs"
                        )
                    for token_id, position in zip(
                        continuation,
                        continuation_logprobs,
                        strict=True,
                    ):
                        if position is None or token_id not in position:
                            raise RuntimeError(
                                "vLLM omitted a perplexity token logprob"
                            )
                        token_logprob = position[token_id]
                        value = float(token_logprob.logprob)
                        if not math.isfinite(value):
                            raise RuntimeError(
                                "vLLM returned a non-finite perplexity logprob"
                            )
                        document_logprobs[document_index].append(value)
                        document_argmax[document_index].append(token_logprob.rank == 1)
                pending_inputs.clear()
                pending_windows.clear()

            for document_index, doc in enumerate(documents):
                if not isinstance(doc.query, str):
                    raise TypeError("perplexity documents must contain a string query")
                encoded = self.tokenizer(
                    doc.query,
                    add_special_tokens=False,
                )["input_ids"]
                token_ids = list(encoded)
                if any(
                    isinstance(token_id, bool) or not isinstance(token_id, int)
                    for token_id in token_ids
                ):
                    raise TypeError(
                        "perplexity tokenizer returned non-integer token ids"
                    )
                windows = _rolling_token_windows(
                    token_ids,
                    prefix_token_id=0,
                    checkpoint_context_length=(self._checkpoint_context_length),
                )
                document_tokens.append(token_ids)
                document_logprobs.append([])
                document_argmax.append([])
                for context, continuation in windows:
                    pending_inputs.append(context + continuation)
                    pending_windows.append((document_index, continuation))
                    if len(pending_inputs) == PERPLEXITY_WINDOW_BATCH_SIZE:
                        flush_windows()
            flush_windows()

            responses: list[ModelResponse] = []
            for doc, token_ids, logprobs, argmax in zip(
                documents,
                document_tokens,
                document_logprobs,
                document_argmax,
                strict=True,
            ):
                if len(logprobs) != len(token_ids) or len(argmax) != len(token_ids):
                    raise RuntimeError(
                        "perplexity windows did not score every document token"
                    )
                responses.append(
                    ModelResponse(
                        input=doc.query,
                        input_tokens=token_ids,
                        output_tokens=[[token_id] for token_id in token_ids],
                        logprobs=logprobs,
                        argmax_logits_eq_gold=argmax,
                    )
                )
            return responses

    def unique_gold_index(doc) -> int | None:
        raw = doc.gold_index
        if isinstance(raw, bool):
            return None
        if isinstance(raw, int):
            return raw
        if (
            isinstance(raw, (list, tuple))
            and len(raw) == 1
            and isinstance(raw[0], int)
            and not isinstance(raw[0], bool)
        ):
            return raw[0]
        return None

    def is_choice_doc(doc) -> bool:
        choices = doc.choices
        gold_index = unique_gold_index(doc)
        return (
            isinstance(choices, list)
            and 2 <= len(choices) <= 26
            and all(
                isinstance(choice, str) and bool(choice.strip()) for choice in choices
            )
            and len({choice.strip() for choice in choices}) == len(choices)
            and gold_index is not None
            and 0 <= gold_index < len(choices)
            and isinstance(doc.query, str)
            and SamplingMethod.LOGPROBS in doc.sampling_methods
            and SamplingMethod.GENERATIVE not in doc.sampling_methods
        )

    def is_multiselect_choice_doc(doc) -> bool:
        choices = doc.choices
        gold_indices = doc.gold_index
        return (
            isinstance(choices, list)
            and 2 <= len(choices) <= 26
            and isinstance(gold_indices, (list, tuple))
            and len(gold_indices) > 1
            and all(
                isinstance(index, int)
                and not isinstance(index, bool)
                and 0 <= index < len(choices)
                for index in gold_indices
            )
            and len(set(gold_indices)) == len(gold_indices)
            and SamplingMethod.LOGPROBS in doc.sampling_methods
            and SamplingMethod.GENERATIVE not in doc.sampling_methods
        )

    def convert_choice_doc(doc) -> None:
        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(doc.choices)])
        options = "\n".join(
            f"{label}. {choice.strip()}"
            for label, choice in zip(labels, doc.choices, strict=True)
        )
        doc.query = (
            f"{doc.query.rstrip()}\n\n{options}\n\n"
            'After reasoning, end with "Answer: <letter>".'
        )
        doc.sampling_methods = [
            (SamplingMethod.GENERATIVE if method == SamplingMethod.LOGPROBS else method)
            for method in doc.sampling_methods
        ]
        doc.sampling_methods = list(dict.fromkeys(doc.sampling_methods))
        doc.specific = dict(doc.specific or {}, rwkv_generative_choice=True)

    def choice_metrics(metric) -> tuple:
        names = (
            (metric.metric_name,)
            if isinstance(metric.metric_name, str)
            else tuple(metric.metric_name)
        )
        if not names or any(not isinstance(name, str) for name in names):
            raise ValueError("choice metric names must be non-empty strings")
        grouped = len(names) > 1 or not isinstance(metric.metric_name, str)
        if grouped and (
            not isinstance(metric.corpus_level_fn, dict)
            or not isinstance(metric.higher_is_better, dict)
        ):
            raise ValueError("grouped choice metric metadata is invalid")
        return tuple(
            SampleLevelMetric(
                metric_name=name,
                sample_level_fn=ExactMatches(),
                category=SamplingMethod.GENERATIVE,
                corpus_level_fn=(
                    metric.corpus_level_fn[name] if grouped else metric.corpus_level_fn
                ),
                higher_is_better=(
                    metric.higher_is_better[name]
                    if grouped
                    else metric.higher_is_better
                ),
            )
            for name in names
        )

    def choice_answer(raw, tokens, choices) -> str:
        if (
            not isinstance(raw, str)
            or not isinstance(tokens, list)
            or not tokens
            or len(tokens) >= MAX_NEW_TOKENS
            or raw.count("</think>") != 1
        ):
            return ""
        suffix = _MARKUP.sub("", raw.split("</think>", 1)[1])
        matches = [
            value.upper()
            for match in _BOXED.finditer(suffix)
            for value in match.groups()
            if value
        ]
        matches += [match.group(1).upper() for match in _EXPLICIT.finditer(suffix)]
        if match := _BARE.fullmatch(suffix):
            matches += [value.upper() for value in match.groups() if value]
        unique = set(matches)
        if len(unique) != 1:
            return ""
        label = unique.pop()
        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(choices)])
        return choices[labels.index(label)] if label in labels else ""

    class RegistryOnlyCache:
        def _init_registry(self, _registry) -> None:
            return None

    class RWKVPipeline(Pipeline):
        def __init__(self, *args, **kwargs):
            model = kwargs.get("model")
            if model is None:
                raise ValueError("RWKVPipeline requires the campaign-owned model")
            self._rwkv_stop_sequence = model.config.rwkv_stop_sequence
            model._cache = RegistryOnlyCache()
            try:
                super().__init__(*args, **kwargs)
            finally:
                model._cache = None

        def _init_tasks_and_requests(self, tasks: str):
            super()._init_tasks_and_requests(tasks)
            for task in self.tasks_dict.values():
                # Registry task configs are module-level objects reused by later
                # Pipeline instances. Keep per-shard accounting and metric
                # adaptation local so the next weight/mode starts from the
                # locked upstream task contract.
                task.config = copy.copy(task.config)
                original_docs = self.documents_dict[task.full_name]
                skipped_multiselect_docs = sum(
                    is_multiselect_choice_doc(doc) for doc in original_docs
                )
                docs = [
                    doc for doc in original_docs if not is_multiselect_choice_doc(doc)
                ]
                self.documents_dict[task.full_name] = docs
                task.config.original_num_docs = len(task.eval_docs())
                task.config.effective_num_docs = len(docs)
                task.config.skipped_multiselect_docs = skipped_multiselect_docs
                if (
                    task.config.original_num_docs <= 0
                    or task.config.effective_num_docs <= 0
                    or task.config.original_num_docs
                    != task.config.effective_num_docs
                    + task.config.skipped_multiselect_docs
                ):
                    raise RuntimeError(
                        f"task {task.full_name} did not account for its full "
                        "evaluation split"
                    )
                for document_index, doc in enumerate(docs):
                    specific = dict(doc.specific or {})
                    reserved = specific.get("helicopter_document_index")
                    if reserved is not None and reserved != document_index:
                        raise RuntimeError(
                            f"task {task.full_name} uses reserved "
                            "helicopter_document_index metadata"
                        )
                    specific["helicopter_document_index"] = document_index
                    doc.specific = specific
                choice_docs = [doc for doc in docs if is_choice_doc(doc)]
                if not choice_docs or any(
                    SamplingMethod.GENERATIVE in doc.sampling_methods for doc in docs
                ):
                    continue
                for doc in choice_docs:
                    convert_choice_doc(doc)
                converted_metrics = []
                retain_logprob_metrics = any(
                    SamplingMethod.LOGPROBS in doc.sampling_methods for doc in docs
                )
                for metric in task.metrics:
                    if metric.category == SamplingMethod.LOGPROBS:
                        if retain_logprob_metrics:
                            converted_metrics.append(metric)
                        converted_metrics.extend(choice_metrics(metric))
                    else:
                        converted_metrics.append(metric)
                task.metrics = tuple(converted_metrics)
                task.config.metrics = task.metrics
                task.sampling_methods = list(
                    dict.fromkeys(metric.category for metric in task.metrics)
                )
            self.sampling_docs.clear()
            for docs in self.documents_dict.values():
                for doc in docs:
                    if SamplingMethod.GENERATIVE in doc.sampling_methods:
                        doc.generation_size = MAX_NEW_TOKENS
                        doc.stop_sequences = [self._rwkv_stop_sequence]
                    for method in doc.sampling_methods:
                        self.sampling_docs[method].append(doc)
            self.evaluation_tracker.task_config_logger.log(self.tasks_dict)

        def _post_process_outputs(self, sampling_method_responses):
            super()._post_process_outputs(sampling_method_responses)
            for method, responses in sampling_method_responses.items():
                for doc, response in zip(
                    self.sampling_docs[method], responses, strict=True
                ):
                    if (
                        method != SamplingMethod.GENERATIVE
                        or not isinstance(doc.specific, dict)
                        or doc.specific.get("rwkv_generative_choice") is not True
                    ):
                        continue
                    response.text_post_processed = [
                        choice_answer(
                            raw,
                            (
                                response.output_tokens[index]
                                if isinstance(response.output_tokens, list)
                                and index < len(response.output_tokens)
                                else None
                            ),
                            doc.choices,
                        )
                        for index, raw in enumerate(response.text)
                    ]

    return (
        RWKVGenerationParameters,
        RWKVVLLMModelConfig,
        RWKVVLLMModel,
        RWKVPipeline,
        choice_answer,
    )


def _actual_capacity(backend) -> tuple[int, int]:
    engine = backend.model.llm_engine
    scheduler = getattr(engine, "scheduler_config", None)
    if scheduler is None:
        scheduler = getattr(
            getattr(engine, "vllm_config", None),
            "scheduler_config",
            None,
        )
    max_num_seqs = getattr(scheduler, "max_num_seqs", None)
    max_num_batched_tokens = getattr(
        scheduler,
        "max_num_batched_tokens",
        None,
    )
    if not isinstance(max_num_seqs, int) or not isinstance(max_num_batched_tokens, int):
        raise RuntimeError("cannot read vLLM resolved active capacity")
    return max_num_seqs, max_num_batched_tokens


def _model_execution(
    unit: EvaluationUnit,
    backend,
) -> dict[str, object]:
    import torch

    max_num_seqs, max_num_batched_tokens = _actual_capacity(backend)
    return {
        "weight_sha256": unit.weight.sha256,
        "weight_display_name": unit.weight.display_name,
        "wkv_mode": unit.wkv_mode,
        "prompt_template": unit.prompt_template,
        "gemm_policy": (
            "fp16-accumulation" if unit.wkv_mode == "fp16" else "fp32-accumulation"
        ),
        "gpu": torch.cuda.get_device_name(0),
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "dependency_versions": {
            name: importlib.metadata.version(name)
            for name in ("lighteval", "vllm", "torch")
        },
    }


def _remove_runtime_directory(campaign_dir: Path, runtime_dir: Path) -> None:
    try:
        _, relative = validate_campaign_child(
            campaign_dir,
            runtime_dir,
        )
    except RuntimeError as error:
        raise RuntimeError("model runtime directory is unsafe") from error
    campaign = campaign_dir.resolve()
    if not relative.parts:
        raise RuntimeError("invalid model runtime cleanup target")
    remove_campaign_child_directory(campaign, runtime_dir)


def _finish_runtime(
    *,
    backend: Any | None,
    campaign_dir: Path,
    runtime_dir: Path,
    on_runtime_finished: Callable[[], None] | None,
) -> None:
    if backend is not None:
        try:
            backend.close_unit()
        except Exception as error:
            raise UnsafeModelCleanupError(
                f"model cleanup failed: {_exception_type(error)}"
            ) from error
    try:
        _remove_runtime_directory(campaign_dir, runtime_dir)
    except Exception as error:
        raise UnsafeModelCleanupError(
            f"runtime directory cleanup failed: {_exception_type(error)}"
        ) from error
    if on_runtime_finished is not None:
        on_runtime_finished()


def _construct_backend(model_type, model_config):
    try:
        return model_type(model_config)
    except UnsafeModelCleanupError:
        raise
    except Exception as error:
        raise UnsafeModelCleanupError(
            "model construction failed before lifecycle ownership could be proven "
            f"safe: {_exception_type(error)}"
        ) from error


def _evaluate_unit(
    *,
    unit: EvaluationUnit,
    shards: tuple[EvaluationShard, ...],
    campaign_dir: Path,
    on_shard_started: Callable[
        [EvaluationShard, Path, dict[str, object]],
        None,
    ]
    | None = None,
    on_shard_completed: Callable[[ShardEvaluation], None] | None = None,
    on_shard_failed: Callable[[ShardFailure], None] | None = None,
    on_runtime_started: Callable[[Path], None] | None = None,
    on_runtime_finished: Callable[[], None] | None = None,
) -> tuple[
    list[ShardEvaluation],
    list[ShardFailure],
]:
    from vllm.transformers_utils.configs.rwkv7 import (
        build_rwkv7_config_from_pth,
    )

    types = _runtime_types()
    (
        GenerationParameters,
        ModelConfig,
        Model,
        Pipeline,
        _,
    ) = _build_runtime_classes(types)
    rwkv_prompt_template, stop_sequence = _official_prompt_template(
        unit.prompt_template
    )
    runtime_dir = campaign_dir / "runtime" / unit.weight.sha256 / unit.wkv_mode
    checkpoint_config = build_rwkv7_config_from_pth(str(unit.weight.path))
    if checkpoint_config is None:
        raise ValueError("evaluation weight is not a supported RWKV7 checkpoint")
    model_config = ModelConfig(
        model_name=unit.weight.path.as_uri(),
        cache_dir=str(runtime_dir / "disabled-sample-cache"),
        wkv_mode=unit.wkv_mode,
        rwkv_prompt_template=rwkv_prompt_template,
        rwkv_stop_sequence=stop_sequence,
        checkpoint_context_length=(checkpoint_config.max_position_embeddings),
        dtype="float16",
        max_model_length=evaluation_max_model_length(
            checkpoint_config.max_position_embeddings
        ),
        max_num_seqs=None,
        max_num_batched_tokens=None,
        enable_prefix_caching=False,
        override_chat_template=True,
        generation_parameters=GenerationParameters(
            temperature=0.96,
            top_p=0.76,
            top_k=32,
            presence_penalty=1.0,
            frequency_penalty=0.1,
            penalty_decay=0.988,
            stop_tokens=[stop_sequence],
            max_new_tokens=MAX_NEW_TOKENS,
        ),
    )
    if on_runtime_started is not None:
        on_runtime_started(runtime_dir)
    backend = None
    outputs: list[ShardEvaluation] = []
    failures: list[ShardFailure] = []
    try:
        backend = _construct_backend(Model, model_config)
        execution = _model_execution(unit, backend)
        for shard in shards:
            shard_dir = (
                campaign_dir
                / unit.weight.sha256
                / unit.wkv_mode
                / hashlib.sha256(shard.shard_id.encode()).hexdigest()[:16]
                / f"attempt-{uuid.uuid4().hex}"
            )
            if on_shard_started is not None:
                on_shard_started(shard, shard_dir, execution)
            shard_dir.mkdir(parents=True)
            pipeline = None
            evaluation_error: Exception | None = None
            error_phase = "pipeline-construction"
            try:
                tracker = types["EvaluationTracker"](
                    output_dir=str(shard_dir),
                    save_details=True,
                )
                parameters = _pipeline_parameters(types)
                pipeline = Pipeline(
                    tasks=",".join(task.identity for task in shard.tasks),
                    pipeline_parameters=parameters,
                    evaluation_tracker=tracker,
                    model=backend,
                )
                error_phase = "evaluation"
                pipeline.evaluate()
                error_phase = "artifact-save"
                pipeline.save_and_push_results()
            except Exception as error:
                evaluation_error = error
            finally:
                if pipeline is not None:
                    pipeline.tasks_dict.clear()
                    pipeline.documents_dict.clear()
                    pipeline.sampling_docs.clear()
                try:
                    backend.reset_shard_state()
                except Exception as error:
                    if evaluation_error is None:
                        evaluation_error = error
                        error_phase = "backend-reset"
                    else:
                        evaluation_error = RuntimeError(
                            f"{evaluation_error}; shard reset failed: {error}"
                        )
                        error_phase = "evaluation-and-backend-reset"
            if evaluation_error is None:
                evaluation = ShardEvaluation(shard, shard_dir, execution)
                outputs.append(evaluation)
                if on_shard_completed is not None:
                    on_shard_completed(evaluation)
            else:
                task_names = ",".join(task.identity for task in shard.tasks)
                error_type = _exception_type(evaluation_error)
                failure = ShardFailure(
                    shard,
                    shard_dir,
                    error_type,
                    error_phase,
                    _exception_site(evaluation_error),
                    f"{shard.shard_id} [{task_names}]: {error_type} "
                    f"at {error_phase}/{_exception_site(evaluation_error)}",
                )
                failures.append(failure)
                if on_shard_failed is not None:
                    on_shard_failed(failure)
    finally:
        _finish_runtime(
            backend=backend,
            campaign_dir=campaign_dir,
            runtime_dir=runtime_dir,
            on_runtime_finished=on_runtime_finished,
        )
    return outputs, failures


def evaluate_unit(
    *,
    unit: EvaluationUnit,
    shards: tuple[EvaluationShard, ...],
    campaign_dir: Path,
    on_shard_started: Callable[
        [EvaluationShard, Path, dict[str, object]],
        None,
    ]
    | None = None,
    on_shard_completed: Callable[[ShardEvaluation], None] | None = None,
    on_shard_failed: Callable[[ShardFailure], None] | None = None,
    on_runtime_started: Callable[[Path], None] | None = None,
    on_runtime_finished: Callable[[], None] | None = None,
) -> tuple[
    list[ShardEvaluation],
    list[ShardFailure],
]:
    verify_weight_identity(unit.weight)
    with _temporary_environment(
        {
            "VLLM_RWKV7_WKV_MODE": unit.wkv_mode,
            "VLLM_RWKV7_EMB_DEVICE": "gpu",
            "VLLM_USE_RAPID_SAMPLER": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            # Registry discovery can initialize CUDA before the model unit.
            # A spawned worker is therefore required; forking a CUDA-initialized
            # parent cannot safely initialize the device again.
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            # RWKV is recurrent and has no positional encoding ceiling. The
            # total LightEval length includes the checkpoint prompt context
            # plus the fixed generation budget, so vLLM's generic positional
            # model guard must be disabled only for this scoped model unit.
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            "RKV_MODE": "off",
            "CMIX_SPARSE": "no-fc",
            "LOW_RANK_WEIGHT": "both",
            "ORIG_LINEAR_GROUPS": "none",
        }
    ):
        return _evaluate_unit(
            unit=unit,
            shards=shards,
            campaign_dir=campaign_dir,
            on_shard_started=on_shard_started,
            on_shard_completed=on_shard_completed,
            on_shard_failed=on_shard_failed,
            on_runtime_started=on_runtime_started,
            on_runtime_finished=on_runtime_finished,
        )


def choice_answer_for_test(raw: str, tokens: list[int], choices: list[str]) -> str:
    types = _runtime_types()
    *_, choice_answer = _build_runtime_classes(types)
    return choice_answer(raw, tokens, choices)


_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _failure_summary(error: BaseException) -> str:
    if isinstance(error, (ManifestError, ScoreboardError)):
        return str(error)
    error_type = type(error)
    return f"{error_type.__module__}.{error_type.__qualname__}"


def _task_identity(unit: EvaluationUnit, task_identity: str) -> str:
    return f"{unit.weight.sha256}:{unit.wkv_mode}:{task_identity}"


def _expected_tasks(plan: EvaluationPlan) -> list[dict[str, object]]:
    expected: list[dict[str, object]] = []
    for unit in plan.units:
        for task in plan.registry.tasks:
            expected.append(
                {
                    "identity": _task_identity(unit, task.identity),
                    "weight_sha256": unit.weight.sha256,
                    "weight_display_name": unit.weight.display_name,
                    "wkv_mode": unit.wkv_mode,
                    "selector": task.selector,
                    "task_name": task.identity,
                    "task_version": task.version,
                    "module_family": task.module_family,
                    "module": task.module,
                    "dataset": task.dataset,
                    "subset": task.subset,
                    "evaluation_splits": list(task.evaluation_splits),
                    "languages": list(task.languages),
                    "upstream_tags": list(task.upstream_tags),
                }
            )
    return expected


def _resume_key(plan: EvaluationPlan) -> str:
    raw = json.dumps(
        {
            "config": plan.config_digest,
            "weights": list(dict.fromkeys(unit.weight.sha256 for unit in plan.units)),
            "registry": plan.registry.digest,
            "eval_contract": plan.eval_contract_digest,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _campaign_payload(plan: EvaluationPlan, resume_key: str) -> dict[str, object]:
    return {
        "schema_version": "lighteval-campaign-v2",
        "resume_key": resume_key,
        "config_digest": plan.config_digest,
        "registry_digest": plan.registry.digest,
        "eval_contract_digest": plan.eval_contract_digest,
        "lighteval_version": plan.registry.lighteval_version,
        "configured_selectors": list(plan.registry.configured_selectors),
        "resolved_selectors": list(plan.registry.resolved_selectors),
        "skipped_selectors": list(plan.registry.skipped_selectors),
        "expected_tasks": _expected_tasks(plan),
    }


def _new_manifest(
    plan: EvaluationPlan, resume_key: str, campaign_id: str
) -> CampaignManifest:
    return CampaignManifest(
        version=MANIFEST_VERSION,
        resume_key=resume_key,
        campaign_id=campaign_id,
        config_digest=plan.config_digest,
        registry_digest=plan.registry.digest,
        eval_contract_digest=plan.eval_contract_digest,
        weight_sha256=list(dict.fromkeys(unit.weight.sha256 for unit in plan.units)),
        configured_selectors=list(plan.registry.configured_selectors),
        resolved_selectors=list(plan.registry.resolved_selectors),
        skipped_selectors=list(plan.registry.skipped_selectors),
        registry_task_identities=[task.identity for task in plan.registry.tasks],
    )


def _check_manifest_contract(
    manifest: CampaignManifest,
    plan: EvaluationPlan,
    resume_key: str,
) -> None:
    expected = {
        "resume_key": resume_key,
        "config_digest": plan.config_digest,
        "registry_digest": plan.registry.digest,
        "eval_contract_digest": plan.eval_contract_digest,
    }
    mismatched = [
        name for name, value in expected.items() if getattr(manifest, name) != value
    ]
    if mismatched:
        raise ManifestError(
            "local manifest does not match evaluation contract: "
            + ", ".join(mismatched)
        )


def _validate_manifest_plan(
    manifest: CampaignManifest,
    plan: EvaluationPlan,
) -> None:
    if manifest.configured_selectors != list(plan.registry.configured_selectors):
        raise ManifestError("campaign manifest configured selectors do not match plan")
    if manifest.resolved_selectors != list(plan.registry.resolved_selectors):
        raise ManifestError("campaign manifest resolved selectors do not match plan")
    if manifest.skipped_selectors != list(plan.registry.skipped_selectors):
        raise ManifestError("campaign manifest skipped selectors do not match plan")
    expected_weights = list(dict.fromkeys(unit.weight.sha256 for unit in plan.units))
    if manifest.weight_sha256 != expected_weights:
        raise ManifestError("campaign manifest weight snapshot does not match plan")
    expected_registry = [task.identity for task in plan.registry.tasks]
    if manifest.registry_task_identities != expected_registry:
        raise ManifestError("campaign manifest registry snapshot does not match plan")
    expected_tasks = {
        _task_identity(unit, task.identity)
        for unit in plan.units
        for task in plan.registry.tasks
    }
    expected_shards = {
        f"{unit.weight.sha256}:{unit.wkv_mode}:{shard.shard_id}"
        for unit in plan.units
        for shard in unit.shards
    }
    expected_units = {f"{unit.weight.sha256}:{unit.wkv_mode}" for unit in plan.units}
    for name in (
        "pending_task_digests",
        "acknowledged_task_digests",
    ):
        unexpected = set(getattr(manifest, name)) - expected_tasks
        if unexpected:
            raise ManifestError(f"campaign manifest {name} contains unknown tasks")
    for name in (
        "shard_paths",
        "attempted_shard_paths",
        "failed_shard_paths",
        "failed_shard_errors",
    ):
        unexpected = set(getattr(manifest, name)) - expected_shards
        if unexpected:
            raise ManifestError(f"campaign manifest {name} contains unknown shards")
    if set(manifest.runtime_paths) - expected_units:
        raise ManifestError("campaign manifest runtime_paths contains unknown units")
    for unit_key, runtime_path in manifest.runtime_paths.items():
        weight_sha256, wkv_mode = unit_key.split(":", 1)
        expected_runtime_path = str(Path("runtime") / weight_sha256 / wkv_mode)
        if runtime_path != expected_runtime_path:
            raise ManifestError(
                "campaign manifest runtime path does not match its unit"
            )
    if set(manifest.shard_paths) & set(manifest.attempted_shard_paths):
        raise ManifestError(
            "campaign manifest has overlapping complete and attempted shards"
        )
    if set(manifest.failed_shard_errors) != set(manifest.failed_shard_paths):
        raise ManifestError(
            "campaign manifest failure records do not match failed shards"
        )
    if set(manifest.model_executions) - expected_units:
        raise ManifestError("campaign manifest model_executions contains unknown units")
    shard_state_keys = (
        set(manifest.shard_paths)
        | set(manifest.attempted_shard_paths)
        | set(manifest.failed_shard_paths)
    )
    for shard_key in shard_state_keys:
        weight_sha256, wkv_mode, _ = shard_key.split(":", 2)
        if f"{weight_sha256}:{wkv_mode}" not in manifest.model_executions:
            raise ManifestError(
                "campaign manifest shard state lacks model execution metadata"
            )
    paths = [
        *manifest.shard_paths.values(),
        *manifest.attempted_shard_paths.values(),
        *(
            path
            for shard_paths in manifest.failed_shard_paths.values()
            for path in shard_paths
        ),
        *manifest.runtime_paths.values(),
    ]
    if len(paths) != len(set(paths)):
        raise ManifestError("campaign manifest reuses a staging path")


def _canonical_campaign_id(raw_campaign_id: object) -> str:
    try:
        campaign_id = str(uuid.UUID(str(raw_campaign_id)))
    except (ValueError, TypeError) as error:
        raise ScoreboardError("Scoreboard returned an invalid campaign id") from error
    if campaign_id != raw_campaign_id:
        raise ScoreboardError("Scoreboard campaign id is not canonical")
    return campaign_id


def _validate_status(
    *,
    status: dict[str, object],
    campaign_id: str,
    plan: EvaluationPlan,
) -> tuple[str, dict[str, str], list[str]]:
    if status.get("campaign_id") != campaign_id:
        raise ScoreboardError("campaign status id does not match request")
    state = status.get("status")
    if state not in {"incomplete", "complete"}:
        raise ScoreboardError("campaign status is invalid")
    task_count = status.get("expected_task_count")
    if isinstance(task_count, bool) or task_count != plan.expected_task_count:
        raise ScoreboardError("campaign status task count does not match plan")
    backend_digests = status.get("acknowledged_task_digests")
    missing = status.get("missing_task_identities")
    if not isinstance(backend_digests, dict) or not all(
        isinstance(identity, str)
        and isinstance(digest, str)
        and _DIGEST.fullmatch(digest) is not None
        for identity, digest in backend_digests.items()
    ):
        raise ScoreboardError("campaign status lacks valid task digests")
    if (
        not isinstance(missing, list)
        or not all(isinstance(identity, str) for identity in missing)
        or len(missing) != len(set(missing))
    ):
        raise ScoreboardError("campaign status lacks valid missing tasks")
    expected = {
        item["identity"]
        for item in _expected_tasks(plan)
        if isinstance(item["identity"], str)
    }
    if set(backend_digests) | set(missing) != expected:
        raise ScoreboardError("campaign status task identities do not match plan")
    if set(backend_digests) & set(missing):
        raise ScoreboardError("campaign status reports overlapping task states")
    if state == "complete" and (
        missing or len(backend_digests) != plan.expected_task_count
    ):
        raise ScoreboardError("complete campaign is missing task publications")
    return state, backend_digests, missing


def _validate_campaign_receipt(
    receipt: dict[str, object],
    *,
    plan: EvaluationPlan,
) -> str:
    campaign_id = _canonical_campaign_id(receipt.get("campaign_id"))
    if receipt.get("status") != "incomplete":
        raise ScoreboardError("new campaign receipt must be incomplete")
    if receipt.get("disposition") not in {"created", "resumed"}:
        raise ScoreboardError("new campaign receipt disposition is invalid")
    task_count = receipt.get("expected_task_count")
    if isinstance(task_count, bool) or task_count != plan.expected_task_count:
        raise ScoreboardError("new campaign receipt task count does not match plan")
    acknowledged = receipt.get("acknowledged_task_digests")
    expected = {
        item["identity"]
        for item in _expected_tasks(plan)
        if isinstance(item["identity"], str)
    }
    if not isinstance(acknowledged, dict) or not all(
        isinstance(identity, str)
        and identity in expected
        and isinstance(digest, str)
        and _DIGEST.fullmatch(digest) is not None
        for identity, digest in acknowledged.items()
    ):
        raise ScoreboardError("new campaign receipt lacks task digests")
    return campaign_id


def _reconcile(
    manifest: CampaignManifest,
    backend_digests: dict[str, str],
) -> None:
    for identity, digest in backend_digests.items():
        pending = manifest.pending_task_digests.get(identity)
        acknowledged = manifest.acknowledged_task_digests.get(identity)
        local = pending or acknowledged
        if local is not None and local != digest:
            raise ManifestError(
                f"backend digest conflicts with local task digest: {identity}"
            )
        manifest.acknowledged_task_digests[identity] = digest
        manifest.pending_task_digests.pop(identity, None)


def _cleanup_completed_shards(
    *,
    plan: EvaluationPlan,
    manifest: CampaignManifest,
    environment: EvaluationEnvironment,
) -> None:
    for unit in plan.units:
        for shard in unit.shards:
            key = f"{unit.weight.sha256}:{unit.wkv_mode}:{shard.shard_id}"
            shard_path = manifest.shard_paths.get(key)
            identities = [_task_identity(unit, task.identity) for task in shard.tasks]
            if all(
                identity in manifest.acknowledged_task_digests
                for identity in identities
            ):
                if shard_path is not None:
                    remove_acknowledged_shard(
                        staging_root=environment.staging_root,
                        campaign_id=manifest.campaign_id,
                        shard_path=shard_path,
                    )
                    manifest.shard_paths.pop(key, None)
                for failed_path in manifest.failed_shard_paths.pop(key, []):
                    remove_acknowledged_shard(
                        staging_root=environment.staging_root,
                        campaign_id=manifest.campaign_id,
                        shard_path=failed_path,
                    )
                manifest.failed_shard_errors.pop(key, None)


def _cleanup_interrupted_attempts(
    *,
    manifest: CampaignManifest,
    environment: EvaluationEnvironment,
) -> None:
    for shard_key, shard_path in tuple(manifest.attempted_shard_paths.items()):
        remove_acknowledged_shard(
            staging_root=environment.staging_root,
            campaign_id=manifest.campaign_id,
            shard_path=shard_path,
        )
        manifest.attempted_shard_paths.pop(shard_key)


def _cleanup_interrupted_runtimes(
    *,
    manifest: CampaignManifest,
    environment: EvaluationEnvironment,
) -> None:
    for unit_key, runtime_path in tuple(manifest.runtime_paths.items()):
        remove_acknowledged_shard(
            staging_root=environment.staging_root,
            campaign_id=manifest.campaign_id,
            shard_path=runtime_path,
        )
        manifest.runtime_paths.pop(unit_key)


def _record_runtime_started(
    *,
    store: ManifestStore,
    manifest: CampaignManifest,
    campaign_dir: Path,
    unit: EvaluationUnit,
    runtime_dir: Path,
) -> None:
    unit_key = f"{unit.weight.sha256}:{unit.wkv_mode}"
    candidate, relative = validate_campaign_child(campaign_dir, runtime_dir)
    if candidate.is_symlink():
        raise ManifestError("model runtime path must not be a symlink")
    relative_path = str(relative)
    expected_path = str(Path("runtime") / unit.weight.sha256 / unit.wkv_mode)
    if relative_path != expected_path:
        raise ManifestError(f"model runtime path is not deterministic: {unit_key}")
    known = manifest.runtime_paths.get(unit_key)
    if known is not None and known != relative_path:
        raise ManifestError(f"model runtime path changed: {unit_key}")
    manifest.runtime_paths[unit_key] = relative_path
    store.save(manifest)


def _record_runtime_finished(
    *,
    store: ManifestStore,
    manifest: CampaignManifest,
    unit: EvaluationUnit,
) -> None:
    unit_key = f"{unit.weight.sha256}:{unit.wkv_mode}"
    if unit_key not in manifest.runtime_paths:
        raise ManifestError(f"model runtime completion was not registered: {unit_key}")
    manifest.runtime_paths.pop(unit_key)
    store.save(manifest)


def _control_metadata(
    *,
    plan: EvaluationPlan,
    campaign_id: str,
) -> dict[str, object]:
    return {
        "schema_version": "lighteval-control-v2",
        "campaign_id": campaign_id,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "config_digest": plan.config_digest,
        "implementation_digest": plan.implementation_digest,
        "registry_digest": plan.registry.digest,
        "eval_contract_digest": plan.eval_contract_digest,
        "prompt_template": plan.prompt_template,
        "configured_selectors": list(plan.registry.configured_selectors),
        "resolved_selectors": list(plan.registry.resolved_selectors),
        "skipped_selectors": list(plan.registry.skipped_selectors),
        "weight_sha256": list(dict.fromkeys(unit.weight.sha256 for unit in plan.units)),
        "wkv_modes": ["fp16", "fp32io16"],
        "expected_task_count": plan.expected_task_count,
        "content_location": "scoreboard-database-only",
    }


def _finish_local_campaign(
    *,
    plan: EvaluationPlan,
    environment: EvaluationEnvironment,
    store: ManifestStore,
    manifest: CampaignManifest,
) -> Path:
    _cleanup_completed_shards(
        plan=plan,
        manifest=manifest,
        environment=environment,
    )
    store.save(manifest)
    if manifest.shard_paths:
        raise ManifestError("acknowledged campaign still has shard artifacts")
    if manifest.failed_shard_paths:
        raise ManifestError("acknowledged campaign still has failed shard artifacts")
    if manifest.failed_shard_errors:
        raise ManifestError("acknowledged campaign still has shard failure records")
    if manifest.runtime_paths:
        raise ManifestError("acknowledged campaign still has model runtime artifacts")
    remove_empty_campaign(environment.staging_root, manifest.campaign_id)
    metadata_path = write_control_metadata(
        staging_root=environment.staging_root,
        campaign_id=manifest.campaign_id,
        metadata=_control_metadata(
            plan=plan,
            campaign_id=manifest.campaign_id,
        ),
    )
    store.delete()
    return metadata_path


def _publish_shard(
    *,
    client: ScoreboardClient,
    store: ManifestStore,
    manifest: CampaignManifest,
    plan: EvaluationPlan,
    environment: EvaluationEnvironment,
    unit: EvaluationUnit,
    shard,
    shard_dir,
    model_execution: dict[str, object],
) -> None:
    publications = publications_from_shard(
        shard_dir=shard_dir,
        campaign_id=manifest.campaign_id,
        unit=unit,
        shard=shard,
        model_execution=model_execution,
        registry_tasks=plan.registry.tasks,
    )
    for identity, payload, digest in publications:
        known = manifest.pending_task_digests.get(identity)
        acknowledged = manifest.acknowledged_task_digests.get(identity)
        if known is not None and known != digest:
            raise ManifestError(f"recomputed task digest changed: {identity}")
        if acknowledged is not None:
            if acknowledged != digest:
                raise ManifestError(f"acknowledged task digest changed: {identity}")
            continue
        manifest.pending_task_digests[identity] = digest
        store.save(manifest)
        task_receipt = client.publish_task(
            campaign_id=manifest.campaign_id,
            task_identity=identity,
            payload=payload,
            digest=digest,
        )
        if (
            task_receipt.get("content_digest") != digest
            or task_receipt.get("task_identity") != identity
            or task_receipt.get("disposition") not in {"created", "unchanged"}
        ):
            raise ScoreboardError(f"invalid publication receipt for {identity}")
        manifest.acknowledged_task_digests[identity] = digest
        manifest.pending_task_digests.pop(identity, None)
        store.save(manifest)
    _cleanup_completed_shards(
        plan=plan,
        manifest=manifest,
        environment=environment,
    )
    store.save(manifest)


def _record_shard_artifact(
    *,
    store: ManifestStore,
    manifest: CampaignManifest,
    campaign_dir: Path,
    unit: EvaluationUnit,
    shard,
    shard_dir: Path,
    model_execution: dict[str, object],
) -> None:
    shard_key = f"{unit.weight.sha256}:{unit.wkv_mode}:{shard.shard_id}"
    unit_key = f"{unit.weight.sha256}:{unit.wkv_mode}"
    candidate, relative = validate_campaign_child(campaign_dir, shard_dir)
    if not relative.parts or candidate.is_symlink() or not candidate.is_dir():
        raise ManifestError("evaluated shard path is not a safe directory")
    known_path = manifest.shard_paths.get(shard_key)
    if known_path is not None and known_path != str(relative):
        raise ManifestError(f"shard artifact path changed: {shard_key}")
    attempted_path = manifest.attempted_shard_paths.get(shard_key)
    if attempted_path != str(relative):
        raise ManifestError(
            f"shard artifact was not the registered attempt: {shard_key}"
        )
    known_execution = manifest.model_executions.get(unit_key)
    if known_execution is not None and known_execution != model_execution:
        raise ManifestError(f"model execution metadata changed: {unit_key}")
    manifest.shard_paths[shard_key] = str(relative)
    manifest.attempted_shard_paths.pop(shard_key)
    manifest.model_executions[unit_key] = model_execution
    store.save(manifest)


def _record_shard_attempt(
    *,
    store: ManifestStore,
    manifest: CampaignManifest,
    campaign_dir: Path,
    unit: EvaluationUnit,
    shard,
    shard_dir: Path,
    model_execution: dict[str, object],
) -> None:
    shard_key = f"{unit.weight.sha256}:{unit.wkv_mode}:{shard.shard_id}"
    unit_key = f"{unit.weight.sha256}:{unit.wkv_mode}"
    _, relative = validate_campaign_child(campaign_dir, shard_dir)
    relative_path = str(relative)
    if shard_key in manifest.shard_paths:
        raise ManifestError(f"cannot start an already completed shard: {shard_key}")
    known_attempt = manifest.attempted_shard_paths.get(shard_key)
    if known_attempt is not None and known_attempt != relative_path:
        raise ManifestError(f"shard attempt path changed: {shard_key}")
    known_execution = manifest.model_executions.get(unit_key)
    if known_execution is not None and known_execution != model_execution:
        raise ManifestError(f"model execution metadata changed: {unit_key}")
    manifest.attempted_shard_paths[shard_key] = relative_path
    manifest.model_executions[unit_key] = model_execution
    store.save(manifest)


def _record_shard_failure(
    *,
    store: ManifestStore,
    manifest: CampaignManifest,
    campaign_dir: Path,
    unit: EvaluationUnit,
    failure,
) -> None:
    shard_key = f"{unit.weight.sha256}:{unit.wkv_mode}:{failure.shard.shard_id}"
    candidate, relative = validate_campaign_child(
        campaign_dir,
        failure.path,
    )
    if not candidate.is_dir() or candidate.is_symlink():
        raise ManifestError("failed shard path is not a safe directory")
    relative_path = str(relative)
    if manifest.attempted_shard_paths.get(shard_key) != relative_path:
        raise ManifestError(f"failed shard was not the registered attempt: {shard_key}")
    manifest.attempted_shard_paths.pop(shard_key)
    paths = manifest.failed_shard_paths.setdefault(shard_key, [])
    if relative_path not in paths:
        paths.append(relative_path)
    manifest.failed_shard_errors[shard_key] = failure.error_type
    store.save(manifest)


def run_campaign(
    *,
    plan: EvaluationPlan,
    environment: EvaluationEnvironment,
) -> int:
    ensure_private_staging_root(environment.staging_root)
    if plan.registry.skipped_selectors:
        print(
            "skipped benchmark selectors unavailable in this LightEval release: "
            + ", ".join(plan.registry.skipped_selectors)
        )
    client = ScoreboardClient(environment)
    resume_key = _resume_key(plan)
    store = ManifestStore(environment.staging_root, resume_key)
    try:
        manifest = store.load()
    except ManifestError as error:
        quarantined = store.quarantine()
        print(
            "isolated an unreadable local campaign manifest without touching "
            f"run content: {quarantined}; reason: {error}"
        )
        manifest = None
    if manifest is not None:
        try:
            _check_manifest_contract(manifest, plan, resume_key)
            _validate_manifest_plan(manifest, plan)
        except ManifestError:
            quarantined = store.quarantine()
            print(
                "isolated a local manifest that does not match the current "
                f"evaluation contract: {quarantined}"
            )
            manifest = None
    if manifest is not None:
        campaign_id = _canonical_campaign_id(manifest.campaign_id)
        state, backend_digests, _ = _validate_status(
            status=client.campaign_status(campaign_id),
            campaign_id=campaign_id,
            plan=plan,
        )
        _reconcile(manifest, backend_digests)
        _cleanup_interrupted_attempts(
            manifest=manifest,
            environment=environment,
        )
        _cleanup_interrupted_runtimes(
            manifest=manifest,
            environment=environment,
        )
        store.save(manifest)
        if state == "complete":
            metadata_path = _finish_local_campaign(
                plan=plan,
                environment=environment,
                store=store,
                manifest=manifest,
            )
            print(
                f"recovered completed campaign {campaign_id}; evaluation content "
                f"retained only by Scoreboard; control metadata: {metadata_path}"
            )
            # A matching local manifest proves this invocation is recovering
            # the prior command after backend finalization, not requesting a
            # new evaluation. Once local cleanup finishes, that command is
            # complete. A later invocation has no manifest and therefore
            # creates a fresh campaign instead of treating this one as cache.
            return 0

    receipt = client.create_campaign(
        _campaign_payload(plan, resume_key),
        resume_key,
    )
    campaign_id = _validate_campaign_receipt(receipt, plan=plan)
    if manifest is None:
        existing_campaign_dir = campaign_directory(
            environment.staging_root,
            campaign_id,
        )
        if existing_campaign_dir.exists():
            if existing_campaign_dir.is_symlink() or not existing_campaign_dir.is_dir():
                raise ManifestError("resumed campaign has an unsafe local run path")
            if any(existing_campaign_dir.iterdir()):
                raise ManifestError(
                    "resumed campaign has local run content without a matching "
                    "manifest; refusing to guess, reuse, or delete it"
                )
        manifest = _new_manifest(plan, resume_key, campaign_id)
    elif manifest.campaign_id != campaign_id:
        raise ManifestError(
            "local manifest campaign does not match resumed backend campaign"
        )
    state, backend_digests, _ = _validate_status(
        status=client.campaign_status(campaign_id),
        campaign_id=campaign_id,
        plan=plan,
    )
    if state != "incomplete":
        raise ScoreboardError("campaign became complete before evaluation started")
    _reconcile(manifest, backend_digests)
    _validate_manifest_plan(manifest, plan)
    _cleanup_interrupted_attempts(
        manifest=manifest,
        environment=environment,
    )
    _cleanup_interrupted_runtimes(
        manifest=manifest,
        environment=environment,
    )
    _cleanup_completed_shards(
        plan=plan,
        manifest=manifest,
        environment=environment,
    )
    store.save(manifest)

    failures: list[str] = []
    campaign_dir = create_campaign_directory(
        environment.staging_root,
        campaign_id,
    )
    for unit in plan.units:
        unit_key = f"{unit.weight.sha256}:{unit.wkv_mode}"
        for shard in unit.shards:
            shard_key = f"{unit.weight.sha256}:{unit.wkv_mode}:{shard.shard_id}"
            relative = manifest.shard_paths.get(shard_key)
            model_execution = manifest.model_executions.get(unit_key)
            if relative is None or model_execution is None:
                continue
            shard_dir, _ = validate_campaign_child(
                campaign_dir,
                campaign_dir / relative,
            )
            if not shard_dir.is_dir() or shard_dir.is_symlink():
                raise ManifestError(
                    f"manifest shard artifact is unavailable: {relative}"
                )
            try:
                _publish_shard(
                    client=client,
                    store=store,
                    manifest=manifest,
                    plan=plan,
                    environment=environment,
                    unit=unit,
                    shard=shard,
                    shard_dir=shard_dir,
                    model_execution=model_execution,
                )
            except ManifestError:
                raise
            except Exception as error:
                failures.append(
                    f"{unit.weight.display_name}/{unit.wkv_mode}/"
                    f"{shard.shard_id} resume: {_failure_summary(error)}"
                )
        pending_shards = [
            shard
            for shard in unit.shards
            if (
                f"{unit.weight.sha256}:{unit.wkv_mode}:{shard.shard_id}"
                not in manifest.shard_paths
                and any(
                    _task_identity(unit, task.identity)
                    not in manifest.acknowledged_task_digests
                    for task in shard.tasks
                )
            )
        ]
        if not pending_shards:
            continue

        def on_shard_started(shard, shard_dir, model_execution):
            _record_shard_attempt(
                store=store,
                manifest=manifest,
                campaign_dir=campaign_dir,
                unit=unit,
                shard=shard,
                shard_dir=shard_dir,
                model_execution=model_execution,
            )

        def on_shard_completed(evaluation):
            shard, shard_dir, model_execution = evaluation
            _record_shard_artifact(
                store=store,
                manifest=manifest,
                campaign_dir=campaign_dir,
                unit=unit,
                shard=shard,
                shard_dir=shard_dir,
                model_execution=model_execution,
            )
            try:
                _publish_shard(
                    client=client,
                    store=store,
                    manifest=manifest,
                    plan=plan,
                    environment=environment,
                    unit=unit,
                    shard=shard,
                    shard_dir=shard_dir,
                    model_execution=model_execution,
                )
            except ManifestError:
                raise
            except Exception as error:
                failures.append(
                    f"{unit.weight.display_name}/{unit.wkv_mode}/"
                    f"{shard.shard_id} publication: "
                    f"{_failure_summary(error)}"
                )

        def on_shard_failed(failure):
            _record_shard_failure(
                store=store,
                manifest=manifest,
                campaign_dir=campaign_dir,
                unit=unit,
                failure=failure,
            )
            failures.append(
                f"{unit.weight.display_name}/{unit.wkv_mode}/{failure.message}"
            )

        def on_runtime_started(runtime_dir):
            _record_runtime_started(
                store=store,
                manifest=manifest,
                campaign_dir=campaign_dir,
                unit=unit,
                runtime_dir=runtime_dir,
            )

        def on_runtime_finished():
            _record_runtime_finished(
                store=store,
                manifest=manifest,
                unit=unit,
            )

        try:
            evaluate_unit(
                unit=unit,
                shards=tuple(pending_shards),
                campaign_dir=campaign_dir,
                on_shard_started=on_shard_started,
                on_shard_completed=on_shard_completed,
                on_shard_failed=on_shard_failed,
                on_runtime_started=on_runtime_started,
                on_runtime_finished=on_runtime_finished,
            )
        except UnsafeModelCleanupError as error:
            raise ManifestError(
                "model lifecycle could not be proven safe; campaign stopped "
                f"before starting another weight or WKV mode: {error}"
            ) from error
        except ManifestError:
            raise
        except Exception as error:
            failures.append(
                f"{unit.weight.display_name}/{unit.wkv_mode}: {_failure_summary(error)}"
            )
            continue

    final_state, final_digests, missing = _validate_status(
        status=client.campaign_status(campaign_id),
        campaign_id=campaign_id,
        plan=plan,
    )
    if final_state != "incomplete":
        raise ScoreboardError("campaign completed without an explicit finalize")
    _reconcile(manifest, final_digests)
    _cleanup_completed_shards(
        plan=plan,
        manifest=manifest,
        environment=environment,
    )
    store.save(manifest)
    if failures or missing:
        for failure in failures:
            print(f"evaluation unit failed: {failure}")
        print(
            f"campaign incomplete: {len(missing) if isinstance(missing, list) else 'unknown'} tasks missing"
        )
        return 1

    finalized = client.finalize(campaign_id)
    if (
        finalized.get("campaign_id") != campaign_id
        or finalized.get("status") != "complete"
        or finalized.get("task_count") != plan.expected_task_count
    ):
        raise ScoreboardError("Scoreboard did not finalize campaign")
    finalized_state, finalized_digests, finalized_missing = _validate_status(
        status=client.campaign_status(campaign_id),
        campaign_id=campaign_id,
        plan=plan,
    )
    if finalized_state != "complete" or finalized_missing:
        raise ScoreboardError("Scoreboard finalized campaign inconsistently")
    _reconcile(manifest, finalized_digests)
    metadata_path = _finish_local_campaign(
        plan=plan,
        environment=environment,
        store=store,
        manifest=manifest,
    )
    print(
        f"campaign {campaign_id} complete; evaluation content retained only by "
        f"Scoreboard; control metadata: {metadata_path}"
    )
    return 0


@contextmanager
def _process_environment(values: Mapping[str, str]):
    missing = object()
    previous: dict[str, str | object] = {
        key: os.environ.get(key, missing) for key in values
    }
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is missing:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run(*, config_path: Path, env: Mapping[str, str], dry_run: bool) -> int:
    with _process_environment(env):
        try:
            config = load_evaluation_config(config_path)
            environment = load_evaluation_environment(env)
            weights = resolve_weights(config, environment)
            readiness = run_preflight(environment)
            registry = load_configured_registry(config.benchmarks)
            plan = build_plan(config, weights, registry)
        except (EvaluationConfigurationError, ScoreboardError, OSError) as error:
            raise SystemExit(str(error)) from error

        if dry_run:
            output = {
                "status": "ready",
                "environment": public_environment(environment),
                "readiness": readiness,
                "plan": public_plan(plan),
            }
            print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        try:
            return run_campaign(plan=plan, environment=environment)
        except (ManifestError, ScoreboardError, OSError) as error:
            raise SystemExit(f"evaluation failed: {error}") from error
