from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping

from .config import ConfigError, LightEvalConfig
from .publish import (
    PublicationError,
    ScoreboardClient,
    content_digest,
    prepare_staging,
    publish_results,
    read_aggregate_metrics,
)


MAX_NEW_TOKENS = 8192
_MARKUP = re.compile(r"\*\*|__|`+")
_BOXED = re.compile(
    r"\\boxed\{\s*(?:([A-Z])|\\(?:text|mathrm)\{\s*([A-Z])\s*\})\s*\}",
    re.IGNORECASE,
)
_EXPLICIT = re.compile(
    r"^\s*(?:(?:thus|therefore|hence|so)[,:]?\s+)?(?:the\s+)?"
    r"(?:(?:final|correct)\s+)?(?:answer|choice|option)"
    r"\s*(?:is|:|=)\s*([A-Z])\b",
    re.IGNORECASE | re.MULTILINE,
)
_BARE = re.compile(
    r"^\s*(?:([A-Z])\.?|\(([A-Z])\)|\[([A-Z])\])\s*$",
    re.IGNORECASE,
)


def run(*, config_path: Path, env: Mapping[str, str], dry_run: bool) -> int:
    with _process_environment(env):
        try:
            config = LightEvalConfig.read(config_path, env)
            tasks, skipped, lighteval_version = _resolve_benchmarks(config.benchmarks)
            expected_tasks = _expected_tasks(config, tasks)
            client: ScoreboardClient | None = None
            readiness: dict[str, object] | None = None
            campaign: dict[str, object] | None = None
            if config.publish:
                if config.scoreboard_url is None or config.scoreboard_token is None:
                    raise ConfigError("published evaluation requires Scoreboard access")
                client = ScoreboardClient(
                    config.scoreboard_url,
                    config.scoreboard_token,
                )
                readiness = client.preflight()
                campaign = _campaign_payload(
                    config,
                    tasks,
                    skipped,
                    expected_tasks,
                    lighteval_version,
                )
            if dry_run:
                print(
                    json.dumps(
                        {
                            "status": "ready",
                            "config": config.public(),
                            "resolved_benchmarks": [
                                selector
                                for selector in config.benchmarks
                                if selector not in skipped
                            ],
                            "skipped_benchmarks": skipped,
                            "tasks": tasks,
                            "execution_units": len(config.weights)
                            * len(config.wkv_modes),
                            "expected_task_count": len(expected_tasks),
                            "scoreboard": readiness,
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            if not config.publish:
                return _run_local(
                    config=config,
                    tasks=tasks,
                )
            if campaign is None or client is None:
                raise ConfigError("published evaluation was not initialized")
            return _run_campaign(
                config=config,
                tasks=tasks,
                expected_tasks=expected_tasks,
                campaign=campaign,
                client=client,
            )
        except (ConfigError, PublicationError) as error:
            raise SystemExit(str(error)) from error
        except Exception as error:
            name = f"{type(error).__module__}.{type(error).__qualname__}"
            detail = str(error).strip()
            suffix = f": {detail}" if detail else ""
            raise SystemExit(f"evaluation failed: {name}{suffix}") from error


def _run_campaign(
    *,
    config: LightEvalConfig,
    tasks: list[dict[str, object]],
    expected_tasks: list[dict[str, object]],
    campaign: dict[str, object],
    client: ScoreboardClient,
) -> int:
    staging_root = prepare_staging(config.staging_root)
    run_key = str(campaign["run_key"])
    receipt = client.create_campaign(campaign, run_key)
    campaign_id = str(receipt["campaign_id"])
    try:
        normalized_campaign_id = str(uuid.UUID(campaign_id))
    except ValueError as error:
        raise PublicationError("Scoreboard returned an invalid campaign id") from error
    if normalized_campaign_id != campaign_id:
        raise PublicationError("Scoreboard returned a non-canonical campaign id")

    run_root = staging_root / campaign_id
    run_root.mkdir(mode=0o700)
    task_names = [str(task["task_name"]) for task in tasks]
    completed = 0
    for weight, weight_hash in zip(
        config.weights,
        config.weight_hashes,
        strict=True,
    ):
        for wkv_mode in config.wkv_modes:
            output_dir = run_root / weight_hash / wkv_mode
            output_dir.mkdir(parents=True)
            model, sampling_config = _evaluate(
                config=config,
                weight=weight,
                weight_hash=weight_hash,
                wkv_mode=wkv_mode,
                task_names=task_names,
                output_dir=output_dir,
            )
            unit_tasks = [
                task
                for task in expected_tasks
                if task["weight_sha256"] == weight_hash and task["wkv_mode"] == wkv_mode
            ]
            completed += publish_results(
                output_dir=output_dir,
                campaign_id=campaign_id,
                expected_tasks=unit_tasks,
                model=model,
                sampling_config=sampling_config,
                client=client,
            )

    if completed != len(expected_tasks):
        raise PublicationError("not every expected evaluation task was published")
    client.finalize(campaign_id, len(expected_tasks))
    _remove_completed_run(run_root, staging_root)
    print(f"campaign {campaign_id} complete; evaluation results retained by Scoreboard")
    return 0


def _run_local(
    *,
    config: LightEvalConfig,
    tasks: list[dict[str, object]],
) -> int:
    if config.result_path is None:
        raise ConfigError("local evaluation requires result_path")
    staging_root = prepare_staging(config.staging_root)
    run_root = staging_root / f"local-{uuid.uuid4()}"
    run_root.mkdir(mode=0o700)
    task_names = [str(task["task_name"]) for task in tasks]
    weight = config.weights[0]
    weight_hash = config.weight_hashes[0]
    wkv_mode = config.wkv_modes[0]
    output_dir = run_root / weight_hash / wkv_mode
    output_dir.mkdir(parents=True)
    _evaluate(
        config=config,
        weight=weight,
        weight_hash=weight_hash,
        wkv_mode=wkv_mode,
        task_names=task_names,
        output_dir=output_dir,
    )
    metrics = read_aggregate_metrics(
        output_dir=output_dir,
        task_names=task_names,
    )
    result = {
        "schema_version": 1,
        "weight_sha256": weight_hash,
        "wkv_mode": wkv_mode,
        "metrics": metrics,
    }
    config.result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.result_path.with_name(
        f".{config.result_path.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(config.result_path)
    _remove_completed_run(run_root, staging_root)
    print(f"evaluation metrics written to {config.result_path}")
    return 0


def _evaluate(
    *,
    config: LightEvalConfig,
    weight: Path,
    weight_hash: str,
    wkv_mode: str,
    task_names: list[str],
    output_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.metrics.metrics_sample import ExactMatches
    from lighteval.metrics.utils.metric_utils import SampleLevelMetric
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
    from lighteval.tasks.requests import SamplingMethod

    raw_prompt_template, stop = config.prompt

    class RWKVPipeline(Pipeline):
        def _init_tasks_and_requests(self, tasks: str):
            super()._init_tasks_and_requests(tasks)
            for task in self.tasks_dict.values():
                task.config = copy.copy(task.config)
                original_docs = self.documents_dict[task.full_name]
                docs = [
                    doc
                    for doc in original_docs
                    if not _is_multiselect(doc, SamplingMethod)
                ]
                if not docs:
                    raise ValueError(
                        f"task {task.full_name} contains no supported documents"
                    )
                self.documents_dict[task.full_name] = docs
                task.config.original_num_docs = len(original_docs)
                task.config.effective_num_docs = len(docs)
                task.config.skipped_multiselect_docs = len(original_docs) - len(docs)
                for document_index, doc in enumerate(docs):
                    doc.specific = dict(
                        doc.specific or {},
                        helicopter_document_index=document_index,
                    )

                choice_docs = [
                    doc for doc in docs if _is_single_choice(doc, SamplingMethod)
                ]
                if choice_docs and not any(
                    SamplingMethod.GENERATIVE in doc.sampling_methods for doc in docs
                ):
                    for doc in choice_docs:
                        _convert_choice(doc, SamplingMethod)
                    converted_metrics = []
                    for metric in task.metrics:
                        if metric.category == SamplingMethod.LOGPROBS:
                            converted_metrics.extend(
                                _choice_metrics(
                                    metric,
                                    SamplingMethod,
                                    ExactMatches,
                                    SampleLevelMetric,
                                )
                            )
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
                        doc.stop_sequences = [stop]
                    for method in doc.sampling_methods:
                        self.sampling_docs[method].append(doc)
            self.evaluation_tracker.task_config_logger.log(self.tasks_dict)

        def _post_process_outputs(self, responses):
            super()._post_process_outputs(responses)
            for method, outputs in responses.items():
                for doc, response in zip(
                    self.sampling_docs[method],
                    outputs,
                    strict=True,
                ):
                    if (
                        method != SamplingMethod.GENERATIVE
                        or not isinstance(doc.specific, dict)
                        or doc.specific.get("helicopter_choice") is not True
                    ):
                        continue
                    response.text_post_processed = [
                        _choice_answer(
                            text,
                            (
                                response.output_tokens[index]
                                if index < len(response.output_tokens)
                                else []
                            ),
                            doc.choices,
                        )
                        for index, text in enumerate(response.text)
                    ]

    sampling_config = {
        "temperature": 0.96,
        "top_p": 0.76,
        "top_k": 32,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.1,
        "repetition_penalty": 1.0,
        "penalty_decay": 0.988,
        "max_new_tokens": MAX_NEW_TOKENS,
        "stop": [stop],
        "ignore_eos": False,
    }
    tracker = EvaluationTracker(output_dir=str(output_dir), save_details=True)
    if config.backend == "vllm_http":
        if config.vllm_pool_manifest is None:
            raise ConfigError("vLLM HTTP evaluation requires a pool manifest")
        from .http_model import VLLMHttpModel
        from .http_pool import PoolManifest

        manifest = PoolManifest.read(config.vllm_pool_manifest)
        if manifest.wkv_mode != wkv_mode:
            raise ConfigError(
                "vLLM pool WKV mode does not match the evaluation configuration"
            )
        backend = VLLMHttpModel(
            manifest=manifest,
            cache_dir=output_dir / "cache",
            raw_prompt_template=raw_prompt_template,
            generation_parameters={
                "temperature": 0.96,
                "top_p": 0.76,
                "top_k": 32,
                "presence_penalty": 1.0,
                "frequency_penalty": 0.1,
                "repetition_penalty": 1.0,
                "penalty_decay": 0.988,
                "stop_token_ids": [0],
                "ignore_eos": False,
            },
        )
        launcher_type = ParallelismManager.NONE
        model_execution = _http_model_execution(
            backend,
            weight,
            weight_hash,
            wkv_mode,
            config.prompt_template,
        )
    else:
        from .local_model import create_local_model

        backend = create_local_model(
            weight=weight,
            output_dir=output_dir,
            raw_prompt_template=raw_prompt_template,
            stop=stop,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        launcher_type = ParallelismManager.VLLM
        model_execution = _model_execution(
            backend,
            weight,
            weight_hash,
            wkv_mode,
            config.prompt_template,
        )
    parameters = PipelineParameters(
        launcher_type=launcher_type,
        max_samples=None,
        remove_reasoning_tags=False,
        load_tasks_multilingual=True,
    )
    with _process_environment(
        {
            "VLLM_RWKV7_WKV_MODE": wkv_mode,
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_USE_RAPID_SAMPLER": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        }
    ):
        pipeline = RWKVPipeline(
            tasks=",".join(task_names),
            pipeline_parameters=parameters,
            evaluation_tracker=tracker,
            model=backend,
        )
        pipeline.evaluate()
        pipeline.save_and_push_results()
        pipeline.show_results()
    return model_execution, sampling_config


def _resolve_benchmarks(
    selectors: tuple[str, ...],
) -> tuple[list[dict[str, object]], list[str], str]:
    from lighteval.tasks.registry import Registry

    version = importlib.metadata.version("lighteval")
    if version != "0.13.1.dev0":
        raise ConfigError(f"LightEval 0.13.1.dev0 is required, found {version}")
    inventory = Registry(
        tasks=None,
        load_multilingual=True,
    ).get_tasks_dump()
    metadata: dict[str, tuple[str, dict[str, object]]] = {}
    for row in inventory:
        if not isinstance(row, dict) or not isinstance(row.get("tasks"), list):
            continue
        module = row.get("module")
        if not isinstance(module, str):
            continue
        docstring = row.get("docstring")
        task_metadata = docstring if isinstance(docstring, dict) else {}
        for task in row["tasks"]:
            if isinstance(task, dict) and isinstance(task.get("name"), str):
                metadata[task["name"]] = (module, task_metadata)

    tasks: list[dict[str, object]] = []
    skipped: list[str] = []
    owners: dict[str, str] = {}
    for selector in selectors:
        try:
            registry = Registry(
                tasks=selector,
                load_multilingual=True,
            )
        except ValueError:
            skipped.append(selector)
            continue
        configs = [
            task_config
            for task_configs in registry.task_to_configs.values()
            for task_config in task_configs
        ]
        if not configs:
            skipped.append(selector)
            continue
        for task_config in configs:
            task_name = task_config.full_name
            if task_name in owners:
                raise ConfigError(
                    f"benchmark selectors overlap on {task_name}: "
                    f"{owners[task_name]}, {selector}"
                )
            owners[task_name] = selector
            try:
                module, task_metadata = metadata[task_config.name]
            except KeyError as error:
                raise ConfigError(
                    f"LightEval metadata is missing for {task_config.name}"
                ) from error
            if (
                not isinstance(task_config.version, (int, str))
                or isinstance(task_config.version, bool)
                or not str(task_config.version)
                or not isinstance(task_config.hf_repo, str)
                or not task_config.hf_repo
                or not isinstance(task_config.evaluation_splits, (list, tuple))
                or not task_config.evaluation_splits
                or any(
                    not isinstance(split, str) or not split
                    for split in task_config.evaluation_splits
                )
            ):
                raise ConfigError(f"LightEval task metadata is invalid for {task_name}")
            tasks.append(
                {
                    "selector": selector,
                    "task_name": task_name,
                    "task_version": str(task_config.version),
                    "module_family": _module_family(module),
                    "module": module,
                    "dataset": task_config.hf_repo,
                    "subset": task_config.hf_subset or "",
                    "evaluation_splits": list(task_config.evaluation_splits),
                    "languages": _metadata_strings(task_metadata.get("languages")),
                    "upstream_tags": _metadata_strings(task_metadata.get("tags")),
                }
            )
    if not tasks:
        raise ConfigError("none of the configured benchmarks exist in LightEval")
    tasks.sort(key=lambda task: str(task["task_name"]))
    return tasks, skipped, version


def _expected_tasks(
    config: LightEvalConfig,
    tasks: list[dict[str, object]],
) -> list[dict[str, object]]:
    expected: list[dict[str, object]] = []
    for weight, weight_hash in zip(
        config.weights,
        config.weight_hashes,
        strict=True,
    ):
        for wkv_mode in config.wkv_modes:
            for task in tasks:
                task_name = str(task["task_name"])
                expected.append(
                    {
                        "identity": f"{weight_hash}:{wkv_mode}:{task_name}",
                        "weight_sha256": weight_hash,
                        "weight_display_name": weight.name,
                        "wkv_mode": wkv_mode,
                        **task,
                    }
                )
    return expected


def _campaign_payload(
    config: LightEvalConfig,
    tasks: list[dict[str, object]],
    skipped: list[str],
    expected_tasks: list[dict[str, object]],
    lighteval_version: str,
) -> dict[str, object]:
    config_digest = content_digest(
        {
            "prompt_template": config.prompt_template,
            "weights": list(config.weight_hashes),
            "benchmarks": list(config.benchmarks),
        }
    )
    registry_digest = content_digest(tasks)
    eval_contract_digest = content_digest(
        {
            "wkv_modes": config.wkv_modes,
            "max_samples": None,
            "max_new_tokens": MAX_NEW_TOKENS,
            "prompt_template": config.prompt_template,
        }
    )
    run_key = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    return {
        "schema_version": "lighteval-campaign-v3",
        "run_key": run_key,
        "config_digest": config_digest,
        "registry_digest": registry_digest,
        "eval_contract_digest": eval_contract_digest,
        "lighteval_version": lighteval_version,
        "configured_selectors": list(config.benchmarks),
        "resolved_selectors": [
            selector for selector in config.benchmarks if selector not in skipped
        ],
        "skipped_selectors": skipped,
        "expected_tasks": expected_tasks,
    }


def _model_execution(
    backend,
    weight: Path,
    weight_hash: str,
    wkv_mode: str,
    prompt_template: str,
) -> dict[str, object]:
    import torch

    engine = backend.model.llm_engine
    scheduler = getattr(engine, "scheduler_config", None)
    if scheduler is None:
        scheduler = engine.vllm_config.scheduler_config
    return {
        "weight_sha256": weight_hash,
        "weight_display_name": weight.name,
        "wkv_mode": wkv_mode,
        "prompt_template": prompt_template,
        "gemm_policy": (
            "fp16-accumulation" if wkv_mode == "fp16" else "fp32-accumulation"
        ),
        "gpu": torch.cuda.get_device_name(0),
        "max_num_seqs": int(scheduler.max_num_seqs),
        "max_num_batched_tokens": int(scheduler.max_num_batched_tokens),
        "dependency_versions": {
            name: importlib.metadata.version(name)
            for name in ("lighteval", "vllm", "torch")
        },
    }


def _http_model_execution(
    backend,
    weight: Path,
    weight_hash: str,
    wkv_mode: str,
    prompt_template: str,
) -> dict[str, object]:
    manifest = backend.pool.manifest
    return {
        "weight_sha256": weight_hash,
        "weight_display_name": weight.name,
        "wkv_mode": wkv_mode,
        "prompt_template": prompt_template,
        "gemm_policy": (
            "fp16-accumulation" if wkv_mode == "fp16" else "fp32-accumulation"
        ),
        "gpu": f"remote-vllm-pool:{len(manifest.replicas)}",
        "max_num_seqs": manifest.total_capacity,
        "max_num_batched_tokens": None,
        "dependency_versions": {
            "lighteval": importlib.metadata.version("lighteval"),
            "vllm": manifest.vllm_version,
            "httpx": importlib.metadata.version("httpx"),
        },
    }


def _is_multiselect(doc, sampling_method) -> bool:
    gold = doc.gold_index
    return (
        isinstance(doc.choices, list)
        and isinstance(gold, (list, tuple))
        and len(gold) > 1
        and sampling_method.LOGPROBS in doc.sampling_methods
        and sampling_method.GENERATIVE not in doc.sampling_methods
    )


def _is_single_choice(doc, sampling_method) -> bool:
    gold = doc.gold_index
    if (
        isinstance(gold, (list, tuple))
        and len(gold) == 1
        and isinstance(gold[0], int)
        and not isinstance(gold[0], bool)
    ):
        gold = gold[0]
    return (
        isinstance(doc.query, str)
        and isinstance(doc.choices, list)
        and 2 <= len(doc.choices) <= 26
        and all(isinstance(choice, str) and choice.strip() for choice in doc.choices)
        and isinstance(gold, int)
        and not isinstance(gold, bool)
        and 0 <= gold < len(doc.choices)
        and sampling_method.LOGPROBS in doc.sampling_methods
        and sampling_method.GENERATIVE not in doc.sampling_methods
    )


def _convert_choice(doc, sampling_method) -> None:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(doc.choices)]
    options = "\n".join(
        f"{label}. {choice.strip()}"
        for label, choice in zip(labels, doc.choices, strict=True)
    )
    doc.query = (
        f"{doc.query.rstrip()}\n\n{options}\n\n"
        'After reasoning, end with "Answer: <letter>".'
    )
    doc.sampling_methods = list(
        dict.fromkeys(
            sampling_method.GENERATIVE if method == sampling_method.LOGPROBS else method
            for method in doc.sampling_methods
        )
    )
    doc.specific = dict(doc.specific or {}, helicopter_choice=True)


def _choice_metrics(metric, sampling_method, exact_matches, sample_metric):
    names = (
        (metric.metric_name,)
        if isinstance(metric.metric_name, str)
        else tuple(metric.metric_name)
    )
    grouped = not isinstance(metric.metric_name, str)
    return tuple(
        sample_metric(
            metric_name=name,
            sample_level_fn=exact_matches(),
            category=sampling_method.GENERATIVE,
            corpus_level_fn=(
                metric.corpus_level_fn[name] if grouped else metric.corpus_level_fn
            ),
            higher_is_better=(
                metric.higher_is_better[name] if grouped else metric.higher_is_better
            ),
        )
        for name in names
    )


def _choice_answer(raw: str, tokens: list[int], choices: list[str]) -> str:
    if (
        not isinstance(raw, str)
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
    matches.extend(match.group(1).upper() for match in _EXPLICIT.finditer(suffix))
    if match := _BARE.fullmatch(suffix):
        matches.extend(value.upper() for value in match.groups() if value)
    if len(set(matches)) != 1:
        return ""
    label = matches[0]
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(choices)]
    return choices[labels.index(label)] if label in labels else ""


def _metadata_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return list(dict.fromkeys(item for item in value if isinstance(item, str)))
    return []


def _module_family(module: str) -> str:
    family = module.removesuffix(".main")
    for prefix in (
        "lighteval.tasks.tasks.",
        "lighteval.tasks.multilingual.tasks.",
        "lighteval.tasks.multilingual.",
    ):
        if family.startswith(prefix):
            return family.removeprefix(prefix)
    return family


def _remove_completed_run(run_root: Path, staging_root: Path) -> None:
    resolved_staging = staging_root.resolve()
    if run_root.is_symlink() or run_root.resolve().parent != resolved_staging:
        raise PublicationError("refusing to clean an unsafe evaluation run directory")
    shutil.rmtree(run_root)


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
