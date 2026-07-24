"""Composition root for the thin LightEval integration."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from openai import AsyncOpenAI

from .datasets import get_benchmark_info
from .datasets.benchmark import Benchmark, BenchmarkInfo, Field
from .datasets.coding import CODING_UNSUPPORTED_REASON


LIGHTEVAL_REVISION = "64f4f5ae173626509fad6e477ca4ee56ebb26129"
DEFAULT_RESULTS_ROOT = Path("src/eval/lighteval/results")
DEFAULT_GENERATION_LIMIT = 32768
_SIGNED_BINARY_METRICS = frozenset(
    {
        "extractive_match",
        "em",
        "gpqa_pass@k:k=1",
        "pass@k:k=1&n=1",
        "pass@k:k=1",
        "prompt_level_strict_acc",
    }
)
_SUPPORTED_GENERATIVE_METRICS = _SIGNED_BINARY_METRICS | frozenset(
    {
        "avg@n:n=1",
        "inst_level_strict_acc",
        "prompt_level_loose_acc",
        "inst_level_loose_acc",
    }
)
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_CANONICAL_TASK_RE = re.compile(
    r"^lighteval/(?P<family>[a-z][a-z0-9-]*)/(?P<benchmark>[^/@]+)@(?P<version>[0-9]+(?:\.[0-9]+)*)$"
)


class UnsupportedTaskError(ValueError):
    """The pinned LightEval revision cannot safely execute the requested task."""


@dataclass(frozen=True, slots=True)
class _BenchmarkSelection:
    canonical_task: str
    info: BenchmarkInfo
    task_version: str

    @property
    def lighteval_task(self) -> str:
        # The public @version is the upstream task config version. The current
        # evaluator deliberately runs every supported task zero-shot.
        return f"{self.info.lighteval_task_name}|0"


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    model: str
    task: str
    endpoint_url: str
    output_root: Path = DEFAULT_RESULTS_ROOT
    checkpoint_sha256: str = ""
    tokenizer_revision: str = ""
    chat_template_revision: str = ""
    server_revision: str = ""
    wkv_mode: str = ""
    precision: str = ""
    gemm_policy: str = ""
    launch_contract: str = ""
    product_revision: str = ""
    product_dirty: bool = False
    cot_mode: str = "none"
    max_concurrent_requests: int = 16
    request_timeout_seconds: float = 3600.0
    max_samples: int | None = None
    generation_limit: int | None = None
    config_digest: str = ""
    scoreboard_url: str | None = None
    scoreboard_token: str | None = None
    endpoint_api_key: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.model.strip()
            or not self.task.strip()
            or not self.endpoint_url.strip()
        ):
            raise ValueError("model, task, and endpoint_url are required")
        if self.cot_mode not in {"none", "cot"}:
            raise ValueError("cot_mode must be none or cot")
        if (
            isinstance(self.max_concurrent_requests, bool)
            or not isinstance(self.max_concurrent_requests, int)
            or self.max_concurrent_requests <= 0
        ):
            raise ValueError("max_concurrent_requests must be positive")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if self.generation_limit is not None and self.generation_limit <= 0:
            raise ValueError("generation_limit must be positive")
        if self.scoreboard_url is not None and not self.scoreboard_token:
            raise ValueError("scoreboard publication requires a bearer token")
        for name, value in (("checkpoint_sha256", self.checkpoint_sha256),):
            if value and not _is_sha256(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.product_revision and not re.fullmatch(
            r"[0-9a-f]{40}", self.product_revision
        ):
            raise ValueError("product_revision must be a lowercase Git commit")
        if self.config_digest and not _is_sha256(self.config_digest):
            raise ValueError("config_digest must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    run_id: str
    run_status: str
    manifest_path: Path | None
    publication_status: str = "not_requested"
    publication_error: str | None = None
    publication_retry_identity: str | None = None
    publication_task_id: int | None = None
    summary: dict[str, int | float | str | None] | None = None

    @property
    def is_success(self) -> bool:
        return self.run_status == "completed" and self.publication_status in {
            "not_requested",
            "published",
        }


def run_evaluation(request: EvaluationRequest) -> EvaluationOutcome:
    """Generate through OpenAI chat completions and apply native LightEval metrics."""

    try:
        selection = _select_benchmark(request.task)
    except UnsupportedTaskError as error:
        return EvaluationOutcome(
            run_id="unsupported",
            run_status="unsupported",
            manifest_path=None,
            summary={"error": str(error)},
        )
    canonical_task = selection.canonical_task
    if selection.info.field == Field.CODING:
        return EvaluationOutcome(
            run_id="unsupported",
            run_status="unsupported",
            manifest_path=None,
            summary={"error": CODING_UNSUPPORTED_REASON},
        )

    run_id = uuid4().hex
    output_root = Path(request.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / run_id
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("generated run id is unsafe")
    run_dir.mkdir(exist_ok=False)

    benchmark = selection.info.create()
    try:
        _validate_identity_inputs(request)
        task, documents = _load_task(
            request=request,
            lighteval_task=selection.lighteval_task,
            benchmark=benchmark,
        )
        if request.generation_limit is not None:
            _override_generation_size(task, request.generation_limit)
        responses, finish_reasons = asyncio.run(
            _generate(request=request, documents=documents)
        )
        sample_metrics, result_dict = _score(
            task=task, documents=documents, responses=responses
        )
        samples = _collect_samples(
            task=task,
            documents=documents,
            responses=responses,
            finish_reasons=finish_reasons,
            sample_metrics=sample_metrics,
            canonical_task=canonical_task,
            request=request,
        )
        if not samples:
            raise ValueError("LightEval completed without scored samples")
        terminal_payload = _terminal_payload(
            task=task, samples=samples, result_dict=result_dict
        )
        terminal_payload["dataset"] = {
            "repository": task.config.hf_repo,
            "subset": task.config.hf_subset,
            "revision": task.config.hf_revision,
            "fingerprint": _dataset_fingerprint(task),
            "split": list(task.config.evaluation_splits),
        }
        _write_json(run_dir / "terminal_evidence.json", terminal_payload)
        identity, accounting = _build_identity_and_accounting(
            request=request,
            canonical_task=canonical_task,
            task=task,
            benchmark=benchmark,
            sample_count=len(samples),
            result_dict=result_dict,
        )
        manifest_path = _write_manifest(
            run_dir=run_dir,
            identity=identity,
            accounting=accounting,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        publication_status = "not_requested"
        publication_error = None
        retry_identity = None
        publication_task_id = None
        if request.scoreboard_url is not None:
            from .scoreboard import publish_manifest

            publication = publish_manifest(
                manifest_path=manifest_path,
                scoreboard_url=request.scoreboard_url,
                bearer_token=request.scoreboard_token or "",
            )
            publication_status = publication.status
            publication_error = publication.error
            retry_identity = publication.retry_identity
            publication_task_id = publication.task_id
        primary_metric = identity["task"]["primary_metric"]
        metric_value = terminal_payload["metrics"].get(primary_metric)
        return EvaluationOutcome(
            run_id=run_id,
            run_status="completed",
            manifest_path=manifest_path,
            publication_status=publication_status,
            publication_error=publication_error,
            publication_retry_identity=retry_identity,
            publication_task_id=publication_task_id,
            summary={
                "generated_samples": len(samples),
                "truncated_samples": terminal_payload["truncated_samples"],
                "truncation_rate": terminal_payload["truncation_rate"],
                primary_metric: metric_value,
            },
        )
    except Exception as error:
        _write_json(
            run_dir / "failure.json",
            {
                "status": "invalid" if isinstance(error, ValueError) else "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        return EvaluationOutcome(
            run_id=run_id,
            run_status="invalid" if isinstance(error, ValueError) else "failed",
            manifest_path=None,
            summary={"error": str(error)},
        )


def _load_task(
    *,
    request: EvaluationRequest,
    lighteval_task: str,
    benchmark: Benchmark,
):
    from lighteval.tasks.lighteval_task import LightevalTask
    from lighteval.tasks.registry import Registry

    tasks = Registry(tasks=lighteval_task, load_multilingual=False).load_tasks()
    LightevalTask.load_datasets(tasks, dataset_loading_processes=1)
    task_values = list(tasks.values())
    if len(task_values) != 1:
        raise ValueError("one benchmark must resolve to one LightEval task")
    task = task_values[0]
    documents = task.get_docs(request.max_samples)
    benchmark.prepare_documents(task=task, documents=documents)
    _ensure_generation_size(task)
    return task, documents


async def _generate(*, request: EvaluationRequest, documents: list[Any]):
    from lighteval.models.model_output import ModelResponse

    semaphore = asyncio.Semaphore(request.max_concurrent_requests)
    generation_prompt = "open_think" if request.cot_mode == "cot" else "fake_think"
    async with AsyncOpenAI(
        base_url=request.endpoint_url.rstrip("/"),
        api_key=request.endpoint_api_key or "EMPTY",
        timeout=request.request_timeout_seconds,
        max_retries=0,
    ) as client:

        async def generate(document: Any):
            async with semaphore:
                response = await client.chat.completions.create(
                    model=request.model,
                    messages=[{"role": "user", "content": document.query}],
                    max_tokens=document.generation_size,
                    temperature=0.0,
                    stop=["\nUser:"],
                    extra_body={
                        "stop_token_ids": [0],
                        "chat_template_kwargs": {
                            "rwkv_generation_prompt": generation_prompt
                        },
                    },
                )
            if len(response.choices) != 1:
                raise ValueError("chat completion must return one choice")
            choice = response.choices[0]
            if choice.finish_reason not in {"stop", "length"}:
                raise ValueError(f"unsupported finish_reason: {choice.finish_reason}")
            if not isinstance(choice.message.content, str):
                raise ValueError("chat completion content must be text")
            return (
                ModelResponse(
                    input=[{"role": "user", "content": document.query}],
                    text=[choice.message.content],
                ),
                choice.finish_reason,
            )

        generated = await asyncio.gather(
            *(generate(document) for document in documents)
        )
    return [item[0] for item in generated], [item[1] for item in generated]


def _score(*, task: Any, documents: list[Any], responses: list[Any]):
    from lighteval.logging.info_loggers import MetricsLogger
    from lighteval.metrics import apply_metric

    sample_metrics = apply_metric(
        docs=documents,
        responses=responses,
        metrics=task.metrics,
    )
    metrics_logger = MetricsLogger()
    for metrics in sample_metrics:
        metrics_logger.log(task.full_name, metrics)
    metrics_logger.aggregate({task.full_name: task})
    return sample_metrics, {
        "results": {
            task.full_name: dict(metrics_logger.metric_aggregated[task.full_name])
        }
    }


def _collect_samples(
    *,
    task: Any,
    documents: list[Any],
    responses: list[Any],
    finish_reasons: list[str],
    sample_metrics: list[dict[str, Any]],
    canonical_task: str,
    request: EvaluationRequest,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for ordinal, (document, response, finish_reason, metrics) in enumerate(
        zip(documents, responses, finish_reasons, sample_metrics, strict=True)
    ):
        reference = None
        if document.choices:
            try:
                reference = document.choices[document.gold_index]
            except (IndexError, TypeError):
                reference = document.choices[0]
        completion = response.text[0]
        samples.append(
            {
                "sample_index": ordinal,
                "sample_id": f"{task.full_name}:{document.id}",
                "attempt": 1,
                "status": "scored",
                "prompt": document.query,
                "raw_completion": completion,
                "generation": {
                    "finish_reason": finish_reason,
                    "truncated": finish_reason == "length",
                    "generation_limit": document.generation_size,
                },
                "scoring": {
                    "scorer_revision": _scorer_revision(task),
                },
                "metrics": {
                    str(key): float(value)
                    for key, value in metrics.items()
                    if _is_number(value)
                },
                "error_code": None,
                "error_message": None,
                "reference_answer": reference,
                "provenance": {
                    "task": canonical_task,
                    "lighteval_revision": LIGHTEVAL_REVISION,
                    "dataset_revision": task.config.hf_revision,
                    "dataset_repository": task.config.hf_repo,
                    "dataset_subset": task.config.hf_subset,
                    "evaluation_split": list(task.config.evaluation_splits),
                    "cot_mode": request.cot_mode,
                },
            }
        )
    return samples


def _terminal_payload(
    *, task: Any, samples: list[dict[str, Any]], result_dict: Mapping[str, Any]
) -> dict[str, Any]:
    aggregate = _result_metrics_for_task(task=task, result_dict=result_dict)
    native_metrics = {
        str(name): float(value)
        for name, value in aggregate.items()
        if not str(name).endswith("_stderr") and _is_number(value)
    }
    primary = _primary_metric_name(task, aggregate=native_metrics)
    truncated = sum(bool(sample["generation"]["truncated"]) for sample in samples)
    return {
        "samples": samples,
        "metrics": {primary: float(aggregate[primary])},
        "native_metrics": native_metrics,
        "generated_samples": len(samples),
        "truncated_samples": truncated,
        "truncation_rate": truncated / len(samples),
        "performance": {"status": "not_attributable"},
    }


def _build_identity_and_accounting(
    *,
    request: EvaluationRequest,
    canonical_task: str,
    task: Any,
    benchmark: Benchmark,
    sample_count: int,
    result_dict: Mapping[str, Any],
):
    fingerprint = _dataset_fingerprint(task)
    if fingerprint is None:
        raise ValueError("LightEval dataset did not expose a stable fingerprint")
    dataset_digest = _digest(
        {
            "repo": task.config.hf_repo,
            "subset": task.config.hf_subset,
            "split": list(task.config.evaluation_splits),
            "fingerprint": fingerprint,
        }
    )
    aggregate = _result_metrics_for_task(task=task, result_dict=result_dict)
    metric_name = _primary_metric_name(task, aggregate=aggregate)
    task_version = _canonical_version(canonical_task)
    provider = {
        "server_revision": request.server_revision,
        "wkv_mode": request.wkv_mode,
        "precision": request.precision,
        "gemm_policy": request.gemm_policy,
        "launch_contract": request.launch_contract,
    }
    identity = {
        "task": {
            "suite": "lighteval",
            "task": task.config.name,
            "version": str(task_version),
            "split": ",".join(task.config.evaluation_splits),
            "fewshot": int(task.config.num_fewshots),
            "prompt_revision": benchmark.query_revision(),
            "scorer_revision": _scorer_revision(task),
            "generation_contract": "helicopter-lighteval-openai-v1",
            "cot_mode": request.cot_mode,
            "dataset_digest": dataset_digest,
            "primary_metric": metric_name,
            "metrics": [
                {
                    "name": metric_name,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "aggregation": "mean",
                    "binary_correctness": True,
                }
            ],
        },
        "model": {
            "served_name": request.model,
            "checkpoint_sha256": request.checkpoint_sha256,
            "tokenizer_revision": request.tokenizer_revision,
            "chat_template_revision": request.chat_template_revision,
        },
        "provider": provider,
        "evaluator": {
            "product_revision": request.product_revision,
            "dirty": request.product_dirty,
        },
        "config_digest": request.config_digest or _digest(asdict(request)),
        "eligibility": _eligibility(request),
        "comparable": True,
    }
    accounting = _accounting(task, sample_count)
    return identity, accounting


def _result_metrics_for_task(
    *, task: Any, result_dict: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Select the task aggregate while ignoring LightEval's ``all`` summary."""

    results = result_dict.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("LightEval results are invalid")

    task_name = getattr(task, "full_name", None)
    if task_name is not None and task_name in results:
        aggregate = results[task_name]
    else:
        candidates = [(name, value) for name, value in results.items() if name != "all"]
        if len(candidates) != 1:
            raise ValueError("LightEval result must contain exactly one task aggregate")
        aggregate = candidates[0][1]

    if not isinstance(aggregate, Mapping):
        raise ValueError("LightEval aggregate metrics are invalid")
    return aggregate


def _accounting(task: Any, sample_count: int) -> dict[str, int]:
    source_rows = 0
    if task.dataset is not None:
        for split in task.config.evaluation_splits:
            source_rows += len(task.dataset[split])
    if source_rows < sample_count:
        source_rows = sample_count
    return {
        "source_rows": source_rows,
        # LightEval owns selection (including max_samples); unselected rows
        # are not dataset rejections and therefore stay in dataset_accepted.
        "dataset_accepted": source_rows,
        "dataset_rejected": 0,
        "selected": sample_count,
        "formatter_accepted": sample_count,
        "formatter_rejected": 0,
        "scored": sample_count,
        "model_invalid": 0,
        "provider_error": 0,
        "cache_error": 0,
        "scorer_error": 0,
        "harness_error": 0,
        "cancelled": 0,
    }


def _validate_identity_inputs(request: EvaluationRequest) -> None:
    required = {
        "checkpoint_sha256": request.checkpoint_sha256,
        "tokenizer_revision": request.tokenizer_revision,
        "chat_template_revision": request.chat_template_revision,
        "server_revision": request.server_revision,
        "wkv_mode": request.wkv_mode,
        "precision": request.precision,
        "gemm_policy": request.gemm_policy,
        "launch_contract": request.launch_contract,
        "product_revision": request.product_revision,
    }
    missing = [name for name, value in required.items() if not str(value).strip()]
    if missing:
        raise ValueError(
            "evaluation identity fields are required: " + ", ".join(missing)
        )
    if not _is_sha256(request.checkpoint_sha256):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
    if not re.fullmatch(r"[0-9a-f]{40}", request.product_revision):
        raise ValueError("product_revision must be a lowercase Git commit")


def _eligibility(request: EvaluationRequest) -> str:
    if (
        request.product_dirty
        or request.max_samples is not None
        or request.generation_limit is not None
    ):
        return "sanity"
    return "official"


def _write_manifest(
    *,
    run_dir: Path,
    identity: Mapping[str, Any],
    accounting: Mapping[str, int],
    completed_at: str,
) -> Path:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(run_dir).as_posix()
        encoded = path.read_bytes()
        artifacts.append(
            {
                "relative_path": relative,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "size_bytes": len(encoded),
            }
        )
    if not any(
        entry["relative_path"] == "terminal_evidence.json" for entry in artifacts
    ):
        raise ValueError("terminal evidence must be written before manifest")
    payload = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "status": "completed",
        "identity_digest": _digest(identity),
        "identities": {"run": identity},
        "accounting": dict(accounting),
        "artifacts": artifacts,
        "completed_at": completed_at,
    }
    path = run_dir / "manifest.json"
    _write_json(path, payload)
    return path


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _select_benchmark(task: str) -> _BenchmarkSelection:
    match = _CANONICAL_TASK_RE.fullmatch(task)
    if match is None:
        raise ValueError(
            "task must use lighteval/<family>/<benchmark>@<task-version> canonical identity"
        )

    family = match.group("family")
    benchmark_name = match.group("benchmark")
    version = match.group("version")
    if family in {"function-calling", "agent"}:
        raise UnsupportedTaskError(
            f"LightEval revision {LIGHTEVAL_REVISION} has no {family} benchmark task"
        )

    if family not in {"math", "knowledge", "coding", "instruction-following"}:
        raise UnsupportedTaskError(f"unsupported LightEval task family: {family}")

    info = get_benchmark_info(family, benchmark_name)
    if info is None:
        raise UnsupportedTaskError(f"unsupported LightEval benchmark: {benchmark_name}")
    configs = _light_eval_task_configs()
    config = configs.get(info.lighteval_task_name)
    if config is None:
        raise UnsupportedTaskError(
            f"task {task} is not registered by pinned LightEval revision {LIGHTEVAL_REVISION}"
        )

    if str(config.version) != version:
        raise UnsupportedTaskError(
            f"task {task} has no matching LightEval config version; "
            f"available version: {config.version}"
        )
    if family != "coding" and not _supports_generation_only(config):
        raise UnsupportedTaskError(
            f"task {task} requires an unsupported metric or generation backend"
        )
    return _BenchmarkSelection(
        canonical_task=task,
        info=info,
        task_version=version,
    )


@lru_cache(maxsize=1)
def _light_eval_task_configs() -> Mapping[str, Any]:
    from lighteval.tasks.registry import Registry

    return Registry.load_all_task_configs(load_multilingual=False)


def _supports_generation_only(config: Any) -> bool:
    metrics = tuple(getattr(config, "metrics", ()))
    if not metrics:
        return False
    for metric in metrics:
        category = str(
            getattr(
                getattr(metric, "category", None),
                "value",
                getattr(metric, "category", ""),
            )
        )
        metric_names = _metric_names(metric)
        if category != "GENERATIVE" or not metric_names:
            return False
        if any(
            metric_name not in _SUPPORTED_GENERATIVE_METRICS
            for metric_name in metric_names
        ):
            return False
    return True


def _metric_names(metric: Any) -> tuple[str, ...]:
    raw_name = getattr(metric, "metric_name", None)
    if isinstance(raw_name, str):
        return (raw_name,)
    if isinstance(raw_name, (list, tuple)):
        names = tuple(str(name) for name in raw_name)
        return names if all(names) else ()
    return ()


def _primary_metric_name(
    task: Any, *, aggregate: Mapping[str, Any] | None = None
) -> str:
    declared = tuple(
        name
        for metric in getattr(task, "metrics", ())
        for name in _metric_names(metric)
        if name in _SIGNED_BINARY_METRICS
    )
    if aggregate is not None:
        declared = tuple(name for name in declared if name in aggregate)
    if len(declared) != 1:
        available = ", ".join(declared) if declared else "none"
        raise ValueError(
            f"task must expose exactly one signed primary metric; found {available}"
        )
    return declared[0]


def _canonical_version(identity: str) -> str:
    return identity.rsplit("@", 1)[1]


def _ensure_generation_size(task: Any) -> None:
    current = getattr(task, "generation_size", None)
    if not isinstance(current, int) or isinstance(current, bool) or current <= 0:
        _set_generation_size(task, DEFAULT_GENERATION_LIMIT)


def _override_generation_size(task: Any, limit: int) -> None:
    _set_generation_size(task, limit)


def _set_generation_size(task: Any, limit: int) -> None:
    task.generation_size = limit
    task.config.generation_size = limit
    # get_docs() caches the generated documents on the task.
    docs = getattr(task, "_docs", None)
    if docs is not None:
        for doc in docs:
            doc.generation_size = limit


def _dataset_fingerprint(task: Any) -> str | None:
    if task.dataset is None:
        return None
    fingerprints = []
    for split in task.config.evaluation_splits:
        dataset = task.dataset[split]
        fingerprint = getattr(dataset, "_fingerprint", None)
        if not isinstance(fingerprint, str) or not fingerprint:
            return None
        fingerprints.append(f"{split}:{fingerprint}")
    return ";".join(fingerprints)


def _scorer_revision(task: Any) -> str:
    primary = _primary_metric_name(task)
    metric_candidates = [
        metric
        for metric in getattr(task, "metrics", ())
        if primary in _metric_names(metric)
    ]
    if len(metric_candidates) != 1:
        raise ValueError(
            f"signed primary metric {primary!r} must have exactly one owning metric"
        )
    metric = metric_candidates[0]
    scorer = getattr(metric, "sample_level_fn", None)
    if scorer is None:
        raise ValueError(
            f"signed primary metric {primary!r} has no sample-level scorer"
        )
    return _digest(
        {
            "lighteval": LIGHTEVAL_REVISION,
            "metric_name": primary,
            "metric": {
                "declared_names": _stable_identity_value(
                    getattr(metric, "metric_name", None)
                ),
                "higher_is_better": _stable_identity_value(
                    getattr(metric, "higher_is_better", None)
                ),
                "category": str(
                    getattr(
                        getattr(metric, "category", None),
                        "value",
                        getattr(metric, "category", None),
                    )
                ),
                "batched_compute": bool(getattr(metric, "batched_compute", False)),
                "corpus_level_fn": _stable_identity_value(
                    getattr(metric, "corpus_level_fn", None)
                ),
            },
            "scorer": _stable_identity_value(scorer),
            "version": str(task.config.version),
        }
    )


def _stable_identity_value(value: Any) -> Any:
    """Serialize callable instances without process-specific memory addresses."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _stable_identity_value(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_identity_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_stable_identity_value(item) for item in value]
        return sorted(
            items, key=lambda item: json.dumps(item, sort_keys=True, default=str)
        )
    if callable(value):
        try:
            source = inspect.getsource(value)
        except (OSError, TypeError):
            source = None
        if source is not None:
            return {
                "callable": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
                "source": source,
            }
    state = getattr(value, "__dict__", None)
    if isinstance(state, Mapping):
        return {
            "type": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "state": _stable_identity_value(state),
        }
    return {"type": f"{value.__class__.__module__}.{value.__class__.__qualname__}"}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_sha256(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
