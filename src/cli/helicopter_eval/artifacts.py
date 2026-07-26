from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .manifest import ManifestError, validate_campaign_child
from .plan import EvaluationShard, EvaluationUnit
from .registry import RegistryTask


class ArtifactError(RuntimeError):
    pass


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    except (TypeError, ValueError) as error:
        raise ArtifactError(
            "standard artifacts contain non-canonical JSON data"
        ) from error


def content_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _validate_standard_file(shard_dir: Path, path: Path) -> Path:
    try:
        candidate, _ = validate_campaign_child(shard_dir, path)
    except ManifestError as error:
        raise ArtifactError(
            "standard artifact path is not a safe shard child"
        ) from error
    if path.is_symlink() or not candidate.is_file():
        raise ArtifactError("standard artifacts must be regular non-symlink files")
    return candidate


def _standard_artifacts(
    shard_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path, list[Path]]:
    import pyarrow.parquet as parquet

    result_files = sorted(shard_dir.glob("results/**/results_*.json"))
    if len(result_files) != 1:
        raise ArtifactError("expected exactly one standard results JSON per shard")
    result_file = _validate_standard_file(shard_dir, result_files[0])
    stamp = result_file.stem.removeprefix("results_")
    model_dir = result_file.parent.relative_to(shard_dir / "results")
    detail_files = sorted(
        (shard_dir / "details" / model_dir / stamp).glob(f"details_*_{stamp}.parquet")
    )
    if not detail_files:
        raise ArtifactError("expected standard details parquet files")
    detail_files = [_validate_standard_file(shard_dir, path) for path in detail_files]
    try:
        results = json.loads(
            result_file.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ArtifactError("standard results JSON is invalid") from error
    if not isinstance(results, dict):
        raise ArtifactError("standard results JSON must be an object")
    try:
        rows = [
            row for path in detail_files for row in parquet.read_table(path).to_pylist()
        ]
    except Exception as error:
        raise ArtifactError("standard details parquet is invalid") from error
    if any(not isinstance(row, dict) for row in rows):
        raise ArtifactError("standard detail rows must be objects")
    return results, rows, result_file, detail_files


def _completion_diagnostics(
    rows: list[dict[str, Any]],
    *,
    effective_limit: int,
) -> dict[str, int | float]:
    completions = 0
    truncated = 0
    violations = 0
    for row in rows:
        response = row.get("model_response")
        if not isinstance(response, dict):
            raise ArtifactError("detail model_response must be an object")
        texts = response.get("text")
        tokens = response.get("output_tokens")
        if texts in (None, []):
            _validate_loglikelihood_response(response)
            continue
        if not isinstance(texts, list) or not isinstance(tokens, list):
            raise ArtifactError("completion text/output_tokens must be arrays")
        if len(texts) != len(tokens):
            raise ArtifactError("completion and output-token counts differ")
        _validate_optional_text_output(
            response,
            key="text_post_processed",
            expected_count=len(texts),
        )
        _validate_optional_text_output(
            response,
            key="reasonings",
            expected_count=len(texts),
            allow_none=True,
        )
        for text, token_ids in zip(texts, tokens, strict=True):
            if not isinstance(text, str):
                raise ArtifactError("completion text must be a string")
            _validate_token_group(token_ids)
            completions += 1
            truncated += int(len(token_ids) >= effective_limit)
            violations += int("\nUser:" in text)
    return {
        "samples": len(rows),
        "completions": completions,
        "truncated": truncated,
        "non_truncated": completions - truncated,
        "truncation_rate": truncated / completions if completions else 0.0,
        "turn_boundary_violations": violations,
        "turn_boundary_violation_rate": violations / completions
        if completions
        else 0.0,
    }


def _validate_token_group(value: object) -> None:
    if not isinstance(value, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in value
    ):
        raise ArtifactError("output tokens must be integer arrays")


def _validate_optional_text_output(
    response: dict[str, Any],
    *,
    key: str,
    expected_count: int,
    allow_none: bool = False,
) -> None:
    value = response.get(key)
    if value is None or (key == "reasonings" and value == []):
        return
    if (
        not isinstance(value, list)
        or len(value) != expected_count
        or any(
            not isinstance(item, str) and not (allow_none and item is None)
            for item in value
        )
    ):
        raise ArtifactError(f"{key} must align one-for-one with completion text")


def _validate_loglikelihood_response(response: dict[str, Any]) -> None:
    logprobs = response.get("logprobs")
    argmax = response.get("argmax_logits_eq_gold")
    if logprobs in (None, []) and argmax in (None, []):
        raise ArtifactError("empty completion lacks log-likelihood evidence")
    if logprobs not in (None, []):
        if not isinstance(logprobs, list) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in logprobs
        ):
            raise ArtifactError("logprobs must be a finite numeric array")
    if argmax not in (None, []):
        if not isinstance(argmax, list) or any(
            not isinstance(value, bool) for value in argmax
        ):
            raise ArtifactError("argmax evidence must be a boolean array")
    evidence_count = (
        len(logprobs) if isinstance(logprobs, list) and logprobs else len(argmax)
    )
    if (
        isinstance(logprobs, list)
        and logprobs
        and isinstance(argmax, list)
        and argmax
        and len(logprobs) != len(argmax)
    ):
        raise ArtifactError("log-likelihood evidence counts differ")
    output_tokens = response.get("output_tokens")
    if not isinstance(output_tokens, list) or not output_tokens:
        raise ArtifactError("log-likelihood output_tokens must be a non-empty array")
    for token_group in output_tokens:
        _validate_token_group(token_group)
        if not token_group:
            raise ArtifactError("log-likelihood output token groups must be non-empty")
    if len(output_tokens) != evidence_count:
        raise ArtifactError("log-likelihood evidence and output-token counts differ")
    _validate_optional_text_output(
        response,
        key="text_post_processed",
        expected_count=0,
    )
    _validate_optional_text_output(
        response,
        key="reasonings",
        expected_count=0,
        allow_none=True,
    )


def _sampling_config(results: dict[str, Any]) -> dict[str, Any]:
    try:
        config = results["config_general"]["model_config"]
        generation = config["generation_parameters"]
    except (KeyError, TypeError) as error:
        raise ArtifactError("results lack model generation configuration") from error
    if not isinstance(config, dict) or not isinstance(generation, dict):
        raise ArtifactError("model generation configuration must be objects")
    sampling = {
        "temperature": generation.get("temperature"),
        "top_p": generation.get("top_p"),
        "top_k": generation.get("top_k"),
        "presence_penalty": generation.get("presence_penalty"),
        "repetition_penalty": generation.get("repetition_penalty")
        or generation.get("frequency_penalty"),
        "backend_frequency_penalty": 0.0,
        "penalty_decay": generation.get("penalty_decay"),
        "max_new_tokens": generation.get("max_new_tokens"),
        "stop": generation.get("stop_tokens"),
        "ignore_eos": False,
        "seed": config.get("seed"),
    }
    required = {
        "temperature": 0.96,
        "top_p": 0.76,
        "top_k": 32,
        "presence_penalty": 1.0,
        "repetition_penalty": 0.1,
        "backend_frequency_penalty": 0.0,
        "penalty_decay": 0.988,
        "max_new_tokens": 8192,
        "stop": ["\nUser:"],
        "ignore_eos": False,
    }
    mismatched = [
        key for key, expected in required.items() if sampling.get(key) != expected
    ]
    if mismatched:
        raise ArtifactError(
            "standard results violate the evaluation sampling contract: "
            + ", ".join(mismatched)
        )
    return sampling


def _validate_model_execution(
    model_execution: dict[str, object],
    unit: EvaluationUnit,
) -> None:
    expected_gemm_policy = (
        "fp16-accumulation" if unit.wkv_mode == "fp16" else "fp32-accumulation"
    )
    expected = {
        "weight_sha256": unit.weight.sha256,
        "weight_display_name": unit.weight.display_name,
        "wkv_mode": unit.wkv_mode,
        "gemm_policy": expected_gemm_policy,
    }
    mismatched = [
        key for key, value in expected.items() if model_execution.get(key) != value
    ]
    if mismatched:
        raise ArtifactError(
            "model execution does not match the planned unit: " + ", ".join(mismatched)
        )
    for key in ("max_num_seqs", "max_num_batched_tokens"):
        value = model_execution.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ArtifactError(f"model execution {key} must be a positive integer")
    gpu = model_execution.get("gpu")
    if not isinstance(gpu, str) or not gpu:
        raise ArtifactError("model execution must identify the GPU")
    dependency_versions = model_execution.get("dependency_versions")
    if (
        not isinstance(dependency_versions, dict)
        or not {"lighteval", "vllm", "torch"}.issubset(dependency_versions)
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            for name, version in dependency_versions.items()
        )
    ):
        raise ArtifactError(
            "model execution must record lighteval, vllm, and torch versions"
        )


def _effective_limit(
    sampling: dict[str, Any],
    task_config: dict[str, Any],
) -> int:
    value = sampling.get("max_new_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        value = task_config.get("generation_size")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArtifactError("effective output limit is missing")
    return value


def _expected_task(
    unit: EvaluationUnit,
    task,
) -> dict[str, object]:
    return {
        "identity": f"{unit.weight.sha256}:{unit.wkv_mode}:{task.identity}",
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


def _is_uncertainty_aggregate(name: str) -> bool:
    return name == "stderr" or name.endswith("_stderr")


def _standard_task_sets(
    shard: EvaluationShard,
    registry_tasks: tuple[RegistryTask, ...],
) -> tuple[set[str], set[str]]:
    config_names = {task.identity for task in shard.tasks}
    aggregate_names = set(config_names)
    registry_names = {task.identity for task in registry_tasks}
    for task in shard.tasks:
        task_name, separator, few_shot = task.identity.rpartition("|")
        if not separator or ":" in task_name:
            continue
        # LightEval treats a root name with colon-qualified siblings as a
        # superset selector. Derive that exact expansion from the locked
        # registry instead of accepting arbitrary extra result tasks.
        members: set[str] = set()
        for identity in registry_names:
            registry_task_name = identity.rpartition("|")[0]
            if registry_task_name == task_name or registry_task_name.startswith(
                f"{task_name}:"
            ):
                members.add(f"{registry_task_name}|{few_shot}")
        if len(members) <= 1:
            continue
        config_names.update(members)
        aggregate_names.add(f"{task_name}:_average|{few_shot}")
    aggregate_names.update(config_names)
    return config_names, aggregate_names


def publications_from_shard(
    *,
    shard_dir: Path,
    campaign_id: str,
    unit: EvaluationUnit,
    shard: EvaluationShard,
    model_execution: dict[str, object],
    registry_tasks: tuple[RegistryTask, ...],
) -> list[tuple[str, dict[str, object], str]]:
    _validate_model_execution(model_execution, unit)
    results, rows, result_file, detail_files = _standard_artifacts(shard_dir)
    raw_task_results = results.get("results")
    raw_task_configs = results.get("config_tasks")
    general_config = results.get("config_general")
    if not isinstance(raw_task_results, dict) or not isinstance(raw_task_configs, dict):
        raise ArtifactError("results lack task aggregates/configs")
    if (
        not isinstance(general_config, dict)
        or general_config.get("max_samples") is not None
    ):
        raise ArtifactError("standard result does not prove max_samples=None")
    expected = {task.identity: task for task in shard.tasks}
    standard_config_names, standard_aggregate_names = _standard_task_sets(
        shard,
        registry_tasks,
    )
    result_names = {name for name in raw_task_results if name != "all"}
    if (
        result_names != standard_aggregate_names
        or set(raw_task_configs) != standard_config_names
    ):
        raise ArtifactError(
            "standard result task set does not match deterministic shard"
        )
    rows_by_task: dict[str, list[dict[str, Any]]] = {
        name: [] for name in standard_config_names
    }
    for row in rows:
        if (
            not isinstance(row.get("doc"), dict)
            or not isinstance(row.get("metric"), dict)
            or not isinstance(row.get("model_response"), dict)
        ):
            raise ArtifactError(
                "detail doc, metric, and model_response must be objects"
            )
        try:
            task_name = row["doc"]["task_name"]
        except (KeyError, TypeError) as error:
            raise ArtifactError("detail row lacks doc.task_name") from error
        if task_name not in rows_by_task:
            raise ArtifactError(f"unexpected detail task: {task_name}")
        rows_by_task[task_name].append(row)
    document_indices_by_task: dict[str, list[int]] = {}
    for task_name in sorted(standard_config_names):
        task_config = raw_task_configs[task_name]
        if not isinstance(task_config, dict):
            raise ArtifactError(f"invalid task config for {task_name}")
        original_docs = task_config.get("original_num_docs")
        effective_docs = task_config.get("effective_num_docs")
        skipped_multiselect_docs = task_config.get("skipped_multiselect_docs")
        document_indices: list[int] = []
        for row in rows_by_task[task_name]:
            try:
                document_index = row["doc"]["specific"]["helicopter_document_index"]
            except (KeyError, TypeError) as error:
                raise ArtifactError(
                    f"task detail lacks stable document index: {task_name}"
                ) from error
            if isinstance(document_index, bool) or not isinstance(document_index, int):
                raise ArtifactError(
                    f"task detail document index is invalid: {task_name}"
                )
            document_indices.append(document_index)
        if (
            isinstance(original_docs, bool)
            or not isinstance(original_docs, int)
            or isinstance(effective_docs, bool)
            or not isinstance(effective_docs, int)
            or isinstance(skipped_multiselect_docs, bool)
            or not isinstance(skipped_multiselect_docs, int)
            or original_docs <= 0
            or effective_docs <= 0
            or skipped_multiselect_docs < 0
            or original_docs != effective_docs + skipped_multiselect_docs
            or set(document_indices) != set(range(effective_docs))
        ):
            raise ArtifactError(
                f"task detail count does not account for the full evaluation "
                f"split: {task_name}"
            )
        document_indices_by_task[task_name] = document_indices
    sampling = _sampling_config(results)
    dependency_versions = model_execution.get("dependency_versions")
    lighteval_version = (
        dependency_versions.get("lighteval")
        if isinstance(dependency_versions, dict)
        else None
    )
    if not isinstance(lighteval_version, str) or not lighteval_version:
        raise ArtifactError("model execution lacks the LightEval version")
    artifact = {
        "lighteval_version": lighteval_version,
        "results_path": str(result_file.relative_to(shard_dir)),
        "details_paths": [str(path.relative_to(shard_dir)) for path in detail_files],
    }
    publications: list[tuple[str, dict[str, object], str]] = []
    for task_name in sorted(expected):
        task_config = raw_task_configs[task_name]
        aggregates = raw_task_results[task_name]
        task_rows = rows_by_task[task_name]
        document_indices = document_indices_by_task[task_name]
        if not isinstance(task_config, dict) or not isinstance(aggregates, dict):
            raise ArtifactError(f"invalid task result/config for {task_name}")
        numeric: dict[str, float] = {}
        for key, value in aggregates.items():
            if (
                not isinstance(key, str)
                or not key
                or key != key.strip()
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise ArtifactError(f"task aggregate is invalid: {task_name}/{key}")
            if not math.isfinite(value):
                raise ArtifactError(f"task aggregate is not finite: {task_name}/{key}")
            numeric[key] = value
        primary_candidates = [
            key for key in numeric if not _is_uncertainty_aggregate(key)
        ]
        if not primary_candidates:
            raise ArtifactError(f"task has no native aggregate: {task_name}")
        primary_metric = primary_candidates[0]
        task = _expected_task(unit, expected[task_name])
        details = [
            {
                "sample_index": index,
                "document_index": document_indices[index],
                "doc": row["doc"],
                "metric": row["metric"],
                "model_response": row["model_response"],
            }
            for index, row in enumerate(task_rows)
        ]
        payload: dict[str, object] = {
            "schema_version": "lighteval-task-v2",
            "campaign_id": campaign_id,
            "task": task,
            "artifact": artifact,
            "task_config": task_config,
            "model": model_execution,
            "sampling_config": sampling,
            "primary_metric": primary_metric,
            "aggregates": numeric,
            "diagnostics": _completion_diagnostics(
                task_rows,
                effective_limit=_effective_limit(sampling, task_config),
            ),
            "details": details,
        }
        identity = str(task["identity"])
        publications.append((identity, payload, content_digest(payload)))
    return publications
