from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import uuid
from pathlib import Path
from typing import Mapping

from helicopter_lighteval.publish import (
    PublicationError,
    ScoreboardClient,
    content_digest,
)

LM_EVAL_VERSION = "0.4.12"


def task_metadata(manager, selectors: tuple[str, ...], tasks: tuple[str, ...]):
    rows: list[dict[str, object]] = []
    selector_leaves = {
        selector: _leaf_tasks(manager, manager.match_tasks([selector]))
        for selector in selectors
    }
    leaf_tasks = list(
        dict.fromkeys(
            task_name
            for resolved in tasks
            for task_name in _leaf_tasks(manager, [resolved])
        )
    )
    for task_name in leaf_tasks:
        entry = manager.task_index[task_name]
        config = entry.cfg or {}
        metadata = config.get("metadata") or {}
        selector = next(
            value for value in selectors if task_name in selector_leaves[value]
        )
        splits = [
            str(config[name])
            for name in ("test_split", "validation_split")
            if config.get(name)
        ]
        rows.append(
            {
                "selector": selector,
                "task_name": task_name,
                "task_version": str(metadata.get("version", "0")),
                "module_family": entry.yaml_path.parent.name,
                "module": _portable_task_path(entry.yaml_path),
                "dataset": str(config.get("dataset_path") or "unknown"),
                "subset": str(config.get("dataset_name") or ""),
                "evaluation_splits": list(dict.fromkeys(splits or ["unknown"])),
                "languages": _strings(metadata.get("languages")),
                "upstream_tags": sorted(entry.tags),
            }
        )
    return rows


def _portable_task_path(path: Path) -> str:
    parts = path.parts
    task_root = max(
        (index for index, part in enumerate(parts) if part == "tasks"),
        default=-1,
    )
    if task_root >= 0:
        return Path(*parts[task_root:]).as_posix()
    return path.name


def _leaf_tasks(manager, names):
    leaves: list[str] = []
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise PublicationError(f"lm-eval task registry contains a cycle at {name}")
        entry = manager.task_index[name]
        kind = entry.kind.name.lower()
        if kind in {"task", "py_task"}:
            if name not in leaves:
                leaves.append(name)
            return
        visiting.add(name)
        if kind == "group":
            members = (entry.cfg or {}).get("task", [])
        elif kind == "tag":
            members = sorted(
                child_name
                for child_name, child in manager.task_index.items()
                if name in child.tags
            )
        else:
            raise PublicationError(f"unsupported lm-eval registry entry kind: {kind}")
        if not isinstance(members, list) or not members:
            raise PublicationError(f"lm-eval registry entry has no members: {name}")
        for child in members:
            if not isinstance(child, str) or child not in manager.task_index:
                raise PublicationError(f"lm-eval registry member is invalid: {child}")
            visit(child)
        visiting.remove(name)

    for name in names:
        visit(name)
    return leaves


def expected_tasks(config, metadata: list[dict[str, object]]):
    return [
        {
            "identity": f"{unit.weight_sha256}:{unit.wkv_mode}:{task['task_name']}",
            "weight_sha256": unit.weight_sha256,
            "weight_display_name": unit.weight.name,
            "wkv_mode": unit.wkv_mode,
            **task,
        }
        for unit in config.execution_units
        for task in metadata
    ]


def campaign_payload(config, metadata, expected):
    resolved = list(dict.fromkeys(str(task["selector"]) for task in metadata))
    skipped = [selector for selector in config.tasks if selector not in resolved]
    return {
        "schema_version": "lm-eval-campaign-v1",
        "run_key": hashlib.sha256(uuid.uuid4().bytes).hexdigest(),
        "config_digest": content_digest(
            {
                "backend": config.backend,
                "tasks": list(config.tasks),
                "weights": list(
                    dict.fromkeys(
                        unit.weight_sha256 for unit in config.execution_units
                    )
                ),
                "wkv_modes": list(
                    dict.fromkeys(unit.wkv_mode for unit in config.execution_units)
                ),
                "batch_size": config.batch_size,
                "eot_token_id": config.eot_token_id,
                "max_gen_toks": config.max_gen_toks,
                "limit": config.limit,
            }
        ),
        "registry_digest": content_digest(metadata),
        "eval_contract_digest": content_digest(
            {
                "evaluator": {"name": "lm-eval", "version": LM_EVAL_VERSION},
                "batch_size": config.batch_size,
                "eot_token_id": config.eot_token_id,
                "max_gen_toks": config.max_gen_toks,
                "limit": config.limit,
            }
        ),
        "evaluator": {"name": "lm-eval", "version": LM_EVAL_VERSION},
        "configured_selectors": list(config.tasks),
        "resolved_selectors": resolved,
        "skipped_selectors": skipped,
        "expected_tasks": expected,
    }


def publish_unit(*, output_dir: Path, campaign_id: str, expected, unit, config, client):
    results = _read_json(output_dir / "results.json")
    artifact = _read_json(output_dir / "artifacts.json")
    aggregates_by_task = results.get("results")
    samples_by_task = results.get("samples")
    configs = results.get("configs")
    sample_counts = results.get("n-samples")
    if not isinstance(aggregates_by_task, dict) or not isinstance(samples_by_task, dict):
        raise PublicationError("lm-eval publication requires results and samples")
    for task in expected:
        task_name = str(task["task_name"])
        aggregates = _aggregates(aggregates_by_task.get(task_name))
        rows = samples_by_task.get(task_name)
        if not aggregates or not isinstance(rows, list) or not rows:
            raise PublicationError(f"lm-eval output is incomplete for {task_name}")
        primary = next((name for name in aggregates if "stderr" not in name), None)
        if primary is None:
            raise PublicationError(f"lm-eval output has no primary metric for {task_name}")
        details = [_detail(task_name, index, row) for index, row in enumerate(rows)]
        original_docs, effective_docs = _document_counts(
            sample_counts, task_name, details
        )
        task_config = (
            dict(configs.get(task_name, {}))
            if isinstance(configs, dict) and isinstance(configs.get(task_name), dict)
            else {}
        )
        task_config.update(
            original_num_docs=original_docs,
            effective_num_docs=effective_docs,
            skipped_multiselect_docs=original_docs - effective_docs,
        )
        sample_path = next(
            (
                item["path"]
                for item in artifact.get("sample_artifacts", [])
                if item.get("task_name") == task_name
            ),
            None,
        )
        if not isinstance(sample_path, str):
            raise PublicationError(f"lm-eval sample artifact is missing for {task_name}")
        payload = {
            "schema_version": "lm-eval-task-v1",
            "campaign_id": campaign_id,
            "task": task,
            "artifact": {
                "evaluator": {"name": "lm-eval", "version": LM_EVAL_VERSION},
                "results_path": "results.json",
                "details_paths": [sample_path],
            },
            "task_config": task_config,
            "model": {
                "weight_sha256": unit.weight_sha256,
                "weight_display_name": unit.weight.name,
                "wkv_mode": unit.wkv_mode,
                "prompt_template": "none",
                "gemm_policy": (
                    "fp16-accumulation"
                    if unit.wkv_mode == "fp16"
                    else "fp32-accumulation"
                ),
                "gpu": "remote-vllm-pool",
                "max_num_seqs": unit.manifest.total_capacity,
                "max_num_batched_tokens": unit.manifest.max_model_len,
                "dependency_versions": {
                    "lm-eval": LM_EVAL_VERSION,
                    "vllm": unit.manifest.vllm_version,
                    "torch": importlib.metadata.version("torch"),
                },
                "evaluator": "lm-eval",
            },
            "sampling_config": {
                "output_type": task_config.get("output_type"),
                "batch_size": config.batch_size,
                "eot_token_id": config.eot_token_id,
                "default_max_gen_toks": config.max_gen_toks,
                "generation_kwargs": task_config.get("generation_kwargs", {}),
            },
            "primary_metric": primary,
            "aggregates": aggregates,
            "diagnostics": _diagnostics(details),
            "details": details,
        }
        client.publish_task(campaign_id, str(task["identity"]), payload)
    return len(expected)


def _detail(task_name: str, index: int, raw: object):
    if not isinstance(raw, dict):
        raise PublicationError("lm-eval sample must be an object")
    document_index = raw.get("doc_id")
    if (
        isinstance(document_index, bool)
        or not isinstance(document_index, int)
        or document_index < 0
    ):
        raise PublicationError("lm-eval sample must contain a non-negative doc_id")
    raw_doc = raw.get("doc")
    doc = dict(raw_doc) if isinstance(raw_doc, dict) else {"value": raw_doc}
    doc.update(
        task_name=task_name,
        specific={"helicopter_document_index": document_index},
    )
    metrics = raw.get("metrics")
    if isinstance(metrics, dict):
        metric = dict(metrics)
    elif isinstance(metrics, list) and all(isinstance(name, str) for name in metrics):
        filter_name = raw.get("filter")
        metric = {
            (
                f"{name},{filter_name}"
                if isinstance(filter_name, str) and filter_name
                else name
            ): raw[name]
            for name in metrics
            if name in raw
        }
    else:
        raise PublicationError("lm-eval sample metrics are invalid")
    response = {key: value for key, value in raw.items() if key not in {"doc", "metrics"}}
    return {
        "sample_index": index,
        "document_index": document_index,
        "doc": doc,
        "metric": metric,
        "model_response": response,
    }


def _document_counts(raw: object, task_name: str, details) -> tuple[int, int]:
    if not isinstance(raw, Mapping) or not isinstance(raw.get(task_name), Mapping):
        raise PublicationError(f"lm-eval n-samples is missing for {task_name}")
    counts = raw[task_name]
    original = counts.get("original")
    effective = counts.get("effective")
    if (
        isinstance(original, bool)
        or not isinstance(original, int)
        or isinstance(effective, bool)
        or not isinstance(effective, int)
        or original <= 0
        or effective <= 0
        or effective > original
    ):
        raise PublicationError(f"lm-eval n-samples is invalid for {task_name}")
    if {detail["document_index"] for detail in details} != set(range(effective)):
        raise PublicationError(f"lm-eval sample doc_id coverage is invalid for {task_name}")
    return original, effective


def _diagnostics(details):
    completions = sum(
        len(detail["model_response"].get("filtered_resps", []))
        if isinstance(detail["model_response"].get("filtered_resps"), list)
        else 0
        for detail in details
    )
    return {
        "samples": len(details),
        "completions": completions,
        "truncated": 0,
        "non_truncated": completions,
        "truncation_rate": 0.0,
        "turn_boundary_violations": 0,
        "turn_boundary_violation_rate": 0.0,
    }


def _aggregates(raw: object):
    if not isinstance(raw, Mapping):
        return {}
    return {
        name: float(value)
        for name, value in raw.items()
        if isinstance(name, str)
        and name not in {"alias", "name", "sample_len"}
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    }


def _strings(raw: object):
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return sorted({value for value in raw if isinstance(value, str) and value})
    return []


def _read_json(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicationError(f"invalid lm-eval artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise PublicationError(f"lm-eval artifact must be an object: {path.name}")
    return value
