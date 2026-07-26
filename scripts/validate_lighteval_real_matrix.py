"""Run the bounded real-checkpoint LightEval acceptance matrix.

This is an internal acceptance harness, not a product evaluation entrypoint.
The product contract remains ``helicopter eval --config ...`` over the complete
default registry.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from helicopter_cli.env import load_env
from helicopter_eval.artifacts import publications_from_shard
from helicopter_eval.config import (
    load_evaluation_config,
    load_evaluation_environment,
    resolve_weights,
)
from helicopter_eval.lighteval_adapter import (
    STOP_SEQUENCE,
    _temporary_environment,
    evaluate_unit,
    evaluation_max_model_length,
)
from helicopter_eval.plan import WKV_MODES, EvaluationUnit, build_plan
from helicopter_eval.registry import load_default_registry
from helicopter_eval.runner import _process_environment


SUCCESS_TASKS = frozenset({"aime24|0", "aime25|0"})
PREREQUISITE_FAILURE_TASK = "aa_omniscience|0"
VALIDATION_CAMPAIGN_ID = "00000000-0000-0000-0000-000000000000"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed two-weight, paired-WKV real LightEval acceptance matrix"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/lighteval.toml"),
    )
    parser.add_argument("--env-file", default=".env.remote")
    return parser.parse_args()


def _selected_shards(unit: EvaluationUnit):
    identities = SUCCESS_TASKS | {PREREQUISITE_FAILURE_TASK}
    shards = tuple(
        shard
        for shard in unit.shards
        if len(shard.tasks) == 1 and shard.tasks[0].identity in identities
    )
    selected = {shard.tasks[0].identity for shard in shards}
    if selected != identities:
        missing = sorted(identities - selected)
        raise RuntimeError(f"validation registry tasks are missing: {missing}")
    return shards


def _publication_summary(
    *,
    output,
    unit: EvaluationUnit,
    registry_tasks,
) -> dict[str, Any]:
    publications = publications_from_shard(
        shard_dir=output.shard_dir,
        campaign_id=VALIDATION_CAMPAIGN_ID,
        unit=unit,
        shard=output.shard,
        model_execution=output.model_execution,
        registry_tasks=registry_tasks,
    )
    if len(publications) != 1:
        raise RuntimeError("validation shard did not produce exactly one publication")
    identity, payload, digest = publications[0]
    details = payload["details"]
    aggregates = payload["aggregates"]
    if (
        not isinstance(details, list)
        or not details
        or not isinstance(aggregates, dict)
        or not aggregates
    ):
        raise RuntimeError("validation publication lacks details or native aggregates")
    query_rows = sum(
        isinstance(row.get("doc"), dict)
        and isinstance(row["doc"].get("query"), str)
        and bool(row["doc"]["query"])
        for row in details
    )
    input_token_rows = sum(
        isinstance(row.get("model_response"), dict)
        and isinstance(row["model_response"].get("input_tokens"), list)
        and bool(row["model_response"]["input_tokens"])
        for row in details
    )
    metric_rows = sum(
        isinstance(row.get("metric"), dict) and bool(row["metric"]) for row in details
    )
    if query_rows != len(details):
        raise RuntimeError("validation details do not preserve every task-native query")
    if input_token_rows != len(details):
        raise RuntimeError(
            "validation details do not preserve every raw prompt token row"
        )
    if metric_rows != len(details):
        raise RuntimeError("validation details do not preserve every sample metric")
    return {
        "task_identity": identity,
        "document_count": len(details),
        "native_aggregates": sorted(aggregates),
        "query_rows": query_rows,
        "input_token_rows": input_token_rows,
        "metric_rows": metric_rows,
        "content_digest": digest,
    }


def _run_unit_matrix(
    *,
    plan,
    weights,
    registry_tasks,
    campaign_dir: Path,
) -> list[dict[str, Any]]:
    if len(weights) != 2:
        raise RuntimeError("validation requires exactly two configured weights")
    expected_units = len(weights) * len(WKV_MODES)
    validated_weights = {weight.sha256 for weight in weights}
    units = tuple(
        unit for unit in plan.units if unit.weight.sha256 in validated_weights
    )
    if len(units) != expected_units:
        raise RuntimeError("validation plan does not contain every paired WKV unit")

    summaries: list[dict[str, Any]] = []
    for unit in units:
        runtime_paths: list[Path] = []
        runtime_finished: list[bool] = []
        execution_records: list[dict[str, object]] = []
        outputs, failures = evaluate_unit(
            unit=unit,
            shards=_selected_shards(unit),
            campaign_dir=campaign_dir,
            on_shard_started=lambda _shard, _path, execution: execution_records.append(
                execution
            ),
            on_runtime_started=runtime_paths.append,
            on_runtime_finished=lambda: runtime_finished.append(True),
        )
        successful_tasks = {output.shard.tasks[0].identity for output in outputs}
        failed_tasks = {failure.shard.tasks[0].identity for failure in failures}
        if successful_tasks != SUCCESS_TASKS:
            raise RuntimeError(
                f"real validation success set mismatch: {sorted(successful_tasks)}"
            )
        if failed_tasks != {PREREQUISITE_FAILURE_TASK}:
            raise RuntimeError(
                f"real validation failure set mismatch: {sorted(failed_tasks)}"
            )
        if len(runtime_paths) != 1 or runtime_finished != [True]:
            raise RuntimeError("model lifecycle was not owned once for the unit")
        if runtime_paths[0].exists():
            raise RuntimeError("model runtime directory remains after unit cleanup")
        if not execution_records:
            raise RuntimeError("model execution metadata was not recorded")
        execution = execution_records[0]
        if any(record != execution for record in execution_records[1:]):
            raise RuntimeError("shards did not reuse one model execution")
        expected_capacity = 2560 if unit.wkv_mode == "fp16" else 1280
        if execution.get("max_num_seqs") != expected_capacity:
            raise RuntimeError(
                "resolved active capacity does not match the 7.2B/96G matrix"
            )

        publications = [
            _publication_summary(
                output=output,
                unit=unit,
                registry_tasks=registry_tasks,
            )
            for output in outputs
        ]
        summaries.append(
            {
                "weight_display_name": unit.weight.display_name,
                "weight_sha256": unit.weight.sha256,
                "wkv_mode": unit.wkv_mode,
                "model_execution": execution,
                "model_load_count": 1,
                "successful_shards": sorted(successful_tasks),
                "task_prerequisite_failures": [
                    {
                        "task_identity": failure.shard.tasks[0].identity,
                        "error_type": failure.error_type,
                        "error_phase": failure.error_phase,
                        "error_site": failure.error_site,
                    }
                    for failure in failures
                ],
                "publications": publications,
                "runtime_cleanup": "verified",
            }
        )
    return summaries


def _waiting_queue_and_state_reset(weight) -> dict[str, Any]:
    import torch
    from vllm import LLM, SamplingParams
    from vllm.transformers_utils.configs.rwkv7 import build_rwkv7_config_from_pth

    checkpoint = build_rwkv7_config_from_pth(str(weight.path))
    if checkpoint is None:
        raise RuntimeError("waiting-queue validation weight is not RWKV7")
    environment = {
        "VLLM_RWKV7_WKV_MODE": "fp16",
        "VLLM_RWKV7_EMB_DEVICE": "gpu",
        "VLLM_USE_RAPID_SAMPLER": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        "RKV_MODE": "off",
        "CMIX_SPARSE": "no-fc",
        "LOW_RANK_WEIGHT": "both",
        "ORIG_LINEAR_GROUPS": "none",
    }
    with _temporary_environment(environment):
        llm = LLM(
            model=weight.path.as_uri(),
            dtype="float16",
            max_model_len=evaluation_max_model_length(
                checkpoint.max_position_embeddings
            ),
            max_num_seqs=2,
            max_num_batched_tokens=checkpoint.max_position_embeddings,
            enforce_eager=True,
            enable_prefix_caching=False,
        )
        prompts = [
            f"User: Give the exact integer result of {number}+{number}.\n"
            "Assistant: <think>"
            for number in range(1, 5)
        ]
        parameters = SamplingParams(
            temperature=0,
            max_tokens=64,
            stop=[STOP_SEQUENCE],
        )
        first = llm.generate(prompts, parameters)
        second = llm.generate(prompts, parameters)
        first_tokens = [output.outputs[0].token_ids for output in first]
        second_tokens = [output.outputs[0].token_ids for output in second]
        if len(first) != 4 or len(second) != 4:
            raise RuntimeError("waiting queue did not return every accepted request")
        if first_tokens != second_tokens:
            raise RuntimeError("reused recurrent rows changed repeated generation")
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    digest = hashlib.sha256(
        json.dumps(first_tokens, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "weight_display_name": weight.display_name,
        "weight_sha256": weight.sha256,
        "wkv_mode": "fp16",
        "submitted_requests": 4,
        "active_capacity": 2,
        "first_returned": len(first_tokens),
        "second_returned": len(second_tokens),
        "identical_token_rows": True,
        "token_digest": digest,
    }


def main() -> None:
    args = _parse_args()
    root = Path.cwd()
    private_env, _ = load_env(
        root,
        args.env_file,
        use_fallbacks=False,
        require_private=True,
    )
    with _process_environment(private_env):
        config = load_evaluation_config(args.config)
        environment = load_evaluation_environment(private_env)
        weights = resolve_weights(config, environment)
        registry = load_default_registry()
        plan = build_plan(config, weights, registry)
        with tempfile.TemporaryDirectory(
            prefix="lighteval-real-matrix-",
            dir=environment.staging_root,
        ) as directory:
            unit_matrix = _run_unit_matrix(
                plan=plan,
                weights=weights,
                registry_tasks=registry.tasks,
                campaign_dir=Path(directory),
            )
        waiting_queue = _waiting_queue_and_state_reset(weights[0])

    print(
        json.dumps(
            {
                "schema_version": 1,
                "configured_weight_count": len(weights),
                "validated_unit_count": len(unit_matrix),
                "paired_wkv_modes": list(WKV_MODES),
                "unit_matrix": unit_matrix,
                "waiting_queue_and_state_reset": waiting_queue,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
