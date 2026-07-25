from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from itertools import groupby
from pathlib import Path

from .config import EvaluationConfig, WeightIdentity
from .registry import RegistrySnapshot, RegistryTask


WKV_MODES = ("fp16", "fp32io16")
MAX_TASKS_PER_SHARD = 1
EVAL_CONTRACT_VERSION = "lighteval-full-registry-v1"
_VLLM_CONTRACT_FILES = (
    "vllm/config/model.py",
    "vllm/engine/arg_utils.py",
    "vllm/entrypoints/llm.py",
    "vllm/tokenizers/registry.py",
    "vllm/tokenizers/rwkv.py",
    "vllm/tokenizers/rwkv_defaults.py",
    "vllm/transformers_utils/configs/rwkv7.py",
    "vllm/transformers_utils/tokenizer.py",
    "vllm/model_executor/models/rwkv7.py",
    "vllm/v1/core/sched/rwkv_decode_wave.py",
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/engine/core.py",
    "vllm/v1/engine/input_processor.py",
    "vllm/v1/worker/gpu/model_states/rwkv.py",
    "vllm/v1/worker/gpu/sample/sampler.py",
    "vllm/v1/worker/gpu_worker.py",
)


@dataclass(frozen=True)
class EvaluationShard:
    shard_id: str
    module_family: str
    tasks: tuple[RegistryTask, ...]


@dataclass(frozen=True)
class EvaluationUnit:
    weight: WeightIdentity
    wkv_mode: str
    shards: tuple[EvaluationShard, ...]


@dataclass(frozen=True)
class EvaluationPlan:
    config_digest: str
    implementation_digest: str
    eval_contract_digest: str
    registry: RegistrySnapshot
    units: tuple[EvaluationUnit, ...]

    @property
    def expected_task_count(self) -> int:
        return sum(len(shard.tasks) for unit in self.units for shard in unit.shards)


def _chunks(tasks: tuple[RegistryTask, ...]) -> tuple[tuple[RegistryTask, ...], ...]:
    return tuple(
        tasks[index : index + MAX_TASKS_PER_SHARD]
        for index in range(0, len(tasks), MAX_TASKS_PER_SHARD)
    )


def build_shards(registry: RegistrySnapshot) -> tuple[EvaluationShard, ...]:
    shards: list[EvaluationShard] = []
    for module_family, rows in groupby(
        sorted(registry.tasks, key=lambda task: (task.module_family, task.identity)),
        key=lambda task: task.module_family,
    ):
        module_tasks = tuple(rows)
        chunks = _chunks(module_tasks)
        for index, tasks in enumerate(chunks):
            suffix = f"{index + 1:03d}-of-{len(chunks):03d}"
            shards.append(
                EvaluationShard(
                    shard_id=f"{module_family}:{suffix}",
                    module_family=module_family,
                    tasks=tasks,
                )
            )
    return tuple(shards)


def _implementation_digest() -> str:
    repository = Path(__file__).resolve().parents[3]
    evaluator_root = repository / "src" / "cli" / "helicopter_eval"
    sources = sorted(evaluator_root.glob("*.py"))
    sources.extend(
        repository / "src" / "infer" / "vllm-rwkv" / relative
        for relative in _VLLM_CONTRACT_FILES
    )
    digest = hashlib.sha256()
    for path in sources:
        relative = path.relative_to(repository)
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_plan(
    config: EvaluationConfig,
    weights: tuple[WeightIdentity, ...],
    registry: RegistrySnapshot,
) -> EvaluationPlan:
    config_public = {
        "schema_version": config.schema_version,
        "weights": [
            {"configured_path": weight.configured_path, "sha256": weight.sha256}
            for weight in weights
        ],
    }
    config_digest = hashlib.sha256(
        json.dumps(config_public, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    implementation_digest = _implementation_digest()
    contract = {
        "version": EVAL_CONTRACT_VERSION,
        "implementation_digest": implementation_digest,
        "wkv_modes": WKV_MODES,
        "gemm_policy_by_wkv_mode": {
            "fp16": "fp16-accumulation",
            "fp32io16": "fp32-accumulation",
        },
        "max_samples": None,
        "save_details": True,
        "remove_reasoning_tags": False,
        "prompt_context": "checkpoint-filename-ctx",
        "max_model_len": "checkpoint-context-plus-max-new-tokens",
        "perplexity": {
            "prompt": "raw-task-query-without-chat-template",
            "scoring": "rolling-loglikelihood-each-token-once",
            "window": "checkpoint-context-minus-context-and-dummy-token",
        },
        "generation": {
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
        },
    }
    eval_contract_digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    shards = build_shards(registry)
    units = tuple(
        EvaluationUnit(weight=weight, wkv_mode=mode, shards=shards)
        for weight in weights
        for mode in WKV_MODES
    )
    return EvaluationPlan(
        config_digest=config_digest,
        implementation_digest=implementation_digest,
        eval_contract_digest=eval_contract_digest,
        registry=registry,
        units=units,
    )


def public_plan(plan: EvaluationPlan) -> dict[str, object]:
    return {
        "config_digest": plan.config_digest,
        "implementation_digest": plan.implementation_digest,
        "eval_contract_digest": plan.eval_contract_digest,
        "registry": {
            "lighteval_version": plan.registry.lighteval_version,
            "task_count": len(plan.registry.tasks),
            "module_count": plan.registry.module_count,
            "digest": plan.registry.digest,
            "domain_rules_version": plan.registry.domain_rules_version,
            "domain_rules_digest": plan.registry.domain_rules_digest,
            "unknown_domain_count": len(plan.registry.unknown_domain_modules),
            "unknown_domain_modules": plan.registry.unknown_domain_modules,
            "tasks": [
                {
                    "identity": task.identity,
                    "module_family": task.module_family,
                    "module": task.module,
                    "dataset": task.dataset,
                    "subset": task.subset,
                    "evaluation_splits": task.evaluation_splits,
                    "languages": task.languages,
                    "upstream_tags": task.upstream_tags,
                    "primary_domain": task.primary_domain,
                }
                for task in plan.registry.tasks
            ],
        },
        "weights": [
            {
                "configured_path": weight.configured_path,
                "display_name": weight.display_name,
                "sha256": weight.sha256,
            }
            for weight in dict.fromkeys(unit.weight for unit in plan.units)
        ],
        "execution_units": [
            {
                "configured_path": unit.weight.configured_path,
                "display_name": unit.weight.display_name,
                "sha256": unit.weight.sha256,
                "wkv_mode": unit.wkv_mode,
                "shard_count": len(unit.shards),
                "task_count": sum(len(shard.tasks) for shard in unit.shards),
            }
            for unit in plan.units
        ],
        "execution_unit_count": len(plan.units),
        "expected_task_count": plan.expected_task_count,
        "shards": [
            {
                "shard_id": shard.shard_id,
                "module_family": shard.module_family,
                "task_identities": [task.identity for task in shard.tasks],
            }
            for shard in (plan.units[0].shards if plan.units else ())
        ],
    }
