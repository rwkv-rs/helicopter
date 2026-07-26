from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from itertools import groupby
from pathlib import Path
import stat
import tomllib
from typing import Any, Iterable, Literal, Mapping, cast
from urllib.parse import urlsplit


PromptTemplate = Literal["bot", "assistant", "function_calling"]
PROMPT_TEMPLATE_STOPS: dict[PromptTemplate, str] = {
    "bot": "✿",
    "assistant": "\nUser:",
    "function_calling": "\n### User",
}
CONFIG_KEYS = frozenset({"schema_version", "prompt_template", "weights", "benchmarks"})
REQUIRED_CONFIG_KEYS = frozenset({"schema_version", "weights", "benchmarks"})
SCHEMA_VERSION = 1


class EvaluationConfigurationError(ValueError):
    """The public eval config or private environment is invalid."""


@dataclass(frozen=True)
class EvaluationConfig:
    schema_version: int
    weights: tuple[str, ...]
    benchmarks: tuple[str, ...]
    prompt_template: PromptTemplate = "bot"


@dataclass(frozen=True)
class WeightIdentity:
    configured_path: str
    path: Path
    display_name: str
    sha256: str


@dataclass(frozen=True)
class EvaluationEnvironment:
    weight_root: Path
    scoreboard_url: str
    scoreboard_token: str
    staging_root: Path


def repository_root() -> Path:
    package_root = Path(__file__).resolve().parent
    for candidate in package_root.parents:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "eval" / "lighteval"
        ).is_dir():
            return candidate
    raise EvaluationConfigurationError(
        "cannot locate the Helicopter repository from the LightEval package"
    )


def load_evaluation_config(path: Path) -> EvaluationConfig:
    try:
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
    except FileNotFoundError as error:
        raise EvaluationConfigurationError(f"eval config not found: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise EvaluationConfigurationError(f"invalid eval TOML: {error}") from error

    unknown = sorted(set(raw) - CONFIG_KEYS)
    if unknown:
        raise EvaluationConfigurationError(
            "unknown eval config fields: " + ", ".join(unknown)
        )
    missing = sorted(REQUIRED_CONFIG_KEYS - set(raw))
    if missing:
        raise EvaluationConfigurationError(
            "missing eval config fields: " + ", ".join(missing)
        )
    version = raw["schema_version"]
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        raise EvaluationConfigurationError(f"schema_version must be {SCHEMA_VERSION}")
    prompt_template = raw.get("prompt_template", "bot")
    if (
        not isinstance(prompt_template, str)
        or prompt_template not in PROMPT_TEMPLATE_STOPS
    ):
        raise EvaluationConfigurationError(
            "prompt_template must be one of: " + ", ".join(PROMPT_TEMPLATE_STOPS)
        )
    weights = _string_array(raw["weights"], name="weights")
    benchmarks = _string_array(raw["benchmarks"], name="benchmarks")
    return EvaluationConfig(
        schema_version=SCHEMA_VERSION,
        prompt_template=cast(PromptTemplate, prompt_template),
        weights=weights,
        benchmarks=benchmarks,
    )


def _string_array(value: object, *, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, str) or not item or item != item.strip()
            for item in value
        )
    ):
        raise EvaluationConfigurationError(
            f"{name} must be a non-empty array of non-empty trimmed strings"
        )
    normalized = tuple(value)
    duplicates = sorted({item for item in normalized if normalized.count(item) > 1})
    if duplicates:
        raise EvaluationConfigurationError(
            f"duplicate {name} are not allowed: " + ", ".join(duplicates)
        )
    return normalized


def load_evaluation_environment(env: Mapping[str, str]) -> EvaluationEnvironment:
    names = {
        "WEIGHT_PATH": env.get("WEIGHT_PATH"),
        "HELICOPTER_SCOREBOARD_URL": env.get("HELICOPTER_SCOREBOARD_URL"),
        "HELICOPTER_SCOREBOARD_TOKEN": env.get("HELICOPTER_SCOREBOARD_TOKEN"),
        "HELICOPTER_EVAL_STAGING_ROOT": env.get("HELICOPTER_EVAL_STAGING_ROOT"),
    }
    missing = sorted(name for name, value in names.items() if not value)
    if missing:
        raise EvaluationConfigurationError(
            "missing private eval environment: " + ", ".join(missing)
        )
    raw_weight_root = Path(str(names["WEIGHT_PATH"])).expanduser()
    raw_staging_root = Path(str(names["HELICOPTER_EVAL_STAGING_ROOT"])).expanduser()
    if not raw_weight_root.is_absolute():
        raise EvaluationConfigurationError("WEIGHT_PATH must be an absolute path")
    if not raw_staging_root.is_absolute():
        raise EvaluationConfigurationError(
            "HELICOPTER_EVAL_STAGING_ROOT must be an absolute path"
        )
    if raw_weight_root.is_symlink():
        raise EvaluationConfigurationError("WEIGHT_PATH must not be a symlink")
    if raw_staging_root.is_symlink():
        raise EvaluationConfigurationError(
            "HELICOPTER_EVAL_STAGING_ROOT must not be a symlink"
        )
    weight_root = raw_weight_root.resolve()
    staging_root = raw_staging_root.resolve()
    product_root = repository_root()
    if not weight_root.is_dir():
        raise EvaluationConfigurationError(
            f"WEIGHT_PATH is not a directory: {weight_root}"
        )
    if staging_root.exists() and not staging_root.is_dir():
        raise EvaluationConfigurationError(
            "HELICOPTER_EVAL_STAGING_ROOT must be a directory"
        )
    if staging_root in {Path("/"), Path.home().resolve(), product_root}:
        raise EvaluationConfigurationError(
            "HELICOPTER_EVAL_STAGING_ROOT must be a dedicated child directory, "
            "not a filesystem, home, or product root"
        )
    if staging_root.exists():
        staging_status = staging_root.stat()
        if (
            staging_status.st_uid != os.geteuid()
            or stat.S_IMODE(staging_status.st_mode) != 0o700
        ):
            raise EvaluationConfigurationError(
                "an existing HELICOPTER_EVAL_STAGING_ROOT must be owned by "
                "the current user and have mode 0700"
            )
    if (
        staging_root == weight_root
        or staging_root.is_relative_to(weight_root)
        or weight_root.is_relative_to(staging_root)
    ):
        raise EvaluationConfigurationError(
            "HELICOPTER_EVAL_STAGING_ROOT and WEIGHT_PATH must not overlap"
        )
    scoreboard_url = str(names["HELICOPTER_SCOREBOARD_URL"]).rstrip("/")
    parsed_url = urlsplit(scoreboard_url)
    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.hostname
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise EvaluationConfigurationError(
            "HELICOPTER_SCOREBOARD_URL must be an http(s) origin/base path "
            "without credentials, query, or fragment"
        )
    scoreboard_token = str(names["HELICOPTER_SCOREBOARD_TOKEN"])
    if any(
        ord(character) < 0x21 or ord(character) > 0x7E for character in scoreboard_token
    ):
        raise EvaluationConfigurationError(
            "HELICOPTER_SCOREBOARD_TOKEN must contain only visible ASCII characters"
        )
    return EvaluationEnvironment(
        weight_root=weight_root,
        scoreboard_url=scoreboard_url,
        scoreboard_token=scoreboard_token,
        staging_root=staging_root,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_weights(
    config: EvaluationConfig, environment: EvaluationEnvironment
) -> tuple[WeightIdentity, ...]:
    root = environment.weight_root
    identities: list[WeightIdentity] = []
    seen_paths: set[Path] = set()
    seen_digests: set[str] = set()
    for configured in config.weights:
        relative = Path(configured)
        if (
            relative.is_absolute()
            or not relative.parts
            or "." in relative.parts
            or ".." in relative.parts
            or relative.as_posix() != configured
        ):
            raise EvaluationConfigurationError(
                "weight path must be a normalized relative child of "
                f"WEIGHT_PATH: {configured}"
            )
        unresolved = root / relative
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise EvaluationConfigurationError(
                    f"weight path must not contain symlinks: {configured}"
                )
        candidate = unresolved.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise EvaluationConfigurationError(
                f"weight path escapes WEIGHT_PATH: {configured}"
            ) from error
        if not candidate.is_file():
            raise EvaluationConfigurationError(f"weight file not found: {configured}")
        if candidate in seen_paths:
            raise EvaluationConfigurationError(
                f"duplicate resolved weight path: {configured}"
            )
        digest = _sha256(candidate)
        if digest in seen_digests:
            raise EvaluationConfigurationError(
                f"duplicate weight content is not allowed: {configured}"
            )
        seen_paths.add(candidate)
        seen_digests.add(digest)
        identities.append(
            WeightIdentity(
                configured_path=configured,
                path=candidate,
                display_name=candidate.name,
                sha256=digest,
            )
        )
    return tuple(identities)


def verify_weight_identity(weight: WeightIdentity) -> None:
    if (
        weight.path.is_symlink()
        or not weight.path.is_file()
        or weight.path.resolve() != weight.path
    ):
        raise EvaluationConfigurationError(
            f"evaluation weight path changed after preflight: {weight.configured_path}"
        )
    if _sha256(weight.path) != weight.sha256:
        raise EvaluationConfigurationError(
            f"evaluation weight content changed after preflight: "
            f"{weight.configured_path}"
        )


def public_environment(environment: EvaluationEnvironment) -> dict[str, str]:
    return {
        "weight_root": str(environment.weight_root),
        "scoreboard_url": environment.scoreboard_url,
        "scoreboard_token": "[REDACTED]",
        "staging_root": str(environment.staging_root),
    }


@dataclass(frozen=True)
class RegistryTask:
    selector: str
    identity: str
    name: str
    version: str
    module_family: str
    module: str
    dataset: str
    subset: str
    evaluation_splits: tuple[str, ...]
    languages: tuple[str, ...]
    upstream_tags: tuple[str, ...]

    def snapshot(self) -> dict[str, Any]:
        return {
            "selector": self.selector,
            "identity": self.identity,
            "name": self.name,
            "version": self.version,
            "module_family": self.module_family,
            "module": self.module,
            "dataset": self.dataset,
            "subset": self.subset,
            "evaluation_splits": self.evaluation_splits,
            "languages": self.languages,
            "upstream_tags": self.upstream_tags,
        }


@dataclass(frozen=True)
class RegistrySnapshot:
    lighteval_version: str
    configured_selectors: tuple[str, ...]
    resolved_selectors: tuple[str, ...]
    skipped_selectors: tuple[str, ...]
    tasks: tuple[RegistryTask, ...]
    module_count: int
    digest: str


@dataclass(frozen=True)
class _InventoryTask:
    name: str
    module_family: str
    module: str
    metadata: dict[str, object]


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


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        normalized = value.strip()
        return (normalized,) if normalized else ()
    if isinstance(value, (list, tuple)):
        return tuple(
            dict.fromkeys(
                item.strip() for item in value if isinstance(item, str) and item.strip()
            )
        )
    return ()


def _evaluation_splits(value: object, task_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise EvaluationConfigurationError(
            f"registry task has no evaluation split: {task_name}"
        )
    if any(
        not isinstance(split, str) or not split.strip() or split != split.strip()
        for split in value
    ):
        raise EvaluationConfigurationError(
            f"registry task has invalid evaluation splits: {task_name}"
        )
    normalized = tuple(value)
    if len(normalized) != len(set(normalized)):
        raise EvaluationConfigurationError(
            f"registry task has duplicate evaluation splits: {task_name}"
        )
    return normalized


def _inventory(rows: Iterable[object]) -> dict[str, _InventoryTask]:
    inventory: dict[str, _InventoryTask] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise EvaluationConfigurationError(
                "LightEval registry module inventory is invalid"
            )
        module = row.get("module")
        if (
            not isinstance(module, str)
            or not module.strip()
            or module != module.strip()
        ):
            raise EvaluationConfigurationError(
                "LightEval registry contains an invalid module name"
            )
        module_family = _module_family(module)
        if not module_family:
            raise EvaluationConfigurationError(
                "LightEval registry contains an empty module name"
            )
        raw_metadata = row.get("docstring")
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        raw_tasks = row.get("tasks")
        if not isinstance(raw_tasks, list):
            raise EvaluationConfigurationError(
                f"LightEval registry module lacks a task list: {module}"
            )
        for task in raw_tasks:
            if not isinstance(task, dict):
                raise EvaluationConfigurationError(
                    f"LightEval registry module has invalid task metadata: {module}"
                )
            task_name = task.get("name")
            if (
                not isinstance(task_name, str)
                or not task_name.strip()
                or task_name != task_name.strip()
            ):
                raise EvaluationConfigurationError(
                    f"LightEval registry module has invalid task metadata: {module}"
                )
            if task_name in inventory:
                known = inventory[task_name]
                if (
                    known.module == module
                    and _strings(known.metadata.get("languages"))
                    == _strings(metadata.get("languages"))
                    and _strings(known.metadata.get("tags"))
                    == _strings(metadata.get("tags"))
                ):
                    continue
                raise EvaluationConfigurationError(
                    f"LightEval registry task appears more than once: {task_name}"
                )
            inventory[task_name] = _InventoryTask(
                name=task_name,
                module_family=module_family,
                module=module,
                metadata=metadata,
            )
    if not inventory:
        raise EvaluationConfigurationError("LightEval registry is empty")
    return inventory


def _resolve_selectors(
    selectors: tuple[str, ...],
    inventory: dict[str, _InventoryTask],
) -> tuple[dict[str, str], tuple[str, ...], tuple[str, ...]]:
    task_names = tuple(sorted(inventory))
    selector_by_task: dict[str, str] = {}
    resolved: list[str] = []
    skipped: list[str] = []
    for selector in selectors:
        if selector in inventory:
            matches = (selector,)
        else:
            matches = tuple(
                name
                for name in task_names
                if ":" in name and name.split(":", 1)[0] == selector
            )
            if len(matches) < 2:
                matches = ()
        if not matches:
            skipped.append(selector)
            continue
        overlap = sorted(set(matches) & set(selector_by_task))
        if overlap:
            previous = selector_by_task[overlap[0]]
            raise EvaluationConfigurationError(
                "benchmark selectors overlap on task "
                f"{overlap[0]}: {previous}, {selector}"
            )
        resolved.append(selector)
        selector_by_task.update({name: selector for name in matches})
    if not selector_by_task:
        raise EvaluationConfigurationError(
            "none of the configured benchmark selectors exist in LightEval"
        )
    return selector_by_task, tuple(resolved), tuple(skipped)


def _snapshot_registry(
    registry: Any,
    lighteval_version: str,
    *,
    configured_selectors: tuple[str, ...] | None = None,
    selector_by_task: dict[str, str] | None = None,
    inventory_rows: Iterable[object] | None = None,
    skipped_selectors: tuple[str, ...] = (),
) -> RegistrySnapshot:
    loaded = registry.load_tasks()
    config_rows = [task.config for task in loaded.values()]
    configs: dict[str, object] = {}
    for config in config_rows:
        raw_name = config.name
        if (
            not isinstance(raw_name, str)
            or not raw_name.strip()
            or raw_name != raw_name.strip()
        ):
            raise EvaluationConfigurationError(
                "LightEval registry contains an invalid task config name"
            )
        if raw_name in configs:
            raise EvaluationConfigurationError(
                "LightEval registry contains duplicate task configs"
            )
        configs[raw_name] = config
    if not configs:
        raise EvaluationConfigurationError("selected LightEval registry is empty")

    inventory = _inventory(
        registry.get_tasks_dump() if inventory_rows is None else inventory_rows
    )
    if selector_by_task is None:
        selector_by_task = {name: name for name in configs}
    if set(configs) != set(selector_by_task):
        missing = sorted(set(selector_by_task) - set(configs))
        extra = sorted(set(configs) - set(selector_by_task))
        raise EvaluationConfigurationError(
            "selected LightEval registry does not match selector expansion; "
            f"missing={missing}, extra={extra}"
        )
    missing_metadata = sorted(set(configs) - set(inventory))
    if missing_metadata:
        raise EvaluationConfigurationError(
            "selected LightEval tasks are absent from module metadata: "
            + ", ".join(missing_metadata)
        )

    tasks: list[RegistryTask] = []
    for name in sorted(configs):
        config = configs[name]
        metadata = inventory[name]
        identity = config.full_name
        if (
            not isinstance(identity, str)
            or not identity.strip()
            or identity != identity.strip()
        ):
            raise EvaluationConfigurationError(
                f"registry task has no stable identity: {name}"
            )
        dataset = config.hf_repo
        if (
            not isinstance(dataset, str)
            or not dataset.strip()
            or dataset != dataset.strip()
        ):
            raise EvaluationConfigurationError(
                f"registry task has no dataset identity: {name}"
            )
        if config.hf_subset is not None and not isinstance(config.hf_subset, str):
            raise EvaluationConfigurationError(
                f"registry task has an invalid dataset subset: {name}"
            )
        subset = config.hf_subset or ""
        if subset != subset.strip():
            raise EvaluationConfigurationError(
                f"registry task has an invalid dataset subset: {name}"
            )
        raw_version = config.version
        if raw_version is None or isinstance(raw_version, bool):
            raise EvaluationConfigurationError(
                f"registry task has an invalid version: {name}"
            )
        version = str(raw_version)
        if not version or version != version.strip():
            raise EvaluationConfigurationError(
                f"registry task has an invalid version: {name}"
            )
        tasks.append(
            RegistryTask(
                selector=selector_by_task[name],
                identity=identity,
                name=name,
                version=version,
                module_family=metadata.module_family,
                module=metadata.module,
                dataset=dataset,
                subset=subset,
                evaluation_splits=_evaluation_splits(
                    config.evaluation_splits,
                    name,
                ),
                languages=_strings(metadata.metadata.get("languages")),
                upstream_tags=_strings(metadata.metadata.get("tags")),
            )
        )
    if len({task.identity for task in tasks}) != len(tasks):
        raise EvaluationConfigurationError(
            "selected LightEval registry contains duplicate task identities"
        )
    configured = (
        configured_selectors
        if configured_selectors is not None
        else tuple(dict.fromkeys(selector_by_task.values()))
    )
    resolved = tuple(
        selector for selector in configured if selector not in skipped_selectors
    )
    payload = {
        "lighteval_version": lighteval_version,
        "configured_selectors": configured,
        "resolved_selectors": resolved,
        "skipped_selectors": skipped_selectors,
        "tasks": [task.snapshot() for task in tasks],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return RegistrySnapshot(
        lighteval_version=lighteval_version,
        configured_selectors=configured,
        resolved_selectors=resolved,
        skipped_selectors=skipped_selectors,
        tasks=tuple(tasks),
        module_count=len({task.module_family for task in tasks}),
        digest=digest,
    )


def load_configured_registry(selectors: tuple[str, ...]) -> RegistrySnapshot:
    try:
        from lighteval.tasks.registry import Registry

        lighteval_version = importlib.metadata.version("lighteval")
        inventory_registry = Registry(
            tasks=None,
            load_multilingual=True,
            custom_tasks=None,
        )
        inventory_rows = inventory_registry.get_tasks_dump()
        inventory = _inventory(inventory_rows)
        selector_by_task, _resolved, skipped = _resolve_selectors(
            selectors,
            inventory,
        )
        selected_registry = Registry(
            tasks=",".join(selector_by_task),
            load_multilingual=True,
            custom_tasks=None,
        )
        return _snapshot_registry(
            selected_registry,
            lighteval_version,
            configured_selectors=selectors,
            selector_by_task=selector_by_task,
            inventory_rows=inventory_rows,
            skipped_selectors=skipped,
        )
    except EvaluationConfigurationError:
        raise
    except Exception as error:
        raise EvaluationConfigurationError(
            "cannot load configured LightEval benchmarks: "
            f"{type(error).__module__}.{type(error).__qualname__}"
        ) from error


WKV_MODES = ("fp16", "fp32io16")
MAX_TASKS_PER_SHARD = 1
EVAL_CONTRACT_VERSION = "lighteval-configured-selectors-v2"
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
    prompt_template: PromptTemplate = "bot"


@dataclass(frozen=True)
class EvaluationPlan:
    config_digest: str
    implementation_digest: str
    eval_contract_digest: str
    prompt_template: PromptTemplate
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
    repository = repository_root()
    evaluator_root = repository / "src" / "eval" / "lighteval"
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
        "prompt_template": config.prompt_template,
        "benchmarks": config.benchmarks,
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
            "prompt_template": config.prompt_template,
            "temperature": 0.96,
            "top_p": 0.76,
            "top_k": 32,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.1,
            "repetition_penalty": 1.0,
            "penalty_decay": 0.988,
            "max_new_tokens": 8192,
            "stop": "vllm-rwkv-prompt-template-owned",
            "ignore_eos": False,
        },
    }
    eval_contract_digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    shards = build_shards(registry)
    units = tuple(
        EvaluationUnit(
            weight=weight,
            wkv_mode=mode,
            shards=shards,
            prompt_template=config.prompt_template,
        )
        for weight in weights
        for mode in WKV_MODES
    )
    return EvaluationPlan(
        config_digest=config_digest,
        implementation_digest=implementation_digest,
        eval_contract_digest=eval_contract_digest,
        prompt_template=config.prompt_template,
        registry=registry,
        units=units,
    )


def public_plan(plan: EvaluationPlan) -> dict[str, object]:
    return {
        "config_digest": plan.config_digest,
        "implementation_digest": plan.implementation_digest,
        "eval_contract_digest": plan.eval_contract_digest,
        "prompt_template": plan.prompt_template,
        "registry": {
            "lighteval_version": plan.registry.lighteval_version,
            "task_count": len(plan.registry.tasks),
            "module_count": plan.registry.module_count,
            "digest": plan.registry.digest,
            "configured_selectors": plan.registry.configured_selectors,
            "resolved_selectors": plan.registry.resolved_selectors,
            "skipped_selectors": plan.registry.skipped_selectors,
            "tasks": [
                {
                    "selector": task.selector,
                    "identity": task.identity,
                    "module_family": task.module_family,
                    "module": task.module,
                    "dataset": task.dataset,
                    "subset": task.subset,
                    "evaluation_splits": task.evaluation_splits,
                    "languages": task.languages,
                    "upstream_tags": task.upstream_tags,
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
                "prompt_template": unit.prompt_template,
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
