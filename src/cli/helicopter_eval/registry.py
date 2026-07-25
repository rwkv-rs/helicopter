from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from typing import Any

from .domains import (
    DOMAIN_RULES_VERSION,
    DomainAssignment,
    assign_domain,
    rules_digest,
)
from .config import EvaluationConfigurationError


@dataclass(frozen=True)
class RegistryTask:
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
    primary_domain: str

    def snapshot(self) -> dict[str, Any]:
        return {
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
            "primary_domain": self.primary_domain,
        }


@dataclass(frozen=True)
class RegistrySnapshot:
    lighteval_version: str
    tasks: tuple[RegistryTask, ...]
    module_count: int
    digest: str
    domain_rules_version: str
    domain_rules_digest: str
    unknown_domain_modules: tuple[str, ...]


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


def _snapshot_registry(registry: Any, lighteval_version: str) -> RegistrySnapshot:
    loaded = registry.load_tasks()
    config_rows = [task.config for task in loaded.values()]
    config_names: list[str] = []
    for config in config_rows:
        raw_name = config.name
        if (
            not isinstance(raw_name, str)
            or not raw_name.strip()
            or raw_name != raw_name.strip()
        ):
            raise EvaluationConfigurationError(
                "default built-in registry contains an invalid task config name"
            )
        config_names.append(raw_name)
    if len(config_names) != len(set(config_names)):
        raise EvaluationConfigurationError(
            "default built-in registry contains duplicate task configs"
        )
    configs = {
        name: config for name, config in zip(config_names, config_rows, strict=True)
    }
    if not configs:
        raise EvaluationConfigurationError("default built-in registry is empty")
    module_metadata: dict[str, tuple[str, dict[str, object]]] = {}
    task_modules: dict[str, str] = {}
    for row in registry.get_tasks_dump():
        if not isinstance(row, dict) or "module" not in row:
            raise EvaluationConfigurationError(
                "default built-in registry module inventory is invalid"
            )
        module = row["module"]
        if (
            not isinstance(module, str)
            or not module.strip()
            or module != module.strip()
        ):
            raise EvaluationConfigurationError(
                "default built-in registry contains an invalid module name"
            )
        family = module.removeprefix("lighteval.tasks.tasks.").removesuffix(".main")
        if not family:
            raise EvaluationConfigurationError(
                "default built-in registry contains an empty module family"
            )
        metadata = row.get("docstring")
        parsed = metadata if isinstance(metadata, dict) else {}
        known_module = module_metadata.get(family)
        if known_module is not None and known_module[0] != module:
            raise EvaluationConfigurationError(
                f"default registry module family is ambiguous: {family}"
            )
        module_metadata[family] = (module, parsed)
        raw_tasks = row.get("tasks")
        if not isinstance(raw_tasks, list):
            raise EvaluationConfigurationError(
                f"default registry module lacks a task list: {module}"
            )
        for task in raw_tasks:
            if not isinstance(task, dict):
                raise EvaluationConfigurationError(
                    f"default registry module has invalid task metadata: {module}"
                )
            task_name = task.get("name")
            if (
                not isinstance(task_name, str)
                or not task_name.strip()
                or task_name != task_name.strip()
            ):
                raise EvaluationConfigurationError(
                    f"default registry module has invalid task metadata: {module}"
                )
            if task_name in task_modules:
                raise EvaluationConfigurationError(
                    f"default registry task appears more than once: {task_name}"
                )
            task_modules[task_name] = family

    if set(configs) != set(task_modules):
        missing = sorted(set(configs) - set(task_modules))
        extra = sorted(set(task_modules) - set(configs))
        raise EvaluationConfigurationError(
            "default built-in registry module inventory mismatch; "
            f"missing={missing}, extra={extra}"
        )

    tasks: list[RegistryTask] = []
    unknown: set[str] = set()
    for name in sorted(configs):
        config = configs[name]
        family = task_modules[name]
        module, metadata = module_metadata[family]
        tags = _strings(metadata.get("tags"))
        assignment: DomainAssignment = assign_domain(family, tags)
        if assignment.primary_domain == "other":
            unknown.add(family)
        evaluation_splits = _evaluation_splits(
            config.evaluation_splits,
            name,
        )
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
                identity=identity,
                name=name,
                version=version,
                module_family=family,
                module=module,
                dataset=dataset,
                subset=subset,
                evaluation_splits=evaluation_splits,
                languages=_strings(metadata.get("languages")),
                upstream_tags=assignment.upstream_tags,
                primary_domain=assignment.primary_domain,
            )
        )
    if len({task.identity for task in tasks}) != len(tasks):
        raise EvaluationConfigurationError(
            "default built-in registry contains duplicate task identities"
        )
    payload = {
        "lighteval_version": lighteval_version,
        "tasks": [task.snapshot() for task in tasks],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return RegistrySnapshot(
        lighteval_version=lighteval_version,
        tasks=tuple(tasks),
        module_count=len({task.module_family for task in tasks}),
        digest=digest,
        domain_rules_version=DOMAIN_RULES_VERSION,
        domain_rules_digest=rules_digest(),
        unknown_domain_modules=tuple(sorted(unknown)),
    )


def load_default_registry() -> RegistrySnapshot:
    try:
        from lighteval.tasks.registry import Registry

        return _snapshot_registry(
            Registry(tasks=None, load_multilingual=False, custom_tasks=None),
            importlib.metadata.version("lighteval"),
        )
    except EvaluationConfigurationError:
        raise
    except Exception as error:
        raise EvaluationConfigurationError(
            "cannot load the complete default LightEval registry: "
            f"{type(error).__module__}.{type(error).__qualname__}"
        ) from error
