from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from typing import Any, Iterable

from .config import EvaluationConfigurationError


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
