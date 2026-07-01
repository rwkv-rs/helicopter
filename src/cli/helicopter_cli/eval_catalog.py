from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any, Iterable


CATALOG_RESOURCE = "data/rwkv_skills_catalog.json"


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    name: str
    field: str
    dataset: str
    default_split: str
    cot_modes: tuple[str, ...]
    scheduler_jobs: tuple[str, ...]
    n_shots: tuple[int, ...]
    avg_ks: tuple[float, ...]
    pass_ks: tuple[int, ...]
    target_eval_attempts: int

    @property
    def dataset_slug(self) -> str:
        return f"{self.dataset}_{self.default_split}"


@dataclass(frozen=True, slots=True)
class RunnerSpec:
    name: str
    group: str
    scheduler_domain: str
    module: str
    is_cot: bool
    fallback_dataset_slugs: tuple[str, ...]
    extra_args: tuple[str, ...]
    batch_flag: str | None
    probe_flag: str | None
    probe_max_generate_flag: str | None
    probe_dataset_required: bool
    probe_extra_args: tuple[str, ...]
    probe_samples_per_task: int
    probe_question_floor: int


@dataclass(frozen=True, slots=True)
class InferenceDefaults:
    engine: str
    protocol: str
    model_path: str
    model_name: str
    base_url: str
    host: str
    port: int
    max_model_len: int
    tensor_parallel_size: int
    wkv_mode: str
    emb_device: str


@dataclass(frozen=True, slots=True)
class EvalJobPlan:
    benchmark: str
    field: str
    dataset_slug: str
    runner: str | None
    status: str
    module: str | None
    extra_args: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RwkvSkillsCatalog:
    schema_version: int
    source: dict[str, Any]
    aliases: dict[str, tuple[str, ...]]
    inference_defaults: InferenceDefaults
    benchmarks: tuple[BenchmarkSpec, ...]
    runners: tuple[RunnerSpec, ...]

    @property
    def runners_by_name(self) -> dict[str, RunnerSpec]:
        return {runner.name: runner for runner in self.runners}

    @property
    def benchmarks_by_name(self) -> dict[str, BenchmarkSpec]:
        return {benchmark.name: benchmark for benchmark in self.benchmarks}

    def field_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for benchmark in self.benchmarks:
            counts[benchmark.field] = counts.get(benchmark.field, 0) + 1
        return dict(sorted(counts.items()))

    def select_benchmarks(
        self,
        *,
        names: Iterable[str] | None = None,
        fields: Iterable[str] | None = None,
    ) -> tuple[BenchmarkSpec, ...]:
        requested_names = tuple(names or ())
        requested_fields = {field for field in (fields or ()) if field}
        by_name = self.benchmarks_by_name
        selected: list[BenchmarkSpec] = []

        if requested_names:
            for name in requested_names:
                if name == "all":
                    selected.extend(self.benchmarks)
                    continue
                targets = self.aliases.get(name, (name,))
                for target in targets:
                    try:
                        selected.append(by_name[target])
                    except KeyError as exc:
                        raise KeyError(f"unknown benchmark: {name}") from exc
        else:
            selected.extend(self.benchmarks)

        if requested_fields:
            selected = [item for item in selected if item.field in requested_fields]

        seen: set[str] = set()
        unique = []
        for item in selected:
            if item.name in seen:
                continue
            seen.add(item.name)
            unique.append(item)
        return tuple(unique)

    def build_job_plan(
        self,
        *,
        names: Iterable[str] | None = None,
        fields: Iterable[str] | None = None,
    ) -> tuple[EvalJobPlan, ...]:
        runners = self.runners_by_name
        rows: list[EvalJobPlan] = []
        for benchmark in self.select_benchmarks(names=names, fields=fields):
            if not benchmark.scheduler_jobs:
                rows.append(
                    EvalJobPlan(
                        benchmark=benchmark.name,
                        field=benchmark.field,
                        dataset_slug=benchmark.dataset_slug,
                        runner=None,
                        status="no_scheduler_job_in_rwkv_skills",
                        module=None,
                        extra_args=(),
                    )
                )
                continue
            for job_name in benchmark.scheduler_jobs:
                runner = runners.get(job_name)
                rows.append(
                    EvalJobPlan(
                        benchmark=benchmark.name,
                        field=benchmark.field,
                        dataset_slug=benchmark.dataset_slug,
                        runner=job_name,
                        status="ready" if runner else "missing_runner",
                        module=runner.module if runner else None,
                        extra_args=runner.extra_args if runner else (),
                    )
                )
        return tuple(rows)


def _tuple(value: Iterable[Any], cast: type = str) -> tuple[Any, ...]:
    return tuple(cast(item) for item in value)


def _benchmark(raw: dict[str, Any]) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=str(raw["name"]),
        field=str(raw["field"]),
        dataset=str(raw["dataset"]),
        default_split=str(raw["default_split"]),
        cot_modes=_tuple(raw.get("cot_modes", ())),
        scheduler_jobs=_tuple(raw.get("scheduler_jobs", ())),
        n_shots=_tuple(raw.get("n_shots", ()), int),
        avg_ks=_tuple(raw.get("avg_ks", ()), float),
        pass_ks=_tuple(raw.get("pass_ks", ()), int),
        target_eval_attempts=int(raw["target_eval_attempts"]),
    )


def _runner(raw: dict[str, Any]) -> RunnerSpec:
    return RunnerSpec(
        name=str(raw["name"]),
        group=str(raw["group"]),
        scheduler_domain=str(raw["scheduler_domain"]),
        module=str(raw["module"]),
        is_cot=bool(raw["is_cot"]),
        fallback_dataset_slugs=_tuple(raw.get("fallback_dataset_slugs", ())),
        extra_args=_tuple(raw.get("extra_args", ())),
        batch_flag=raw.get("batch_flag"),
        probe_flag=raw.get("probe_flag"),
        probe_max_generate_flag=raw.get("probe_max_generate_flag"),
        probe_dataset_required=bool(raw["probe_dataset_required"]),
        probe_extra_args=_tuple(raw.get("probe_extra_args", ())),
        probe_samples_per_task=int(raw["probe_samples_per_task"]),
        probe_question_floor=int(raw["probe_question_floor"]),
    )


def _inference_defaults(raw: dict[str, Any]) -> InferenceDefaults:
    return InferenceDefaults(
        engine=str(raw["engine"]),
        protocol=str(raw["protocol"]),
        model_path=str(raw["model_path"]),
        model_name=str(raw["model_name"]),
        base_url=str(raw["base_url"]),
        host=str(raw["host"]),
        port=int(raw["port"]),
        max_model_len=int(raw["max_model_len"]),
        tensor_parallel_size=int(raw["tensor_parallel_size"]),
        wkv_mode=str(raw["wkv_mode"]),
        emb_device=str(raw["emb_device"]),
    )


def load_rwkv_skills_catalog() -> RwkvSkillsCatalog:
    raw_text = resources.files("helicopter_cli").joinpath(CATALOG_RESOURCE).read_text()
    raw = json.loads(raw_text)
    return RwkvSkillsCatalog(
        schema_version=int(raw["schema_version"]),
        source=dict(raw["source"]),
        aliases={str(key): tuple(str(item) for item in value) for key, value in raw["aliases"].items()},
        inference_defaults=_inference_defaults(raw["inference_defaults"]),
        benchmarks=tuple(_benchmark(item) for item in raw["benchmarks"]),
        runners=tuple(_runner(item) for item in raw["runners"]),
    )


def job_plan_to_dict(row: EvalJobPlan) -> dict[str, Any]:
    return {
        "benchmark": row.benchmark,
        "field": row.field,
        "dataset_slug": row.dataset_slug,
        "runner": row.runner,
        "status": row.status,
        "module": row.module,
        "extra_args": list(row.extra_args),
    }
