from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Mapping

from .execution import Eligibility
from .data_sources.resources import BundledResource
from .framework import LightEvalTaskRuntime
from .task_runtime import ScoringTransform, TaskRuntime, unchanged_scoring_text
from .tasks.coding import CodingRuntime
from .tasks.function_calling import FunctionCallingRuntime
from .tasks.math import transform_math_scoring_text


class MetricDirection(StrEnum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True, slots=True)
class MetricContract:
    primary_metric: str
    direction: MetricDirection
    minimum: float
    maximum: float
    binary_correctness_metric: str | None = None
    additional_metrics: frozenset[str] = frozenset()
    aggregation_rule: str = "mean"

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ValueError("metric minimum must not exceed maximum")
        if self.aggregation_rule not in {"mean", "sum"}:
            raise ValueError(f"unsupported aggregation rule: {self.aggregation_rule}")

    @property
    def allowed_metrics(self) -> frozenset[str]:
        names = {self.primary_metric, *self.additional_metrics}
        if self.binary_correctness_metric is not None:
            names.add(self.binary_correctness_metric)
        return frozenset(names)

    def validate(self, metrics: Mapping[str, float]) -> None:
        unknown = sorted(set(metrics) - self.allowed_metrics)
        if unknown:
            raise ValueError(f"unknown task metrics: {', '.join(unknown)}")
        if self.primary_metric not in metrics:
            raise ValueError(f"missing primary metric: {self.primary_metric}")
        primary = metrics[self.primary_metric]
        if not self.minimum <= primary <= self.maximum:
            raise ValueError(
                f"primary metric is outside [{self.minimum}, {self.maximum}]"
            )
        if self.binary_correctness_metric is not None:
            binary = metrics.get(self.binary_correctness_metric)
            if binary not in {0.0, 1.0}:
                raise ValueError("binary correctness metric must be exactly 0 or 1")

    def correctness(self, metrics: Mapping[str, float]) -> bool | None:
        if self.binary_correctness_metric is None:
            return None
        value = metrics.get(self.binary_correctness_metric)
        if value is None:
            raise ValueError(
                f"missing binary correctness metric: {self.binary_correctness_metric}"
            )
        if value not in {0.0, 1.0}:
            raise ValueError("binary correctness metric must be exactly 0 or 1")
        return value == 1.0

    def aggregate(self, values: list[float]) -> float:
        if not values:
            raise ValueError("cannot aggregate an empty metric partition")
        if self.aggregation_rule == "sum":
            return sum(values)
        return sum(values) / len(values)


@dataclass(frozen=True, slots=True)
class TaskContract:
    identity: str
    dataset_revision: str
    prompt_revision: str
    scorer_revision: str
    generation_contract: str
    dataset_owner: str
    prompt_owner: str
    scorer_owner: str
    harness_revision: str | None
    eligibility: Eligibility
    metric: MetricContract

    @property
    def can_publish_official(self) -> bool:
        return (
            self.eligibility is Eligibility.OFFICIAL
            and bool(
                self.dataset_revision and self.prompt_revision and self.scorer_revision
            )
            and bool(self.dataset_owner and self.prompt_owner and self.scorer_owner)
        )


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    contract: TaskContract
    family: str
    upstream_module: str
    upstream_task_name: str
    dataset_repository: str
    dataset_subset: str
    dataset_split: str
    dataset_revision: str
    generation_limit: int
    runtime_factory: Callable[[], TaskRuntime]
    scoring_transform: Callable[[str, str, bool, str], ScoringTransform]
    asset_name: str
    source_file: str
    snapshot_sha256: str
    expected_rows: int
    required_columns: frozenset[str]
    bundled_resource: BundledResource | None = None

    def load_runtime(self) -> TaskRuntime:
        return self.runtime_factory()


LIGHTEVAL_REVISION = "64f4f5ae173626509fad6e477ca4ee56ebb26129"


def _official_contract(
    *, identity: str, task_version: int, dataset_revision: str, primary_metric: str
) -> TaskContract:
    revision = f"lighteval-{LIGHTEVAL_REVISION}-task-v{task_version}"
    return TaskContract(
        identity=identity,
        dataset_revision=dataset_revision,
        prompt_revision=revision,
        scorer_revision=revision,
        generation_contract="rwkv-stop-v1",
        dataset_owner="huggingface-dataset-snapshot",
        prompt_owner="pinned-lighteval-task",
        scorer_owner="pinned-lighteval-task",
        harness_revision=None,
        eligibility=Eligibility.OFFICIAL,
        metric=MetricContract(
            primary_metric=primary_metric,
            direction=MetricDirection.HIGHER_IS_BETTER,
            minimum=0.0,
            maximum=1.0,
            binary_correctness_metric=primary_metric,
            aggregation_rule="mean",
        ),
    )


def _lighteval_runtime(
    *,
    module_name: str,
    task_name: str,
    dataset_repository: str,
    generation_limit: int,
    primary_metric: str,
) -> Callable[[], TaskRuntime]:
    return lambda: LightEvalTaskRuntime(
        module_name=module_name,
        task_name=task_name,
        dataset_repository=dataset_repository,
        generation_limit=generation_limit,
        primary_metric=primary_metric,
    )


def _proxy_contract(
    *,
    identity: str,
    dataset_revision: str,
    scorer_revision: str,
    metric: str,
    harness_revision: str | None = None,
) -> TaskContract:
    return TaskContract(
        identity=identity,
        dataset_revision=dataset_revision,
        prompt_revision=f"{identity}-prompt-v1",
        scorer_revision=scorer_revision,
        generation_contract="rwkv-stop-v1",
        dataset_owner="lighteval-runner-bundled-resource",
        prompt_owner="lighteval-runner-proxy-task",
        scorer_owner="lighteval-runner-proxy-task",
        harness_revision=harness_revision,
        eligibility=Eligibility.PROXY,
        metric=MetricContract(
            primary_metric=metric,
            direction=MetricDirection.HIGHER_IS_BETTER,
            minimum=0.0,
            maximum=1.0,
            binary_correctness_metric=metric if harness_revision is None else None,
            aggregation_rule="mean",
        ),
    )


CANONICAL_TASKS: Mapping[str, TaskDefinition] = {
    "lighteval/math/gsm8k@0": TaskDefinition(
        contract=_official_contract(
            identity="lighteval/math/gsm8k@0",
            task_version=0,
            dataset_revision="e53f048856ff4f594e959d75785d2c2d37b678ee",
            primary_metric="extractive_match",
        ),
        family="math",
        upstream_module="lighteval.tasks.tasks.gsm8k",
        upstream_task_name="gsm8k",
        dataset_repository="openai/gsm8k",
        dataset_subset="main",
        dataset_split="test",
        dataset_revision="e53f048856ff4f594e959d75785d2c2d37b678ee",
        generation_limit=256,
        runtime_factory=_lighteval_runtime(
            module_name="lighteval.tasks.tasks.gsm8k",
            task_name="gsm8k",
            dataset_repository="openai/gsm8k",
            generation_limit=256,
            primary_metric="extractive_match",
        ),
        scoring_transform=transform_math_scoring_text,
        asset_name="lighteval_gsm8k_test",
        source_file="main/test-00000-of-00001.parquet",
        snapshot_sha256="ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
        expected_rows=1319,
        required_columns=frozenset({"question", "answer"}),
    ),
    "lighteval/math/math-500@2": TaskDefinition(
        contract=_official_contract(
            identity="lighteval/math/math-500@2",
            task_version=2,
            dataset_revision="33b8b6415e2c3765cb6b0ac1a63b550167e7eb87",
            primary_metric="pass@k:k=1&n=1",
        ),
        family="math",
        upstream_module="lighteval.tasks.tasks.math_500",
        upstream_task_name="math_500",
        dataset_repository="HuggingFaceH4/MATH-500",
        dataset_subset="default",
        dataset_split="test",
        dataset_revision="33b8b6415e2c3765cb6b0ac1a63b550167e7eb87",
        generation_limit=32768,
        runtime_factory=_lighteval_runtime(
            module_name="lighteval.tasks.tasks.math_500",
            task_name="math_500",
            dataset_repository="HuggingFaceH4/MATH-500",
            generation_limit=32768,
            primary_metric="pass@k:k=1&n=1",
        ),
        scoring_transform=transform_math_scoring_text,
        asset_name="lighteval_math_500_test",
        source_file="test.jsonl",
        snapshot_sha256="35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132",
        expected_rows=500,
        required_columns=frozenset(
            {"problem", "solution", "answer", "subject", "level", "unique_id"}
        ),
    ),
    "lighteval/knowledge/mmlu-abstract-algebra@0": TaskDefinition(
        contract=_official_contract(
            identity="lighteval/knowledge/mmlu-abstract-algebra@0",
            task_version=0,
            dataset_revision="740c944ee6cacc8334f25bc00d8b3b6f20999d60",
            primary_metric="em",
        ),
        family="knowledge",
        upstream_module="lighteval.tasks.tasks.mmlu",
        upstream_task_name="mmlu:abstract_algebra",
        dataset_repository="lighteval/mmlu",
        dataset_subset="abstract_algebra",
        dataset_split="test",
        dataset_revision="740c944ee6cacc8334f25bc00d8b3b6f20999d60",
        generation_limit=5,
        runtime_factory=_lighteval_runtime(
            module_name="lighteval.tasks.tasks.mmlu",
            task_name="mmlu:abstract_algebra",
            dataset_repository="lighteval/mmlu",
            generation_limit=5,
            primary_metric="em",
        ),
        scoring_transform=unchanged_scoring_text,
        asset_name="lighteval_mmlu_abstract_algebra_test",
        source_file="abstract_algebra/test-00000-of-00001.parquet",
        snapshot_sha256="2d2cc95a39503ecbd1999b674894c9579dd3244aa76a9e525bbf19bb990f6720",
        expected_rows=100,
        required_columns=frozenset({"question", "subject", "choices", "answer"}),
    ),
    "helicopter-proxy/function-calling/exact-json@1": TaskDefinition(
        contract=_proxy_contract(
            identity="helicopter-proxy/function-calling/exact-json@1",
            dataset_revision="bundled:function-calling-v1",
            scorer_revision="exact-function-call-v1",
            metric="exact_function_call",
        ),
        family="function-calling",
        upstream_module="",
        upstream_task_name="",
        dataset_repository="helicopter/lighteval-runner",
        dataset_subset="bundled",
        dataset_split="test",
        dataset_revision="bundled:function-calling-v1",
        generation_limit=256,
        runtime_factory=FunctionCallingRuntime,
        scoring_transform=unchanged_scoring_text,
        asset_name="function_calling_v1",
        source_file="tasks/assets/function_calling_v1.jsonl",
        snapshot_sha256="a574d09feebb54877e39ac2b7dcde9ca41024ab9644a77c9ddc73d5dcd414191",
        expected_rows=1,
        required_columns=frozenset({"id", "prompt", "tool", "expected"}),
        bundled_resource=BundledResource(
            "tasks/assets/function_calling_v1.jsonl",
            "a574d09feebb54877e39ac2b7dcde9ca41024ab9644a77c9ddc73d5dcd414191",
        ),
    ),
    "helicopter-proxy/coding/python-stdio@1": TaskDefinition(
        contract=_proxy_contract(
            identity="helicopter-proxy/coding/python-stdio@1",
            dataset_revision="bundled:coding-v1",
            scorer_revision="sandbox-pass-rate-v1",
            metric="sandbox_pass_rate",
            harness_revision="landlock-seccomp-python-v3",
        ),
        family="coding",
        upstream_module="",
        upstream_task_name="",
        dataset_repository="helicopter/lighteval-runner",
        dataset_subset="bundled",
        dataset_split="test",
        dataset_revision="bundled:coding-v1",
        generation_limit=512,
        runtime_factory=CodingRuntime,
        scoring_transform=unchanged_scoring_text,
        asset_name="coding_v1",
        source_file="tasks/assets/coding_v1.jsonl",
        snapshot_sha256="b1d0b8296dab584aec5c9a768c739ba0e610fc430cdd05ad343a5143a0715712",
        expected_rows=1,
        required_columns=frozenset({"id", "prompt", "cases"}),
        bundled_resource=BundledResource(
            "tasks/assets/coding_v1.jsonl",
            "b1d0b8296dab584aec5c9a768c739ba0e610fc430cdd05ad343a5143a0715712",
        ),
    ),
}


def get_task_definition(identity: str, *, allow_proxy: bool = False) -> TaskDefinition:
    try:
        definition = CANONICAL_TASKS[identity]
    except KeyError as error:
        raise ValueError(f"unknown canonical task identity: {identity}") from error
    if definition.contract.eligibility is not Eligibility.OFFICIAL and not allow_proxy:
        raise ValueError("proxy task requires explicit opt-in")
    return definition
