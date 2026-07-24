"""Benchmark registry grouped by the evaluation fields used by rwkv-eval."""

from .benchmark import BenchmarkInfo, Field
from .coding import get_livecodebench_info
from .instruction_following import ALL_BENCHMARKS as INSTRUCTION_BENCHMARKS
from .knowledge import ALL_BENCHMARKS as KNOWLEDGE_BENCHMARKS
from .knowledge import get_mmlu_info
from .maths import ALL_BENCHMARKS as MATHS_BENCHMARKS


ALL_BENCHMARKS = (
    *MATHS_BENCHMARKS,
    *KNOWLEDGE_BENCHMARKS,
    *INSTRUCTION_BENCHMARKS,
)


def get_benchmark_info(field_name: str, benchmark_name: str) -> BenchmarkInfo | None:
    field_names = {
        "math": Field.MATHS,
        "knowledge": Field.KNOWLEDGE,
        "coding": Field.CODING,
        "instruction-following": Field.INSTRUCTION_FOLLOWING,
    }
    field = field_names.get(field_name)
    if field is None:
        return None

    for info in ALL_BENCHMARKS:
        if info.field == field and info.name.value == benchmark_name:
            return info
    if field == Field.KNOWLEDGE:
        return get_mmlu_info(benchmark_name)
    if field == Field.CODING:
        return get_livecodebench_info(benchmark_name)
    return None
