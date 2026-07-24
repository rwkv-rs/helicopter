from ..benchmark import BenchmarkInfo, BenchmarkName, Field
from .gpqa import ALL_BENCHMARKS as GPQA_BENCHMARKS
from .mmlu import Mmlu
from .mmlu_pro import MMLU_PRO_INFO


ALL_BENCHMARKS = (MMLU_PRO_INFO, *GPQA_BENCHMARKS)


def get_mmlu_info(benchmark_name: str) -> BenchmarkInfo | None:
    if not benchmark_name.startswith("mmlu-") or benchmark_name == "mmlu-pro":
        return None
    subject = benchmark_name.removeprefix("mmlu-").replace("-", "_")
    if not subject:
        return None
    info = BenchmarkInfo(
        name=BenchmarkName(benchmark_name),
        field=Field.KNOWLEDGE,
        display_name=f"MMLU {subject}",
        lighteval_task_name=f"mmlu:{subject}",
        create=Mmlu,
    )
    return info
