from ...benchmark import BenchmarkInfo, BenchmarkName, Field
from .benchmark import Gpqa


class GpqaExtended(Gpqa):
    pass


GPQA_EXTENDED_INFO = BenchmarkInfo(
    name=BenchmarkName("gpqa-extended"),
    field=Field.KNOWLEDGE,
    display_name="GPQA Extended",
    lighteval_task_name="gpqa:extended",
    create=GpqaExtended,
)
