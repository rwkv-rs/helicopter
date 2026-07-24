from ...benchmark import BenchmarkInfo, BenchmarkName, Field
from .benchmark import Gpqa


class GpqaDiamond(Gpqa):
    pass


GPQA_DIAMOND_INFO = BenchmarkInfo(
    name=BenchmarkName("gpqa-diamond"),
    field=Field.KNOWLEDGE,
    display_name="GPQA Diamond",
    lighteval_task_name="gpqa:diamond",
    create=GpqaDiamond,
)
