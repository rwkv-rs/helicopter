from ...benchmark import BenchmarkInfo, BenchmarkName, Field
from .benchmark import Gpqa


class GpqaMain(Gpqa):
    pass


GPQA_MAIN_INFO = BenchmarkInfo(
    name=BenchmarkName("gpqa-main"),
    field=Field.KNOWLEDGE,
    display_name="GPQA Main",
    lighteval_task_name="gpqa:main",
    create=GpqaMain,
)
