"""
name:
AMC 2023

dataset:
math-ai/amc23

abstract:
The 40 problems from the 2023 AMC 10A, AMC 10B, AMC 12A, and AMC 12B
competitions.

languages:
english

tags:
math, reasoning
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.aime import MATH_PROMPT_TEMPLATE


def amc23_prompt(line, task_name: str | None = None) -> Doc:
    return Doc(
        task_name=task_name,
        query=MATH_PROMPT_TEMPLATE.format(prompt=line["question"]),
        choices=[str(line["answer"])],
        gold_index=0,
    )


amc23 = LightevalTaskConfig(
    name="amc23",
    prompt_function=amc23_prompt,
    hf_repo="math-ai/amc23",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1})],
    version=1,
)


TASKS_TABLE = [amc23]
