from __future__ import annotations


CATALOG_SCOPE = "direct_hf_lighteval_non_function_calling"
CATALOG_SOURCE = "lighteval"
CATALOG_TARGET_KIND = "task"
CATALOG_RUN_STATUS = "direct_lighteval_task"
TARGET_PER_DOMAIN = 100
EXPECTED_FIELDS = ("math", "coding", "instruction_following", "knowledge")
REQUIRED_TASKS = (
    "mmlu:professional_law",
    "gpqa:diamond",
    "arc:challenge",
    "truthfulqa:mc",
    "natural_questions",
    "squad_v2",
)
