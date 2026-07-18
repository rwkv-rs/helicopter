from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .benchmark_catalog_defaults import (
    CATALOG_RUN_STATUS,
    CATALOG_SCOPE,
    CATALOG_SOURCE,
    CATALOG_TARGET_KIND,
    EXPECTED_FIELDS,
    REQUIRED_TASKS,
    TARGET_PER_DOMAIN,
)
from .lighteval_tasks import load_registry


DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CUSTOM_TASKS = Path(__file__).with_name("lighteval_rwkv_skills_tasks.py")


@dataclass(frozen=True)
class DomainSpec:
    field: str
    label: str
    description: str


@dataclass(frozen=True)
class SelectionRule:
    field: str
    source_family: str
    prefixes: tuple[str, ...] = ()
    exact: tuple[str, ...] = ()


EXCLUDED_DIRECT_PATTERNS = (
    r"\bbfcl\b",
    r"api[_-]?bank",
    r"complexfuncbench",
    r"toolalpaca",
    r"function[_-]?call",
    r"tool[_-]?call",
    r"mcp[_-]?bench",
    r"tau[23]?[_-]?bench",
    r"agentbench",
    r"browsecomp",
    r"swe[_-]?bench",
    r"^mathqa$",
    r"^ifeval-fr$",
    r"^qasper$",
    r"^cmmlu_zho_mcf:",
    r"^global_mmlu_all_kor_mcf:",
    r"^global_mmlu_all_nor_mcf:",
    r"^global_mmlu_all_tam_mcf:",
    r"^global_mmlu_all_tha_mcf:",
    r"^global_mmlu_all_urd_mcf:",
)

DOMAIN_SPECS = {
    "math": DomainSpec(
        "math",
        "Math",
        "Recognized math and quantitative-reasoning suites: MATH, GSM8K/MGSM, AIME, OlympiadBench, "
        "Minerva/SVAMP, AGIEval math rows, and math-heavy MMLU/CEval/Global MMLU subjects.",
    ),
    "coding": DomainSpec(
        "coding",
        "Coding / CS",
        "Recognized code-generation plus computer-science benchmark rows: HumanEval, MBPP+, "
        "LiveCodeBench, and CS/security/programming subjects from MMLU-family public suites.",
    ),
    "instruction_following": DomainSpec(
        "instruction_following",
        "Instruction / Task Following",
        "Recognized instruction-following and general task-following suites: IFEval, IFBench, MT-Bench, "
        "Arena-Hard, MixEval, BIG-Bench, BBH, and SuperGLUE-style task following.",
    ),
    "knowledge": DomainSpec(
        "knowledge",
        "Knowledge",
        "Recognized knowledge, QA, exam, commonsense, and factual-recall suites: MMLU, GPQA, ARC, "
        "HellaSwag, TruthfulQA, OpenBookQA, PIQA, SciQ, Natural Questions, TriviaQA, SQuAD, "
        "CoQA, NarrativeQA, PubMedQA, AGIEval, and CEval.",
    ),
}
GLOBAL_MMLU_SAFE_LANGUAGE_SUFFIXES = (
    "ara_mcf",
    "ben_mcf",
    "ces_mcf",
    "deu_mcf",
    "eng_mcf",
    "fra_mcf",
    "hin_mcf",
    "ind_mcf",
    "ita_mcf",
    "jpn_mcf",
    "nld_mcf",
    "pol_mcf",
    "por_mcf",
    "ron_mcf",
    "rus_mcf",
    "spa_mcf",
    "srp_mcf",
    "swa_mcf",
    "swe_mcf",
    "tel_mcf",
    "tgl_mcf",
    "tur_mcf",
    "ukr_mcf",
    "vie_mcf",
    "zho_mcf",
)

GLOBAL_MMLU_MATH_SCIENCE_SUBJECTS = (
    "abstract_algebra",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_mathematics",
    "college_physics",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
)

GLOBAL_MMLU_CS_SUBJECTS = (
    "college_computer_science",
    "computer_security",
    "high_school_computer_science",
    "machine_learning",
)


def global_mmlu_exact(subjects: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        f"global_mmlu_all_{suffix}:{subject}"
        for suffix in GLOBAL_MMLU_SAFE_LANGUAGE_SUFFIXES
        for subject in subjects
    )


SCOREBOARD_DOMAIN_SCOPE = {
    "included": EXPECTED_FIELDS,
    "excluded": (
        {
            "field": "agent",
            "reason": (
                "Agent/tool-use benchmarks require official external harnesses or custom runtime judges; "
                "they are tracked in benchmarks/agent_benchmarks.json, not the direct HF/LightEval manifest."
            ),
        },
        {
            "field": "function_call",
            "reason": (
                "Function-calling benchmarks use the native OpenAI tools path and benchmark-specific scorers, "
                "not ordinary LightEval task rows."
            ),
        },
    ),
}

SELECTION_RULES = (
    SelectionRule("math", "MATH / Math Odyssey", ("math:",), ("math_500", "math_odyssey")),
    SelectionRule(
        "math",
        "GSM8K / MGSM",
        ("mgsm:",),
        (
            "gsm8k",
            "mgsm",
            "mgsm_ben",
            "mgsm_deu",
            "mgsm_eng",
            "mgsm_fra",
            "mgsm_jpn",
            "mgsm_rus",
            "mgsm_spa",
            "mgsm_swa",
            "mgsm_tel",
            "mgsm_tha",
            "mgsm_zho",
        ),
    ),
    SelectionRule("math", "AIME", ("aime",)),
    SelectionRule("math", "OlympiadBench", ("olympiad_bench:",)),
    SelectionRule("math", "Minerva / SVAMP", exact=("minerva_math", "svamp", "algebra222", "amc23", "beyond_aime")),
    SelectionRule("math", "AGIEval Math", exact=("agieval:aqua-rat", "agieval:gaokao-mathqa", "agieval:sat-math")),
    SelectionRule(
        "math",
        "CEval Math/Science",
        exact=tuple(
            f"ceval_zho_mcf:{name}"
            for name in (
                "advanced_mathematics",
                "college_chemistry",
                "college_physics",
                "discrete_mathematics",
                "high_school_biology",
                "high_school_chemistry",
                "high_school_mathematics",
                "high_school_physics",
                "middle_school_biology",
                "middle_school_chemistry",
                "middle_school_mathematics",
                "middle_school_physics",
                "probability_and_statistics",
            )
        ),
    ),
    SelectionRule(
        "math",
        "MMLU Math/Science",
        exact=tuple(
            f"mmlu:{name}"
            for name in (
                "abstract_algebra",
                "astronomy",
                "college_biology",
                "college_chemistry",
                "college_mathematics",
                "college_physics",
                "conceptual_physics",
                "econometrics",
                "electrical_engineering",
                "elementary_mathematics",
                "high_school_biology",
                "high_school_chemistry",
                "high_school_mathematics",
                "high_school_physics",
                "high_school_statistics",
            )
        ),
    ),
    SelectionRule(
        "math",
        "Global MMLU Math/Science",
        exact=global_mmlu_exact(GLOBAL_MMLU_MATH_SCIENCE_SUBJECTS),
    ),
    SelectionRule("coding", "HumanEval / MBPP / LiveCodeBench", ("human_eval", "lcb:"), ("mbpp_plus", "longcodeqa")),
    SelectionRule(
        "coding",
        "MMLU CS",
        exact=tuple(
            f"mmlu:{name}"
            for name in (
                "college_computer_science",
                "computer_security",
                "high_school_computer_science",
                "machine_learning",
            )
        ),
    ),
    SelectionRule(
        "coding",
        "CEval CS",
        exact=tuple(
            f"ceval_zho_mcf:{name}"
            for name in (
                "college_programming",
                "computer_architecture",
                "computer_network",
                "operating_system",
            )
        ),
    ),
    SelectionRule(
        "coding",
        "Global MMLU CS",
        exact=global_mmlu_exact(GLOBAL_MMLU_CS_SUBJECTS),
    ),
    SelectionRule("instruction_following", "IFEval / IFBench", ("ifeval", "ifbench")),
    SelectionRule(
        "instruction_following",
        "MT-Bench / Arena-Hard / MixEval",
        ("mixeval_easy:", "mixeval_hard:"),
        ("mt_bench", "arena_hard_v2"),
    ),
    SelectionRule("instruction_following", "BIG-Bench", ("bigbench:",)),
    SelectionRule("instruction_following", "BIG-Bench Extra Hard", ("bigbench_extra_hard:",)),
    SelectionRule("instruction_following", "BBH", ("bigbench_hard:",)),
    SelectionRule(
        "instruction_following",
        "SuperGLUE / MuSR / DROP / RACE",
        ("super_glue:", "musr:"),
        ("drop", "race:high", "race:middle"),
    ),
    SelectionRule("instruction_following", "BBQ", ("bbq:",), ("bbq",)),
    SelectionRule("knowledge", "MMLU", ("mmlu:",)),
    SelectionRule("knowledge", "GPQA", ("gpqa",)),
    SelectionRule("knowledge", "ARC / HellaSwag / WinoGrande", ("arc:",), ("hellaswag", "winogrande")),
    SelectionRule("knowledge", "TruthfulQA / OpenBookQA / PIQA / SciQ", ("truthfulqa:",), ("openbookqa", "piqa", "sciq")),
    SelectionRule(
        "knowledge",
        "General / Scientific QA",
        exact=(
            "natural_questions",
            "triviaqa",
            "squad_v2",
            "coqa",
            "narrativeqa",
            "commonsenseqa",
            "pubmedqa",
            "simpleqa",
        ),
    ),
    SelectionRule("knowledge", "AGIEval", ("agieval:",)),
    SelectionRule("knowledge", "CEval", ("ceval_zho_mcf:",)),
)


def matches_any(value: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in patterns)


def task_sort_key(task: str) -> tuple[int, int, str]:
    preferred_prefixes = (
        "aime",
        "gsm8k",
        "mgsm",
        "math",
        "human_eval",
        "mbpp",
        "lcb",
        "ifeval",
        "ifbench",
        "mt_bench",
        "mmlu:",
        "arc:",
        "gpqa",
    )
    priority = next((index for index, prefix in enumerate(preferred_prefixes) if task.startswith(prefix)), 99)
    return priority, len(task), task


def rule_matches(rule: SelectionRule, task: str) -> bool:
    return task in rule.exact or any(task.startswith(prefix) for prefix in rule.prefixes)


def has_perplexity_metric(task_config: object) -> bool:
    for metric in getattr(task_config, "metrics", ()) or ():
        category = getattr(metric, "category", None)
        if getattr(category, "name", None) == "PERPLEXITY":
            return True
    return False


def is_direct_lighteval_candidate(task: str, task_config: object) -> bool:
    return not matches_any(task, EXCLUDED_DIRECT_PATTERNS) and not has_perplexity_metric(task_config)


def load_lighteval_task_names(
    *,
    custom_tasks: str | Path | None = None,
    load_multilingual: bool = True,
) -> list[str]:
    registry = load_registry(
        custom_tasks=str(custom_tasks or DEFAULT_CUSTOM_TASKS),
        load_multilingual=load_multilingual,
    )
    return sorted(
        task
        for task, task_config in registry._task_registry.items()
        if is_direct_lighteval_candidate(task, task_config)
    )


def _select_tasks_for_field(
    *,
    field: str,
    candidates: list[str],
    used: set[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for rule in [item for item in SELECTION_RULES if item.field == field]:
        matches = [task for task in candidates if task not in used and rule_matches(rule, task)]
        for task in sorted(matches, key=task_sort_key):
            rows.append({"name": task, "source_family": rule.source_family})
            used.add(task)
            if len(rows) >= TARGET_PER_DOMAIN:
                break
        if len(rows) >= TARGET_PER_DOMAIN:
            break
    if len(rows) < TARGET_PER_DOMAIN:
        raise SystemExit(
            f"{field} only has {len(rows)} recognized direct LightEval candidates; "
            f"need {TARGET_PER_DOMAIN}"
        )
    return rows


def select_domain_tasks(tasks: list[str]) -> dict[str, list[dict[str, str]]]:
    selected: dict[str, list[dict[str, str]]] = {}
    used: set[str] = set()
    candidates = [task for task in tasks if not matches_any(task, EXCLUDED_DIRECT_PATTERNS)]
    for field in EXPECTED_FIELDS:
        selected[field] = _select_tasks_for_field(field=field, candidates=candidates, used=used)
    return selected


def build_benchmark_rows(selected: Mapping[str, Iterable[Mapping[str, str]]]) -> list[dict[str, str]]:
    return [
        {
            "name": task["name"],
            "field": field,
            "source": CATALOG_SOURCE,
            "source_family": task["source_family"],
            "target_kind": CATALOG_TARGET_KIND,
            "run_status": CATALOG_RUN_STATUS,
        }
        for field, tasks in selected.items()
        for task in tasks
    ]


def _relative_custom_tasks_label(custom_tasks: Path, *, root: Path) -> str:
    try:
        return str(custom_tasks.relative_to(root))
    except ValueError:
        return str(custom_tasks)


def build_manifest(
    *,
    custom_tasks: str | Path | None = None,
    load_multilingual: bool = True,
    root: str | Path | None = None,
) -> dict[str, object]:
    root_path = Path(root or DEFAULT_ROOT).resolve()
    custom_tasks_path = Path(custom_tasks or DEFAULT_CUSTOM_TASKS).resolve()
    task_names = load_lighteval_task_names(custom_tasks=custom_tasks_path, load_multilingual=load_multilingual)
    benchmarks = build_benchmark_rows(select_domain_tasks(task_names))
    counts = Counter(row["field"] for row in benchmarks)
    return {
        "description": (
            "Curated directly runnable non-function-calling LightEval benchmark manifest. "
            "This file uses an explicit allowlist of widely used public benchmark families; it is "
            "not generated from arbitrary registry-name matches. Agent/tool-use benchmarks remain "
            "in benchmarks/agent_benchmarks.json because they require official external harnesses "
            "or custom judges rather than ordinary HF-backed LightEval task rows."
        ),
        "target_per_domain": TARGET_PER_DOMAIN,
        "scope": CATALOG_SCOPE,
        "scoreboard_domain_scope": {
            "included": list(SCOREBOARD_DOMAIN_SCOPE["included"]),
            "excluded": list(SCOREBOARD_DOMAIN_SCOPE["excluded"]),
        },
        "custom_tasks": _relative_custom_tasks_label(custom_tasks_path, root=root_path),
        "domains": [
            {
                "field": spec.field,
                "label": spec.label,
                "description": spec.description,
                "count": counts[spec.field],
            }
            for spec in DOMAIN_SPECS.values()
        ],
        "benchmarks": benchmarks,
    }
