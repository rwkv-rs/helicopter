from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


RunKind = Literal["free_response", "multiple_choice", "instruction_following", "code_generation", "longcodeqa"]


@dataclass(frozen=True, slots=True)
class CatalogRunSpec:
    benchmark: str
    field: str
    dataset_slug: str
    status: str
    kind: RunKind | None
    reason: str
    dataset_name: str | None = None
    dataset_config: str | None = None
    source_type: str = "hf"
    source_url: str | None = None
    source_urls: tuple[str, ...] = ()
    source_path: str | None = None
    source_split: str | None = None
    question_field: str = "question"
    answer_field: str = "answer"
    choices_field: str = "choices"
    choice_fields: tuple[str, ...] = ()
    choice_labels: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    answer_marker: str | None = "####"
    row_adapter: str | None = None
    adapter_seed: int = 42
    strict: bool = True
    job_name: str | None = None
    max_tokens: int | None = None


_DIRECT_HF_SPECS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "kind": "free_response",
        "dataset_name": "openai/gsm8k",
        "dataset_config": "main",
        "question_field": "question",
        "answer_field": "answer",
        "answer_marker": "####",
        "job_name": "free_response_judge",
        "max_tokens": 512,
    },
    "mmlu": {
        "kind": "multiple_choice",
        "dataset_name": "cais/mmlu",
        "dataset_config": "all",
        "question_field": "question",
        "choices_field": "choices",
        "answer_field": "answer",
        "choice_labels": "ABCD",
        "job_name": "multi_choice_plain",
        "max_tokens": 32,
    },
    "mmlu_pro": {
        "kind": "multiple_choice",
        "dataset_name": "TIGER-Lab/MMLU-Pro",
        "dataset_config": None,
        "question_field": "question",
        "choices_field": "options",
        "answer_field": "answer",
        "choice_labels": "ABCDEFGHIJ",
        "job_name": "multi_choice_plain",
        "max_tokens": 32,
    },
    "ceval": {
        "kind": "multiple_choice",
        "dataset_name": "ceval/ceval-exam",
        "dataset_config": "*",
        "question_field": "question",
        "choice_fields": ("A", "B", "C", "D"),
        "answer_field": "answer",
        "choice_labels": "ABCD",
        "job_name": "multi_choice_plain",
        "max_tokens": 32,
    },
    "cmmlu": {
        "kind": "multiple_choice",
        "source_type": "cmmlu_zip",
        "source_url": "https://huggingface.co/datasets/lmlmcat/cmmlu/resolve/main/cmmlu_v1_0_1.zip",
        "dataset_name": "lmlmcat/cmmlu",
        "question_field": "question",
        "choice_fields": ("A", "B", "C", "D"),
        "answer_field": "answer",
        "choice_labels": "ABCD",
        "job_name": "multi_choice_plain",
        "max_tokens": 32,
        "reason": "url_zip_csv_remote",
    },
    "gpqa_diamond": {
        "kind": "multiple_choice",
        "dataset_name": "Idavidrein/gpqa",
        "dataset_config": "gpqa_diamond",
        "source_split": "train",
        "row_adapter": "gpqa",
        "choice_labels": "ABCD",
        "job_name": "multi_choice_plain",
        "max_tokens": 32,
    },
    "gpqa_extended": {
        "kind": "multiple_choice",
        "dataset_name": "Idavidrein/gpqa",
        "dataset_config": "gpqa_extended",
        "source_split": "train",
        "row_adapter": "gpqa",
        "choice_labels": "ABCD",
        "job_name": "multi_choice_plain",
        "max_tokens": 32,
    },
    "gpqa_main": {
        "kind": "multiple_choice",
        "dataset_name": "Idavidrein/gpqa",
        "dataset_config": "gpqa_main",
        "source_split": "train",
        "row_adapter": "gpqa",
        "choice_labels": "ABCD",
        "job_name": "multi_choice_plain",
        "max_tokens": 32,
    },
    "include": {
        "kind": "multiple_choice",
        "dataset_name": "CohereForAI/include-base-44",
        "dataset_config": "*",
        "row_adapter": "include",
        "choice_labels": "ABCD",
        "job_name": "multi_choice_plain",
        "max_tokens": 32,
    },
    "mmlu_redux": {
        "kind": "multiple_choice",
        "dataset_name": "edinburgh-dawg/mmlu-redux-2.0",
        "dataset_config": "*",
        "row_adapter": "mmlu_redux",
        "choice_labels": "ABCD",
        "job_name": "multi_choice_plain",
        "max_tokens": 32,
    },
    "mmmlu": {
        "kind": "multiple_choice",
        "source_type": "mmmlu",
        "dataset_name": "giuliolovisotto/openai_multilingual_mmlu",
        "question_field": "question",
        "choice_fields": ("A", "B", "C", "D"),
        "answer_field": "answer",
        "choice_labels": "ABCD",
        "job_name": "multi_choice_plain",
        "max_tokens": 32,
        "reason": "hf_multilingual_mmlu",
    },
    "supergpqa": {
        "kind": "multiple_choice",
        "dataset_name": "m-a-p/SuperGPQA",
        "dataset_config": None,
        "source_split": "train",
        "row_adapter": "supergpqa",
        "choice_labels": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "job_name": "multi_choice_plain",
        "max_tokens": 32,
    },
    "ifeval": {
        "kind": "instruction_following",
        "dataset_name": "google-research/instruction_following_eval",
        "source_url": (
            "https://raw.githubusercontent.com/google-research/google-research/"
            "master/instruction_following_eval/data/input_data.jsonl"
        ),
        "job_name": "instruction_following",
        "max_tokens": 1024,
        "reason": "url_jsonl_rule_scored",
    },
    "ifbench": {
        "kind": "instruction_following",
        "dataset_name": "allenai/IFBench",
        "source_url": "https://raw.githubusercontent.com/allenai/IFBench/refs/heads/main/data/IFBench_test.jsonl",
        "job_name": "instruction_following",
        "max_tokens": 1024,
        "strict": False,
        "reason": "url_jsonl_rule_scored",
    },
    "human_eval": {
        "kind": "code_generation",
        "source_type": "human_eval_url_gzip",
        "source_url": "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
        "dataset_name": "human_eval",
        "job_name": "code_human_eval",
        "max_tokens": 512,
        "reason": "url_gzip_jsonl_code_execution",
    },
    "human_eval_cn": {
        "kind": "code_generation",
        "source_type": "human_eval_cn_url",
        "source_url": "https://hf-mirror.com/datasets/zai-org/humaneval-x/resolve/main/data/python/data/humaneval.jsonl",
        "dataset_name": "human_eval_cn",
        "job_name": "code_human_eval",
        "max_tokens": 512,
        "reason": "url_jsonl_code_execution",
    },
    "human_eval_fix": {
        "kind": "code_generation",
        "source_type": "human_eval_fix_hf",
        "dataset_name": "bigcode/humanevalpack",
        "dataset_config": "python",
        "job_name": "code_human_eval",
        "max_tokens": 512,
        "reason": "hf_code_execution",
    },
    "human_eval_plus": {
        "kind": "code_generation",
        "source_type": "human_eval_plus_evalplus",
        "dataset_name": "evalplus/human_eval_plus",
        "job_name": "code_human_eval",
        "max_tokens": 512,
        "reason": "evalplus_code_execution",
    },
    "mbpp": {
        "kind": "code_generation",
        "source_type": "mbpp_evalplus",
        "dataset_name": "evalplus/mbpp_plus",
        "job_name": "code_mbpp",
        "max_tokens": 512,
        "reason": "evalplus_assertion_code_execution",
    },
    "mbpp_plus": {
        "kind": "code_generation",
        "source_type": "mbpp_evalplus",
        "dataset_name": "evalplus/mbpp_plus",
        "job_name": "code_mbpp",
        "max_tokens": 512,
        "reason": "evalplus_base_plus_code_execution",
    },
    "livecodebench": {
        "kind": "code_generation",
        "source_type": "livecodebench_hf",
        "dataset_name": "livecodebench/code_generation_lite",
        "dataset_config": "release_latest",
        "job_name": "code_livecodebench",
        "max_tokens": 1024,
        "reason": "hf_livecodebench_code_execution",
    },
    "longcodeqa": {
        "kind": "longcodeqa",
        "source_type": "hf_zip",
        "dataset_name": "Steefano/LCB",
        "source_path": "LongCodeQA.zip",
        "job_name": "function_longcodebench",
        "max_tokens": 64,
        "reason": "hf_zip_longcodeqa_exact_letter",
    },
    "amc23": {
        "kind": "free_response",
        "source_type": "qwen_math",
        "dataset_name": "amc23",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "job_name": "free_response_judge",
        "max_tokens": 512,
        "reason": "qwen_math_remote",
    },
    "aime24": {
        "kind": "free_response",
        "source_type": "package_jsonl",
        "source_path": "data/free_response/aime24_test.jsonl",
        "dataset_name": "aime24",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "job_name": "free_response",
        "max_tokens": 512,
        "reason": "package_jsonl_static",
    },
    "aime25": {
        "kind": "free_response",
        "source_type": "package_jsonl",
        "source_path": "data/free_response/aime25_test.jsonl",
        "dataset_name": "aime25",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "job_name": "free_response",
        "max_tokens": 512,
        "reason": "package_jsonl_static",
    },
    "algebra222": {
        "kind": "free_response",
        "source_type": "url_csv",
        "source_url": "https://raw.githubusercontent.com/joyheyueya/declarative-math-word-problem/main/algebra222.csv",
        "dataset_name": "algebra222",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "row_adapter": "algebra222",
        "job_name": "free_response",
        "max_tokens": 512,
        "reason": "url_csv_remote",
    },
    "answer_judge": {
        "kind": "free_response",
        "dataset_name": "nvidia/judges-verdict",
        "source_split": "train",
        "question_field": "problem",
        "answer_field": "expected_judgement",
        "answer_marker": None,
        "row_adapter": "answer_judge",
        "job_name": "free_response",
        "max_tokens": 32,
        "reason": "hf_train_with_annotations",
    },
    "asdiv": {
        "kind": "free_response",
        "source_type": "url_xml",
        "source_url": "https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/master/dataset/ASDiv.xml",
        "dataset_name": "asdiv",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "row_adapter": "asdiv_xml",
        "job_name": "free_response",
        "max_tokens": 512,
        "reason": "url_xml_remote",
    },
    "beyond_aime": {
        "kind": "free_response",
        "dataset_name": "ByteDance-Seed/BeyondAIME",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "row_adapter": "answer_to_expected",
        "job_name": "free_response",
        "max_tokens": 512,
    },
    "brumo25": {
        "kind": "free_response",
        "dataset_name": "MathArena/brumo_2025",
        "source_split": "train",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "row_adapter": "answer_to_expected",
        "job_name": "free_response",
        "max_tokens": 512,
    },
    "comp_math_24_25": {
        "kind": "free_response",
        "source_type": "package_jsonl",
        "source_path": "data/free_response/comp-math-24-25_test.jsonl",
        "dataset_name": "comp_math_24_25",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "job_name": "free_response_judge",
        "max_tokens": 512,
        "reason": "package_jsonl_static",
    },
    "gaokao2023en": {
        "kind": "free_response",
        "source_type": "qwen_math",
        "dataset_name": "gaokao2023en",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "job_name": "free_response_judge",
        "max_tokens": 512,
        "reason": "qwen_math_remote",
    },
    "gsm_plus": {
        "kind": "free_response",
        "source_type": "url_jsonl",
        "source_url": "https://huggingface.co/datasets/qintongli/GSM-Plus/resolve/main/data/test-00000-of-00001.jsonl?download=true",
        "dataset_name": "gsm_plus",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "row_adapter": "gsm_plus",
        "job_name": "free_response",
        "max_tokens": 512,
        "reason": "url_jsonl_remote",
    },
    "hle": {
        "status": "needs_dataset_access",
        "kind": "free_response",
        "dataset_name": "cais/hle",
        "source_split": "test",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "row_adapter": "hle",
        "job_name": "free_response",
        "max_tokens": 512,
        "reason": "cais/hle is gated on Hugging Face and requires dataset access",
    },
    "hmmt_feb25": {
        "kind": "free_response",
        "dataset_name": "MathArena/hmmt_feb_2025",
        "source_split": "train",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "row_adapter": "answer_to_expected",
        "job_name": "free_response",
        "max_tokens": 512,
    },
    "mawps": {
        "kind": "free_response",
        "source_type": "url_jsonl",
        "source_urls": (
            "https://raw.githubusercontent.com/microsoft/ToRA/main/src/data/mawps/addsub.jsonl",
            "https://raw.githubusercontent.com/microsoft/ToRA/main/src/data/mawps/singleeq.jsonl",
            "https://raw.githubusercontent.com/microsoft/ToRA/main/src/data/mawps/singleop.jsonl",
            "https://raw.githubusercontent.com/microsoft/ToRA/main/src/data/mawps/multiarith.jsonl",
        ),
        "dataset_name": "mawps",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "row_adapter": "mawps",
        "job_name": "free_response",
        "max_tokens": 512,
        "reason": "url_jsonl_remote",
    },
    "polymath": {
        "kind": "free_response",
        "source_type": "polymath",
        "dataset_name": "Qwen/PolyMath",
        "source_split": "all",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "row_adapter": "polymath",
        "job_name": "free_response",
        "max_tokens": 512,
        "reason": "hf_all_configs",
    },
    "simpleqa": {
        "kind": "free_response",
        "dataset_name": "codelion/SimpleQA-Verified",
        "source_split": "train",
        "question_field": "question",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "row_adapter": "simpleqa_verified",
        "job_name": "free_response",
        "max_tokens": 128,
        "reason": "hf_verified_split",
    },
    "college_math": {
        "kind": "free_response",
        "source_type": "qwen_math",
        "dataset_name": "college_math",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "job_name": "free_response",
        "max_tokens": 512,
        "reason": "qwen_math_remote",
    },
    "hendrycks_math": {
        "kind": "free_response",
        "source_type": "qwen_math",
        "dataset_name": "math",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "job_name": "free_response",
        "max_tokens": 512,
        "reason": "qwen_math_remote",
    },
    "minerva_math": {
        "kind": "free_response",
        "source_type": "qwen_math",
        "dataset_name": "minerva_math",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "job_name": "free_response_judge",
        "max_tokens": 512,
        "reason": "qwen_math_remote",
    },
    "olympiadbench": {
        "kind": "free_response",
        "source_type": "qwen_math",
        "dataset_name": "olympiadbench",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "job_name": "free_response_judge",
        "max_tokens": 512,
        "reason": "qwen_math_remote",
    },
    "math_500": {
        "kind": "free_response",
        "source_type": "url_jsonl",
        "source_url": "https://github.com/openai/prm800k/raw/main/prm800k/math_splits/test.jsonl",
        "dataset_name": "math_500",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "row_adapter": "answer_solution",
        "job_name": "free_response_judge",
        "max_tokens": 512,
        "reason": "url_jsonl_remote",
    },
    "math_odyssey": {
        "kind": "free_response",
        "source_type": "url_jsonl",
        "source_url": "https://raw.githubusercontent.com/protagolabs/odyssey-math/main/final-odyssey-math-with-levels.jsonl",
        "dataset_name": "math_odyssey",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "row_adapter": "math_odyssey",
        "job_name": "free_response",
        "max_tokens": 512,
        "reason": "url_jsonl_remote",
    },
    "omni_math": {
        "kind": "free_response",
        "source_type": "url_jsonl",
        "source_url": "https://raw.githubusercontent.com/KbsdJames/Omni-MATH/refs/heads/main/Omni-Math.jsonl",
        "dataset_name": "omni_math",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "row_adapter": "answer_solution",
        "job_name": "free_response",
        "max_tokens": 512,
        "reason": "url_jsonl_remote",
    },
    "svamp": {
        "kind": "free_response",
        "source_type": "url_json",
        "source_url": "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json",
        "dataset_name": "svamp",
        "question_field": "problem",
        "answer_field": "expected_answer",
        "answer_marker": None,
        "row_adapter": "svamp",
        "job_name": "free_response",
        "max_tokens": 512,
        "reason": "url_json_remote",
    },
}


def resolve_catalog_run_spec(benchmark: Any) -> CatalogRunSpec:
    raw = _DIRECT_HF_SPECS.get(str(benchmark.name))
    dataset_slug = str(benchmark.dataset_slug)
    if raw:
        return CatalogRunSpec(
            benchmark=str(benchmark.name),
            field=str(benchmark.field),
            dataset_slug=dataset_slug,
            status=str(raw.get("status") or "implemented"),
            kind=raw["kind"],
            reason=str(raw.get("reason") or "direct_hf"),
            dataset_name=raw["dataset_name"],
            dataset_config=raw.get("dataset_config"),
            source_type=str(raw.get("source_type") or "hf"),
            source_url=raw.get("source_url"),
            source_urls=tuple(str(item) for item in raw.get("source_urls", ())),
            source_path=raw.get("source_path"),
            source_split=str(raw.get("source_split") or benchmark.default_split),
            question_field=str(raw.get("question_field", "question")),
            answer_field=str(raw.get("answer_field", "answer")),
            choices_field=str(raw.get("choices_field", "choices")),
            choice_fields=tuple(str(item) for item in raw.get("choice_fields", ())),
            choice_labels=str(raw.get("choice_labels", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")),
            answer_marker=raw.get("answer_marker"),
            row_adapter=raw.get("row_adapter"),
            adapter_seed=int(raw.get("adapter_seed") or 42),
            strict=bool(raw.get("strict", True)),
            job_name=str(raw.get("job_name") or ""),
            max_tokens=int(raw.get("max_tokens") or 512),
        )

    if str(benchmark.field) in {"knowledge", "maths"}:
        return CatalogRunSpec(
            benchmark=str(benchmark.name),
            field=str(benchmark.field),
            dataset_slug=dataset_slug,
            status="needs_dataset_adapter",
            kind=None,
            reason="dataset requires rwkv-skills-specific materialization or answer normalization",
        )
    return CatalogRunSpec(
        benchmark=str(benchmark.name),
        field=str(benchmark.field),
        dataset_slug=dataset_slug,
        status="needs_specialized_runner",
        kind=None,
        reason=f"{benchmark.field} is not covered by generic HF free-response/multiple-choice runners",
    )


def catalog_run_spec_to_dict(spec: CatalogRunSpec) -> dict[str, Any]:
    return {
        "benchmark": spec.benchmark,
        "field": spec.field,
        "dataset_slug": spec.dataset_slug,
        "status": spec.status,
        "kind": spec.kind,
        "reason": spec.reason,
        "hf_dataset": spec.dataset_name,
        "hf_config": spec.dataset_config,
        "source_type": spec.source_type,
        "source_url": spec.source_url,
        "source_urls": list(spec.source_urls),
        "source_path": spec.source_path,
        "source_split": spec.source_split,
        "row_adapter": spec.row_adapter,
        "strict": spec.strict,
    }


def dry_run_catalog_spec(
    spec: CatalogRunSpec,
    *,
    base_url: str,
    model: str,
    limit: int | None,
) -> dict[str, Any]:
    if spec.status != "implemented" or spec.kind is None:
        raise RuntimeError(f"{spec.benchmark} is not runnable yet: {spec.reason}")
    config = _run_config(spec, base_url=base_url, model=model, limit=limit)
    if spec.kind == "free_response":
        from .free_response import dry_run_summary

        return dry_run_summary(config)
    if spec.kind == "instruction_following":
        from .instruction_following import dry_run_summary

        return dry_run_summary(config)
    if spec.kind == "code_generation":
        from .code_generation import dry_run_summary

        return dry_run_summary(config)
    if spec.kind == "longcodeqa":
        from .longcodeqa import dry_run_summary

        return dry_run_summary(config)
    from .multiple_choice import dry_run_summary

    return dry_run_summary(config)


def run_catalog_spec(
    spec: CatalogRunSpec,
    *,
    base_url: str,
    model: str,
    limit: int | None,
    repo_root: Path,
) -> dict[str, Any]:
    if spec.status != "implemented" or spec.kind is None:
        raise RuntimeError(f"{spec.benchmark} is not runnable yet: {spec.reason}")
    config = _run_config(spec, base_url=base_url, model=model, limit=limit)
    if spec.kind == "free_response":
        from .free_response import run_free_response

        return run_free_response(config, repo_root=repo_root)
    if spec.kind == "instruction_following":
        from .instruction_following import run_instruction_following

        return run_instruction_following(config, repo_root=repo_root)
    if spec.kind == "code_generation":
        from .code_generation import run_code_generation

        return run_code_generation(config, repo_root=repo_root)
    if spec.kind == "longcodeqa":
        from .longcodeqa import run_longcodeqa

        return run_longcodeqa(config, repo_root=repo_root)
    from .multiple_choice import run_multiple_choice

    return run_multiple_choice(config, repo_root=repo_root)


def _run_config(spec: CatalogRunSpec, *, base_url: str, model: str, limit: int | None) -> Any:
    if spec.kind == "free_response":
        from .free_response import FreeResponseRunConfig
        from .gsm8k import REFERENCE_ANSWER_FIXES

        return FreeResponseRunConfig(
            base_url=base_url,
            model=model,
            benchmark=spec.benchmark,
            dataset_name=str(spec.dataset_name),
            dataset_config=spec.dataset_config,
            source_type=spec.source_type,
            source_url=spec.source_url,
            source_urls=spec.source_urls,
            source_path=spec.source_path,
            row_adapter=spec.row_adapter,
            question_field=spec.question_field,
            answer_field=spec.answer_field,
            limit=limit,
            split=str(spec.source_split),
            max_tokens=int(spec.max_tokens or 512),
            answer_marker=spec.answer_marker,
            reference_answer_overrides=REFERENCE_ANSWER_FIXES if spec.benchmark == "gsm8k" else None,
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "free_response_judge",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
        )
    if spec.kind == "multiple_choice":
        from .multiple_choice import MultipleChoiceRunConfig

        return MultipleChoiceRunConfig(
            base_url=base_url,
            model=model,
            benchmark=spec.benchmark,
            dataset_name=str(spec.dataset_name),
            dataset_config=spec.dataset_config,
            source_type=spec.source_type,
            source_url=spec.source_url,
            question_field=spec.question_field,
            choices_field=spec.choices_field,
            answer_field=spec.answer_field,
            limit=limit,
            choice_fields=spec.choice_fields,
            row_adapter=spec.row_adapter,
            adapter_seed=spec.adapter_seed,
            split=str(spec.source_split),
            max_tokens=int(spec.max_tokens or 32),
            choice_labels=spec.choice_labels,
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "multi_choice_plain",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
        )
    if spec.kind == "instruction_following":
        from .instruction_following import InstructionFollowingRunConfig

        return InstructionFollowingRunConfig(
            base_url=base_url,
            model=model,
            benchmark=spec.benchmark,
            dataset_name=str(spec.dataset_name),
            source_url=str(spec.source_url),
            limit=limit,
            split=str(spec.source_split),
            max_tokens=int(spec.max_tokens or 1024),
            strict=spec.strict,
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "instruction_following",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
        )
    if spec.kind == "code_generation":
        from .code_generation import CodeGenerationRunConfig

        return CodeGenerationRunConfig(
            base_url=base_url,
            model=model,
            benchmark=spec.benchmark,
            dataset_name=str(spec.dataset_name),
            source_type=spec.source_type,
            source_url=spec.source_url,
            limit=limit,
            split=str(spec.source_split),
            max_tokens=int(spec.max_tokens or 512),
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "code_human_eval",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
            cot_mode="CoT" if spec.benchmark == "livecodebench" else "NoCoT",
        )
    if spec.kind == "longcodeqa":
        from .longcodeqa import LongCodeQARunConfig

        return LongCodeQARunConfig(
            base_url=base_url,
            model=model,
            benchmark=spec.benchmark,
            limit=limit,
            split=str(spec.source_split),
            source_path=None,
            max_tokens=int(spec.max_tokens or 64),
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "function_longcodebench",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
        )
    raise RuntimeError(f"{spec.benchmark} is not runnable yet: {spec.reason}")
