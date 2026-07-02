from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Literal


RunKind = Literal[
    "free_response",
    "multiple_choice",
    "instruction_following",
    "code_generation",
    "swe_bench",
    "tau_bench",
    "longcodeqa",
    "longbench",
    "arena_hard",
    "agentbench",
    "mcp_bench",
    "browsecomp",
    "browsecomp_plus",
    "apibank",
    "bfcl_ast",
    "bfcl_exec",
    "bfcl_v3",
    "toolalpaca",
    "translation",
    "complexfuncbench",
]


_SAMPLE_SIZE_SUPPORTED_KINDS = frozenset(
    {
        "agentbench",
        "apibank",
        "arena_hard",
        "bfcl_ast",
        "bfcl_exec",
        "bfcl_v3",
        "browsecomp",
        "browsecomp_plus",
        "complexfuncbench",
        "free_response",
        "multiple_choice",
        "code_generation",
        "instruction_following",
        "mcp_bench",
        "swe_bench",
        "tau_bench",
        "toolalpaca",
        "translation",
        "longcodeqa",
        "longbench",
    }
)


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
    "flores200": {
        "status": "needs_dataset_access",
        "kind": "translation",
        "source_type": "hf_flores200",
        "dataset_name": "openlanguagedata/flores_plus",
        "source_split": "devtest",
        "job_name": "translation_chrf",
        "max_tokens": 512,
        "reason": "openlanguagedata/flores_plus is gated on Hugging Face and requires dataset access",
    },
    "wmt24pp": {
        "kind": "translation",
        "source_type": "hf_wmt24pp",
        "dataset_name": "google/wmt24pp",
        "source_split": "test",
        "job_name": "translation_chrf",
        "max_tokens": 512,
        "reason": "hf_translation_chrf",
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
    "swe_bench": {
        "kind": "swe_bench",
        "source_type": "swe_bench_hf",
        "dataset_name": "princeton-nlp/SWE-bench",
        "row_adapter": "swe_bench",
        "job_name": "code_swe_bench",
        "max_tokens": 2048,
        "reason": "swebench_patch_generation_official_harness",
    },
    "swe_bench_lite": {
        "kind": "swe_bench",
        "source_type": "swe_bench_hf",
        "dataset_name": "princeton-nlp/SWE-bench_Lite",
        "row_adapter": "swe_bench_lite",
        "job_name": "code_swe_bench",
        "max_tokens": 2048,
        "reason": "swebench_patch_generation_official_harness",
    },
    "swe_bench_verified": {
        "kind": "swe_bench",
        "source_type": "swe_bench_hf",
        "dataset_name": "princeton-nlp/SWE-bench_Verified",
        "row_adapter": "swe_bench_verified",
        "job_name": "code_swe_bench",
        "max_tokens": 2048,
        "reason": "swebench_patch_generation_official_harness",
    },
    "swe_bench_lite_oracle": {
        "kind": "swe_bench",
        "source_type": "swe_bench_hf",
        "dataset_name": "princeton-nlp/SWE-bench_Lite_oracle",
        "row_adapter": "swe_bench_lite_oracle",
        "job_name": "code_swe_bench",
        "max_tokens": 2048,
        "reason": "swebench_patch_generation_official_harness",
    },
    "swe_bench_lite_bm25_13k": {
        "kind": "swe_bench",
        "source_type": "swe_bench_hf",
        "dataset_name": "princeton-nlp/SWE-bench_Lite_bm25_13K",
        "row_adapter": "swe_bench_lite_bm25_13k",
        "job_name": "code_swe_bench",
        "max_tokens": 2048,
        "reason": "swebench_patch_generation_official_harness",
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
    "longbench": {
        "kind": "longbench",
        "source_type": "hf_longbench",
        "dataset_name": "THUDM/LongBench",
        "job_name": "function_longbench",
        "max_tokens": 128,
        "reason": "hf_longbench_em_f1",
    },
    "longbench_qa": {
        "kind": "longbench",
        "source_type": "hf_longbench",
        "dataset_name": "THUDM/LongBench",
        "row_adapter": "longbench_qa",
        "job_name": "function_longbench",
        "max_tokens": 128,
        "reason": "hf_longbench_qa_em_f1",
    },
    "longbench_qa_balanced": {
        "kind": "longbench",
        "source_type": "hf_longbench",
        "dataset_name": "THUDM/LongBench",
        "row_adapter": "longbench_qa_balanced",
        "job_name": "function_longbench",
        "max_tokens": 128,
        "reason": "hf_longbench_qa_round_robin_em_f1",
    },
    "mcp_bench": {
        "kind": "mcp_bench",
        "source_type": "mcp_bench_official",
        "dataset_name": "mcp_bench",
        "job_name": "function_mcp_bench",
        "max_tokens": 1024,
        "reason": "official_mcp_bench_runtime_judged",
    },
    "mcp_bench_single": {
        "kind": "mcp_bench",
        "source_type": "mcp_bench_official",
        "dataset_name": "mcp_bench_single",
        "job_name": "function_mcp_bench",
        "max_tokens": 1024,
        "reason": "official_mcp_bench_runtime_judged",
    },
    "mcp_bench_multi_2server": {
        "kind": "mcp_bench",
        "source_type": "mcp_bench_official",
        "dataset_name": "mcp_bench_multi_2server",
        "job_name": "function_mcp_bench",
        "max_tokens": 1024,
        "reason": "official_mcp_bench_runtime_judged",
    },
    "mcp_bench_multi_3server": {
        "kind": "mcp_bench",
        "source_type": "mcp_bench_official",
        "dataset_name": "mcp_bench_multi_3server",
        "job_name": "function_mcp_bench",
        "max_tokens": 1024,
        "reason": "official_mcp_bench_runtime_judged",
    },
    "browsecomp": {
        "kind": "browsecomp",
        "source_type": "browsecomp_csv",
        "source_url": "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv",
        "dataset_name": "openai/simple-evals-browsecomp",
        "job_name": "function_browsecomp",
        "max_tokens": 2048,
        "reason": "url_csv_encrypted_two_stage_judged",
    },
    "browsecomp_zh": {
        "kind": "browsecomp",
        "source_type": "browsecomp_zh_xlsx",
        "source_url": (
            "https://raw.githubusercontent.com/PALIN2018/BrowseComp-ZH/main/data/"
            "browsecomp-zh-encrypted.xlsx"
        ),
        "dataset_name": "PALIN2018/BrowseComp-ZH",
        "job_name": "function_browsecomp",
        "max_tokens": 2048,
        "reason": "url_xlsx_encrypted_two_stage_judged",
    },
    "browsecomp_plus": {
        "kind": "browsecomp_plus",
        "source_type": "browsecomp_plus_hf",
        "dataset_name": "texttron/BrowseComp-Plus",
        "source_split": "test",
        "job_name": "function_browsecomp_plus",
        "max_tokens": 1024,
        "reason": "official_browsecomp_plus_tool_loop",
    },
    "arena_hard_v2": {
        "kind": "arena_hard",
        "source_type": "arena_hard_url",
        "dataset_name": "arena_hard",
        "source_url": "https://raw.githubusercontent.com/lm-sys/arena-hard-auto/main/data/arena-hard-v0.1/question.jsonl",
        "source_split": "test",
        "job_name": "instruction_arena_hard",
        "max_tokens": 2048,
        "reason": "arena_hard_pairwise_judged",
    },
    "agentbench_db": {
        "kind": "agentbench",
        "source_type": "agentbench_official",
        "dataset_name": "agentbench_db",
        "job_name": "function_agentbench",
        "max_tokens": 1024,
        "reason": "official_agentbench_controller",
    },
    "agentbench_kg": {
        "kind": "agentbench",
        "source_type": "agentbench_official",
        "dataset_name": "agentbench_kg",
        "job_name": "function_agentbench",
        "max_tokens": 1024,
        "reason": "official_agentbench_controller",
    },
    "apibank_l1": {
        "kind": "apibank",
        "source_type": "apibank_git",
        "dataset_name": "AlibabaResearch/DAMO-ConvAI/api-bank",
        "row_adapter": "apibank_level1",
        "job_name": "function_api_bank",
        "max_tokens": 768,
        "reason": "official_api_bank_execution",
    },
    "apibank_l2": {
        "kind": "apibank",
        "source_type": "apibank_git",
        "dataset_name": "AlibabaResearch/DAMO-ConvAI/api-bank",
        "row_adapter": "apibank_level2",
        "job_name": "function_api_bank",
        "max_tokens": 768,
        "reason": "official_api_bank_execution",
    },
    "apibank_level1": {
        "kind": "apibank",
        "source_type": "apibank_git",
        "dataset_name": "AlibabaResearch/DAMO-ConvAI/api-bank",
        "row_adapter": "apibank_level1",
        "job_name": "function_api_bank",
        "max_tokens": 768,
        "reason": "official_api_bank_execution",
    },
    "apibank_level2": {
        "kind": "apibank",
        "source_type": "apibank_git",
        "dataset_name": "AlibabaResearch/DAMO-ConvAI/api-bank",
        "row_adapter": "apibank_level2",
        "job_name": "function_api_bank",
        "max_tokens": 768,
        "reason": "official_api_bank_execution",
    },
    "bfcl_simple_python": {
        "kind": "bfcl_ast",
        "source_type": "bfcl_v4_github",
        "dataset_name": "ShishirPatil/gorilla/bfcl_v4",
        "row_adapter": "simple_python",
        "job_name": "function_bfcl_ast",
        "max_tokens": 768,
        "reason": "official_bfcl_v4_ast",
    },
    "bfcl_multiple": {
        "kind": "bfcl_ast",
        "source_type": "bfcl_v4_github",
        "dataset_name": "ShishirPatil/gorilla/bfcl_v4",
        "row_adapter": "multiple",
        "job_name": "function_bfcl_ast",
        "max_tokens": 768,
        "reason": "official_bfcl_v4_ast",
    },
    "bfcl_exec_simple_ast": {
        "kind": "bfcl_ast",
        "source_type": "bfcl_v4_github",
        "dataset_name": "ShishirPatil/gorilla/bfcl_v4",
        "row_adapter": "exec_simple",
        "job_name": "function_bfcl_ast",
        "max_tokens": 768,
        "reason": "official_bfcl_v4_exec_ast",
    },
    "bfcl_exec_multiple_ast": {
        "kind": "bfcl_ast",
        "source_type": "bfcl_v4_github",
        "dataset_name": "ShishirPatil/gorilla/bfcl_v4",
        "row_adapter": "exec_multiple",
        "job_name": "function_bfcl_ast",
        "max_tokens": 768,
        "reason": "official_bfcl_v4_exec_ast",
    },
    "bfcl_exec_simple": {
        "kind": "bfcl_exec",
        "source_type": "bfcl_v4_exec_github",
        "dataset_name": "ShishirPatil/gorilla/bfcl_v4",
        "row_adapter": "exec_simple",
        "job_name": "function_bfcl_exec",
        "max_tokens": 768,
        "reason": "official_bfcl_v4_exec",
    },
    "bfcl_exec_multiple": {
        "kind": "bfcl_exec",
        "source_type": "bfcl_v4_exec_github",
        "dataset_name": "ShishirPatil/gorilla/bfcl_v4",
        "row_adapter": "exec_multiple",
        "job_name": "function_bfcl_exec",
        "max_tokens": 768,
        "reason": "official_bfcl_v4_exec",
    },
    "bfcl_exec_parallel": {
        "kind": "bfcl_exec",
        "source_type": "bfcl_v4_exec_github",
        "dataset_name": "ShishirPatil/gorilla/bfcl_v4",
        "row_adapter": "exec_parallel",
        "job_name": "function_bfcl_exec",
        "max_tokens": 768,
        "reason": "official_bfcl_v4_exec",
    },
    "bfcl_exec_parallel_multiple": {
        "kind": "bfcl_exec",
        "source_type": "bfcl_v4_exec_github",
        "dataset_name": "ShishirPatil/gorilla/bfcl_v4",
        "row_adapter": "exec_parallel_multiple",
        "job_name": "function_bfcl_exec",
        "max_tokens": 768,
        "reason": "official_bfcl_v4_exec",
    },
    "bfcl_v3": {
        "kind": "bfcl_v3",
        "source_type": "bfcl_v3_official",
        "dataset_name": "ShishirPatil/gorilla/bfcl_v3",
        "source_split": "test",
        "job_name": "function_bfcl_v3",
        "max_tokens": 1024,
        "reason": "official_bfcl_v3_multi_turn",
    },
    "toolalpaca_eval_simulated": {
        "kind": "toolalpaca",
        "source_type": "toolalpaca_git",
        "dataset_name": "toolalpaca_eval_simulated",
        "row_adapter": "eval_simulated",
        "job_name": "function_toolalpaca",
        "max_tokens": 1024,
        "reason": "official_toolalpaca_request_execution",
    },
    "toolalpaca_eval_real": {
        "kind": "toolalpaca",
        "source_type": "toolalpaca_git",
        "dataset_name": "toolalpaca_eval_real",
        "row_adapter": "eval_real",
        "job_name": "function_toolalpaca",
        "max_tokens": 1024,
        "reason": "official_toolalpaca_request_execution",
    },
    "complexfuncbench_official": {
        "kind": "complexfuncbench",
        "source_type": "hf_complexfuncbench",
        "dataset_name": "complexfuncbench_official",
        "job_name": "function_complexfuncbench",
        "max_tokens": 1024,
        "reason": "hf_complexfuncbench_local_golden_conversation",
    },
    "complexfuncbench_subset": {
        "kind": "complexfuncbench",
        "source_type": "hf_complexfuncbench",
        "dataset_name": "complexfuncbench_subset",
        "job_name": "function_complexfuncbench",
        "max_tokens": 1024,
        "reason": "hf_complexfuncbench_local_golden_conversation",
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

for _tau_name, _tau_meta in {
    "tau_bench_airline": ("function_tau_bench", "test"),
    "tau_bench_retail": ("function_tau_bench", "test"),
    "tau_bench_telecom": ("function_tau_bench", "test"),
    "tau2_bench_airline": ("function_tau2_bench", "base"),
    "tau2_bench_retail": ("function_tau2_bench", "base"),
    "tau2_bench_telecom": ("function_tau2_bench", "base"),
    "tau3_bench_airline": ("function_tau3_bench", "base"),
    "tau3_bench_retail": ("function_tau3_bench", "base"),
    "tau3_bench_telecom": ("function_tau3_bench", "base"),
    "tau3_bench_banking_knowledge": ("function_tau3_bench", "base"),
    "tau3_bench_mock": ("function_tau3_bench", "base"),
    "tau3_bench_mock_long_context": ("function_tau3_bench", "base"),
}.items():
    _DIRECT_HF_SPECS[_tau_name] = {
        "kind": "tau_bench",
        "source_type": "tau_official_manifest",
        "dataset_name": _tau_name,
        "source_split": _tau_meta[1],
        "job_name": _tau_meta[0],
        "max_tokens": 512,
        "reason": "official_tau_runtime_environment_judged",
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
    sample_size: int | None = None,
    sample_seed: int = 42,
    longbench_source_path: str | None = None,
    longbench_infer_protocol: str | None = None,
    longbench_temperature: float | None = None,
    longbench_top_p: float | None = None,
    longbench_presence_penalty: float | None = None,
    longbench_frequency_penalty: float | None = None,
    longbench_seed_requests: bool = False,
    longbench_stop_suffixes: tuple[str, ...] = (),
    agentbench_controller_url: str | None = None,
    mcp_runtime_root: str | None = None,
    mcp_worker_script: str | None = None,
    mcp_max_rounds: int | None = None,
    mcp_tool_router_mode: str | None = None,
    mcp_tool_router_max_tools: int | None = None,
    mcp_tool_router_trigger_tool_count: int | None = None,
    mcp_tool_router_trigger_catalog_chars: int | None = None,
    mcp_tool_router_context_chars: int | None = None,
    mcp_tool_router_description_chars: int | None = None,
    mcp_long_context_router_mode: str | None = None,
    mcp_long_context_min_chars: int | None = None,
    mcp_long_context_chunk_chars: int | None = None,
    mcp_long_context_overlap_lines: int | None = None,
    mcp_long_context_max_evidence_chunks: int | None = None,
    mcp_long_context_max_evidence_chars: int | None = None,
    judge_base_url: str | None = None,
    judge_model: str | None = None,
    judge_api_key: str | None = None,
    swebench_run_harness: bool = False,
    swebench_predictions_dir: str | None = None,
    swebench_harness_run_id: str | None = None,
    swebench_max_workers: int | None = None,
    swebench_cache_level: str | None = None,
    swebench_clean: bool = False,
    swebench_timeout_s: float | None = None,
    swebench_max_context_chars: int | None = None,
    tau_runtime_root: str | None = None,
    tau_data_root: str | None = None,
    tau_user_base_url: str | None = None,
    tau_user_model: str | None = None,
    tau_user_api_key: str | None = None,
    tau_max_steps: int | None = None,
    tau_max_errors: int | None = None,
    tau_history_max_chars: int | None = None,
    tau_prompt_max_chars: int | None = None,
    candidate_router_mode: str | None = None,
    candidate_router_chunk_tools: int | None = None,
    candidate_router_batch_size: int | None = None,
    candidate_router_context_chars: int | None = None,
    candidate_router_prompt_max_chars: int | None = None,
    candidate_router_candidate_max_tokens: int | None = None,
    candidate_router_aggregate_max_tokens: int | None = None,
    candidate_router_max_candidates: int | None = None,
    candidate_router_tool_schema_mode: str | None = None,
) -> dict[str, Any]:
    if spec.status != "implemented" or spec.kind is None:
        raise RuntimeError(f"{spec.benchmark} is not runnable yet: {spec.reason}")
    config = _run_config(
        spec,
        base_url=base_url,
        model=model,
        limit=limit,
        sample_size=sample_size,
        sample_seed=sample_seed,
        longbench_source_path=longbench_source_path,
        longbench_infer_protocol=longbench_infer_protocol,
        longbench_temperature=longbench_temperature,
        longbench_top_p=longbench_top_p,
        longbench_presence_penalty=longbench_presence_penalty,
        longbench_frequency_penalty=longbench_frequency_penalty,
        longbench_seed_requests=longbench_seed_requests,
        longbench_stop_suffixes=longbench_stop_suffixes,
        agentbench_controller_url=agentbench_controller_url,
        mcp_runtime_root=mcp_runtime_root,
        mcp_worker_script=mcp_worker_script,
        mcp_max_rounds=mcp_max_rounds,
        mcp_tool_router_mode=mcp_tool_router_mode,
        mcp_tool_router_max_tools=mcp_tool_router_max_tools,
        mcp_tool_router_trigger_tool_count=mcp_tool_router_trigger_tool_count,
        mcp_tool_router_trigger_catalog_chars=mcp_tool_router_trigger_catalog_chars,
        mcp_tool_router_context_chars=mcp_tool_router_context_chars,
        mcp_tool_router_description_chars=mcp_tool_router_description_chars,
        mcp_long_context_router_mode=mcp_long_context_router_mode,
        mcp_long_context_min_chars=mcp_long_context_min_chars,
        mcp_long_context_chunk_chars=mcp_long_context_chunk_chars,
        mcp_long_context_overlap_lines=mcp_long_context_overlap_lines,
        mcp_long_context_max_evidence_chunks=mcp_long_context_max_evidence_chunks,
        mcp_long_context_max_evidence_chars=mcp_long_context_max_evidence_chars,
        judge_base_url=judge_base_url,
        judge_model=judge_model,
        judge_api_key=judge_api_key,
        swebench_run_harness=swebench_run_harness,
        swebench_predictions_dir=swebench_predictions_dir,
        swebench_harness_run_id=swebench_harness_run_id,
        swebench_max_workers=swebench_max_workers,
        swebench_cache_level=swebench_cache_level,
        swebench_clean=swebench_clean,
        swebench_timeout_s=swebench_timeout_s,
        swebench_max_context_chars=swebench_max_context_chars,
        tau_runtime_root=tau_runtime_root,
        tau_data_root=tau_data_root,
        tau_user_base_url=tau_user_base_url,
        tau_user_model=tau_user_model,
        tau_user_api_key=tau_user_api_key,
        tau_max_steps=tau_max_steps,
        tau_max_errors=tau_max_errors,
        tau_history_max_chars=tau_history_max_chars,
        tau_prompt_max_chars=tau_prompt_max_chars,
        candidate_router_mode=candidate_router_mode,
        candidate_router_chunk_tools=candidate_router_chunk_tools,
        candidate_router_batch_size=candidate_router_batch_size,
        candidate_router_context_chars=candidate_router_context_chars,
        candidate_router_prompt_max_chars=candidate_router_prompt_max_chars,
        candidate_router_candidate_max_tokens=candidate_router_candidate_max_tokens,
        candidate_router_aggregate_max_tokens=candidate_router_aggregate_max_tokens,
        candidate_router_max_candidates=candidate_router_max_candidates,
        candidate_router_tool_schema_mode=candidate_router_tool_schema_mode,
    )
    if spec.kind == "free_response":
        from .free_response import dry_run_summary

        return dry_run_summary(config)
    if spec.kind == "instruction_following":
        from .instruction_following import dry_run_summary

        return dry_run_summary(config)
    if spec.kind == "code_generation":
        from .code_generation import dry_run_summary

        return dry_run_summary(config)
    if spec.kind == "swe_bench":
        from .swe_bench import dry_run_summary

        return dry_run_summary(config)
    if spec.kind == "tau_bench":
        from .tau_bench import dry_run_summary

        return dry_run_summary(config)
    if spec.kind == "longcodeqa":
        from .longcodeqa import dry_run_summary

        return dry_run_summary(config)
    if spec.kind == "longbench":
        from .longbench import dry_run_summary

        return dry_run_summary(config)
    if spec.kind == "arena_hard":
        from .arena_hard import dry_run_summary

        return dry_run_summary(config)
    if spec.kind == "agentbench":
        from .agentbench import dry_run_summary

        return dry_run_summary(config)
    if spec.kind == "mcp_bench":
        from .mcp_bench import dry_run_summary

        return dry_run_summary(config)
    if spec.kind == "browsecomp":
        from .browsecomp import dry_run_summary

        return dry_run_summary(config)
    if spec.kind == "browsecomp_plus":
        from .browsecomp_plus import dry_run_summary

        return dry_run_summary(config)
    if spec.kind == "apibank":
        from .apibank import dry_run_summary

        return dry_run_summary(config)
    if spec.kind == "bfcl_ast":
        from .bfcl_ast import dry_run_summary

        return dry_run_summary(config)
    if spec.kind == "bfcl_exec":
        from .bfcl_exec import dry_run_summary

        return dry_run_summary(config)
    if spec.kind == "bfcl_v3":
        from .bfcl_v3 import dry_run_summary

        return dry_run_summary(config)
    if spec.kind == "toolalpaca":
        from .toolalpaca import dry_run_summary

        return dry_run_summary(config)
    if spec.kind == "translation":
        from .translation import dry_run_summary

        return dry_run_summary(config)
    if spec.kind == "complexfuncbench":
        from .complexfuncbench import dry_run_summary

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
    sample_size: int | None = None,
    sample_seed: int = 42,
    longbench_source_path: str | None = None,
    longbench_infer_protocol: str | None = None,
    longbench_temperature: float | None = None,
    longbench_top_p: float | None = None,
    longbench_presence_penalty: float | None = None,
    longbench_frequency_penalty: float | None = None,
    longbench_seed_requests: bool = False,
    longbench_stop_suffixes: tuple[str, ...] = (),
    agentbench_controller_url: str | None = None,
    mcp_runtime_root: str | None = None,
    mcp_worker_script: str | None = None,
    mcp_max_rounds: int | None = None,
    mcp_tool_router_mode: str | None = None,
    mcp_tool_router_max_tools: int | None = None,
    mcp_tool_router_trigger_tool_count: int | None = None,
    mcp_tool_router_trigger_catalog_chars: int | None = None,
    mcp_tool_router_context_chars: int | None = None,
    mcp_tool_router_description_chars: int | None = None,
    mcp_long_context_router_mode: str | None = None,
    mcp_long_context_min_chars: int | None = None,
    mcp_long_context_chunk_chars: int | None = None,
    mcp_long_context_overlap_lines: int | None = None,
    mcp_long_context_max_evidence_chunks: int | None = None,
    mcp_long_context_max_evidence_chars: int | None = None,
    judge_base_url: str | None = None,
    judge_model: str | None = None,
    judge_api_key: str | None = None,
    swebench_run_harness: bool = False,
    swebench_predictions_dir: str | None = None,
    swebench_harness_run_id: str | None = None,
    swebench_max_workers: int | None = None,
    swebench_cache_level: str | None = None,
    swebench_clean: bool = False,
    swebench_timeout_s: float | None = None,
    swebench_max_context_chars: int | None = None,
    tau_runtime_root: str | None = None,
    tau_data_root: str | None = None,
    tau_user_base_url: str | None = None,
    tau_user_model: str | None = None,
    tau_user_api_key: str | None = None,
    tau_max_steps: int | None = None,
    tau_max_errors: int | None = None,
    tau_history_max_chars: int | None = None,
    tau_prompt_max_chars: int | None = None,
    candidate_router_mode: str | None = None,
    candidate_router_chunk_tools: int | None = None,
    candidate_router_batch_size: int | None = None,
    candidate_router_context_chars: int | None = None,
    candidate_router_prompt_max_chars: int | None = None,
    candidate_router_candidate_max_tokens: int | None = None,
    candidate_router_aggregate_max_tokens: int | None = None,
    candidate_router_max_candidates: int | None = None,
    candidate_router_tool_schema_mode: str | None = None,
) -> dict[str, Any]:
    if spec.status != "implemented" or spec.kind is None:
        raise RuntimeError(f"{spec.benchmark} is not runnable yet: {spec.reason}")
    config = _run_config(
        spec,
        base_url=base_url,
        model=model,
        limit=limit,
        sample_size=sample_size,
        sample_seed=sample_seed,
        longbench_source_path=longbench_source_path,
        longbench_infer_protocol=longbench_infer_protocol,
        longbench_temperature=longbench_temperature,
        longbench_top_p=longbench_top_p,
        longbench_presence_penalty=longbench_presence_penalty,
        longbench_frequency_penalty=longbench_frequency_penalty,
        longbench_seed_requests=longbench_seed_requests,
        longbench_stop_suffixes=longbench_stop_suffixes,
        agentbench_controller_url=agentbench_controller_url,
        mcp_runtime_root=mcp_runtime_root,
        mcp_worker_script=mcp_worker_script,
        mcp_max_rounds=mcp_max_rounds,
        mcp_tool_router_mode=mcp_tool_router_mode,
        mcp_tool_router_max_tools=mcp_tool_router_max_tools,
        mcp_tool_router_trigger_tool_count=mcp_tool_router_trigger_tool_count,
        mcp_tool_router_trigger_catalog_chars=mcp_tool_router_trigger_catalog_chars,
        mcp_tool_router_context_chars=mcp_tool_router_context_chars,
        mcp_tool_router_description_chars=mcp_tool_router_description_chars,
        mcp_long_context_router_mode=mcp_long_context_router_mode,
        mcp_long_context_min_chars=mcp_long_context_min_chars,
        mcp_long_context_chunk_chars=mcp_long_context_chunk_chars,
        mcp_long_context_overlap_lines=mcp_long_context_overlap_lines,
        mcp_long_context_max_evidence_chunks=mcp_long_context_max_evidence_chunks,
        mcp_long_context_max_evidence_chars=mcp_long_context_max_evidence_chars,
        judge_base_url=judge_base_url,
        judge_model=judge_model,
        judge_api_key=judge_api_key,
        swebench_run_harness=swebench_run_harness,
        swebench_predictions_dir=swebench_predictions_dir,
        swebench_harness_run_id=swebench_harness_run_id,
        swebench_max_workers=swebench_max_workers,
        swebench_cache_level=swebench_cache_level,
        swebench_clean=swebench_clean,
        swebench_timeout_s=swebench_timeout_s,
        swebench_max_context_chars=swebench_max_context_chars,
        tau_runtime_root=tau_runtime_root,
        tau_data_root=tau_data_root,
        tau_user_base_url=tau_user_base_url,
        tau_user_model=tau_user_model,
        tau_user_api_key=tau_user_api_key,
        tau_max_steps=tau_max_steps,
        tau_max_errors=tau_max_errors,
        tau_history_max_chars=tau_history_max_chars,
        tau_prompt_max_chars=tau_prompt_max_chars,
        candidate_router_mode=candidate_router_mode,
        candidate_router_chunk_tools=candidate_router_chunk_tools,
        candidate_router_batch_size=candidate_router_batch_size,
        candidate_router_context_chars=candidate_router_context_chars,
        candidate_router_prompt_max_chars=candidate_router_prompt_max_chars,
        candidate_router_candidate_max_tokens=candidate_router_candidate_max_tokens,
        candidate_router_aggregate_max_tokens=candidate_router_aggregate_max_tokens,
        candidate_router_max_candidates=candidate_router_max_candidates,
        candidate_router_tool_schema_mode=candidate_router_tool_schema_mode,
    )
    if spec.kind == "free_response":
        from .free_response import run_free_response

        return run_free_response(config, repo_root=repo_root)
    if spec.kind == "instruction_following":
        from .instruction_following import run_instruction_following

        return run_instruction_following(config, repo_root=repo_root)
    if spec.kind == "code_generation":
        from .code_generation import run_code_generation

        return run_code_generation(config, repo_root=repo_root)
    if spec.kind == "swe_bench":
        from .swe_bench import run_swe_bench

        return run_swe_bench(config, repo_root=repo_root)
    if spec.kind == "tau_bench":
        from .tau_bench import run_tau_bench

        return run_tau_bench(config, repo_root=repo_root)
    if spec.kind == "longcodeqa":
        from .longcodeqa import run_longcodeqa

        return run_longcodeqa(config, repo_root=repo_root)
    if spec.kind == "longbench":
        from .longbench import run_longbench

        return run_longbench(config, repo_root=repo_root)
    if spec.kind == "arena_hard":
        from .arena_hard import run_arena_hard

        return run_arena_hard(config, repo_root=repo_root)
    if spec.kind == "agentbench":
        from .agentbench import run_agentbench

        return run_agentbench(config, repo_root=repo_root)
    if spec.kind == "mcp_bench":
        from .mcp_bench import run_mcp_bench

        return run_mcp_bench(config, repo_root=repo_root)
    if spec.kind == "browsecomp":
        from .browsecomp import run_browsecomp

        return run_browsecomp(config, repo_root=repo_root)
    if spec.kind == "browsecomp_plus":
        from .browsecomp_plus import run_browsecomp_plus

        return run_browsecomp_plus(config, repo_root=repo_root)
    if spec.kind == "apibank":
        from .apibank import run_apibank

        return run_apibank(config, repo_root=repo_root)
    if spec.kind == "bfcl_ast":
        from .bfcl_ast import run_bfcl_ast

        return run_bfcl_ast(config, repo_root=repo_root)
    if spec.kind == "bfcl_exec":
        from .bfcl_exec import run_bfcl_exec

        return run_bfcl_exec(config, repo_root=repo_root)
    if spec.kind == "bfcl_v3":
        from .bfcl_v3 import run_bfcl_v3

        return run_bfcl_v3(config, repo_root=repo_root)
    if spec.kind == "toolalpaca":
        from .toolalpaca import run_toolalpaca

        return run_toolalpaca(config, repo_root=repo_root)
    if spec.kind == "translation":
        from .translation import run_translation

        return run_translation(config, repo_root=repo_root)
    if spec.kind == "complexfuncbench":
        from .complexfuncbench import run_complexfuncbench

        return run_complexfuncbench(config, repo_root=repo_root)
    from .multiple_choice import run_multiple_choice

    return run_multiple_choice(config, repo_root=repo_root)


def export_catalog_sample_manifest(
    spec: CatalogRunSpec,
    *,
    base_url: str,
    model: str,
    limit: int | None,
    output_path: str,
    sample_size: int | None = None,
    sample_seed: int = 42,
    longbench_source_path: str | None = None,
    longbench_infer_protocol: str | None = None,
    longbench_temperature: float | None = None,
    longbench_top_p: float | None = None,
    longbench_presence_penalty: float | None = None,
    longbench_frequency_penalty: float | None = None,
    longbench_seed_requests: bool = False,
    longbench_stop_suffixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    if spec.status != "implemented" or spec.kind is None:
        raise RuntimeError(f"{spec.benchmark} is not runnable yet: {spec.reason}")
    if spec.kind != "longbench":
        raise ValueError("--write-sample-manifest is only supported for longbench benchmarks")
    config = _run_config(
        spec,
        base_url=base_url,
        model=model,
        limit=limit,
        sample_size=sample_size,
        sample_seed=sample_seed,
        longbench_source_path=longbench_source_path,
        longbench_infer_protocol=longbench_infer_protocol,
        longbench_temperature=longbench_temperature,
        longbench_top_p=longbench_top_p,
        longbench_presence_penalty=longbench_presence_penalty,
        longbench_frequency_penalty=longbench_frequency_penalty,
        longbench_seed_requests=longbench_seed_requests,
        longbench_stop_suffixes=longbench_stop_suffixes,
    )
    from .longbench import export_sample_manifest

    return export_sample_manifest(config, output_path)


def _run_config(
    spec: CatalogRunSpec,
    *,
    base_url: str,
    model: str,
    limit: int | None,
    sample_size: int | None = None,
    sample_seed: int = 42,
    longbench_source_path: str | None = None,
    longbench_infer_protocol: str | None = None,
    longbench_temperature: float | None = None,
    longbench_top_p: float | None = None,
    longbench_presence_penalty: float | None = None,
    longbench_frequency_penalty: float | None = None,
    longbench_seed_requests: bool = False,
    longbench_stop_suffixes: tuple[str, ...] = (),
    agentbench_controller_url: str | None = None,
    mcp_runtime_root: str | None = None,
    mcp_worker_script: str | None = None,
    mcp_max_rounds: int | None = None,
    mcp_tool_router_mode: str | None = None,
    mcp_tool_router_max_tools: int | None = None,
    mcp_tool_router_trigger_tool_count: int | None = None,
    mcp_tool_router_trigger_catalog_chars: int | None = None,
    mcp_tool_router_context_chars: int | None = None,
    mcp_tool_router_description_chars: int | None = None,
    mcp_long_context_router_mode: str | None = None,
    mcp_long_context_min_chars: int | None = None,
    mcp_long_context_chunk_chars: int | None = None,
    mcp_long_context_overlap_lines: int | None = None,
    mcp_long_context_max_evidence_chunks: int | None = None,
    mcp_long_context_max_evidence_chars: int | None = None,
    judge_base_url: str | None = None,
    judge_model: str | None = None,
    judge_api_key: str | None = None,
    swebench_run_harness: bool = False,
    swebench_predictions_dir: str | None = None,
    swebench_harness_run_id: str | None = None,
    swebench_max_workers: int | None = None,
    swebench_cache_level: str | None = None,
    swebench_clean: bool = False,
    swebench_timeout_s: float | None = None,
    swebench_max_context_chars: int | None = None,
    tau_runtime_root: str | None = None,
    tau_data_root: str | None = None,
    tau_user_base_url: str | None = None,
    tau_user_model: str | None = None,
    tau_user_api_key: str | None = None,
    tau_max_steps: int | None = None,
    tau_max_errors: int | None = None,
    tau_history_max_chars: int | None = None,
    tau_prompt_max_chars: int | None = None,
    candidate_router_mode: str | None = None,
    candidate_router_chunk_tools: int | None = None,
    candidate_router_batch_size: int | None = None,
    candidate_router_context_chars: int | None = None,
    candidate_router_prompt_max_chars: int | None = None,
    candidate_router_candidate_max_tokens: int | None = None,
    candidate_router_aggregate_max_tokens: int | None = None,
    candidate_router_max_candidates: int | None = None,
    candidate_router_tool_schema_mode: str | None = None,
) -> Any:
    if sample_size is not None and spec.kind not in _SAMPLE_SIZE_SUPPORTED_KINDS:
        raise ValueError(f"--sample-size is not supported for {spec.kind} benchmark: {spec.benchmark}")
    if longbench_source_path is not None and spec.kind != "longbench":
        raise ValueError("--longbench-source-path is only supported for longbench benchmarks")
    if longbench_infer_protocol is not None and longbench_infer_protocol not in {"chat", "completions"}:
        raise ValueError("--longbench-infer-protocol must be one of: chat, completions")
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
            sample_size=sample_size,
            sample_seed=sample_seed,
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
            sample_size=sample_size,
            sample_seed=sample_seed,
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
            sample_size=sample_size,
            sample_seed=sample_seed,
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
            sample_size=sample_size,
            sample_seed=sample_seed,
            split=str(spec.source_split),
            max_tokens=int(spec.max_tokens or 512),
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "code_human_eval",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
            cot_mode="CoT" if spec.benchmark == "livecodebench" else "NoCoT",
        )
    if spec.kind == "swe_bench":
        from .swe_bench import SweBenchRunConfig

        env_run_harness = os.getenv("HELICOPTER_SWEBENCH_RUN_HARNESS", "").strip().lower()
        run_harness = bool(swebench_run_harness or env_run_harness in {"1", "true", "yes", "on"})
        return SweBenchRunConfig(
            base_url=base_url,
            model=model,
            benchmark=spec.benchmark,
            dataset_name=str(spec.row_adapter or spec.benchmark),
            limit=limit,
            sample_size=sample_size,
            sample_seed=sample_seed,
            split=str(spec.source_split),
            run_harness=run_harness,
            predictions_dir=swebench_predictions_dir,
            harness_run_id=swebench_harness_run_id,
            harness_max_workers=int(swebench_max_workers or 1),
            harness_cache_level=swebench_cache_level,
            harness_clean=bool(swebench_clean),
            harness_timeout_s=swebench_timeout_s,
            max_context_chars=int(swebench_max_context_chars or 24000),
            max_tokens=int(spec.max_tokens or 2048),
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "code_swe_bench",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
        )
    if spec.kind == "tau_bench":
        from .tau_bench import TauBenchRunConfig

        return TauBenchRunConfig(
            base_url=base_url,
            model=model,
            benchmark=spec.benchmark,
            dataset_name=str(spec.dataset_name),
            limit=limit,
            sample_size=sample_size,
            sample_seed=sample_seed,
            split=str(spec.source_split),
            runtime_root=tau_runtime_root,
            data_root=tau_data_root,
            user_base_url=tau_user_base_url,
            user_model=tau_user_model,
            user_api_key=tau_user_api_key,
            judge_base_url=judge_base_url,
            judge_model=judge_model,
            judge_api_key=judge_api_key,
            max_steps=int(tau_max_steps or 200),
            max_errors=int(tau_max_errors or 10),
            history_max_chars=int(tau_history_max_chars or 16000),
            prompt_max_chars=int(tau_prompt_max_chars or 24576),
            max_tokens=int(spec.max_tokens or 512),
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "function_tau_bench",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
        )
    if spec.kind == "longcodeqa":
        from .longcodeqa import LongCodeQARunConfig

        return LongCodeQARunConfig(
            base_url=base_url,
            model=model,
            benchmark=spec.benchmark,
            limit=limit,
            sample_size=sample_size,
            sample_seed=sample_seed,
            split=str(spec.source_split),
            source_path=None,
            max_tokens=int(spec.max_tokens or 64),
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "function_longcodebench",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
        )
    if spec.kind == "longbench":
        from .longbench import LONG_BENCH_QA_DATASETS, LongBenchRunConfig

        include_datasets: tuple[str, ...] = ()
        balance_by_dataset = False
        if spec.row_adapter in {"longbench_qa", "longbench_qa_balanced"}:
            include_datasets = tuple(sorted(LONG_BENCH_QA_DATASETS))
        if spec.row_adapter == "longbench_qa_balanced":
            balance_by_dataset = True
        return LongBenchRunConfig(
            base_url=base_url,
            model=model,
            benchmark=spec.benchmark,
            limit=limit,
            sample_size=sample_size,
            sample_seed=sample_seed,
            split=str(spec.source_split),
            source_path=longbench_source_path,
            include_datasets=include_datasets,
            balance_by_dataset=balance_by_dataset,
            infer_protocol=longbench_infer_protocol or "chat",
            temperature=0.0 if longbench_temperature is None else float(longbench_temperature),
            top_p=1.0 if longbench_top_p is None else float(longbench_top_p),
            presence_penalty=0.0 if longbench_presence_penalty is None else float(longbench_presence_penalty),
            frequency_penalty=0.0 if longbench_frequency_penalty is None else float(longbench_frequency_penalty),
            seed_requests=bool(longbench_seed_requests),
            stop_suffixes=tuple(longbench_stop_suffixes),
            max_tokens=int(spec.max_tokens or 128),
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "function_longbench",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
        )
    if spec.kind == "arena_hard":
        from .arena_hard import ARENA_HARD_BASELINE_URL, ArenaHardRunConfig

        return ArenaHardRunConfig(
            base_url=base_url,
            model=model,
            benchmark=spec.benchmark,
            source_url=str(spec.source_url),
            baseline_url=ARENA_HARD_BASELINE_URL,
            limit=limit,
            sample_size=sample_size,
            sample_seed=sample_seed,
            split=str(spec.source_split),
            max_tokens=int(spec.max_tokens or 2048),
            judge_base_url=judge_base_url,
            judge_model=judge_model,
            judge_api_key=judge_api_key,
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "instruction_arena_hard",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
        )
    if spec.kind == "agentbench":
        from .agentbench import AgentBenchRunConfig

        return AgentBenchRunConfig(
            base_url=base_url,
            model=model,
            benchmark=spec.benchmark,
            dataset_name=str(spec.dataset_name),
            limit=limit,
            sample_size=sample_size,
            sample_seed=sample_seed,
            split=str(spec.source_split),
            controller_url=agentbench_controller_url,
            max_tokens=int(spec.max_tokens or 1024),
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "function_agentbench",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
        )
    if spec.kind == "mcp_bench":
        from .mcp_bench import McpBenchRunConfig

        return McpBenchRunConfig(
            base_url=base_url,
            model=model,
            benchmark=spec.benchmark,
            dataset_name=str(spec.dataset_name),
            limit=limit,
            sample_size=sample_size,
            sample_seed=sample_seed,
            split=str(spec.source_split),
            runtime_root=mcp_runtime_root,
            worker_script=mcp_worker_script,
            judge_base_url=judge_base_url,
            judge_model=judge_model,
            judge_api_key=judge_api_key,
            max_rounds=int(mcp_max_rounds or 8),
            decision_max_tokens=int(spec.max_tokens or 1024),
            final_max_tokens=int(spec.max_tokens or 1024),
            tool_router_mode=str(mcp_tool_router_mode or "lexical"),
            tool_router_max_tools=int(mcp_tool_router_max_tools or 16),
            tool_router_trigger_tool_count=int(mcp_tool_router_trigger_tool_count or 20),
            tool_router_trigger_catalog_chars=int(mcp_tool_router_trigger_catalog_chars or 6000),
            tool_router_context_chars=int(mcp_tool_router_context_chars or 6000),
            tool_router_description_chars=int(mcp_tool_router_description_chars or 240),
            long_context_router_mode=str(mcp_long_context_router_mode or "lexical"),
            long_context_min_chars=int(mcp_long_context_min_chars or 4000),
            long_context_chunk_chars=int(mcp_long_context_chunk_chars or 1200),
            long_context_overlap_lines=int(mcp_long_context_overlap_lines if mcp_long_context_overlap_lines is not None else 2),
            long_context_max_evidence_chunks=int(mcp_long_context_max_evidence_chunks or 4),
            long_context_max_evidence_chars=int(mcp_long_context_max_evidence_chars or 6000),
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "function_mcp_bench",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
        )
    if spec.kind == "browsecomp":
        from .browsecomp import BrowseCompRunConfig

        return BrowseCompRunConfig(
            base_url=base_url,
            model=model,
            benchmark=spec.benchmark,
            source_type=spec.source_type,
            source_url=spec.source_url,
            limit=limit,
            sample_size=sample_size,
            sample_seed=sample_seed,
            split=str(spec.source_split),
            cot_max_tokens=int(spec.max_tokens or 2048),
            answer_max_tokens=1024,
            judge_base_url=judge_base_url,
            judge_model=judge_model,
            judge_api_key=judge_api_key,
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "function_browsecomp",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
        )
    if spec.kind == "browsecomp_plus":
        from .browsecomp_plus import BrowseCompPlusRunConfig

        return BrowseCompPlusRunConfig(
            base_url=base_url,
            model=model,
            benchmark=spec.benchmark,
            limit=limit,
            sample_size=sample_size,
            sample_seed=sample_seed,
            split=str(spec.source_split),
            max_tokens=int(spec.max_tokens or 1024),
            judge_base_url=judge_base_url,
            judge_model=judge_model,
            judge_api_key=judge_api_key,
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "function_browsecomp_plus",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
        )
    if spec.kind == "apibank":
        from .apibank import ApiBankRunConfig

        level = 2 if spec.row_adapter == "apibank_level2" else 1
        return ApiBankRunConfig(
            base_url=base_url,
            model=model,
            benchmark=spec.benchmark,
            level=level,
            limit=limit,
            sample_size=sample_size,
            sample_seed=sample_seed,
            split=str(spec.source_split),
            max_tokens=int(spec.max_tokens or 768),
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "function_api_bank",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
        )
    if spec.kind == "bfcl_ast":
        from .bfcl_ast import BfclAstRunConfig

        return BfclAstRunConfig(
            base_url=base_url,
            model=model,
            benchmark=spec.benchmark,
            category=str(spec.row_adapter or spec.benchmark),
            limit=limit,
            sample_size=sample_size,
            sample_seed=sample_seed,
            split=str(spec.source_split),
            max_tokens=int(spec.max_tokens or 768),
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "function_bfcl_ast",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
        )
    if spec.kind == "bfcl_exec":
        from .bfcl_exec import BfclExecRunConfig

        return BfclExecRunConfig(
            base_url=base_url,
            model=model,
            benchmark=spec.benchmark,
            category=str(spec.row_adapter or spec.benchmark),
            limit=limit,
            sample_size=sample_size,
            sample_seed=sample_seed,
            split=str(spec.source_split),
            max_tokens=int(spec.max_tokens or 768),
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "function_bfcl_exec",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
        )
    if spec.kind == "bfcl_v3":
        from .bfcl_v3 import BfclV3RunConfig

        return BfclV3RunConfig(
            base_url=base_url,
            model=model,
            benchmark=spec.benchmark,
            limit=limit,
            sample_size=sample_size,
            sample_seed=sample_seed,
            split=str(spec.source_split),
            max_tokens=int(spec.max_tokens or 1024),
            candidate_router_mode=str(candidate_router_mode or "parallel"),
            candidate_router_chunk_tools=int(candidate_router_chunk_tools or 2),
            candidate_router_batch_size=int(candidate_router_batch_size or 16),
            candidate_router_context_chars=int(candidate_router_context_chars or 6000),
            candidate_router_prompt_max_chars=int(candidate_router_prompt_max_chars or 8192),
            candidate_router_candidate_max_tokens=int(candidate_router_candidate_max_tokens or 192),
            candidate_router_aggregate_max_tokens=int(candidate_router_aggregate_max_tokens or 192),
            candidate_router_max_candidates=int(candidate_router_max_candidates or 12),
            candidate_router_tool_schema_mode=str(candidate_router_tool_schema_mode or "compact"),
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "function_bfcl_v3",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
        )
    if spec.kind == "toolalpaca":
        from .toolalpaca import ToolAlpacaRunConfig

        return ToolAlpacaRunConfig(
            base_url=base_url,
            model=model,
            benchmark=spec.benchmark,
            dataset_name=str(spec.dataset_name),
            limit=limit,
            sample_size=sample_size,
            sample_seed=sample_seed,
            split=str(spec.source_split),
            max_tokens=int(spec.max_tokens or 1024),
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "function_toolalpaca",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
        )
    if spec.kind == "translation":
        from .translation import DEFAULT_WMT24PP_TARGET_LANGUAGES, TranslationRunConfig

        target_languages = (
            tuple(DEFAULT_WMT24PP_TARGET_LANGUAGES)
            if spec.source_type == "hf_wmt24pp"
            else ("en", "de", "es", "fr", "it", "ja")
        )
        return TranslationRunConfig(
            base_url=base_url,
            model=model,
            benchmark=spec.benchmark,
            source_type=spec.source_type,
            dataset_name=str(spec.dataset_name),
            limit=limit,
            sample_size=sample_size,
            sample_seed=sample_seed,
            split=str(spec.source_split),
            target_languages=target_languages,
            max_tokens=int(spec.max_tokens or 512),
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "translation_chrf",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
        )
    if spec.kind == "complexfuncbench":
        from .complexfuncbench import COMPLEXFUNCBENCH_SOURCE_URL, ComplexFuncBenchRunConfig

        return ComplexFuncBenchRunConfig(
            base_url=base_url,
            model=model,
            benchmark=spec.benchmark,
            dataset_name=str(spec.dataset_name),
            limit=limit,
            sample_size=sample_size,
            sample_seed=sample_seed,
            split=str(spec.source_split),
            source_url=spec.source_url or COMPLEXFUNCBENCH_SOURCE_URL,
            max_tokens=int(spec.max_tokens or 1024),
            scoreboard_dataset=spec.dataset_slug,
            job_name=spec.job_name or "function_complexfuncbench",
            job_id=f"helicopter-{spec.benchmark}",
            runner="helicopter_eval.catalog_runner",
        )
    raise RuntimeError(f"{spec.benchmark} is not runnable yet: {spec.reason}")
