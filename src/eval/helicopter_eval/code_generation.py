from __future__ import annotations

import asyncio
import ast
import contextlib
from dataclasses import dataclass
from decimal import Decimal
import faulthandler
import gzip
import hashlib
import io
import json
import linecache
import multiprocessing
import os
from pathlib import Path
import platform
import re
import signal
import tempfile
import traceback
from types import ModuleType
from typing import Any, Sequence
import urllib.request

from .openai_client import chat_completion
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_PYTHON_FENCED_CODE_RE = re.compile(
    r"```[ \t]*(?:python|py)[^\S\r\n]*\r?\n(?P<code>.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_FENCED_CODE_RE = re.compile(
    r"```[ \t]*(?:python|py)?[^\S\r\n]*\r?\n(?P<code>.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_LEADING_END_THINK_RE = re.compile(r"^[\s\r\n]*</think>[ \t]*\r?\n?", re.IGNORECASE)
_STANDALONE_CODE_FENCE_RE = re.compile(r"^[ \t]*```[ \t]*(?:python|py)?[ \t]*$", re.IGNORECASE)
_DEF_RE = re.compile(r"def\s+(?P<name>[\w_]+)\s*\(")
_LCB_DATASET_ID = "livecodebench/code_generation_lite"
_LCB_DATASET_CONFIG = "release_latest"
_LCB_VERSION_TAG = "release_v6"
_LCB_SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program "
    "that matches the specification and passes all tests."
)
_LCB_FORMAT_WITH_STARTER = (
    "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
)
_LCB_FORMAT_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to STDOUT "
    "(do not directly test on the sample inputs). Enclose your code within delimiters as follows. "
    "Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
)
_LCB_FINAL_ANSWER_PREFIX = "\n</think>\n```python\n"
_LCB_IMPORT_STRING = (
    "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\n"
    "from heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\n"
    "from random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\n"
    "from typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\n"
    "import bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\n"
    "import operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"
)


@dataclass(frozen=True, slots=True)
class CodeGenerationSample:
    sample_index: int
    task_id: str
    prompt: str
    check_prefix: str | None = None
    starter_code: str | None = None
    entry_point: str | None = None
    canonical_solution: str | None = None
    test: str | None = None
    assertion: Any | None = None
    test_list: Any | None = None
    public_test_cases: Any | None = None
    private_test_cases: Any | None = None
    metadata: dict[str, Any] | None = None
    difficulty: str | None = None
    base_input: Any | None = None
    plus_input: Any | None = None
    contract: str | None = None
    atol: float = 0.0


@dataclass(frozen=True, slots=True)
class CodeGenerationResult:
    sample_index: int
    task_id: str
    prompt: str
    completion: str
    answer: str
    reference_answer: str
    is_passed: bool
    fail_reason: str
    base_passed: bool | None = None

    def to_scoreboard(self) -> ScoreboardEvalResult:
        return ScoreboardEvalResult(
            sample_index=self.sample_index,
            prompt=self.prompt,
            completion=self.completion,
            answer=self.answer,
            reference_answer=self.reference_answer,
            is_passed=self.is_passed,
            fail_reason=self.fail_reason,
        )


@dataclass(frozen=True, slots=True)
class CodeGenerationRunConfig:
    base_url: str
    model: str
    benchmark: str
    dataset_name: str
    limit: int | None = None
    source_type: str = "human_eval_url_gzip"
    source_url: str | None = None
    split: str = "test"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    timeout_s: float = 600.0
    eval_timeout_s: float = 3.0
    scoreboard_dataset: str | None = None
    job_name: str = "code_human_eval"
    job_id: str | None = None
    runner: str = "helicopter_eval.code_generation"
    cot_mode: str = "NoCoT"


def _compress_newlines(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _format_code_prompt(prompt: str) -> str:
    clean = _compress_newlines(prompt).strip()
    return (
        "User: You are a top-level code master. Complete the following code without any additional text or explanation:\n"
        f"{clean}\n\n"
        "Assistant: <think>\n</think>\n```python"
    )


def _format_code_prompt_no_echo(prompt: str) -> str:
    clean = _compress_newlines(prompt).strip()
    return (
        "User: You are a top-level code master. Complete the following code without any additional text or explanation:\n"
        f"{clean}\n\n"
        "Assistant: <think></think>\n```python"
    )


def extract_function_signature(code: str | None) -> str | None:
    if not code:
        return None
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") and stripped.endswith(":"):
            return stripped
    return None


def build_prompt(sample: CodeGenerationSample, config: CodeGenerationRunConfig) -> str:
    if config.benchmark == "livecodebench":
        return _format_lcb_cot_prompt(sample.prompt, sample.starter_code)
    if config.benchmark.startswith("mbpp"):
        signature = extract_function_signature(sample.canonical_solution)
        body = (
            f"{sample.prompt}\nFunction signature: {signature}\nWrite the full function definition."
            if signature
            else sample.prompt
        )
        return _format_code_prompt_no_echo(body)
    return _format_code_prompt(sample.prompt)


def _format_lcb_body(question: str, starter_code: str | None) -> str:
    clean = (question or "").strip()
    body = f"### Question:\n{clean}\n\n"
    if starter_code and starter_code.strip():
        body += f"### Format: {_LCB_FORMAT_WITH_STARTER}\n"
        body += f"```python\n{starter_code}\n```\n\n"
    else:
        body += f"### Format: {_LCB_FORMAT_WITHOUT_STARTER}\n"
        body += "```python\n# YOUR CODE HERE\n```\n\n"
    body += "### Answer: (use the provided format with backticks)\n\n"
    return body


def _format_lcb_cot_prompt(question: str, starter_code: str | None) -> str:
    return f"User: {_LCB_SYSTEM_MESSAGE}\n{_format_lcb_body(question, starter_code)}Assistant: <think"


def _format_lcb_final_prompt(cot_prompt: str, cot_completion: str) -> str:
    return f"{cot_prompt}{cot_completion}{_LCB_FINAL_ANSWER_PREFIX}"


def extract_code_completion(text: str) -> str:
    if not text:
        return ""
    body = str(text)
    body = _THINK_BLOCK_RE.sub("", body)
    body = _LEADING_END_THINK_RE.sub("", body, count=1)
    matches = list(_PYTHON_FENCED_CODE_RE.finditer(body)) or list(_FENCED_CODE_RE.finditer(body))
    if matches:
        return matches[-1].group("code").strip("\r\n").rstrip()
    return _strip_standalone_code_fence_edges(body)


def _strip_standalone_code_fence_edges(text: str) -> str:
    lines = str(text or "").rstrip().splitlines()
    while lines and _STANDALONE_CODE_FENCE_RE.match(lines[0]):
        lines.pop(0)
    while lines and _STANDALONE_CODE_FENCE_RE.match(lines[-1]):
        lines.pop()
    return "\n".join(lines).rstrip()


def _reference_answer(sample: CodeGenerationSample) -> str:
    if sample.public_test_cases is not None or sample.private_test_cases is not None:
        _, ref_answer = _lcb_input_output(sample)
        return ref_answer
    if sample.canonical_solution:
        return sample.canonical_solution.strip()
    for value in (sample.test, sample.assertion, sample.test_list):
        if value:
            if isinstance(value, str):
                return value.strip()
            try:
                return json.dumps(value, ensure_ascii=False, sort_keys=True)
            except TypeError:
                return str(value)
    return ""


def scoreboard_dataset_name(config: CodeGenerationRunConfig) -> str:
    dataset = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    return dataset


def job_id(config: CodeGenerationRunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def completion_sampling_config(config: CodeGenerationRunConfig) -> dict[str, Any]:
    return {
        "answer": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_tokens,
        }
    }


def task_sampling_config(config: CodeGenerationRunConfig) -> dict[str, Any]:
    return {
        "avg_k": 1,
        "pass_ks": [1],
        "prompt_profile": "helicopter",
        "sampling_config": completion_sampling_config(config),
    }


def _iter_jsonl_bytes(raw: bytes):
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _read_url(url: str, *, timeout_s: float) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return response.read()


def _configure_evalplus_cache() -> Path:
    cache_root = Path(os.environ.get("HELICOPTER_EVALPLUS_CACHE", "/tmp/helicopter-evalplus-cache"))
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("EVALPLUS_CACHE", str(cache_root))
    try:
        import evalplus.data.utils as evalplus_data_utils

        evalplus_data_utils.CACHE_DIR = str(cache_root)
        import evalplus.evaluate as evalplus_evaluate

        evalplus_evaluate.CACHE_DIR = str(cache_root)
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in integration environments
        raise SystemExit("code-generation eval requires `evalplus`; install the rwkv dependency group.") from exc
    return cache_root


def _sample_from_payload(index: int, payload: dict[str, Any]) -> CodeGenerationSample:
    task_id = payload.get("task_id") or payload.get("problem_id") or payload.get("id")
    prompt = payload.get("prompt") or payload.get("question") or payload.get("instruction") or payload.get("text")
    if task_id is None or prompt is None:
        raise ValueError(f"code-generation record missing task_id/prompt: {payload}")
    entry_point = _extract_entry_point(payload)
    return CodeGenerationSample(
        sample_index=index,
        task_id=str(task_id),
        prompt=str(prompt),
        check_prefix=str(payload["declaration"]) if payload.get("declaration") is not None else None,
        starter_code=str(payload["starter_code"]) if payload.get("starter_code") is not None else None,
        entry_point=entry_point,
        canonical_solution=(
            str(payload["canonical_solution"]) if payload.get("canonical_solution") is not None else None
        ),
        test=str(payload["test"]) if payload.get("test") is not None else None,
        assertion=payload.get("assertion"),
        test_list=payload.get("test_list") or payload.get("test_cases") or payload.get("tests") or payload.get("unit_tests"),
        public_test_cases=payload.get("public_test_cases"),
        private_test_cases=payload.get("private_test_cases"),
        metadata=_parse_dict_maybe(payload.get("metadata")),
        difficulty=str(payload["difficulty"]) if payload.get("difficulty") is not None else None,
        base_input=payload.get("base_input"),
        plus_input=payload.get("plus_input"),
        contract=str(payload["contract"]) if payload.get("contract") is not None else None,
        atol=float(payload.get("atol") or 0.0),
    )


def _parse_dict_maybe(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            return payload
    return None


def _extract_entry_point(payload: dict[str, Any]) -> str | None:
    raw = payload.get("entry_point")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    for key in ("declaration", "prompt"):
        text = payload.get(key)
        if not isinstance(text, str):
            continue
        match = _DEF_RE.search(text)
        if match:
            return match.group("name")
    return None


def _iter_human_eval_rows(config: CodeGenerationRunConfig):
    if not config.source_url:
        raise ValueError("human_eval_url_gzip source requires source_url")
    raw = gzip.decompress(_read_url(config.source_url, timeout_s=config.timeout_s))
    yield from _iter_jsonl_bytes(raw)


def _iter_human_eval_cn_rows(config: CodeGenerationRunConfig):
    if not config.source_url:
        raise ValueError("human_eval_cn_url source requires source_url")
    yield from _iter_jsonl_bytes(_read_url(config.source_url, timeout_s=config.timeout_s))


def _iter_human_eval_fix_rows(config: CodeGenerationRunConfig):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised in integration environments
        raise SystemExit("human_eval_fix requires the `datasets` package; install the rwkv dependency group.") from exc
    for row in load_dataset("bigcode/humanevalpack", "python", split=config.split):
        payload = dict(row)
        payload["prompt"] = _format_bugfix_prompt(
            payload.get("prompt"),
            payload.get("buggy_solution"),
            payload.get("entry_point"),
        )
        yield payload


def _format_bugfix_prompt(prompt: Any, buggy_solution: Any, entry_point: Any) -> str:
    prompt_text = str(prompt or "").rstrip()
    buggy = str(buggy_solution or "").rstrip()
    entry = str(entry_point or "").strip()
    parts: list[str] = []
    if prompt_text:
        parts.append(prompt_text)
    if buggy:
        parts.append("# Buggy implementation:")
        parts.append(buggy)
    if entry:
        parts.append(f"# Fix the function `{entry}` so it passes all tests.")
    return "\n".join(parts).strip()


def _iter_human_eval_plus_rows(_config: CodeGenerationRunConfig):
    _configure_evalplus_cache()
    from evalplus.data import get_human_eval_plus

    yield from get_human_eval_plus().values()


def _jsonable(value: Any) -> Any:
    if isinstance(value, complex):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _iter_mbpp_rows(config: CodeGenerationRunConfig):
    _configure_evalplus_cache()
    from evalplus.data.mbpp import get_mbpp_plus

    keep_plus = config.benchmark == "mbpp_plus"
    for task_id, problem in get_mbpp_plus().items():
        payload = _jsonable(dict(problem))
        payload.setdefault("task_id", str(task_id))
        prompt = payload.get("prompt") or payload.get("question") or ""
        payload["prompt"] = str(prompt).replace("    ", "\t")
        payload["question"] = payload["prompt"]
        if not keep_plus:
            payload.pop("base_input", None)
            payload.pop("plus_input", None)
        yield payload


def _iter_livecodebench_rows(config: CodeGenerationRunConfig):
    if config.split != "test":
        raise ValueError("livecodebench only supports test split")
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised in integration environments
        raise SystemExit("livecodebench requires the `datasets` package; install the rwkv dependency group.") from exc
    version_tag = os.environ.get("RWKV_SKILLS_LIVECODEBENCH_VERSION_TAG", _LCB_VERSION_TAG).strip()
    dataset = load_dataset(
        path=_LCB_DATASET_ID,
        name=_LCB_DATASET_CONFIG,
        split=config.split,
        version_tag=version_tag,
    )
    for row in sorted(dataset, key=lambda item: str(item.get("question_id", ""))):
        payload = dict(row)
        question_id = str(payload.get("question_id", "") or "")
        payload["task_id"] = question_id
        payload["prompt"] = str(payload.get("question_content") or payload.get("question_title") or "")
        payload.setdefault("question_id", question_id)
        payload["release_version"] = version_tag
        payload["source_dataset"] = _LCB_DATASET_ID
        yield payload


def _iter_rows(config: CodeGenerationRunConfig):
    if config.source_type == "human_eval_url_gzip":
        return _iter_human_eval_rows(config)
    if config.source_type == "human_eval_cn_url":
        return _iter_human_eval_cn_rows(config)
    if config.source_type == "human_eval_fix_hf":
        return _iter_human_eval_fix_rows(config)
    if config.source_type == "human_eval_plus_evalplus":
        return _iter_human_eval_plus_rows(config)
    if config.source_type == "mbpp_evalplus":
        return _iter_mbpp_rows(config)
    if config.source_type == "livecodebench_hf":
        return _iter_livecodebench_rows(config)
    raise ValueError(f"unsupported code-generation source_type: {config.source_type}")


def load_samples(config: CodeGenerationRunConfig) -> list[CodeGenerationSample]:
    if config.limit is not None and int(config.limit) < 0:
        raise ValueError("limit must be non-negative")
    limit = None if config.limit is None else int(config.limit)
    samples: list[CodeGenerationSample] = []
    for payload in _iter_rows(config):
        if limit is not None and len(samples) >= limit:
            break
        samples.append(_sample_from_payload(len(samples), dict(payload)))
    return samples


def _format_plus_assertions(sample: CodeGenerationSample) -> str | None:
    if not sample.plus_input or not sample.canonical_solution or not sample.entry_point:
        return None
    programs = []
    if sample.prompt:
        combo = f"{sample.prompt}\n"
        if sample.contract:
            combo += f"{sample.contract}\n"
        combo += sample.canonical_solution
        programs.append(combo)
    programs.append(sample.canonical_solution)

    reference = None
    for program in programs:
        try:
            scope: dict[str, Any] = {}
            exec(program, scope)
            candidate = scope.get(sample.entry_point)
            if callable(candidate):
                reference = candidate
                break
        except Exception:
            continue
    if reference is None:
        return None

    lines: list[str] = []
    for source_args in sample.plus_input:
        args = source_args if isinstance(source_args, (list, tuple)) else [source_args]
        try:
            expected = reference(*args)
        except Exception:
            continue
        arg_expr = ", ".join(repr(arg) for arg in args)
        if isinstance(expected, float) and sample.atol:
            lines.append(f"    assert abs(candidate({arg_expr}) - {expected!r}) <= {sample.atol}")
        else:
            lines.append(f"    assert candidate({arg_expr}) == {expected!r}")
    return "\n".join(lines) if lines else None


def _build_human_eval_check_program(sample: CodeGenerationSample, completion: str) -> str:
    if not sample.entry_point:
        raise ValueError("entry_point is required for HumanEval scoring")
    test = sample.test or ""
    plus_block = _format_plus_assertions(sample)
    if plus_block:
        combined_tests = (
            f"{test}\n\n"
            "def _helicopter_plus(candidate):\n"
            f"{plus_block}\n\n"
            "def _helicopter_check(candidate):\n"
            "    check(candidate)\n"
            "    _helicopter_plus(candidate)\n"
        )
        check_call = f"_helicopter_check({sample.entry_point})"
    else:
        combined_tests = test
        check_call = f"check({sample.entry_point})"
    solution = completion
    if re.search(rf"def\s+{re.escape(sample.entry_point)}\s*\(", completion):
        imports = _leading_imports(sample.check_prefix or sample.prompt)
        if imports:
            solution = f"{imports}\n{completion}"
    else:
        solution = f"{sample.check_prefix or sample.prompt}{completion}"
    return f"{solution}\n{combined_tests}\n{check_call}"


def _leading_imports(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("def "):
            break
        if stripped.startswith("import ") or stripped.startswith("from "):
            lines.append(line)
    return "\n".join(lines).strip()


def _unsafe_execute_program(check_program: str, timeout: float, result):
    with _create_tempdir():
        import os as _os
        import shutil

        rmtree = shutil.rmtree
        rmdir = _os.rmdir
        chdir = _os.chdir
        _reliability_guard()

        try:
            exec_globals: dict[str, Any] = {}
            with _swallow_io():
                with _time_limit(timeout):
                    filename = "<helicopter_code_generation>"
                    linecache.cache[filename] = (
                        len(check_program),
                        None,
                        check_program.splitlines(keepends=True),
                        filename,
                    )
                    exec(compile(check_program, filename, "exec"), exec_globals)
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except BaseException as exc:  # noqa: BLE001
            exc_type = type(exc).__name__
            exc_msg = str(exc).strip()
            header = f"failed: {exc_type}: {exc_msg}" if exc_msg else f"failed: {exc_type}"
            tb = traceback.format_exc().rstrip()
            result.append(f"{header}\n{tb}" if tb else header)

        shutil.rmtree = rmtree
        _os.rmdir = rmdir
        _os.chdir = chdir


def _execute_program(check_program: str, timeout: float) -> tuple[bool, str]:
    manager = multiprocessing.Manager()
    try:
        result = manager.list()
        process = multiprocessing.Process(target=_unsafe_execute_program, args=(check_program, timeout, result))
        process.start()
        process.join(timeout=timeout + 1)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        if not result:
            result.append("timed out")
        detail = str(result[0])
        return detail == "passed", detail
    finally:
        manager.shutdown()


def _score_human_eval(sample: CodeGenerationSample, completion: str, config: CodeGenerationRunConfig) -> tuple[bool, str, bool | None]:
    try:
        return (*_execute_program(_build_human_eval_check_program(sample, completion), config.eval_timeout_s), None)
    except Exception as exc:  # noqa: BLE001
        return False, f"failed: {type(exc).__name__}: {exc}", None


def _build_mbpp_check_program(sample: CodeGenerationSample, completion: str) -> str:
    test_sections: list[str] = []
    for section in (sample.test_list, sample.assertion):
        if not section:
            continue
        if isinstance(section, str):
            test_sections.append(section)
        else:
            test_sections.append("\n".join(str(item) for item in section))
    tests_str = "\n".join(test_sections)
    if not tests_str:
        return completion
    return f"{completion.rstrip()}\n{tests_str.rstrip()}\n"


def _score_mbpp_base(sample: CodeGenerationSample, completion: str, config: CodeGenerationRunConfig) -> tuple[bool, str, bool | None]:
    return (*_execute_program(_build_mbpp_check_program(sample, completion), config.eval_timeout_s), None)


def _parse_json_maybe(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _decode_lcb_private_tests(raw: str) -> list[dict[str, Any]]:
    import base64
    import pickle
    import zlib

    try:
        payload = pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8"))))
        decoded = json.loads(payload)
    except Exception:
        return []
    return decoded if isinstance(decoded, list) else []


def _parse_lcb_tests(raw: object) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return _decode_lcb_private_tests(raw)
        return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []
    return []


def _lcb_input_output(sample: CodeGenerationSample) -> tuple[dict[str, str], str]:
    metadata = _parse_json_maybe(sample.metadata or {})
    if not isinstance(metadata, dict):
        metadata = {}
    public_tests = _parse_lcb_tests(sample.public_test_cases)
    private_tests = _parse_lcb_tests(sample.private_test_cases)
    tests = public_tests + private_tests
    input_output = {
        "input_output": json.dumps(
            {
                "inputs": [str(item.get("input", "")) for item in tests],
                "outputs": [str(item.get("output", "")) for item in tests],
                "fn_name": metadata.get("func_name"),
            }
        )
    }
    ref_answer = (
        f"LiveCodeBench official tests for task_id={sample.task_id}; "
        f"public_tests={len(public_tests)}, private_tests={len(private_tests)}."
    )
    return input_output, ref_answer


def _format_lcb_exception(exc: BaseException, *, include_traceback: bool = False) -> str:
    exc_type = type(exc).__name__
    exc_msg = str(exc).strip()
    header = f"failed: {exc_type}: {exc_msg}" if exc_msg else f"failed: {exc_type}"
    if not include_traceback:
        return header
    tb = traceback.format_exc().rstrip()
    return f"{header}\n{tb}" if tb else header


def _lcb_clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If) and ast.unparse(last_block.test).strip() == "__name__ == '__main__'":
            code = ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)
    except Exception:
        pass
    return code


def _lcb_make_function(code: str) -> str:
    try:
        import_stmts = []
        other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                other_stmts.append(stmt)
        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        return _LCB_IMPORT_STRING + "\n" + ast.unparse(import_stmts) + "\n" + ast.unparse(function_ast)
    except Exception:
        return code


def _lcb_compile_code(code: str, timeout: float):
    with _time_limit(timeout):
        tmp_sol = ModuleType("tmp_sol", "")
        with _swallow_io():
            exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            return tmp_sol.Solution()
        return tmp_sol


def _lcb_get_function(compiled_sol: object, fn_name: str):
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception:
        return None


def _lcb_decimal_line(line: str) -> tuple[bool, list[Decimal]]:
    try:
        return True, [Decimal(item) for item in line.split()]
    except Exception:
        return False, []


def _lcb_stripped_lines(value: str) -> list[str]:
    return [line.strip() for line in value.strip().split("\n")]


@contextlib.contextmanager
def _capture_output(stdin_text: str):
    stdout = io.StringIO()
    stdin = io.StringIO(stdin_text)
    stdin.buffer = io.BytesIO(stdin_text.encode("utf-8"))  # type: ignore[attr-defined]
    with contextlib.redirect_stdout(stdout):
        with contextlib.redirect_stderr(io.StringIO()):
            with redirect_stdin(stdin):
                yield stdout


def _grade_lcb_call_based(code: str, all_inputs: list[Any], all_outputs: list[Any], fn_name: str, timeout: float) -> str:
    code = _LCB_IMPORT_STRING + "\n\n" + code
    try:
        compiled_sol = _lcb_compile_code(code, timeout)
    except TimeoutException:
        return "timed out"
    except BaseException as exc:  # noqa: BLE001
        return _format_lcb_exception(exc)
    method = _lcb_get_function(compiled_sol, fn_name)
    if method is None:
        return "failed: Missing entry point"
    try:
        parsed_inputs = [[json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs]
        parsed_outputs = [json.loads(output) for output in all_outputs]
    except Exception:
        return "failed: Invalid test case"
    for gt_inp, gt_out in zip(parsed_inputs, parsed_outputs):
        try:
            with _swallow_io():
                with _time_limit(timeout):
                    prediction = method(*gt_inp)
            if isinstance(prediction, tuple):
                prediction = list(prediction)
            if prediction != gt_out:
                return "failed: Wrong Answer"
        except TimeoutException:
            return "timed out"
        except BaseException as exc:  # noqa: BLE001
            return _format_lcb_exception(exc)
    return "passed"


def _grade_lcb_stdio(code: str, all_inputs: list[Any], all_outputs: list[Any], timeout: float) -> str:
    code = _lcb_make_function(_lcb_clean_if_name(code))
    try:
        compiled_sol = _lcb_compile_code(code, timeout)
    except TimeoutException:
        return "timed out"
    except BaseException as exc:  # noqa: BLE001
        return _format_lcb_exception(exc)
    method = _lcb_get_function(compiled_sol, "wrapped_function")
    if method is None:
        return "failed: Missing entry point"
    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        try:
            with _capture_output(str(gt_inp)) as captured_output:
                with _time_limit(timeout):
                    method()
            prediction = captured_output.getvalue()
        except TimeoutException:
            return "timed out"
        except BaseException as exc:  # noqa: BLE001
            return _format_lcb_exception(exc)
        pred_lines = _lcb_stripped_lines(prediction)
        gt_lines = _lcb_stripped_lines(str(gt_out))
        if len(pred_lines) != len(gt_lines):
            return "failed: Wrong Answer"
        for pred_line, gt_line in zip(pred_lines, gt_lines):
            if pred_line == gt_line:
                continue
            pred_ok, pred_decimals = _lcb_decimal_line(pred_line)
            gt_ok, gt_decimals = _lcb_decimal_line(gt_line)
            if not pred_ok or not gt_ok or pred_decimals != gt_decimals:
                return "failed: Wrong Answer"
    return "passed"


def _run_lcb_test(sample: dict[str, str], completion: str, timeout: float) -> str:
    if not completion:
        return "failed: no test code provided"
    try:
        in_outs = json.loads(sample["input_output"])
    except Exception:
        return "failed: Invalid test case"
    if not isinstance(in_outs, dict):
        return "failed: Invalid test case"
    all_inputs = in_outs.get("inputs") or []
    all_outputs = in_outs.get("outputs") or []
    fn_name = in_outs.get("fn_name")
    if fn_name:
        return _grade_lcb_call_based(completion, all_inputs, all_outputs, str(fn_name), timeout)
    return _grade_lcb_stdio(completion, all_inputs, all_outputs, timeout)


def _unsafe_execute_lcb(sample: dict[str, str], completion: str, timeout: float, result):
    with _create_tempdir():
        import os as _os
        import shutil

        rmtree = shutil.rmtree
        rmdir = _os.rmdir
        chdir = _os.chdir
        _reliability_guard()
        try:
            result.append(_run_lcb_test(sample, completion, timeout))
        except TimeoutException:
            result.append("timed out")
        except BaseException as exc:  # noqa: BLE001
            result.append(_format_lcb_exception(exc, include_traceback=True))
        shutil.rmtree = rmtree
        _os.rmdir = rmdir
        _os.chdir = chdir


def _score_livecodebench(sample: CodeGenerationSample, completion: str, config: CodeGenerationRunConfig) -> tuple[bool, str, bool | None]:
    lcb_sample, _ = _lcb_input_output(sample)
    manager = multiprocessing.Manager()
    try:
        result = manager.list()
        process = multiprocessing.Process(
            target=_unsafe_execute_lcb,
            args=(lcb_sample, completion, config.eval_timeout_s, result),
        )
        process.start()
        process.join(timeout=config.eval_timeout_s + 1)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        if not result:
            result.append("timed out")
        detail = str(result[0])
        return detail == "passed", detail, None
    finally:
        manager.shutdown()


_MP_ARENA_DIRS_SANITIZED = False


def _sanitize_mp_arena_dir_candidates() -> None:
    global _MP_ARENA_DIRS_SANITIZED
    if _MP_ARENA_DIRS_SANITIZED:
        return
    import multiprocessing.heap as mp_heap

    if hasattr(mp_heap, "Arena") and hasattr(mp_heap.Arena, "_dir_candidates"):
        mp_heap.Arena._dir_candidates = []
    _MP_ARENA_DIRS_SANITIZED = True


def _untrusted_check_mbpp(
    solution: str,
    *,
    inputs: list[Any],
    entry_point: str,
    expected: list[Any],
    atol: float,
    ref_time: list[float],
    timeout: float,
) -> tuple[str, list[bool]]:
    from multiprocessing.sharedctypes import RawArray, RawValue

    from evalplus.eval import FAIL, PASS, TIMEOUT, _UNKNOWN, _mapping, unsafe_execute

    _sanitize_mp_arena_dir_candidates()

    time_limits = [max(0.1, 2.0 * float(item)) for item in ref_time]
    overall_timeout = min(float(timeout), sum(time_limits)) + 1
    progress = RawValue("i", 0)
    stat = RawValue("i", _UNKNOWN)
    details = RawArray("b", [0 for _ in range(len(inputs))])

    process = multiprocessing.Process(
        target=unsafe_execute,
        args=(
            "mbpp",
            entry_point,
            solution,
            inputs,
            expected,
            time_limits,
            atol,
            True,
            stat,
            details,
            progress,
        ),
    )
    process.start()
    process.join(timeout=overall_timeout + 1)
    if process.is_alive():
        process.terminate()
    if process.is_alive():
        process.kill()

    status = _mapping.get(stat.value)
    detail_slice = [bool(item) for item in details[: progress.value]]
    if not status:
        status = TIMEOUT
    if status == PASS and (len(detail_slice) != len(inputs) or not all(detail_slice)):
        status = FAIL
    return status, detail_slice


def _score_mbpp_plus(
    sample: CodeGenerationSample,
    completion: str,
    expected_output: dict[str, Any],
    config: CodeGenerationRunConfig,
) -> tuple[bool, str, bool | None]:
    from evalplus.eval import PASS

    base_inputs = sample.base_input or []
    plus_inputs = sample.plus_input or []
    entry_point = sample.entry_point or ""
    base_status, _ = _untrusted_check_mbpp(
        completion,
        inputs=base_inputs,
        entry_point=entry_point,
        expected=expected_output["base"],
        atol=sample.atol,
        ref_time=expected_output["base_time"],
        timeout=config.eval_timeout_s,
    )
    plus_status, _ = _untrusted_check_mbpp(
        completion,
        inputs=plus_inputs,
        entry_point=entry_point,
        expected=expected_output["plus"],
        atol=sample.atol,
        ref_time=expected_output["plus_time"],
        timeout=config.eval_timeout_s,
    )
    base_passed = base_status == PASS
    plus_passed = plus_status == PASS
    passed = base_passed and plus_passed
    detail = "passed" if passed else f"base={base_status}; plus={plus_status}"
    return passed, detail, base_passed


def _mbpp_plus_expected_outputs(samples: Sequence[CodeGenerationSample]) -> dict[str, dict[str, Any]]:
    _configure_evalplus_cache()
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.evaluate import get_groundtruth

    problems = {
        sample.task_id: {
            "task_id": sample.task_id,
            "prompt": sample.prompt,
            "canonical_solution": sample.canonical_solution,
            "entry_point": sample.entry_point,
            "base_input": sample.base_input or [],
            "plus_input": sample.plus_input or [],
            "atol": sample.atol,
        }
        for sample in samples
    }
    digest_payload = json.dumps(problems, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    dataset_hash = "helicopter_mbpp_plus_" + hashlib.md5(digest_payload).hexdigest()
    return get_groundtruth(problems, dataset_hash, MBPP_OUTPUT_NOT_NONE_TASKS)


def score_completion(
    sample: CodeGenerationSample,
    completion: str,
    config: CodeGenerationRunConfig,
    *,
    expected_outputs: dict[str, dict[str, Any]] | None = None,
) -> tuple[bool, str, bool | None]:
    if config.benchmark == "mbpp":
        return _score_mbpp_base(sample, completion, config)
    if config.benchmark == "mbpp_plus":
        if expected_outputs is None:
            raise ValueError("mbpp_plus expected_outputs are required")
        return _score_mbpp_plus(sample, completion, expected_outputs[sample.task_id], config)
    if config.benchmark == "livecodebench":
        return _score_livecodebench(sample, completion, config)
    return _score_human_eval(sample, completion, config)


def generate_completion(
    sample: CodeGenerationSample,
    config: CodeGenerationRunConfig,
    *,
    expected_outputs: dict[str, dict[str, Any]] | None = None,
) -> CodeGenerationResult:
    prompt = build_prompt(sample, config)
    if config.benchmark == "livecodebench":
        cot_completion = chat_completion(
            base_url=config.base_url,
            model=config.model,
            prompt=prompt,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
        )
        prompt = _format_lcb_final_prompt(prompt, cot_completion)
    raw_completion = chat_completion(
        base_url=config.base_url,
        model=config.model,
        prompt=prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
    )
    answer = extract_code_completion(raw_completion)
    is_passed, detail, base_passed = score_completion(
        sample,
        answer,
        config,
        expected_outputs=expected_outputs,
    )
    return CodeGenerationResult(
        sample_index=sample.sample_index,
        task_id=sample.task_id,
        prompt=prompt,
        completion=raw_completion,
        answer=answer,
        reference_answer=_reference_answer(sample),
        is_passed=is_passed,
        fail_reason="" if is_passed else detail,
        base_passed=base_passed,
    )


def evaluate_samples(
    samples: Sequence[CodeGenerationSample],
    config: CodeGenerationRunConfig,
) -> list[CodeGenerationResult]:
    expected_outputs = _mbpp_plus_expected_outputs(samples) if config.benchmark == "mbpp_plus" else None
    return [generate_completion(sample, config, expected_outputs=expected_outputs) for sample in samples]


def write_results(results: Sequence[CodeGenerationResult], *, config: CodeGenerationRunConfig, repo_root: Path) -> int:
    extra_metrics: dict[str, Any] = {"pass@1": sum(result.is_passed for result in results) / len(results)} if results else {"pass@1": 0.0}
    base_values = [result.base_passed for result in results if result.base_passed is not None]
    if base_values:
        extra_metrics["base_pass@1"] = sum(bool(item) for item in base_values) / len(base_values)
    task_id = asyncio.run(
        write_scoreboard_results(
            [result.to_scoreboard() for result in results],
            config=ScoreboardWriteConfig(
                dataset=scoreboard_dataset_name(config),
                model=config.model,
                job_name=config.job_name,
                job_id=job_id(config),
                benchmark=config.benchmark,
                runner=config.runner,
                cot_mode=config.cot_mode,
                sampling_config=task_sampling_config(config),
                completion_sampling_config=completion_sampling_config(config),
                extra_metrics=extra_metrics,
            ),
            repo_root=repo_root,
        )
    )
    return int(task_id)


def run_code_generation(config: CodeGenerationRunConfig, *, repo_root: Path) -> dict[str, Any]:
    samples = load_samples(config)
    results = evaluate_samples(samples, config)
    task_id = write_results(results, config=config, repo_root=repo_root)
    passed = sum(1 for result in results if result.is_passed)
    payload: dict[str, Any] = {
        "task_id": task_id,
        "benchmark": config.benchmark,
        "dataset": scoreboard_dataset_name(config),
        "model": config.model,
        "total": len(results),
        "passed": passed,
        "pass@1": passed / len(results) if results else 0.0,
    }
    base_values = [result.base_passed for result in results if result.base_passed is not None]
    if base_values:
        payload["base_pass@1"] = sum(bool(item) for item in base_values) / len(base_values)
    return payload


def dry_run_summary(config: CodeGenerationRunConfig) -> dict[str, Any]:
    return {
        "benchmark": config.benchmark,
        "dataset_name": config.dataset_name,
        "source_type": config.source_type,
        "split": config.split,
        "limit": config.limit,
        "base_url": config.base_url,
        "model": config.model,
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": job_id(config),
    }


@contextlib.contextmanager
def _time_limit(seconds: float):
    def signal_handler(signum, frame):  # noqa: ANN001, ARG001
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def _swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def _create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with _chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs):  # noqa: ANN002, ARG002
        raise OSError

    def readline(self, *args, **kwargs):  # noqa: ANN002, ARG002
        raise OSError

    def readlines(self, *args, **kwargs):  # noqa: ANN002, ARG002
        raise OSError

    def readable(self, *args, **kwargs):  # noqa: ANN002, ARG002
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore[type-arg]
    _stream = "stdin"


@contextlib.contextmanager
def _chdir(root: str):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)


def _reliability_guard(maximum_memory_bytes: int | None = None) -> None:
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None

    __builtins__["help"] = None
