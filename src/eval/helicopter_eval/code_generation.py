from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class CodeGenerationSample:
    sample_index: int
    task_id: str
    prompt: str
    check_prefix: str | None = None
    entry_point: str | None = None
    canonical_solution: str | None = None
    test: str | None = None
    assertion: Any | None = None
    test_list: Any | None = None
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
    if config.benchmark.startswith("mbpp"):
        signature = extract_function_signature(sample.canonical_solution)
        body = (
            f"{sample.prompt}\nFunction signature: {signature}\nWrite the full function definition."
            if signature
            else sample.prompt
        )
        return _format_code_prompt_no_echo(body)
    return _format_code_prompt(sample.prompt)


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
        entry_point=entry_point,
        canonical_solution=(
            str(payload["canonical_solution"]) if payload.get("canonical_solution") is not None else None
        ),
        test=str(payload["test"]) if payload.get("test") is not None else None,
        assertion=payload.get("assertion"),
        test_list=payload.get("test_list") or payload.get("test_cases") or payload.get("tests") or payload.get("unit_tests"),
        base_input=payload.get("base_input"),
        plus_input=payload.get("plus_input"),
        contract=str(payload["contract"]) if payload.get("contract") is not None else None,
        atol=float(payload.get("atol") or 0.0),
    )


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
    return _score_human_eval(sample, completion, config)


def generate_completion(
    sample: CodeGenerationSample,
    config: CodeGenerationRunConfig,
    *,
    expected_outputs: dict[str, dict[str, Any]] | None = None,
) -> CodeGenerationResult:
    prompt = build_prompt(sample, config)
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
