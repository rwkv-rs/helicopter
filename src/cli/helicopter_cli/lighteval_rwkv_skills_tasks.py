from __future__ import annotations

import contextlib
import ast
import copy
import faulthandler
import importlib
import io
import json
import linecache
import math
import multiprocessing
import os
import platform
import re
import signal
import string
import subprocess
import sys
import tempfile
import traceback
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from string import ascii_uppercase
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, Sequence

from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, prompt_template

from lighteval.metrics.metrics import Metrics, math_scorer
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SamplingMethod
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


LOCAL_DATA_ROOT = Path(__file__).resolve().parents[3] / "benchmarks/lighteval_data"

HF_MATH_TASKS = {
    "algebra222": {
        "repo": "sirdug/Algebra222",
        "split": "train",
    },
    "amc23": {
        "repo": "math-ai/amc23",
        "split": "test",
    },
    "beyond_aime": {
        "repo": "ByteDance-Seed/BeyondAIME",
        "split": "test",
    },
    "brumo25": {
        "repo": "MathArena/brumo_2025",
        "split": "train",
    },
    "college_math": {
        "repo": "di-zhang-fdu/College_Math_Test",
        "split": "test",
    },
    "comp_math_24_25": {
        "repo": str(LOCAL_DATA_ROOT / "comp_math_24_25"),
        "split": "test",
    },
    "gaokao2023en": {
        "repo": "test-time-compute/test_gaokao2023en",
        "split": "test",
    },
    "hmmt_feb25": {
        "repo": "MathArena/hmmt_feb_2025",
        "split": "train",
    },
    "math_odyssey": {
        "repo": "MathOdyssey/MathOdyssey",
        "split": "test",
    },
    "mawps": {
        "repo": str(LOCAL_DATA_ROOT / "mawps"),
        "split": "test",
    },
    "minerva_math": {
        "repo": "math-ai/minervamath",
        "split": "test",
    },
    "omni_math": {
        "repo": "KbsdJames/Omni-MATH",
        "split": "test",
    },
}

HF_POLYMATH_LANGUAGES = (
    "ar",
    "bn",
    "de",
    "en",
    "es",
    "fr",
    "id",
    "it",
    "ja",
    "ko",
    "ms",
    "pt",
    "ru",
    "sw",
    "te",
    "th",
    "vi",
    "zh",
)

HF_MULTIPLE_CHOICE_TASKS = {
    "include": {
        "repo": str(LOCAL_DATA_ROOT / "include"),
        "split": "test",
    },
    "supergpqa": {
        "repo": "m-a-p/SuperGPQA",
        "split": "train",
    },
}

HF_MMMLU_TASKS = {
    "mmmlu_ar": "AR_XY",
    "mmmlu_bn": "BN_BD",
    "mmmlu_de": "DE_DE",
    "mmmlu_es": "ES_LA",
    "mmmlu_fr": "FR_FR",
    "mmmlu_hi": "HI_IN",
    "mmmlu_id": "ID_ID",
    "mmmlu_it": "IT_IT",
    "mmmlu_ja": "JA_JP",
    "mmmlu_ko": "KO_KR",
    "mmmlu_pt": "PT_BR",
    "mmmlu_sw": "SW_KE",
    "mmmlu_yo": "YO_NG",
    "mmmlu_zh": "ZH_CN",
}

WMT24PP_TARGET_LANGUAGES = {
    "de_DE": "German",
    "es_MX": "Spanish",
    "fr_FR": "French",
    "it_IT": "Italian",
    "ja_JP": "Japanese",
}

HUMANEVAL_TIMEOUT_SECONDS = 3.0
MBPP_TIMEOUT_SECONDS = 3.0
LONGCODEQA_HF_REPO = "Steefano/LCB"
LONGCODEQA_DATA_FILE = "LongCodeQA.zip"
LONGCODEQA_LIGHTEVAL_DATASET = "rwkv_skills_longcodeqa_hf_zip"
LONGCODEQA_DEFAULT_MAX_PROMPT_CHARS = 8_000
LONGCODEQA_CHUNK_CHARS = 1_200
LONGCODEQA_CHUNK_OVERLAP_CHARS = 160
LONG_BENCH_HF_REPO = "THUDM/LongBench"
LONG_BENCH_DATASETS = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
)
LONG_BENCH_QA_DATASETS = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "triviaqa",
)
BFCL_LIGHTEVAL_DATASET = "rwkv_skills_bfcl_v4_ast"
BFCL_EXEC_LIGHTEVAL_DATASET = "rwkv_skills_bfcl_v4_exec"
BFCL_GORILLA_REPO_URL = "https://github.com/ShishirPatil/gorilla.git"
BFCL_GORILLA_REVISION = "main"
BFCL_GORILLA_ROOT_NAME = "gorilla"
BFCL_DEFAULT_MODEL_NAME = "gorilla-openfunctions-v2"
BFCL_MAX_TOOL_DESCRIPTION_CHARS = 700
BFCL_MAX_TOOL_SCHEMA_CHARS = 1_200
BFCL_AST_TASKS = {
    "bfcl_simple_python": {
        "category": "simple_python",
        "question": ("BFCL_v4_simple_python.json",),
        "answer": ("possible_answer", "BFCL_v4_simple_python.json"),
    },
    "bfcl_multiple": {
        "category": "multiple",
        "question": ("BFCL_v4_multiple.json",),
        "answer": ("possible_answer", "BFCL_v4_multiple.json"),
    },
    "bfcl_exec_simple_ast": {
        "category": "exec_simple",
        "question": ("unused_datasets", "question", "BFCL_v4_exec_simple.json"),
        "answer": ("unused_datasets", "possible_answer", "BFCL_v4_exec_simple.json"),
    },
    "bfcl_exec_multiple_ast": {
        "category": "exec_multiple",
        "question": ("unused_datasets", "question", "BFCL_v4_exec_multiple.json"),
        "answer": ("unused_datasets", "possible_answer", "BFCL_v4_exec_multiple.json"),
    },
}
BFCL_EXEC_TASKS = {
    "bfcl_exec_simple": {
        "category": "exec_simple",
        "question": ("unused_datasets", "question", "BFCL_v4_exec_simple.json"),
        "answer": ("unused_datasets", "possible_answer", "BFCL_v4_exec_simple.json"),
    },
    "bfcl_exec_multiple": {
        "category": "exec_multiple",
        "question": ("unused_datasets", "question", "BFCL_v4_exec_multiple.json"),
        "answer": ("unused_datasets", "possible_answer", "BFCL_v4_exec_multiple.json"),
    },
    "bfcl_exec_parallel": {
        "category": "exec_parallel",
        "question": ("unused_datasets", "question", "BFCL_v4_exec_parallel.json"),
        "answer": ("unused_datasets", "possible_answer", "BFCL_v4_exec_parallel.json"),
    },
    "bfcl_exec_parallel_multiple": {
        "category": "exec_parallel_multiple",
        "question": ("unused_datasets", "question", "BFCL_v4_exec_parallel_multiple.json"),
        "answer": ("unused_datasets", "possible_answer", "BFCL_v4_exec_parallel_multiple.json"),
    },
}
APIBANK_LIGHTEVAL_DATASET = "rwkv_skills_apibank"
APIBANK_REPO_URL = "https://github.com/AlibabaResearch/DAMO-ConvAI.git"
APIBANK_REPO_REVISION = "main"
APIBANK_REPO_ROOT_NAME = "DAMO-ConvAI"
APIBANK_TASKS = {
    "apibank_l1": {"level": 1},
    "apibank_l2": {"level": 2},
    "apibank_level1": {"level": 1},
    "apibank_level2": {"level": 2},
}
_APIBANK_SCHEMA_SANDBOX_CACHE: dict[str, "ApiBankSandbox"] = {}
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_FENCED_CODE_RE = re.compile(
    r"```[ \t]*(?:python|py)?[^\S\r\n]*\r?\n(?P<code>.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_DEF_RE = re.compile(r"def\s+(?P<name>[\w_]+)\s*\(")
_LEADING_END_THINK_RE = re.compile(r"^[\s\r\n]*</think>[ \t]*\r?\n?", re.IGNORECASE)
_STANDALONE_CODE_FENCE_RE = re.compile(r"^[ \t]*```[ \t]*(?:python|py)?[ \t]*$", re.IGNORECASE)
_LONGCODEQA_CHOICE_LINE_RE = re.compile(r"^\s*([A-Z])\)", re.MULTILINE)
_LONGCODEQA_ANSWER_PREFIX_RE = re.compile(
    r"(?:final\s+answer|correct\s+answer|answer)\s*(?:is)?\s*[:\-]?\s*(?:\*\*)?\s*\(?([A-Z])\)?(?:\*\*)?\b",
    re.IGNORECASE,
)
_LONGCODEQA_FENCED_BLOCK_RE = re.compile(r"^\s*```(?:json|text)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)
_LONGCODEQA_INLINE_BOLD_CHOICE_RE = re.compile(r"\*\*\s*\(?([A-Z])\)?\s*[\).:]", re.IGNORECASE)
_LONG_BENCH_ANSWER_PREFIX_RE = re.compile(r"^\s*(?:final\s+answer|answer)\s*[:\-]\s*", re.IGNORECASE)
_LONG_BENCH_ROLE_PREFIX_RE = re.compile(r"^\s*Assistant\s*:\s*", re.IGNORECASE)
_LONG_BENCH_WORD_OR_CJK_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
_BFCL_JSON_PREFILL_FIELDS = frozenset({"action", "function", "function_call", "name", "tool", "tool_calls", "tool_name", "type"})
_LONGCODEQA_STOPWORDS = {
    "about",
    "after",
    "also",
    "because",
    "before",
    "between",
    "code",
    "current",
    "does",
    "from",
    "have",
    "into",
    "more",
    "question",
    "repo",
    "repository",
    "should",
    "than",
    "that",
    "their",
    "there",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your
response should be of the form "ANSWER: $ANSWER" (without quotes)
where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to
the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def extract_code_completion(text: str) -> str:
    """Recover executable Python from chatty code benchmark completions."""

    if not text:
        return ""
    body = str(text)
    body = _THINK_BLOCK_RE.sub("", body)
    body = _LEADING_END_THINK_RE.sub("", body, count=1)
    matches = list(_FENCED_CODE_RE.finditer(body))
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


def _extract_entry_point(line: dict[str, Any]) -> str | None:
    raw = line.get("entry_point")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    for key in ("declaration", "prompt"):
        text = line.get(key)
        if not isinstance(text, str):
            continue
        match = _DEF_RE.search(text)
        if match:
            return match.group("name")
    return None


def _prompt_prefix_before_entrypoint(prompt: str, entry_point: str) -> str:
    match = re.search(rf"(?m)^def\s+{re.escape(entry_point)}\s*\(", prompt)
    if not match:
        return ""
    return prompt[: match.start()].rstrip()


def _solution_source(problem: dict[str, Any], completion: str) -> str:
    prompt = str(problem.get("prompt") or "")
    entry_point = str(problem.get("entry_point") or "")
    code = extract_code_completion(completion)
    if prompt and code.startswith(prompt):
        code = code[len(prompt) :].lstrip("\r\n")

    if entry_point and re.search(rf"(?m)^def\s+{re.escape(entry_point)}\s*\(", code):
        prefix = _prompt_prefix_before_entrypoint(prompt, entry_point)
        return f"{prefix}\n\n{code}".lstrip() if prefix else code
    return f"{prompt}{code}"


def _build_humaneval_check_program(problem: dict[str, Any], completion: str) -> str:
    test = str(problem.get("test") or "")
    entry_point = str(problem.get("entry_point") or "")
    if not entry_point:
        raise ValueError("HumanEval problem is missing entry_point")
    solution = _solution_source(problem, completion).rstrip()
    return f"{solution}\n{test}\ncheck({entry_point})"


def _check_python_program_correctness(
    task_id: object,
    check_program: str,
    *,
    timeout: float,
    completion_id: int | None = None,
) -> dict[str, Any]:
    result_queue: multiprocessing.Queue[str] = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_unsafe_execute_python,
        args=(check_program, timeout, result_queue),
    )
    process.start()
    process.join(timeout=timeout + 1)
    if process.is_alive():
        process.kill()
        process.join()

    result = result_queue.get() if not result_queue.empty() else "timed out"
    return {
        "task_id": task_id,
        "passed": result == "passed",
        "result": result,
        "completion_id": completion_id,
    }


def _check_humaneval_correctness(
    problem: dict[str, Any],
    completion: str,
    *,
    timeout: float = HUMANEVAL_TIMEOUT_SECONDS,
    completion_id: int | None = None,
) -> dict[str, Any]:
    try:
        check_program = _build_humaneval_check_program(problem, completion)
    except BaseException as exc:  # noqa: BLE001
        return {
            "task_id": problem.get("task_id"),
            "passed": False,
            "result": f"failed: {type(exc).__name__}: {exc}",
            "completion_id": completion_id,
        }

    return _check_python_program_correctness(
        problem.get("task_id"),
        check_program,
        timeout=timeout,
        completion_id=completion_id,
    )


def _unsafe_execute_python(check_program: str, timeout: float, result_queue: multiprocessing.Queue) -> None:
    with _create_tempdir():
        import shutil

        original_rmtree = shutil.rmtree
        original_rmdir = os.rmdir
        original_chdir = os.chdir

        reliability_guard()
        try:
            exec_globals: dict[str, Any] = {}
            with _swallow_io():
                with _time_limit(timeout):
                    filename = "<helicopter_humaneval>"
                    linecache.cache[filename] = (
                        len(check_program),
                        None,
                        check_program.splitlines(keepends=True),
                        filename,
                    )
                    exec(compile(check_program, filename, "exec"), exec_globals)
            result_queue.put("passed")
        except TimeoutException:
            result_queue.put("timed out")
        except BaseException as exc:  # noqa: BLE001
            exc_type = type(exc).__name__
            exc_msg = str(exc).strip()
            header = f"failed: {exc_type}: {exc_msg}" if exc_msg else f"failed: {exc_type}"
            tb = traceback.format_exc().rstrip()
            result_queue.put(f"{header}\n{tb}" if tb else header)
        finally:
            shutil.rmtree = original_rmtree
            os.rmdir = original_rmdir
            os.chdir = original_chdir


class TimeoutException(Exception):
    pass


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
    stream = _WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with _redirect_stdin(stream):
                yield


@contextlib.contextmanager
def _create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with _chdir(dirname):
            yield dirname


class _WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs):  # noqa: ANN002, ARG002
        raise OSError

    def readline(self, *args, **kwargs):  # noqa: ANN002, ARG002
        raise OSError

    def readlines(self, *args, **kwargs):  # noqa: ANN002, ARG002
        raise OSError

    def readable(self, *args, **kwargs):  # noqa: ANN002, ARG002
        return False


class _redirect_stdin(contextlib._RedirectStream):  # type: ignore[attr-defined]
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


def reliability_guard(maximum_memory_bytes: int | None = None) -> None:
    """Disable destructive functions in the child process; this is not a sandbox."""

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins
    import shutil
    import subprocess

    builtins.exit = None
    builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None  # type: ignore[assignment]
    os.system = None  # type: ignore[assignment]
    os.putenv = None  # type: ignore[assignment]
    os.remove = None  # type: ignore[assignment]
    os.removedirs = None  # type: ignore[assignment]
    os.rmdir = None  # type: ignore[assignment]
    os.fchdir = None  # type: ignore[assignment]
    os.setuid = None  # type: ignore[assignment]
    os.fork = None  # type: ignore[assignment]
    os.forkpty = None  # type: ignore[assignment]
    os.killpg = None  # type: ignore[assignment]
    os.rename = None  # type: ignore[assignment]
    os.renames = None  # type: ignore[assignment]
    os.truncate = None  # type: ignore[assignment]
    os.replace = None  # type: ignore[assignment]
    os.unlink = None  # type: ignore[assignment]
    os.fchmod = None  # type: ignore[assignment]
    os.fchown = None  # type: ignore[assignment]
    os.chmod = None  # type: ignore[assignment]
    os.chown = None  # type: ignore[assignment]
    os.chroot = None  # type: ignore[assignment]
    os.lchflags = None  # type: ignore[attr-defined]
    os.lchmod = None  # type: ignore[attr-defined]
    os.lchown = None  # type: ignore[attr-defined]
    os.getcwd = None  # type: ignore[assignment]
    os.chdir = None  # type: ignore[assignment]

    shutil.rmtree = None  # type: ignore[assignment]
    shutil.move = None  # type: ignore[assignment]
    shutil.chown = None  # type: ignore[assignment]
    subprocess.Popen = None  # type: ignore[assignment]

    builtins.help = None
    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def _extract_choice_letter(text: str, *, max_choices: int) -> str | None:
    valid = set(ascii_uppercase[:max_choices])
    normalized = text.strip().upper()
    marker = re.search(r"\b(?:ANSWER|ANS|OPTION)(?:\s+IS)?\s*[:\-\)]?\s*([A-Z])\b", normalized)
    if marker and marker.group(1) in valid:
        return marker.group(1)
    for match in re.finditer(r"(?:^|[^A-Z])([A-Z])(?:\s*(?:[.)\]:-]|$))", normalized):
        if match.group(1) in valid:
            return match.group(1)
    return None


class MultipleChoiceLetterMatch(SampleLevelComputation):
    def compute(self, doc: Doc, model_response, **kwargs) -> float:
        gold_indices = doc.gold_index if isinstance(doc.gold_index, list) else [doc.gold_index]
        gold_letters = {ascii_uppercase[index] for index in gold_indices}
        for prediction in model_response.final_text:
            letter = _extract_choice_letter(prediction, max_choices=len(doc.choices))
            if letter in gold_letters:
                return 1.0
        return 0.0


MULTIPLE_CHOICE_LETTER_MATCH = SampleLevelMetric(
    metric_name="mc_letter_match",
    sample_level_fn=MultipleChoiceLetterMatch(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=_mean,
    higher_is_better=True,
)


def _normalize_newlines(text: object) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n")


def _longcodeqa_max_prompt_chars() -> int:
    raw = os.getenv("RWKV_LIGHTEVAL_LONGCODEQA_MAX_PROMPT_CHARS", "")
    try:
        value = int(raw)
    except ValueError:
        value = LONGCODEQA_DEFAULT_MAX_PROMPT_CHARS
    return max(2_000, value)


def _longcodeqa_choice_letters(question: object) -> tuple[str, ...]:
    matches = [match.group(1).upper() for match in _LONGCODEQA_CHOICE_LINE_RE.finditer(str(question or ""))]
    seen: set[str] = set()
    ordered: list[str] = []
    for letter in matches:
        if letter in seen:
            continue
        seen.add(letter)
        ordered.append(letter)
    return tuple(ordered or ascii_uppercase[:4])


def _longcodeqa_terms(question: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", question.lower()):
        if term in _LONGCODEQA_STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _iter_overlapping_chunks(text: str, *, chunk_chars: int, overlap_chars: int) -> list[str]:
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]
    chunks: list[str] = []
    step = max(1, chunk_chars - overlap_chars)
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_chars]
        if chunk:
            chunks.append(chunk)
        if start + chunk_chars >= len(text):
            break
    return chunks


def _middle_truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    notice = "\n[... middle truncated to fit prompt budget ...]\n"
    if max_chars <= len(notice) + 8:
        return text[:max_chars]
    head = max_chars // 2
    tail = max_chars - head - len(notice)
    return text[:head] + notice + text[-tail:]


def _longcodeqa_context_excerpt(repo_text: str, question: str, *, max_chars: int) -> str:
    if len(repo_text) <= max_chars:
        return repo_text
    chunks = _iter_overlapping_chunks(
        repo_text,
        chunk_chars=LONGCODEQA_CHUNK_CHARS,
        overlap_chars=LONGCODEQA_CHUNK_OVERLAP_CHARS,
    )
    terms = _longcodeqa_terms(question)
    if not chunks or not terms:
        return _middle_truncate_text(repo_text, max_chars)

    scored: list[tuple[int, int, str]] = []
    for index, chunk in enumerate(chunks):
        lowered = chunk.lower()
        score = sum(lowered.count(term) for term in terms)
        if score:
            scored.append((score, index, chunk))
    if not scored:
        return _middle_truncate_text(repo_text, max_chars)

    selected: list[tuple[int, str]] = []
    used = 0
    separator_chars = len("\n\n[... selected repository excerpt ...]\n")
    for _score, index, chunk in sorted(scored, key=lambda item: (-item[0], item[1])):
        added = len(chunk) + (separator_chars if selected else 0)
        if used + added > max_chars:
            continue
        selected.append((index, chunk))
        used += added
    if not selected:
        return _middle_truncate_text(scored[0][2], max_chars)

    selected.sort(key=lambda item: item[0])
    excerpt = "\n\n[... selected repository excerpt ...]\n".join(chunk for _index, chunk in selected)
    return _middle_truncate_text(excerpt, max_chars)


def _longcodeqa_answer_contract(prompt: str, allowed_letters: Sequence[str]) -> str:
    choices = ", ".join(allowed_letters)
    return (
        f"{prompt.rstrip()}\n\n"
        f"Answer with exactly one option letter from: {choices}. Do not explain.\n"
        "Answer:"
    )


def _extract_longcodeqa_json_answer(text: str, *, allowed_letters: set[str]) -> str:
    stripped = text.strip()
    if not stripped.startswith(("{", "[")):
        return ""
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return ""
    return _extract_longcodeqa_letter_from_value(payload, allowed_letters=allowed_letters)


def _extract_longcodeqa_letter_from_value(value: object, *, allowed_letters: set[str]) -> str:
    if isinstance(value, str):
        candidate = value.strip().upper()
        return candidate if len(candidate) == 1 and candidate in allowed_letters else ""
    if isinstance(value, Mapping):
        for key in ("answer", "choice", "letter", "prediction", "final_answer"):
            if key in value:
                letter = _extract_longcodeqa_letter_from_value(value[key], allowed_letters=allowed_letters)
                if letter:
                    return letter
        arguments = value.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        if isinstance(arguments, Mapping):
            letter = _extract_longcodeqa_letter_from_value(arguments, allowed_letters=allowed_letters)
            if letter:
                return letter
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            letter = _extract_longcodeqa_letter_from_value(item, allowed_letters=allowed_letters)
            if letter:
                return letter
    return ""


def extract_longcodeqa_answer(text: str, *, allowed_letters: Sequence[str]) -> str:
    body = _normalize_newlines(text).strip()
    if body.startswith("Assistant:"):
        body = body[len("Assistant:") :].strip()
    body = _THINK_BLOCK_RE.sub("", body).strip()
    fence_match = _LONGCODEQA_FENCED_BLOCK_RE.match(body)
    if fence_match:
        body = fence_match.group(1).strip()
    allowed = {letter.upper() for letter in allowed_letters}

    json_letter = _extract_longcodeqa_json_answer(body, allowed_letters=allowed)
    if json_letter:
        return json_letter

    lines = [line.strip() for line in body.splitlines() if line.strip()]
    for line in reversed(lines):
        match = _LONGCODEQA_ANSWER_PREFIX_RE.search(line)
        if match and match.group(1).upper() in allowed:
            return match.group(1).upper()
    for line in reversed(lines):
        stripped = line.strip().strip("`").strip()
        match = re.fullmatch(r"\(?([A-Z])\)?[\.\)]?", stripped, flags=re.IGNORECASE)
        if match and match.group(1).upper() in allowed:
            return match.group(1).upper()
    stripped = body.lstrip()
    match = re.match(r"\(?([A-Z])\)?[\.\):,\s]", stripped, flags=re.IGNORECASE)
    if match and match.group(1).upper() in allowed:
        return match.group(1).upper()
    bold_letters = [
        match.group(1).upper()
        for match in _LONGCODEQA_INLINE_BOLD_CHOICE_RE.finditer(body)
        if match.group(1).upper() in allowed
    ]
    if bold_letters and len(set(bold_letters)) == 1:
        return bold_letters[0]
    return ""


class LongCodeQALetterMatch(SampleLevelComputation):
    def compute(self, doc: Doc, model_response, **kwargs) -> float:
        allowed_letters = [choice.strip().upper() for choice in doc.choices]
        gold_letter = allowed_letters[int(doc.gold_index)]
        for prediction in model_response.final_text:
            if extract_longcodeqa_answer(prediction, allowed_letters=allowed_letters) == gold_letter:
                return 1.0
        return 0.0


LONGCODEQA_LETTER_MATCH = SampleLevelMetric(
    metric_name="longcodeqa_accuracy",
    sample_level_fn=LongCodeQALetterMatch(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=_mean,
    higher_is_better=True,
)


def _longbench_task_suffix(dataset: str) -> str:
    return dataset.replace("-", "_")


def _coerce_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items: list[str] = []
        for item in value:
            items.extend(_coerce_string_list(item))
        return items
    stripped = str(value).strip()
    return [stripped] if stripped else []


def _strip_longbench_json_fence(text: str) -> str:
    stripped = str(text or "").strip()
    for _ in range(3):
        before = stripped
        stripped = _LONG_BENCH_ROLE_PREFIX_RE.sub("", stripped, count=1).strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[ \t]*(?:json)?[^\S\r\n]*\r?\n?", "", stripped, count=1, flags=re.IGNORECASE)
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
        if stripped == before:
            break
    return stripped


def _longbench_answer_from_mapping(payload: Mapping[str, object]) -> str | None:
    value = payload.get("answer")
    if value is not None:
        answer = _normalize_newlines(value).strip()
        return answer or None
    arguments = payload.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = None
    if isinstance(arguments, Mapping) and arguments.get("answer") is not None:
        answer = _normalize_newlines(arguments.get("answer")).strip()
        return answer or None
    return None


def normalize_longbench_answer(text: str) -> str:
    body = _THINK_BLOCK_RE.sub("", str(text or "")).strip()
    if not body:
        return ""
    stripped = _strip_longbench_json_fence(body)
    if stripped.lstrip().startswith("{"):
        try:
            payload, _end = json.JSONDecoder().raw_decode(stripped.lstrip())
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, Mapping):
            answer = _longbench_answer_from_mapping(payload)
            if answer:
                return answer
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    for line in reversed(lines):
        match = _LONG_BENCH_ANSWER_PREFIX_RE.match(line)
        if match:
            answer = line[match.end() :].strip().strip("`")
            if answer:
                return answer
    return _LONG_BENCH_ANSWER_PREFIX_RE.sub("", lines[-1]).strip().strip("`") if lines else ""


def _normalize_for_longbench_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _longbench_tokens(text: str) -> list[str]:
    return _LONG_BENCH_WORD_OR_CJK_RE.findall(_normalize_for_longbench_match(text))


def _longbench_f1(prediction: str, reference: str) -> float:
    pred_tokens = _longbench_tokens(prediction)
    ref_tokens = _longbench_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _longbench_best_scores(prediction: str, references: Sequence[str]) -> tuple[float, float]:
    normalized_prediction = normalize_longbench_answer(prediction)
    best_f1 = 0.0
    exact = 0.0
    for reference in references:
        ref = str(reference or "").strip()
        if not ref:
            continue
        if _normalize_for_longbench_match(normalized_prediction) == _normalize_for_longbench_match(ref):
            exact = 1.0
            best_f1 = 1.0
            break
        best_f1 = max(best_f1, _longbench_f1(normalized_prediction, ref))
    return exact, best_f1


class LongBenchExactMatch(SampleLevelComputation):
    def compute(self, doc: Doc, model_response, **kwargs) -> float:
        references = [str(choice) for choice in doc.choices]
        return max((_longbench_best_scores(prediction, references)[0] for prediction in model_response.final_text), default=0.0)


class LongBenchF1(SampleLevelComputation):
    def compute(self, doc: Doc, model_response, **kwargs) -> float:
        references = [str(choice) for choice in doc.choices]
        return max((_longbench_best_scores(prediction, references)[1] for prediction in model_response.final_text), default=0.0)


LONG_BENCH_EXACT_MATCH = SampleLevelMetric(
    metric_name="longbench_exact_match",
    sample_level_fn=LongBenchExactMatch(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=_mean,
    higher_is_better=True,
)


LONG_BENCH_F1 = SampleLevelMetric(
    metric_name="longbench_f1",
    sample_level_fn=LongBenchF1(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=_mean,
    higher_is_better=True,
)


def _normalize_rwkv_text(text: object) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _truncate_text(text: str, max_chars: int) -> str:
    body = str(text or "")
    if max_chars <= 0 or len(body) <= max_chars:
        return body
    marker = "...[truncated]..."
    if max_chars <= len(marker) + 8:
        return body[:max_chars]
    head = max_chars - len(marker)
    return body[:head] + marker


def _coerce_list(raw: object) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    return []


def _read_json_or_jsonl_items(path: Path) -> list[Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        lines = [line for line in raw.splitlines() if line.strip()]
        if not lines:
            return []
        try:
            return [json.loads(line) for line in lines]
        except json.JSONDecodeError:
            raise
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        return [dict(payload)]
    raise ValueError(f"unsupported JSON payload in {path}")


def _bfcl_source_candidates() -> tuple[Path, ...]:
    candidates: list[Path] = []
    for raw in (
        os.environ.get("RWKV_BFCL_SMALL_SOURCE_ROOT"),
        os.environ.get("RWKV_BFCL_V4_SOURCE_ROOT"),
        os.environ.get("BFCL_V4_SOURCE_ROOT"),
    ):
        if raw:
            candidates.append(Path(raw).expanduser())
    candidates.extend(
        [
            Path(__file__).resolve().parents[3]
            / "references"
            / "gorilla"
            / "berkeley-function-call-leaderboard"
            / "bfcl_eval"
            / "data",
            Path("/tmp/rwkv-official-refs/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"),
            Path("/tmp/gorilla-official/berkeley-function-call-leaderboard/bfcl_eval/data"),
            Path("/home/chase/GitHub/rwkv-skills/references/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"),
        ]
    )
    return tuple(dict.fromkeys(path.expanduser().resolve() for path in candidates))


def _bfcl_data_root_from_candidate(candidate: Path) -> Path:
    root = candidate.expanduser().resolve()
    if root.name == "data" and root.parent.name == "bfcl_eval":
        return root
    if (root / "bfcl_eval" / "data").is_dir():
        return root / "bfcl_eval" / "data"
    if (root / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data").is_dir():
        return root / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
    return root


def _bfcl_required_paths(data_root: Path, dataset_name: str) -> tuple[Path, Path]:
    task = BFCL_AST_TASKS.get(dataset_name) or BFCL_EXEC_TASKS[dataset_name]
    return (data_root.joinpath(*task["question"]), data_root.joinpath(*task["answer"]))


def _bfcl_data_root(dataset_name: str) -> Path:
    for candidate in _bfcl_source_candidates():
        data_root = _bfcl_data_root_from_candidate(candidate)
        question_path, answer_path = _bfcl_required_paths(data_root, dataset_name)
        if question_path.exists() and answer_path.exists():
            return data_root.resolve()
    return _download_bfcl_source(dataset_name)


def _download_bfcl_source(dataset_name: str) -> Path:
    target = Path(os.environ.get("RWKV_LIGHTEVAL_BFCL_CACHE", "/tmp/helicopter-bfcl-official")).expanduser().resolve()
    repo_root = target / BFCL_GORILLA_ROOT_NAME
    if not (repo_root / ".git").exists():
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", BFCL_GORILLA_REVISION, BFCL_GORILLA_REPO_URL, str(repo_root)],
            check=True,
        )
    data_root = repo_root / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
    question_path, answer_path = _bfcl_required_paths(data_root, dataset_name)
    if not question_path.exists() or not answer_path.exists():
        raise FileNotFoundError(f"BFCL source is incomplete for {dataset_name}: {question_path}, {answer_path}")
    return data_root.resolve()


def _bfcl_official_root(data_root: Path) -> Path:
    root = data_root.expanduser().resolve()
    if root.name == "data" and root.parent.name == "bfcl_eval":
        return root.parent.parent.resolve()
    if (root / "bfcl_eval" / "eval_checker").is_dir():
        return root.resolve()
    for parent in root.parents:
        if (parent / "bfcl_eval" / "eval_checker").is_dir():
            return parent.resolve()
    return root.parent.parent.resolve()


def _render_bfcl_question(raw: object) -> str:
    if isinstance(raw, str):
        return raw.strip()
    parts: list[str] = []
    for turn in _coerce_list(raw):
        for message in _coerce_list(turn):
            if isinstance(message, Mapping):
                role = str(message.get("role") or "user").strip().lower() or "user"
                content = str(message.get("content") or "").strip()
                if content:
                    parts.append(f"{role.title()}: {content}")
            elif str(message or "").strip():
                parts.append(str(message).strip())
    return "\n".join(parts).strip()


def _normalize_tool_schema(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {"name": "unknown_tool", "description": "", "parameters": {"type": "object", "properties": {}, "required": []}}
    function = raw.get("function") if isinstance(raw.get("function"), Mapping) else None
    source = function or raw
    parameters = source.get("parameters") or {"type": "object", "properties": {}, "required": []}
    if not isinstance(parameters, Mapping):
        parameters = {"type": "object", "properties": {}, "required": []}
    parameters = dict(parameters)
    if str(parameters.get("type") or "").lower() == "dict":
        parameters["type"] = "object"
    parameters.setdefault("properties", {})
    parameters.setdefault("required", [])
    normalized = {
        "name": str(source.get("name") or raw.get("name") or "unknown_tool"),
        "description": str(source.get("description") or raw.get("description") or ""),
        "parameters": parameters,
    }
    metadata = raw.get("metadata") or source.get("metadata")
    if isinstance(metadata, Mapping):
        normalized["metadata"] = dict(metadata)
    return normalized


def _literal_from_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in {"true", "True"}:
            return True
        if node.id in {"false", "False"}:
            return False
        if node.id in {"null", "None"}:
            return None
    if isinstance(node, ast.List):
        return [_literal_from_ast(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return [_literal_from_ast(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal_from_ast(key): _literal_from_ast(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal_from_ast(node.operand)
        return -value if isinstance(value, (int, float)) else value
    if isinstance(node, ast.BinOp):
        left = _literal_from_ast(node.left)
        right = _literal_from_ast(node.right)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
    return ast.literal_eval(node)


def _render_ast_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _render_ast_call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _parse_python_call(text: str) -> tuple[str, dict[str, Any]]:
    parsed = ast.parse(str(text).strip(), mode="eval")
    if not isinstance(parsed.body, ast.Call):
        raise ValueError(f"BFCL ground-truth expression is not a function call: {text}")
    name = _render_ast_call_name(parsed.body.func)
    arguments: dict[str, Any] = {}
    for keyword in parsed.body.keywords:
        if keyword.arg is not None:
            arguments[keyword.arg] = _literal_from_ast(keyword.value)
    return name, arguments


def _normalize_bfcl_ground_truth_calls(raw: object) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in _coerce_list(raw):
        if isinstance(item, str):
            name, arguments = _parse_python_call(item)
            calls.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "argument_options": {key: [value] for key, value in arguments.items()},
                }
            )
            continue
        if isinstance(item, Mapping):
            if "name" in item:
                arguments = item.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                argument_options = {
                    str(key): [value]
                    for key, value in dict(arguments).items()
                } if isinstance(arguments, Mapping) else {}
                calls.append(
                    {
                        "name": str(item.get("name") or ""),
                        "arguments": dict(arguments) if isinstance(arguments, Mapping) else {},
                        "argument_options": argument_options,
                    }
                )
                continue
            if len(item) == 1:
                name, argument_options = next(iter(item.items()))
                if isinstance(argument_options, Mapping):
                    normalized_options = {str(key): _coerce_list(value) or [value] for key, value in argument_options.items()}
                    arguments = {key: _canonical_bfcl_option_value(value) for key, value in normalized_options.items()}
                else:
                    normalized_options = {}
                    arguments = {}
                calls.append({"name": str(name), "arguments": arguments, "argument_options": normalized_options})
    return calls


def _canonical_bfcl_option_value(options: Sequence[Any]) -> Any:
    for option in options:
        if option not in (None, "", {}, []):
            return option
    return options[0] if options else None


def _load_bfcl_ast_rows(dataset_name: str) -> list[dict[str, Any]]:
    if dataset_name not in BFCL_AST_TASKS:
        raise ValueError(f"unknown BFCL AST dataset: {dataset_name}")
    data_root = _bfcl_data_root(dataset_name)
    official_root = _bfcl_official_root(data_root)
    task = BFCL_AST_TASKS[dataset_name]
    question_path, answer_path = _bfcl_required_paths(data_root, dataset_name)
    questions = _read_json_or_jsonl_items(question_path)
    answer_lookup = {
        str(item.get("id") or item.get("task_id") or ""): item
        for item in _read_json_or_jsonl_items(answer_path)
        if isinstance(item, Mapping)
    }
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(questions):
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id") or item.get("task_id") or f"{task['category']}_{index}")
        answer = answer_lookup.get(task_id)
        if answer is None:
            raise ValueError(f"missing BFCL possible-answer entry for {task_id}")
        instruction = _render_bfcl_question(item.get("question"))
        if not instruction:
            raise ValueError(f"BFCL row {task_id!r} is missing question content")
        raw_functions = _coerce_list(item.get("function"))
        raw_ground_truth = _coerce_list(answer.get("ground_truth"))
        rows.append(
            {
                "task_id": task_id,
                "instruction": instruction,
                "tools": [_normalize_tool_schema(tool) for tool in raw_functions],
                "expected_tool_calls": _normalize_bfcl_ground_truth_calls(raw_ground_truth),
                "metadata": {
                    "source_format": "official_bfcl_v4_ast",
                    "category": task["category"],
                    "source_path": str(question_path),
                    "possible_answer_path": str(answer_path),
                    "official_root": str(official_root),
                    "official_source": "gorilla/berkeley-function-call-leaderboard",
                    "bfcl_official_function": raw_functions,
                    "bfcl_official_ground_truth": raw_ground_truth,
                    "bfcl_official_language": str(item.get("language") or item.get("programming_language") or "python").lower(),
                },
            }
        )
    return rows


def _load_bfcl_dataset_dict(dataset_name: str):
    from datasets import Dataset, DatasetDict

    rows: list[dict[str, Any]] = []
    for row in _load_bfcl_ast_rows(dataset_name):
        rendered = dict(row)
        for key in ("tools", "expected_tool_calls", "metadata"):
            rendered[key] = json.dumps(rendered.get(key) or ([] if key != "metadata" else {}), ensure_ascii=False, sort_keys=True)
        rows.append(rendered)
    return DatasetDict({"test": Dataset.from_list(rows)})


def _bfcl_json_list_field(value: object) -> list[Any]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return _coerce_list(value)


def _bfcl_json_mapping_field(value: object) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _render_bfcl_tool_catalog(tools: Sequence[Mapping[str, Any]]) -> str:
    rendered_tools: list[dict[str, Any]] = []
    for tool in tools:
        parameters = tool.get("parameters")
        if not isinstance(parameters, Mapping):
            parameters = {"type": "object", "properties": {}, "required": []}
        raw_properties = parameters.get("properties")
        rendered_arguments: Any = dict(raw_properties) if isinstance(raw_properties, Mapping) else dict(parameters)
        rendered_schema = json.dumps(rendered_arguments, ensure_ascii=False, sort_keys=True)
        if len(rendered_schema) > BFCL_MAX_TOOL_SCHEMA_CHARS:
            rendered_arguments = {
                "_truncated": True,
                "preview": _truncate_text(rendered_schema, BFCL_MAX_TOOL_SCHEMA_CHARS),
            }
        rendered_tools.append(
            {
                "name": str(tool.get("name") or ""),
                "description": _truncate_text(_normalize_rwkv_text(tool.get("description")), BFCL_MAX_TOOL_DESCRIPTION_CHARS),
                "arguments": rendered_arguments,
                **(
                    {"required": list(parameters.get("required") or [])}
                    if isinstance(parameters.get("required"), list) and parameters.get("required")
                    else {}
                ),
            }
        )
    return json.dumps(rendered_tools, ensure_ascii=False, indent=2, sort_keys=False)


def _render_bfcl_output_schema() -> str:
    tool_call_schema = {
        "type": "object",
        "required": ["name", "arguments"],
        "additionalProperties": False,
        "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
    }
    return json.dumps({"oneOf": [tool_call_schema, {"type": "array", "items": tool_call_schema, "minItems": 1}]}, indent=2)


def bfcl_ast_prompt(line: dict, task_name: str | None = None) -> Doc:
    instruction = _normalize_rwkv_text(line.get("instruction"))
    tools = _bfcl_json_list_field(line.get("tools"))
    expected_tool_calls = _bfcl_json_list_field(line.get("expected_tool_calls"))
    metadata = _bfcl_json_mapping_field(line.get("metadata"))
    system_prompt = "\n".join(
        [
            "Tools:",
            _render_bfcl_tool_catalog([tool for tool in tools if isinstance(tool, Mapping)]),
            "Output JSON schema:",
            _render_bfcl_output_schema(),
            "Return exactly one JSON value that validates against the schema.",
            "For one tool call, return one JSON object.",
            "For multiple required tool calls, return a JSON array containing every required call in execution order.",
            "Each arguments object must contain only final argument values for that tool.",
            "Do not copy tool schemas, descriptions, type/items/properties/required/default fields, or wrapper objects into arguments.",
            "Use only listed tool names.",
            "Return no prose, no markdown, and no extra text outside the JSON value.",
        ]
    )
    query = f"System: {system_prompt}\n\nUser: {instruction}\n\nJSON:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[json.dumps(expected_tool_calls, ensure_ascii=False, sort_keys=True)],
        gold_index=0,
        specific={
            "task_id": line.get("task_id"),
            "instruction": instruction,
            "tools": json.dumps(tools, ensure_ascii=False, sort_keys=True),
            "expected_tool_calls": json.dumps(expected_tool_calls, ensure_ascii=False, sort_keys=True),
            "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            "prompt_chars": len(query),
        },
    )


def _strip_json_fence(text: str) -> str:
    normalized = _normalize_rwkv_text(text)
    if normalized.startswith("```"):
        lines = normalized.split("\n")
        if lines and lines[0].strip().lower() in {"```", "```json", "```javascript", "```js"}:
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            normalized = _normalize_rwkv_text("\n".join(lines))
    if normalized.endswith("```"):
        normalized = _normalize_rwkv_text(normalized[: -len("```")])
    return normalized


def _find_leading_json_value_end(text: str) -> int | None:
    if not text or text[0] not in "{[":
        return None
    expected_stack: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            expected_stack.append("}")
            continue
        if char == "[":
            expected_stack.append("]")
            continue
        if char in "}]":
            if not expected_stack or expected_stack[-1] != char:
                return None
            expected_stack.pop()
            if not expected_stack:
                return index + 1
    return None


def _apply_bfcl_json_prefill(response: str) -> str:
    normalized = _normalize_rwkv_text(response)
    stripped = normalized.lstrip()
    if not stripped or stripped.startswith(("{", "[", "```")):
        return normalized
    field_match = re.match(r'^"(?P<field>[A-Za-z_][A-Za-z0-9_]*)"\s*:', stripped)
    if field_match and field_match.group("field") in _BFCL_JSON_PREFILL_FIELDS:
        return "{" + stripped
    return normalized


def _bfcl_functional_call_to_payload(text: str) -> dict[str, Any] | None:
    match = re.match(r"(?s)^(?P<name>[A-Za-z_][\w.:-]*)\s*\((?P<arguments>.*)\)\s*$", _normalize_rwkv_text(text))
    if match is None:
        return None
    try:
        parsed_call = ast.parse(f"_rwkv_tool_call({match.group('arguments')})", mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(parsed_call, ast.Call):
        return None
    arguments: dict[str, Any] = {}
    for keyword in parsed_call.keywords:
        if keyword.arg is None:
            return None
        arguments[str(keyword.arg)] = _literal_from_ast(keyword.value)
    return {"name": match.group("name"), "arguments": arguments}


def _extract_bfcl_json_value_text(response: str) -> str:
    normalized = _normalize_rwkv_text(response)
    if normalized.startswith("Assistant:"):
        normalized = _normalize_rwkv_text(normalized[len("Assistant:") :])
    normalized = _THINK_BLOCK_RE.sub("", normalized).strip()
    normalized = _strip_json_fence(normalized)
    tagged = re.match(r"(?is)^<tool_call>\s*(.*?)\s*</tool_call>\s*$", normalized)
    if tagged is not None:
        normalized = _strip_json_fence(tagged.group(1))
    functional = _bfcl_functional_call_to_payload(normalized)
    if functional is not None:
        return json.dumps(functional, ensure_ascii=False, separators=(",", ":"))
    normalized = _apply_bfcl_json_prefill(normalized)
    if not normalized.startswith(("{", "[")):
        raise ValueError(f"model response must be a JSON function call object or array: {normalized}")
    end = _find_leading_json_value_end(normalized)
    if end is None:
        raise ValueError(f"model response must be a complete JSON function call object or array: {normalized}")
    candidate = normalized[:end]
    trailing = _normalize_rwkv_text(normalized[end:])
    trailing = _strip_json_fence(trailing) if trailing.startswith("```") else trailing
    if trailing and trailing != "```":
        raise ValueError(f"model response has extra text after JSON function call object or array: {trailing}")
    return candidate


def decode_bfcl_tool_call_response(response: str) -> list[dict[str, Any]]:
    candidate = _extract_bfcl_json_value_text(response)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        payload = _literal_from_ast(ast.parse(candidate, mode="eval").body)
    return _coerce_bfcl_function_call_payloads(payload)


def _coerce_bfcl_function_call_payloads(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [_coerce_bfcl_function_call_mapping(item) for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        raise ValueError("tool-call selection payload must be a JSON object")
    if isinstance(payload.get("tool_calls"), list):
        return [_coerce_bfcl_function_call_mapping(item) for item in payload["tool_calls"] if isinstance(item, Mapping)]
    return [_coerce_bfcl_function_call_mapping(payload)]


def _coerce_bfcl_function_call_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    function_payload = payload.get("function")
    if not isinstance(function_payload, Mapping):
        function_payload = payload.get("function_call")
    if isinstance(function_payload, Mapping):
        name = function_payload.get("name") or payload.get("name")
        arguments = function_payload.get("arguments", payload.get("arguments", {}))
    else:
        name = payload.get("name") or payload.get("tool_name") or payload.get("tool")
        arguments = payload.get("arguments", {})
    name_text = str(name or "").strip()
    if not name_text:
        raise ValueError("tool-call selection missing name")
    if arguments is None:
        arguments = {}
    if isinstance(arguments, str):
        raw_arguments = arguments.strip()
        if raw_arguments:
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ValueError("tool-call string arguments must be a JSON object") from exc
        else:
            arguments = {}
    if not isinstance(arguments, Mapping):
        raise ValueError("tool-call arguments must be a JSON object")
    return {"name": name_text, "arguments": dict(arguments)}


@contextlib.contextmanager
def _bfcl_ast_model_config_stub():
    module_name = "bfcl_eval.constants.model_config"
    previous = sys.modules.get(module_name)
    stub = ModuleType(module_name)
    stub.MODEL_CONFIG_MAPPING = defaultdict(lambda: SimpleNamespace(underscore_to_dot=False))
    sys.modules[module_name] = stub
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


@contextlib.contextmanager
def _bfcl_official_import_context(root: str):
    added = False
    if root and root not in sys.path:
        sys.path.insert(0, root)
        added = True
    try:
        yield
    finally:
        if added:
            try:
                sys.path.remove(root)
            except ValueError:
                pass


def _officialize_bfcl_tool_schema(tool: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(tool)
    parameters = payload.get("parameters")
    if isinstance(parameters, Mapping):
        parameters = dict(parameters)
        if str(parameters.get("type") or "").lower() == "object":
            parameters["type"] = "dict"
        payload["parameters"] = parameters
    return payload


def _bfcl_record_function_description(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = _bfcl_json_mapping_field(record.get("metadata"))
    raw = metadata.get("bfcl_official_function")
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    return [_officialize_bfcl_tool_schema(tool) for tool in _bfcl_json_list_field(record.get("tools")) if isinstance(tool, Mapping)]


def _bfcl_record_possible_answer(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = _bfcl_json_mapping_field(record.get("metadata"))
    raw = metadata.get("bfcl_official_ground_truth")
    if isinstance(raw, list):
        answers = [dict(item) for item in raw if isinstance(item, Mapping)]
        if answers:
            return answers
    possible: list[dict[str, Any]] = []
    for item in _bfcl_json_list_field(record.get("expected_tool_calls")):
        if not isinstance(item, Mapping):
            continue
        options = item.get("argument_options")
        if isinstance(options, Mapping):
            possible.append({str(item.get("name") or ""): {str(key): list(_coerce_list(value) or [value]) for key, value in options.items()}})
        else:
            possible.append({str(item.get("name") or ""): {key: [value] for key, value in dict(item.get("arguments") or {}).items()}})
    return possible


def _bfcl_record_language(record: Mapping[str, Any]):
    metadata = _bfcl_json_mapping_field(record.get("metadata"))
    return str(metadata.get("bfcl_official_language") or "python").strip().upper()


def _bfcl_record_category(record: Mapping[str, Any]) -> str:
    metadata = _bfcl_json_mapping_field(record.get("metadata"))
    category = str(metadata.get("category") or "").strip()
    return category or str(record.get("task_id") or "").rsplit("_", 1)[0]


def _bfcl_model_output(decoded_calls: Sequence[Mapping[str, Any]]) -> list[dict[str, dict[str, Any]]]:
    result: list[dict[str, dict[str, Any]]] = []
    for call in decoded_calls:
        name = str(call.get("name") or call.get("tool_name") or "").strip()
        arguments = call.get("arguments")
        result.append({name: dict(arguments) if isinstance(arguments, Mapping) else {}})
    return result


def run_bfcl_official_ast_checker(record: Mapping[str, Any], decoded_calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metadata = _bfcl_json_mapping_field(record.get("metadata"))
    root = str(metadata.get("official_root") or "").strip()
    if not root:
        source_path = str(metadata.get("source_path") or "")
        root = str(_bfcl_official_root(Path(source_path).parent)) if source_path else ""
    with _bfcl_official_import_context(root), _bfcl_ast_model_config_stub():
        from bfcl_eval.constants.enums import Language
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

        language = getattr(Language, _bfcl_record_language(record), Language.PYTHON)
        result = ast_checker(
            _bfcl_record_function_description(record),
            _bfcl_model_output(decoded_calls),
            _bfcl_record_possible_answer(record),
            language,
            _bfcl_record_category(record),
            BFCL_DEFAULT_MODEL_NAME,
        )
    return dict(result) if isinstance(result, Mapping) else {"valid": False, "error": [str(result)]}


class BfclAstAccuracy(SampleLevelComputation):
    def compute(self, doc: Doc, model_response, **kwargs) -> float:
        record = dict(doc.specific or {})
        for prediction in model_response.final_text:
            try:
                decoded_calls = decode_bfcl_tool_call_response(prediction)
                checker_result = run_bfcl_official_ast_checker(record, decoded_calls)
            except Exception:
                continue
            if bool(checker_result.get("valid")):
                return 1.0
        return 0.0


BFCL_AST_ACCURACY = SampleLevelMetric(
    metric_name="bfcl_ast_accuracy",
    sample_level_fn=BfclAstAccuracy(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=_mean,
    higher_is_better=True,
)


class BfclExecCallResult:
    __slots__ = ("call", "success", "result", "error")

    def __init__(self, call: str, success: bool, result: Any = None, error: str | None = None) -> None:
        self.call = call
        self.success = success
        self.result = result
        self.error = error


class BfclExecEvaluation:
    __slots__ = ("reward", "is_passed", "fail_reason", "details")

    def __init__(self, reward: float, is_passed: bool, fail_reason: str, details: dict[str, Any]) -> None:
        self.reward = reward
        self.is_passed = is_passed
        self.fail_reason = fail_reason
        self.details = details


def _ground_truth_to_exec_calls(raw: Any) -> list[str]:
    calls: list[str] = []
    for item in _coerce_list(raw):
        if isinstance(item, str):
            text = item.strip()
            if text:
                calls.append(text)
            continue
        if not isinstance(item, Mapping) or len(item) != 1:
            continue
        name, raw_arguments = next(iter(item.items()))
        arguments: dict[str, Any] = {}
        if isinstance(raw_arguments, Mapping):
            for key, value in raw_arguments.items():
                options = _coerce_list(value)
                arguments[str(key)] = options[0] if options else value
        calls.append(render_bfcl_exec_call({"name": str(name), "arguments": arguments}))
    return calls


def _expected_tool_calls_from_exec(calls: Sequence[str]) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    for call in calls:
        try:
            name, args, kwargs = _parse_exec_call(call)
        except Exception:
            continue
        arguments = dict(kwargs)
        if args:
            arguments["_args"] = list(args)
        expected.append(
            {
                "name": name,
                "arguments": arguments,
                "argument_options": {key: [value] for key, value in arguments.items()},
            }
        )
    return expected


def _load_bfcl_exec_rows(dataset_name: str) -> list[dict[str, Any]]:
    if dataset_name not in BFCL_EXEC_TASKS:
        raise ValueError(f"unknown BFCL exec dataset: {dataset_name}")
    data_root = _bfcl_data_root(dataset_name)
    official_root = _bfcl_official_root(data_root)
    task = BFCL_EXEC_TASKS[dataset_name]
    question_path, answer_path = _bfcl_required_paths(data_root, dataset_name)
    questions = _read_json_or_jsonl_items(question_path)
    answer_lookup = {
        str(item.get("id") or item.get("task_id") or ""): item
        for item in _read_json_or_jsonl_items(answer_path)
        if isinstance(item, Mapping)
    }
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(questions):
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id") or item.get("task_id") or f"{task['category']}_{index}")
        answer = answer_lookup.get(task_id)
        if answer is None:
            raise ValueError(f"missing BFCL executable possible-answer entry for {task_id}")
        instruction = _render_bfcl_question(item.get("question"))
        if not instruction:
            raise ValueError(f"BFCL executable row {task_id!r} is missing question content")
        expected_calls = _ground_truth_to_exec_calls(answer.get("ground_truth"))
        execution_types = [str(value).strip() for value in _coerce_list(answer.get("execution_result_type"))]
        execution_types = [value or "exact_match" for value in execution_types]
        rows.append(
            {
                "task_id": task_id,
                "instruction": instruction,
                "tools": [_normalize_tool_schema(tool) for tool in _coerce_list(item.get("function"))],
                "expected_tool_calls": _expected_tool_calls_from_exec(expected_calls),
                "expected_executable_calls": [call for call in expected_calls if call],
                "execution_result_type": execution_types,
                "metadata": {
                    "source_format": "official_bfcl_v4_exec",
                    "category": task["category"],
                    "source_path": str(question_path),
                    "possible_answer_path": str(answer_path),
                    "official_root": str(official_root),
                    "official_source": "gorilla/berkeley-function-call-leaderboard",
                    "execution_result_type": execution_types,
                    "bfcl_official_function": _coerce_list(item.get("function")),
                    "bfcl_official_ground_truth": _coerce_list(answer.get("ground_truth")),
                },
            }
        )
    return rows


def _load_bfcl_exec_dataset_dict(dataset_name: str):
    from datasets import Dataset, DatasetDict

    rows: list[dict[str, Any]] = []
    for row in _load_bfcl_exec_rows(dataset_name):
        rendered = dict(row)
        for key in ("tools", "expected_tool_calls", "expected_executable_calls", "execution_result_type", "metadata"):
            rendered[key] = json.dumps(rendered.get(key) or ([] if key != "metadata" else {}), ensure_ascii=False, sort_keys=True)
        rows.append(rendered)
    return DatasetDict({"test": Dataset.from_list(rows)})


def _bfcl_exec_json_list_field(value: object) -> list[Any]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return _coerce_list(value)


def _bfcl_exec_is_parallel(record: Mapping[str, Any]) -> bool:
    metadata = _bfcl_json_mapping_field(record.get("metadata"))
    category = str(metadata.get("category") or record.get("task_id") or "").lower()
    return "parallel" in category


def _force_bfcl_exec_array_prefix(record: Mapping[str, Any]) -> bool:
    return _bfcl_exec_is_parallel(record) and len(_bfcl_exec_json_list_field(record.get("expected_executable_calls"))) > 1


def _complete_bfcl_exec_forced_prefix(record: Mapping[str, Any], completion: str) -> str:
    if not _force_bfcl_exec_array_prefix(record):
        return completion
    stripped = completion.lstrip()
    if stripped.startswith("["):
        return completion
    if stripped.startswith("{") and not completion.rstrip().endswith("]"):
        return "[\n" + completion.rstrip() + "\n]"
    return "[\n" + completion


def bfcl_exec_prompt(line: dict, task_name: str | None = None) -> Doc:
    instruction = _normalize_rwkv_text(line.get("instruction"))
    tools = _bfcl_json_list_field(line.get("tools"))
    expected_tool_calls = _bfcl_json_list_field(line.get("expected_tool_calls"))
    expected_executable_calls = _bfcl_exec_json_list_field(line.get("expected_executable_calls"))
    execution_result_type = _bfcl_exec_json_list_field(line.get("execution_result_type"))
    metadata = _bfcl_json_mapping_field(line.get("metadata"))
    system_prompt = "\n".join(
        [
            "Tools:",
            _render_bfcl_tool_catalog([tool for tool in tools if isinstance(tool, Mapping)]),
            "Output JSON schema:",
            _render_bfcl_output_schema(),
            "Return exactly one JSON value that validates against the schema.",
            "For one tool call, return one JSON object.",
            "For multiple required tool calls, return a JSON array containing every required call in execution order; do not stop after the first call.",
            "Each arguments object must contain only final argument values for that tool.",
            "The evaluator will execute the selected tool call arguments and compare execution results.",
            "Do not copy tool schemas, descriptions, type/items/properties/required/default fields, or wrapper objects into arguments.",
            "Use only listed tool names.",
            "Return no prose, no markdown, and no extra text outside the JSON value.",
        ]
    )
    query = f"System: {system_prompt}\n\nUser: {instruction}\n\nJSON:"
    if _force_bfcl_exec_array_prefix(
        {
            "task_id": line.get("task_id"),
            "metadata": metadata,
            "expected_executable_calls": expected_executable_calls,
        }
    ):
        query += "\n["
    return Doc(
        task_name=task_name,
        query=query,
        choices=[json.dumps(expected_tool_calls, ensure_ascii=False, sort_keys=True)],
        gold_index=0,
        specific={
            "task_id": line.get("task_id"),
            "instruction": instruction,
            "tools": json.dumps(tools, ensure_ascii=False, sort_keys=True),
            "expected_tool_calls": json.dumps(expected_tool_calls, ensure_ascii=False, sort_keys=True),
            "expected_executable_calls": json.dumps(expected_executable_calls, ensure_ascii=False, sort_keys=True),
            "execution_result_type": json.dumps(execution_result_type, ensure_ascii=False, sort_keys=True),
            "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            "prompt_chars": len(query),
        },
    )


def render_bfcl_exec_call(call: Mapping[str, Any]) -> str:
    name = str(call.get("name") or call.get("tool_name") or "").strip()
    arguments = call.get("arguments")
    if not isinstance(arguments, Mapping):
        arguments = {}
    rendered = ", ".join(f"{key}={_python_literal(value)}" for key, value in arguments.items())
    return f"{name}({rendered})" if rendered else f"{name}()"


def evaluate_bfcl_exec_calls(record: Mapping[str, Any], decoded_calls: Sequence[Mapping[str, Any]], *, parse_error: str | None = None) -> BfclExecEvaluation:
    sandbox = BfclExecSandbox()
    expected_calls = [str(call) for call in _bfcl_exec_json_list_field(record.get("expected_executable_calls"))]
    model_calls = [render_bfcl_exec_call(call) for call in decoded_calls]
    match_types = [str(value) for value in _bfcl_exec_json_list_field(record.get("execution_result_type"))] or ["exact_match"] * len(expected_calls)
    while len(match_types) < len(expected_calls):
        match_types.append("exact_match")

    details: dict[str, Any] = {
        "expected_executable_calls": expected_calls,
        "decoded_executable_calls": model_calls,
        "execution_result_type": match_types,
        "tool_count_ok": len(model_calls) == len(expected_calls),
        "call_matches": [],
        "parse_error": parse_error or "",
    }
    if parse_error:
        return BfclExecEvaluation(0.0, False, parse_error, details)

    expected_results = [sandbox.execute(call) for call in expected_calls]
    model_results = [sandbox.execute(call) for call in model_calls]
    details["expected_execution_results"] = [_bfcl_exec_result_payload(item) for item in expected_results]
    details["decoded_execution_results"] = [_bfcl_exec_result_payload(item) for item in model_results]

    failure_bits: list[str] = []
    passed_count = 0
    if _bfcl_exec_is_parallel(record):
        matched_model_indices: set[int] = set()
        for expected_index, expected in enumerate(expected_results):
            match_type = match_types[expected_index] if expected_index < len(match_types) else "exact_match"
            candidate_reasons: list[str] = []
            for model_index, actual in enumerate(model_results):
                if model_index in matched_model_indices:
                    continue
                ok, reason = _execution_result_matches(actual, expected, match_type)
                if ok:
                    matched_model_indices.add(model_index)
                    details["call_matches"].append(
                        {
                            "expected_index": expected_index,
                            "decoded_index": model_index,
                            "ok": True,
                            "reason": reason,
                            "match_type": match_type,
                        }
                    )
                    passed_count += 1
                    break
                candidate_reasons.append(f"decoded_{model_index}:{reason}")
            else:
                reason = candidate_reasons[0].split(":", 1)[1] if candidate_reasons else "missing_call"
                details["call_matches"].append(
                    {
                        "expected_index": expected_index,
                        "decoded_index": None,
                        "ok": False,
                        "reason": reason,
                        "match_type": match_type,
                        "candidate_reasons": candidate_reasons,
                    }
                )
                failure_bits.append(f"call_{expected_index}:{reason}")
        for model_index in range(len(model_results)):
            if model_index in matched_model_indices:
                continue
            details["call_matches"].append({"expected_index": None, "decoded_index": model_index, "ok": False, "reason": "unexpected_extra_call"})
            failure_bits.append(f"call_{model_index}:unexpected_extra_call")
        denominator = max(1, len(expected_results))
        reward = passed_count / denominator
        is_passed = len(model_results) == len(expected_results) and passed_count == len(expected_results)
        return BfclExecEvaluation(float(reward), bool(is_passed), "; ".join(failure_bits), details)

    max_len = max(len(expected_results), len(model_results))
    for index in range(max_len):
        if index >= len(expected_results):
            details["call_matches"].append({"index": index, "ok": False, "reason": "unexpected_extra_call"})
            failure_bits.append(f"call_{index}:unexpected_extra_call")
            continue
        if index >= len(model_results):
            details["call_matches"].append({"index": index, "ok": False, "reason": "missing_call"})
            failure_bits.append(f"call_{index}:missing_call")
            continue
        match_type = match_types[index] if index < len(match_types) else "exact_match"
        ok, reason = _execution_result_matches(model_results[index], expected_results[index], match_type)
        details["call_matches"].append({"index": index, "ok": ok, "reason": reason, "match_type": match_type})
        if ok:
            passed_count += 1
        else:
            failure_bits.append(f"call_{index}:{reason}")
    denominator = max(1, len(expected_results))
    reward = passed_count / denominator
    is_passed = len(model_results) == len(expected_results) and passed_count == len(expected_results)
    return BfclExecEvaluation(float(reward), bool(is_passed), "; ".join(failure_bits), details)


class BfclExecSandbox:
    def __init__(self) -> None:
        self._functions: dict[str, Callable[..., Any]] = {
            "add_binary_numbers": lambda a, b: bin(int(str(a), 2) + int(str(b), 2))[2:],
            "adjust_for_inflation": lambda amount, inflation_rate, years: amount * ((1 + inflation_rate) ** years),
            "apply_discount": lambda price, discount: price * (1 - discount),
            "book_room": lambda **kwargs: {"booking_confirmed": True, **kwargs},
            "calc_binomial_probability": self._calc_binomial_probability,
            "calculate_basal_metabolic_rate": self._calculate_basal_metabolic_rate,
            "calculate_cosine_similarity": self._calculate_cosine_similarity,
            "calculate_daily_energy_expenditure": lambda bmr, activity_factor: bmr * activity_factor,
            "calculate_density": lambda mass, volume: mass / volume,
            "calculate_displacement": lambda initial_velocity, acceleration, time: initial_velocity * time + 0.5 * acceleration * time**2,
            "calculate_electrostatic_potential_energy": lambda charge, voltage: charge * voltage,
            "calculate_final_velocity": lambda initial_velocity, acceleration, time: initial_velocity + acceleration * time,
            "calculate_future_value": lambda present_value, interest_rate, periods: present_value * ((1 + interest_rate) ** periods),
            "calculate_intercept": lambda x1, y1, slope: y1 - slope * x1,
            "calculate_interest_rate": lambda principal, amount, time: (amount / principal) ** (1 / time) - 1,
            "calculate_investment_value": self._calculate_investment_value,
            "calculate_mean": lambda numbers: sum(numbers) / len(numbers),
            "calculate_nutritional_needs": self._calculate_nutritional_needs,
            "calculate_permutations": lambda n, k: math.factorial(int(n)) // math.factorial(int(n) - int(k)),
            "calculate_slope": lambda x1, y1, x2, y2: (y2 - y1) / (x2 - x1),
            "calculate_standard_deviation": self._calculate_standard_deviation,
            "calculate_total": lambda price, quantity: price * quantity,
            "calculate_total_price": lambda price, quantity: price * quantity,
            "calculate_triangle_area": lambda base, height: 0.5 * base * height,
            "compound_interest": lambda principal, rate, time, n=1: principal * ((1 + rate / n) ** (n * time)),
            "confirm_booking": lambda **kwargs: {"confirmed": True, **kwargs},
            "convert_binary_to_decimal": lambda binary: int(str(binary), 2),
            "convert_coordinates": lambda latitude, longitude: {"latitude": latitude, "longitude": longitude},
            "convert_decimal_to_hex": lambda decimal: hex(int(decimal)),
            "convert_temperature": self._convert_temperature,
            "estimate_derivative": self._estimate_derivative,
            "generate_random_number": lambda min_value=0, max_value=100: int((int(min_value) + int(max_value)) / 2),
            "geometry_area_circle": lambda radius: math.pi * radius * radius,
            "get_distance": lambda pointA, pointB: math.dist(pointA, pointB),
            "get_fibonacci_number": self._get_fibonacci_number,
            "get_fibonacci_sequence": self._get_fibonacci_sequence,
            "get_prime_factors": self._get_prime_factors,
            "inflation_adjustment": lambda amount, inflation_rate, years: amount * ((1 + inflation_rate) ** years),
            "linear_regression": self._linear_regression,
            "mat_mul": self._mat_mul,
            "math_factorial": lambda n: math.factorial(int(n)),
            "math_gcd": lambda a, b: math.gcd(int(a), int(b)),
            "math_lcm": lambda a, b: abs(int(a) * int(b)) // math.gcd(int(a), int(b)),
            "maxPoints": self._max_points,
            "mortgage_calculator": self._mortgage_calculator,
            "order_food": self._order_food,
            "polygon_area": self._polygon_area,
            "predict_value": lambda slope, intercept, x: slope * x + intercept,
            "quadratic_roots": self._quadratic_roots,
            "sort_array": lambda array, reverse=False: sorted(array, reverse=bool(reverse)),
            "validate_polygon": lambda vertices: len(vertices) >= 3,
        }

    def execute(self, call: str) -> BfclExecCallResult:
        try:
            name, args, kwargs = _parse_exec_call(call)
            func = self._functions.get(name)
            result = func(*args, **kwargs) if func is not None else self._execute_fixture(name, args, kwargs)
            return BfclExecCallResult(call=call, success=True, result=_json_safe(result))
        except Exception as exc:  # noqa: BLE001
            return BfclExecCallResult(call=call, success=False, error=str(exc))

    def _execute_fixture(self, name: str, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
        if name in {"get_stock_price_by_stock_name", "get_company_name_by_stock_name", "get_stock_history"}:
            stock = str(kwargs.get("stock_name") or (args[0] if args else "")).upper()
            if name == "get_company_name_by_stock_name":
                return {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "GOOG": "Alphabet Inc."}.get(stock, stock)
            if name == "get_stock_history":
                return {"symbol": stock, "history": [{"close": self._stock_price(stock)}]}
            return self._stock_price(stock)
        if name == "get_weather_data":
            coordinates = kwargs.get("coordinates") or (args[0] if args else [0, 0])
            if isinstance(coordinates, Mapping):
                latitude = coordinates.get("latitude", coordinates.get("lat", 0))
                longitude = coordinates.get("longitude", coordinates.get("lon", coordinates.get("long", 0)))
            else:
                latitude = coordinates[0] if len(coordinates) > 0 else 0
                longitude = coordinates[1] if len(coordinates) > 1 else 0
            return {"temperature": round(float(latitude) * 0.1 + float(longitude) * 0.01, 3), "unit": "celsius"}
        if name in {"get_coordinate_by_ip_address", "get_zipcode_by_ip_address"}:
            ip_address = str(kwargs.get("ip_address") or (args[0] if args else ""))
            return "private range" if ip_address.startswith("192.168.") else {"ip_address": ip_address}
        if name in {"get_coordinates_from_city", "retrieve_city_based_on_zipcode", "get_time_zone_by_coord"}:
            return self._location_fixture(name, args, kwargs)
        if name in {"get_covid_death_by_country", "get_active_covid_case_by_country"}:
            country = str(kwargs.get("country") or (args[0] if args else "")).lower()
            base = sum(ord(char) for char in country)
            return base * (1000 if "death" in name else 500)
        if name in {"get_rating_by_amazon_ASIN", "get_price_by_amazon_ASIN", "get_product_name_by_amazon_ASIN"}:
            asin = str(kwargs.get("ASIN") or kwargs.get("asin") or (args[0] if args else ""))
            if "rating" in name:
                return str(round(3.5 + (sum(ord(c) for c in asin) % 15) / 10, 1))
            if "price" in name:
                return f"${50 + (sum(ord(c) for c in asin) % 500)}.00"
            return f"Product {asin}"
        if name in {"get_movie_director", "get_director_by_movie_name", "get_movie_rating", "get_movie_genre"}:
            movie = str(kwargs.get("movie_name") or (args[0] if args else "")).lower()
            movies = {
                "avatar": {"director": "James Cameron", "rating": "PG-13", "genre": "Science Fiction"},
                "pulp fiction": {"director": "Quentin Tarantino", "rating": "R", "genre": "Crime"},
            }
            info = movies.get(movie, {"director": "Unknown", "rating": "Unknown", "genre": "Unknown"})
            if "rating" in name:
                return info["rating"]
            if "genre" in name:
                return info["genre"]
            return info["director"]
        if name == "retrieve_holiday_by_year":
            country = kwargs.get("country") or (args[0] if args else "")
            year = int(kwargs.get("year") or (args[1] if len(args) > 1 else 2023))
            return [{"countryCode": str(country), "date": f"{year:04d}-01-01", "localName": "New Year", "name": "New Year's Day"}]
        if name == "find_term_on_urban_dictionary":
            term = str(kwargs.get("term") or (args[0] if args else ""))
            return {"term": term, "definition": f"Definition for {term}"}
        if name == "convert_currency":
            amount = float(kwargs.get("amount") or (args[0] if args else 0.0))
            from_currency = str(kwargs.get("from_currency") or (args[1] if len(args) > 1 else "")).upper()
            to_currency = str(kwargs.get("to_currency") or (args[2] if len(args) > 2 else "")).upper()
            rates = {("USD", "EUR"): 0.92, ("EUR", "USD"): 1.08, ("USD", "GBP"): 0.79, ("GBP", "USD"): 1.27}
            return amount * rates.get((from_currency, to_currency), 1.0)
        raise ValueError(f"unsupported official BFCL executable function: {name}")

    @staticmethod
    def _stock_price(stock: str) -> float:
        return {"AAPL": 169.02, "MSFT": 421.9, "GOOG": 175.4, "META": 477.2, "NFLX": 610.1, "BABA": 75.0}.get(stock, 100.0)

    @staticmethod
    def _location_fixture(name: str, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
        if name == "retrieve_city_based_on_zipcode":
            return {"90210": "BEVERLY HILLS", "10001": "NEW YORK", "08540": "PRINCETON"}.get(str(kwargs.get("zipcode") or ""), "UNKNOWN")
        if name == "get_coordinates_from_city":
            city = str(kwargs.get("city_name") or (args[0] if args else ""))
            return {"city": city, "latitude": "0.0", "longitude": "0.0"}
        return "UTC"

    @staticmethod
    def _calc_binomial_probability(n: int, k: int, p: float) -> float:
        return math.comb(int(n), int(k)) * (float(p) ** int(k)) * ((1 - float(p)) ** (int(n) - int(k)))

    @staticmethod
    def _calculate_cosine_similarity(vectorA: Sequence[float], vectorB: Sequence[float]) -> float:
        dot = sum(float(a) * float(b) for a, b in zip(vectorA, vectorB))
        norm_a = math.sqrt(sum(float(a) ** 2 for a in vectorA))
        norm_b = math.sqrt(sum(float(b) ** 2 for b in vectorB))
        return dot / (norm_a * norm_b)

    @staticmethod
    def _calculate_standard_deviation(numbers: Sequence[float]) -> float:
        mean = sum(numbers) / len(numbers)
        return math.sqrt(sum((float(item) - mean) ** 2 for item in numbers) / len(numbers))

    @staticmethod
    def _calculate_basal_metabolic_rate(weight: float, height: float, age: int, gender: str) -> float:
        offset = 5 if str(gender).lower() == "male" else -161
        return 10 * weight + 6.25 * height - 5 * age + offset

    def _calculate_nutritional_needs(self, weight: float, height: float, age: int, gender: str, activity_level: float, goal: str) -> dict[str, float]:
        bmr = self._calculate_basal_metabolic_rate(weight, height, age, gender)
        calories = bmr * float(activity_level)
        if str(goal).lower().startswith("lose"):
            calories -= 500
        elif str(goal).lower().startswith("gain"):
            calories += 500
        return {"calories": calories}

    @staticmethod
    def _calculate_investment_value(initial_investment: float, annual_contribution: float, years: int, annual_return: float, inflation_rate: Sequence[float] | float = (), adjust_for_inflation: bool = True) -> float:
        value = float(initial_investment)
        if isinstance(inflation_rate, (int, float)):
            rates: Sequence[float] = [float(inflation_rate)]
        elif inflation_rate is None:
            rates = []
        else:
            rates = inflation_rate
        for index in range(int(years)):
            inflation = float(rates[index]) if adjust_for_inflation and index < len(rates) else 0.0
            value = (value + float(annual_contribution)) * (1 + float(annual_return) - inflation)
        return value

    @staticmethod
    def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
        source = str(from_unit).lower()
        target = str(to_unit).lower()
        celsius = (value - 32) * 5 / 9 if source.startswith("f") else value
        return celsius * 9 / 5 + 32 if target.startswith("f") else celsius

    @staticmethod
    def _estimate_derivative(function: str, x: float) -> float:
        source = str(function).strip()
        if not source.startswith("lambda"):
            source = f"lambda x: {source}"
        fn = eval(source, {"__builtins__": {}, "math": math}, {})  # noqa: S307 - restricted BFCL math-expression sandbox.
        h = 1e-5
        return (fn(x + h) - fn(x - h)) / (2 * h)

    @staticmethod
    def _get_fibonacci_number(n: int) -> int:
        seq = BfclExecSandbox._get_fibonacci_sequence(int(n) + 1)
        return seq[-1] if seq else 0

    @staticmethod
    def _get_fibonacci_sequence(n: int) -> list[int]:
        values = [0, 1]
        while len(values) < int(n):
            values.append(values[-1] + values[-2])
        return values[: int(n)]

    @staticmethod
    def _get_prime_factors(n: int | None = None, *, number: int | None = None, formatted: bool | str | None = None) -> list[int] | str:
        factors: list[int] = []
        value = int(number if number is not None else n)
        divisor = 2
        while divisor * divisor <= value:
            while value % divisor == 0:
                factors.append(divisor)
                value //= divisor
            divisor += 1
        if value > 1:
            factors.append(value)
        if _truthy_exec_flag(formatted):
            return " * ".join(str(item) for item in factors)
        return factors

    @staticmethod
    def _order_food(item: Sequence[str] | None = None, quantity: Sequence[float] | None = None, price: Sequence[float] | None = None, *, items: Sequence[str] | None = None, **_kwargs: Any) -> float:
        selected_items = item if item is not None else items
        quantities = quantity or [1 for _ in (selected_items or [])]
        prices = price or [0 for _ in (selected_items or [])]
        return sum(float(q) * float(p) for q, p in zip(quantities, prices))

    @staticmethod
    def _linear_regression(x: Sequence[float], y: Sequence[float], point: float) -> float:
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        denom = sum((item - mean_x) ** 2 for item in x)
        slope = 0.0 if denom == 0 else sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / denom
        return slope * point + (mean_y - slope * mean_x)

    @staticmethod
    def _mat_mul(matA: Sequence[Sequence[float]], matB: Sequence[Sequence[float]]) -> list[list[float]]:
        columns = list(zip(*matB))
        return [[sum(a * b for a, b in zip(row, col)) for col in columns] for row in matA]

    @staticmethod
    def _max_points(points: Sequence[Sequence[float]]) -> int:
        if len(points) <= 2:
            return len(points)
        best = 2
        for i, a in enumerate(points):
            slopes: dict[tuple[float, float], int] = {}
            for b in points[i + 1 :]:
                dx = float(b[0]) - float(a[0])
                dy = float(b[1]) - float(a[1])
                key = (0.0, 1.0) if dx == 0 else (1.0, round(dy / dx, 12))
                slopes[key] = slopes.get(key, 1) + 1
                best = max(best, slopes[key])
        return best

    @staticmethod
    def _mortgage_calculator(loan_amount: float, interest_rate: float, loan_period: int) -> float:
        monthly_rate = float(interest_rate) / 12
        payments = int(loan_period) * 12
        if monthly_rate == 0:
            return loan_amount / payments
        return loan_amount * monthly_rate * ((1 + monthly_rate) ** payments) / (((1 + monthly_rate) ** payments) - 1)

    @staticmethod
    def _polygon_area(vertices: Sequence[Sequence[float]]) -> float:
        area = 0.0
        for index, point in enumerate(vertices):
            next_point = vertices[(index + 1) % len(vertices)]
            area += float(point[0]) * float(next_point[1]) - float(next_point[0]) * float(point[1])
        return abs(area) / 2

    @staticmethod
    def _quadratic_roots(a: float, b: float, c: float) -> list[float | str]:
        disc = b * b - 4 * a * c
        if disc < 0:
            return ["complex"]
        root = math.sqrt(disc)
        return [(-b + root) / (2 * a), (-b - root) / (2 * a)]


def _execution_result_matches(actual: BfclExecCallResult, expected: BfclExecCallResult, match_type: str) -> tuple[bool, str]:
    if not expected.success:
        return False, f"expected_execution_error({expected.error})"
    if not actual.success:
        return False, f"decoded_execution_error({actual.error})"
    normalized = str(match_type or "exact_match").strip().lower()
    if normalized == "structural_match":
        return (True, "ok") if _same_structure(actual.result, expected.result) else (False, "structure_mismatch")
    if normalized == "real_time_match":
        return (True, "ok") if _real_time_value_matches(actual.result, expected.result) else (False, "real_time_mismatch")
    return (True, "ok") if _value_matches(actual.result, expected.result) else (False, "exact_mismatch")


def _same_structure(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        return all(key in actual and _same_structure(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        if not expected or not actual:
            return True
        return _same_structure(actual[0], expected[0])
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return True
    return type(actual) is type(expected) or isinstance(actual, type(expected))


def _real_time_value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if _both_plain_ints(actual, expected):
            return actual == expected
        try:
            actual_float = float(actual)
            expected_float = float(expected)
        except (OverflowError, ValueError):
            return actual == expected
        if not math.isfinite(actual_float) or not math.isfinite(expected_float):
            return actual == expected
        baseline = max(abs(expected_float), 1.0)
        return abs(actual_float - expected_float) / baseline <= 0.20
    return _value_matches(actual, expected) or _same_structure(actual, expected)


def _value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if _both_plain_ints(actual, expected):
            return actual == expected
        try:
            actual_float = float(actual)
            expected_float = float(expected)
        except (OverflowError, ValueError):
            return actual == expected
        if not math.isfinite(actual_float) or not math.isfinite(expected_float):
            return actual == expected
        return math.isclose(actual_float, expected_float, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(actual, str) and isinstance(expected, str):
        return _normalize_rwkv_text(actual).strip() == _normalize_rwkv_text(expected).strip()
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return dict(actual) == dict(expected)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(_value_matches(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _both_plain_ints(actual: Any, expected: Any) -> bool:
    return isinstance(actual, int) and not isinstance(actual, bool) and isinstance(expected, int) and not isinstance(expected, bool)


def _truthy_exec_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "null"}
    return bool(value)


def _parse_exec_call(text: str) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
    parsed = ast.parse(str(text).strip(), mode="eval")
    if not isinstance(parsed.body, ast.Call):
        raise ValueError(f"not a function call: {text}")
    name = _render_ast_call_name(parsed.body.func)
    args = tuple(_literal_from_ast(item) for item in parsed.body.args)
    kwargs = {keyword.arg: _literal_from_ast(keyword.value) for keyword in parsed.body.keywords if keyword.arg is not None}
    return name, args, kwargs


def _python_literal(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (bool, int, float)) or value is None:
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_python_literal(item) for item in value) + "]"
    if isinstance(value, tuple):
        rendered = ", ".join(_python_literal(item) for item in value)
        return f"({rendered}{',' if len(value) == 1 else ''})"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{_python_literal(key)}: {_python_literal(item)}" for key, item in value.items()) + "}"
    return repr(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _bfcl_exec_result_payload(result: BfclExecCallResult) -> dict[str, Any]:
    return {"call": result.call, "success": result.success, "result": result.result, "error": result.error or ""}


class BfclExecAccuracy(SampleLevelComputation):
    def compute(self, doc: Doc, model_response, **kwargs) -> float:
        record = dict(doc.specific or {})
        for prediction in model_response.final_text:
            completion = _complete_bfcl_exec_forced_prefix(record, prediction)
            try:
                decoded_calls = decode_bfcl_tool_call_response(completion)
                evaluation = evaluate_bfcl_exec_calls(record, decoded_calls)
            except Exception:
                continue
            if evaluation.reward > 0:
                return float(evaluation.reward)
        return 0.0


BFCL_EXEC_ACCURACY = SampleLevelMetric(
    metric_name="bfcl_exec_accuracy",
    sample_level_fn=BfclExecAccuracy(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=_mean,
    higher_is_better=True,
)


def _apibank_source_candidates() -> tuple[Path, ...]:
    candidates: list[Path] = []
    for raw in (os.environ.get("API_BANK_SOURCE_ROOT"), os.environ.get("RWKV_API_BANK_SOURCE_ROOT")):
        if raw:
            candidates.append(Path(raw).expanduser())
    repo_root = Path(__file__).resolve().parents[3]
    candidates.extend(
        [
            repo_root / "references" / "DAMO-ConvAI" / "api-bank",
            repo_root / "references" / "API-Bank",
            repo_root / "references" / "api-bank",
            repo_root.parent / "DAMO-ConvAI" / "api-bank",
            repo_root.parent / "API-Bank",
            repo_root.parent / "api-bank",
            Path("/tmp/helicopter-apibank-official/DAMO-ConvAI/api-bank"),
            Path("/tmp/rwkv-official-refs/DAMO-ConvAI/api-bank"),
            Path("/tmp/ref-DAMO-ConvAI/api-bank"),
            Path("/home/chase/GitHub/rwkv-skills/references/DAMO-ConvAI/api-bank"),
            Path("/home/chase/GitHub/rwkv-skills/references/API-Bank"),
            Path("/home/chase/GitHub/rwkv-skills/references/api-bank"),
        ]
    )
    return tuple(dict.fromkeys(path.expanduser().resolve() for path in candidates))


def _apibank_source_root_from_candidate(candidate: Path) -> Path:
    root = candidate.expanduser().resolve()
    if root.name == "level-1-given-desc" and root.parent.name == "lv1-lv2-samples":
        return root.parent.parent.resolve()
    if (root / "api-bank" / "lv1-lv2-samples" / "level-1-given-desc").is_dir():
        return (root / "api-bank").resolve()
    if (root / APIBANK_REPO_ROOT_NAME / "api-bank" / "lv1-lv2-samples" / "level-1-given-desc").is_dir():
        return (root / APIBANK_REPO_ROOT_NAME / "api-bank").resolve()
    return root


def _apibank_required_paths(source_root: Path) -> tuple[Path, ...]:
    return (
        source_root / "apis",
        source_root / "init_database",
        source_root / "lv1-lv2-samples" / "level-1-given-desc",
    )


def _apibank_source_root() -> Path:
    for candidate in _apibank_source_candidates():
        source_root = _apibank_source_root_from_candidate(candidate)
        if all(path.exists() for path in _apibank_required_paths(source_root)):
            return source_root.resolve()
    return _download_apibank_source()


def _download_apibank_source() -> Path:
    target = Path(os.environ.get("RWKV_LIGHTEVAL_APIBANK_CACHE", "/tmp/helicopter-apibank-official")).expanduser().resolve()
    repo_root = target / APIBANK_REPO_ROOT_NAME
    if not (repo_root / ".git").exists():
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "--branch",
                APIBANK_REPO_REVISION,
                APIBANK_REPO_URL,
                str(repo_root),
            ],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(repo_root), "sparse-checkout", "set", "api-bank"],
        check=True,
    )
    source_root = repo_root / "api-bank"
    missing = [path for path in _apibank_required_paths(source_root) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"APIBank source is incomplete: {', '.join(str(path) for path in missing)}")
    return source_root.resolve()


def _apibank_level_from_name(file_name: str) -> int | None:
    if "level-1" in file_name:
        return 1
    if "level-2" in file_name:
        return 2
    return None


def _render_apibank_history(history: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for item in history:
        role = str(item.get("role") or "").strip()
        if role == "User":
            text = str(item.get("text") or "").lstrip().rstrip(" ")
            lines.append(f"User: {text}" if text else "User:")
        elif role == "AI":
            text = str(item.get("text") or "").lstrip().rstrip(" ")
            lines.append(f"Assistant: {text}" if text else "Assistant:")
        elif role == "API":
            args = ", ".join(f"{key}={_official_arg_repr(value)}" for key, value in dict(item.get("param_dict") or {}).items())
            lines.append(f"API: [{item.get('api_name')}({args})] Response: {item.get('result')}")
    return "\n".join(lines).strip()


def _official_arg_repr(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if value is None:
        return "None"
    return repr(value)


def _apibank_json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    return "string"


def _apibank_api_json_type(value: Any) -> str:
    return {
        "int": "integer",
        "float": "number",
        "str": "string",
        "list": "array",
        "list(str)": "array",
        "bool": "boolean",
    }.get(str(value), "string")


@contextlib.contextmanager
def _temporary_apibank_import_path(source_root: Path) -> Iterator[None]:
    old_cwd = Path.cwd()
    root_text = str(source_root)
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "apis" or name.startswith("apis.")
    }
    for name in previous_modules:
        sys.modules.pop(name, None)
    sys.path.insert(0, root_text)
    os.chdir(source_root)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        try:
            sys.path.remove(root_text)
        except ValueError:
            pass
        for name in list(sys.modules):
            if name == "apis" or name.startswith("apis."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)


class ApiBankSandbox:
    _ARGUMENT_ALIASES: dict[str, dict[str, str]] = {
        "CancelTimedSwitch": {"device_id": "name"},
        "TimedSwitch": {"device_id": "name"},
    }

    def __init__(self, source_root: str | Path) -> None:
        self.source_root = Path(source_root).expanduser().resolve()
        self._api_classes: dict[str, type] | None = None
        self._tools: dict[str, Any] = {}
        self._init_databases: dict[str, Any] | None = None

    def api_call(self, api_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        try:
            tool = self.init_tool(api_name)
            api_info = self._api_info(api_name)
            normalized_arguments = self.normalize_arguments(api_name, arguments)
            processed = {
                key: self._coerce_arg(value, api_info.get("input_parameters", {}).get(key, {}).get("type"))
                for key, value in normalized_arguments.items()
            }
            return {"success": True, "result": tool.call(**processed), "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "result": None, "error": str(exc)}

    def check_api_call_correctness(self, api_name: str, actual: Any, expected: Any) -> bool:
        return bool(self.init_tool(api_name).check_api_call_correctness(actual, expected))

    def get_api_description(self, api_name: str) -> dict[str, Any] | None:
        try:
            info = dict(self._api_info(api_name))
        except Exception:
            return None
        info.pop("class", None)
        info.pop("init_database", None)
        return info

    def init_tool(self, api_name: str) -> Any:
        if api_name in self._tools:
            return self._tools[api_name]
        info = self._api_info(api_name)
        args: list[Any] = []
        if "init_database" in info:
            args.append(info["init_database"])
        if api_name != "CheckToken" and "token" in info.get("input_parameters", {}) and "CheckToken" in self._api_classes_by_name():
            args.append(self.init_tool("CheckToken"))
        tool = info["class"](*args)
        self._tools[api_name] = tool
        return tool

    def _api_info(self, api_name: str) -> dict[str, Any]:
        cls = self._api_classes_by_name().get(api_name)
        if cls is None:
            raise ValueError(f"invalid APIBank tool name: {api_name}")
        info: dict[str, Any] = {
            "name": api_name,
            "class": cls,
            "description": getattr(cls, "description", ""),
            "input_parameters": getattr(cls, "input_parameters", {}),
            "output_parameters": getattr(cls, "output_parameters", {}),
        }
        database_name = getattr(cls, "database_name", None)
        init_databases = self._load_init_databases()
        if database_name in init_databases:
            info["init_database"] = init_databases[database_name]
        return info

    def _api_classes_by_name(self) -> dict[str, type]:
        if self._api_classes is not None:
            return self._api_classes
        classes: dict[str, type] = {}
        with _temporary_apibank_import_path(self.source_root):
            api_base = importlib.import_module("apis.api").API
            for file_path in sorted((self.source_root / "apis").glob("*.py")):
                if file_path.name in {"__init__.py", "api.py", "tool_search.py"}:
                    continue
                try:
                    module = importlib.import_module(f"apis.{file_path.stem}")
                except Exception:
                    continue
                for value in vars(module).values():
                    if isinstance(value, type) and issubclass(value, api_base) and value is not api_base:
                        classes[value.__name__] = value
        self._api_classes = classes
        return classes

    def _load_init_databases(self) -> dict[str, Any]:
        if self._init_databases is not None:
            return self._init_databases
        databases: dict[str, Any] = {}
        db_dir = self.source_root / "init_database"
        if db_dir.is_dir():
            for file_path in db_dir.glob("*.json"):
                databases[file_path.stem] = json.loads(file_path.read_text(encoding="utf-8"))
        self._init_databases = databases
        return databases

    @classmethod
    def normalize_arguments(cls, api_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        aliases = cls._ARGUMENT_ALIASES.get(str(api_name), {})
        return {aliases.get(str(key), str(key)): value for key, value in dict(arguments).items()}

    @staticmethod
    def _coerce_arg(value: Any, arg_type: Any) -> Any:
        if arg_type == "int":
            return int(value)
        if arg_type == "float":
            return float(value)
        if arg_type == "bool":
            return value if isinstance(value, bool) else str(value) == "True"
        if str(arg_type) in {"list", "list(str)"}:
            return _coerce_apibank_list_arg(value)
        return value


def _coerce_apibank_list_arg(value: Any) -> Any:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed
    return value


def _normalize_apibank_expected_result(api_name: str, expected: Any) -> Any:
    if not isinstance(expected, Mapping):
        return expected
    normalized = copy.deepcopy(dict(expected))
    input_payload = normalized.get("input")
    if isinstance(input_payload, Mapping):
        normalized["input"] = ApiBankSandbox.normalize_arguments(api_name, input_payload)
    return normalized


def _apibank_schema_sandbox_for(source_root: Path) -> ApiBankSandbox:
    key = str(source_root.expanduser().resolve())
    sandbox = _APIBANK_SCHEMA_SANDBOX_CACHE.get(key)
    if sandbox is None:
        sandbox = ApiBankSandbox(source_root)
        _APIBANK_SCHEMA_SANDBOX_CACHE[key] = sandbox
    return sandbox


def _apibank_tool_schema(source_root: Path, api_name: str, fallback_args: Mapping[str, Any]) -> dict[str, Any]:
    description = _apibank_schema_sandbox_for(source_root).get_api_description(api_name)
    if not description:
        return {
            "name": api_name,
            "description": f"APIBank tool {api_name}",
            "parameters": {
                "type": "object",
                "properties": {str(key): {"type": _apibank_json_type(value)} for key, value in fallback_args.items()},
                "required": [str(key) for key in fallback_args],
            },
        }
    parameters = description.get("input_parameters")
    properties: dict[str, Any] = {}
    required: list[str] = []
    if isinstance(parameters, Mapping):
        for key, spec in parameters.items():
            spec = spec if isinstance(spec, Mapping) else {}
            properties[str(key)] = {
                "type": _apibank_api_json_type(spec.get("type")),
                "description": str(spec.get("description") or ""),
            }
            required.append(str(key))
    return {
        "name": api_name,
        "description": str(description.get("description") or ""),
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


def _load_apibank_rows(dataset_name: str) -> list[dict[str, Any]]:
    if dataset_name not in APIBANK_TASKS:
        raise ValueError(f"unknown APIBank dataset: {dataset_name}")
    level = int(APIBANK_TASKS[dataset_name]["level"])
    source_root = _apibank_source_root()
    source_dir = source_root / "lv1-lv2-samples" / "level-1-given-desc"
    rows: list[dict[str, Any]] = []
    for file_path in sorted(source_dir.glob("*.jsonl")):
        if _apibank_level_from_name(file_path.name) != level:
            continue
        history = [item for item in _read_json_or_jsonl_items(file_path) if isinstance(item, Mapping)]
        api_names = sorted(
            {
                str(item.get("api_name") or "").strip()
                for item in history
                if item.get("role") == "API"
            }
        )
        for turn_index, item in enumerate(history):
            if item.get("role") != "API":
                continue
            api_name = str(item.get("api_name") or "").strip()
            if not api_name:
                continue
            param_dict = item.get("param_dict")
            if not isinstance(param_dict, Mapping):
                param_dict = {}
            rows.append(
                {
                    "task_id": f"{dataset_name}__{file_path.stem}_{turn_index:03d}",
                    "instruction": _render_apibank_history(history[:turn_index]),
                    "tools": [_apibank_tool_schema(source_root, name, param_dict) for name in api_names],
                    "expected_tool_calls": [
                        {
                            "name": api_name,
                            "arguments": dict(param_dict),
                            "argument_options": {key: [value] for key, value in dict(param_dict).items()},
                        }
                    ],
                    "metadata": {
                        "source_format": "official_api_bank",
                        "source_path": str(file_path),
                        "source_dir": str(source_dir),
                        "source_root": str(source_root),
                        "level": level,
                        "turn_index": turn_index,
                        "api_name": api_name,
                        "expected_result": item.get("result"),
                    },
                }
            )
    return rows


def _load_apibank_dataset_dict(dataset_name: str):
    from datasets import Dataset, DatasetDict

    rows: list[dict[str, Any]] = []
    for row in _load_apibank_rows(dataset_name):
        rendered = dict(row)
        for key in ("tools", "expected_tool_calls", "metadata"):
            rendered[key] = json.dumps(rendered.get(key) or ([] if key != "metadata" else {}), ensure_ascii=False, sort_keys=True)
        rows.append(rendered)
    return DatasetDict({"test": Dataset.from_list(rows)})


def apibank_prompt(line: dict, task_name: str | None = None) -> Doc:
    instruction = _normalize_rwkv_text(line.get("instruction"))
    tools = _bfcl_json_list_field(line.get("tools"))
    expected_tool_calls = _bfcl_json_list_field(line.get("expected_tool_calls"))
    metadata = _bfcl_json_mapping_field(line.get("metadata"))
    system_prompt = "\n".join(
        [
            "Tools:",
            _render_bfcl_tool_catalog([tool for tool in tools if isinstance(tool, Mapping)]),
            "Output JSON schema:",
            _render_bfcl_output_schema(),
            "Return exactly one JSON value that validates against the schema.",
            "For one tool call, return one JSON object.",
            "Each arguments object must contain only final argument values for that tool.",
            "API-Bank date convention: if a month/day or relative date has no explicit year and the conversation does not state today's date, use year 2023.",
            "For dates and times, use only dates/times stated or implied by the conversation or function outputs; do not use the real current date.",
            "Do not copy tool schemas, descriptions, type/items/properties/required/default fields, or wrapper objects into arguments.",
            "Use only listed tool names.",
            "Return no prose, no markdown, and no extra text outside the JSON value.",
        ]
    )
    query = f"System: {system_prompt}\n\nUser: {instruction}\n\nJSON:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[json.dumps(expected_tool_calls, ensure_ascii=False, sort_keys=True)],
        gold_index=0,
        specific={
            "task_id": line.get("task_id"),
            "instruction": instruction,
            "tools": json.dumps(tools, ensure_ascii=False, sort_keys=True),
            "expected_tool_calls": json.dumps(expected_tool_calls, ensure_ascii=False, sort_keys=True),
            "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            "prompt_chars": len(query),
        },
    )


def _apibank_source_root_from_metadata(metadata: Mapping[str, Any]) -> Path:
    source_root = str(metadata.get("source_root") or "").strip()
    if source_root:
        return Path(source_root).expanduser().resolve()
    source_dir = Path(str(metadata.get("source_dir") or "")).expanduser()
    if source_dir.name == "level-1-given-desc" and source_dir.parent.name == "lv1-lv2-samples":
        return source_dir.parent.parent.resolve()
    return _apibank_source_root()


def evaluate_apibank_calls(record: Mapping[str, Any], decoded_calls: Sequence[Mapping[str, Any]], *, parse_error: str | None = None) -> dict[str, Any]:
    expected_calls = _bfcl_json_list_field(record.get("expected_tool_calls"))
    expected = expected_calls[0] if expected_calls and isinstance(expected_calls[0], Mapping) else None
    expected_name = str(expected.get("name") or "") if expected else ""
    metadata = _bfcl_json_mapping_field(record.get("metadata"))
    expected_result = _normalize_apibank_expected_result(expected_name, metadata.get("expected_result"))
    details: dict[str, Any] = {
        "expected_tool_calls": expected_calls,
        "decoded_tool_calls": [
            {"name": str(item.get("name") or ""), "arguments": dict(item.get("arguments") or {})}
            for item in decoded_calls
            if isinstance(item, Mapping)
        ],
        "parse_error": parse_error or "",
    }
    if parse_error:
        return {"score": 0.0, "passed": False, "reason": parse_error, "details": details}
    if not expected or not decoded_calls:
        return {"score": 0.0, "passed": False, "reason": "missing_call", "details": details}
    actual = decoded_calls[0]
    actual_name = str(actual.get("name") or "").strip()
    actual_args = actual.get("arguments")
    if not isinstance(actual_args, Mapping):
        actual_args = {}
    if actual_name != expected_name:
        return {"score": 0.0, "passed": False, "reason": f"api_name_mismatch:{actual_name}!={expected_name}", "details": details}

    sandbox = ApiBankSandbox(_apibank_source_root_from_metadata(metadata))
    call_result = sandbox.api_call(actual_name, dict(actual_args))
    details["execution_result"] = dict(call_result)
    details["expected_result"] = expected_result
    if not call_result["success"]:
        return {"score": 0.0, "passed": False, "reason": call_result["error"] or "api_execution_failed", "details": details}
    try:
        ok = sandbox.check_api_call_correctness(actual_name, copy.deepcopy(call_result["result"]), copy.deepcopy(expected_result))
    except Exception as exc:  # noqa: BLE001
        details["check_error"] = str(exc)
        ok = False
    return {"score": 1.0 if ok else 0.0, "passed": bool(ok), "reason": "" if ok else "api_result_mismatch", "details": details}


class ApiBankAccuracy(SampleLevelComputation):
    def compute(self, doc: Doc, model_response, **kwargs) -> float:
        record = dict(doc.specific or {})
        for prediction in model_response.final_text:
            try:
                decoded_calls = decode_bfcl_tool_call_response(prediction)
                result = evaluate_apibank_calls(record, decoded_calls)
            except Exception:
                continue
            if bool(result.get("passed")):
                return 1.0
        return 0.0


APIBANK_ACCURACY = SampleLevelMetric(
    metric_name="apibank_accuracy",
    sample_level_fn=ApiBankAccuracy(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=_mean,
    higher_is_better=True,
)


def _install_longcodeqa_data_files_patch() -> None:
    """Scope LightEval 0.13 dataset patches to custom rwkv-skills datasets."""

    import lighteval.tasks.lighteval_task as lighteval_task_module

    original_load_dataset = lighteval_task_module.load_dataset
    if getattr(original_load_dataset, "_helicopter_rwkv_skills_patch", False):
        return

    def patched_load_dataset(*args, **kwargs):
        path = kwargs.get("path")
        if path is None and args:
            path = args[0]
        if path == LONGCODEQA_LIGHTEVAL_DATASET:
            from datasets import load_dataset as datasets_load_dataset

            return datasets_load_dataset(
                LONGCODEQA_HF_REPO,
                data_files={"test": LONGCODEQA_DATA_FILE},
                revision=kwargs.get("revision"),
            )
        if path == BFCL_LIGHTEVAL_DATASET:
            dataset_name = kwargs.get("name")
            if dataset_name is None and len(args) > 1:
                dataset_name = args[1]
            return _load_bfcl_dataset_dict(str(dataset_name or ""))
        if path == BFCL_EXEC_LIGHTEVAL_DATASET:
            dataset_name = kwargs.get("name")
            if dataset_name is None and len(args) > 1:
                dataset_name = args[1]
            return _load_bfcl_exec_dataset_dict(str(dataset_name or ""))
        if path == APIBANK_LIGHTEVAL_DATASET:
            dataset_name = kwargs.get("name")
            if dataset_name is None and len(args) > 1:
                dataset_name = args[1]
            return _load_apibank_dataset_dict(str(dataset_name or ""))
        return original_load_dataset(*args, **kwargs)

    patched_load_dataset._helicopter_rwkv_skills_patch = True  # type: ignore[attr-defined]
    patched_load_dataset._helicopter_original_load_dataset = original_load_dataset  # type: ignore[attr-defined]
    lighteval_task_module.load_dataset = patched_load_dataset


def _mean_annotation_score(annotations: object) -> float | None:
    if not isinstance(annotations, list) or not annotations:
        return None
    scores: list[float] = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        try:
            scores.append(float(annotation.get("score")))
        except (TypeError, ValueError):
            continue
    return _mean(scores) if scores else None


def _judgement_text(value: bool) -> str:
    return "Judgement: Yes" if value else "Judgement: No"


def _extract_judgement(text: str) -> str | None:
    normalized = text.strip().lower()
    marker = re.search(r"\bjudg(?:e)?ment\s*[:\-]\s*(yes|no)\b", normalized)
    if marker:
        return marker.group(1)
    marker = re.search(r"\b(yes|no)\b", normalized)
    return marker.group(1) if marker else None


class JudgementMatch(SampleLevelComputation):
    def compute(self, doc: Doc, model_response, **kwargs) -> float:
        gold = _extract_judgement(str(doc.choices[doc.gold_index]))
        if gold is None:
            return 0.0
        for prediction in model_response.final_text:
            if _extract_judgement(prediction) == gold:
                return 1.0
        return 0.0


JUDGEMENT_MATCH = SampleLevelMetric(
    metric_name="judgement_match",
    sample_level_fn=JudgementMatch(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=_mean,
    higher_is_better=True,
)


class HumanEvalPassAt1(SampleLevelComputation):
    def compute(self, doc: Doc, model_response, **kwargs) -> float:
        problem = dict(doc.specific or {})
        for completion_id, prediction in enumerate(model_response.final_text):
            result = _check_humaneval_correctness(problem, prediction, completion_id=completion_id)
            if result["passed"]:
                return 1.0
        return 0.0


HUMANEVAL_PASS_AT_1 = SampleLevelMetric(
    metric_name="pass@1",
    sample_level_fn=HumanEvalPassAt1(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=_mean,
    higher_is_better=True,
    batched_compute=False,
)


def _mbpp_task_id(line: dict[str, Any]) -> str:
    task_id = str(line.get("task_id") or "").strip()
    return task_id if "/" in task_id else f"Mbpp/{task_id}"


def _build_mbpp_check_program(problem: dict[str, Any], completion: str, *, include_plus_tests: bool) -> str:
    code = extract_code_completion(completion).rstrip()
    if include_plus_tests:
        tests = str(problem.get("test") or "").strip()
        return f"{code}\n{tests}\n"

    imports = problem.get("test_imports") or []
    import_block = "\n".join(str(item) for item in imports if item)
    test_list = problem.get("test_list") or []
    if isinstance(test_list, str):
        test_block = test_list
    else:
        test_block = "\n".join(str(item) for item in test_list)
    return "\n".join(part for part in (import_block, code, test_block) if part).rstrip() + "\n"


def _check_mbpp_correctness(
    problem: dict[str, Any],
    completion: str,
    *,
    include_plus_tests: bool,
    timeout: float = MBPP_TIMEOUT_SECONDS,
    completion_id: int | None = None,
) -> dict[str, Any]:
    try:
        check_program = _build_mbpp_check_program(problem, completion, include_plus_tests=include_plus_tests)
    except BaseException as exc:  # noqa: BLE001
        return {
            "task_id": problem.get("task_id"),
            "passed": False,
            "result": f"failed: {type(exc).__name__}: {exc}",
            "completion_id": completion_id,
        }
    return _check_python_program_correctness(
        problem.get("task_id"),
        check_program,
        timeout=timeout,
        completion_id=completion_id,
    )


class MbppPassAt1(SampleLevelComputation):
    def __init__(self, *, include_plus_tests: bool) -> None:
        self.include_plus_tests = include_plus_tests

    def compute(self, doc: Doc, model_response, **kwargs) -> float:
        problem = dict(doc.specific or {})
        for completion_id, prediction in enumerate(model_response.final_text):
            result = _check_mbpp_correctness(
                problem,
                prediction,
                include_plus_tests=self.include_plus_tests,
                completion_id=completion_id,
            )
            if result["passed"]:
                return 1.0
        return 0.0


MBPP_PASS_AT_1 = SampleLevelMetric(
    metric_name="pass@1",
    sample_level_fn=MbppPassAt1(include_plus_tests=False),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=_mean,
    higher_is_better=True,
    batched_compute=False,
)


MBPP_PLUS_PASS_AT_1 = SampleLevelMetric(
    metric_name="pass@1",
    sample_level_fn=MbppPassAt1(include_plus_tests=True),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=_mean,
    higher_is_better=True,
    batched_compute=False,
)


def _question(record: dict) -> str:
    for key in ("question", "problem", "problem_statement", "prompt", "input"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return ""


def _format_answer(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _answer(record: dict) -> str:
    for key in ("expected_answer", "answer", "final_answer", "target", "result"):
        value = record.get(key)
        if value is not None:
            return _format_answer(value)
    return ""


def qwen_math_prompt(line: dict, task_name: str | None = None) -> Doc:
    question = _question(line)
    answer = _answer(line)
    return Doc(
        task_name=task_name,
        query=f"Question: {question}\nAnswer:",
        choices=[answer],
        gold_index=0,
        specific={
            "id": line.get("id"),
            "source": line.get("source") or line.get("sourcename"),
        },
    )


def record_to_sample(record: dict) -> Sample:
    return Sample(
        input=_question(record),
        target=_answer(record),
        metadata={
            "id": record.get("id"),
            "source": record.get("source") or record.get("sourcename"),
        },
    )


def _choice_texts(record: dict) -> list[str]:
    if isinstance(record.get("options"), list):
        return [str(choice).strip() for choice in record["options"]]
    choices: list[str] = []
    for letter in ascii_uppercase:
        value = record.get(letter)
        if value is None:
            break
        choices.append(str(value).strip())
    return choices


def supergpqa_prompt(line: dict, task_name: str | None = None) -> Doc:
    choices = _choice_texts(line)
    answer_letter = str(line.get("answer_letter") or line.get("answer") or "").strip().upper()
    if not answer_letter and line.get("answer") in choices:
        answer_letter = ascii_uppercase[choices.index(str(line["answer"]).strip())]
    gold_index = ascii_uppercase.index(answer_letter)

    query = f"Question: {line['question']}"
    query += "".join(f"\n{letter}. {choice}" for letter, choice in zip(ascii_uppercase, choices))
    query += "\nAnswer:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {letter}" for letter in ascii_uppercase[: len(choices)]],
        gold_index=gold_index,
        specific={
            "uuid": line.get("uuid"),
            "discipline": line.get("discipline"),
            "field": line.get("field"),
            "subfield": line.get("subfield"),
            "difficulty": line.get("difficulty"),
        },
    )


def mmmlu_prompt(line: dict, task_name: str | None = None) -> Doc:
    choices = [str(line[letter]).strip() for letter in ascii_uppercase[:4]]
    gold_index = ascii_uppercase.index(str(line["Answer"]).strip().upper())
    query = f"Question: {line['Question']}"
    query += "".join(f"\n{letter}. {choice}" for letter, choice in zip(ascii_uppercase, choices))
    query += "\nAnswer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {letter}" for letter in ascii_uppercase[:4]],
        gold_index=gold_index,
        specific={"subject": line.get("Subject")},
    )


def wmt24pp_prompt(line: dict, task_name: str | None = None) -> Doc:
    target_language = str(line.get("lp") or "").split("-", 1)[-1]
    target_name = WMT24PP_TARGET_LANGUAGES.get(target_language, target_language)
    return Doc(
        task_name=task_name,
        query=f"English phrase: {str(line['source']).rstrip()}\n{target_name} phrase:",
        choices=[str(line["target"]).rstrip()],
        gold_index=0,
        instruction=f"Translate English to {target_name}, do not explain, only output the translation.",
        specific={
            "lp": line.get("lp"),
            "domain": line.get("domain"),
            "document_id": line.get("document_id"),
            "segment_id": line.get("segment_id"),
            "is_bad_source": line.get("is_bad_source"),
        },
    )


def human_eval_prompt(line: dict, task_name: str | None = None) -> Doc:
    prompt = str(line["prompt"]).rstrip()
    return Doc(
        task_name=task_name,
        query=(
            "Complete the following Python function. Return only executable Python code; "
            "do not explain.\n\n"
            f"```python\n{prompt}\n```"
        ),
        choices=[str(line.get("canonical_solution") or "")],
        gold_index=0,
        specific={
            "task_id": line.get("task_id"),
            "prompt": line.get("prompt"),
            "canonical_solution": line.get("canonical_solution"),
            "test": line.get("test"),
            "entry_point": _extract_entry_point(line),
            "declaration": line.get("declaration"),
            "example_test": line.get("example_test"),
            "text": line.get("text"),
        },
    )


def human_eval_fix_prompt(line: dict, task_name: str | None = None) -> Doc:
    prompt = str(line.get("prompt") or "").rstrip()
    buggy_solution = str(line.get("buggy_solution") or "").rstrip()
    entry_point = _extract_entry_point(line) or ""
    query = (
        "Fix the following Python function. Return only the corrected executable Python code; do not explain.\n\n"
        f"Original prompt:\n```python\n{prompt}\n```"
    )
    if buggy_solution:
        query += f"\n\nBuggy implementation:\n```python\n{buggy_solution}\n```"
    if entry_point:
        query += f"\n\nFix `{entry_point}` so it passes all tests."
    return Doc(
        task_name=task_name,
        query=query,
        choices=[str(line.get("canonical_solution") or "")],
        gold_index=0,
        specific={
            "task_id": line.get("task_id"),
            "prompt": line.get("prompt"),
            "canonical_solution": line.get("canonical_solution"),
            "buggy_solution": line.get("buggy_solution"),
            "test": line.get("test"),
            "entry_point": entry_point,
            "declaration": line.get("declaration"),
            "example_test": line.get("example_test"),
            "bug_type": line.get("bug_type"),
            "failure_symptoms": line.get("failure_symptoms"),
        },
    )


def mbpp_prompt(line: dict, task_name: str | None = None) -> Doc:
    prompt = str(line.get("prompt") or line.get("text") or "").strip()
    return Doc(
        task_name=task_name,
        query=(
            "Write a Python function that satisfies the problem statement. "
            "Return only executable Python code; do not explain.\n\n"
            f"Problem:\n{prompt}"
        ),
        choices=[str(line.get("code") or "")],
        gold_index=0,
        specific={
            "task_id": _mbpp_task_id(line),
            "prompt": line.get("prompt") or line.get("text"),
            "code": line.get("code"),
            "source_file": line.get("source_file"),
            "test_imports": line.get("test_imports") or [],
            "test_list": line.get("test_list") or [],
            "test": line.get("test") or "",
        },
    )


def longcodeqa_prompt(line: dict, task_name: str | None = None) -> Doc:
    question = _normalize_newlines(line.get("question")).strip()
    repo_text = _normalize_newlines(line.get("repo_text"))
    prompt_goal = _normalize_newlines(line.get("prompt_goal")).strip() or (
        "You are provided repository context and a multiple-choice question about it."
    )
    allowed_letters = _longcodeqa_choice_letters(question)
    gold_letter = str(line.get("correct_letter") or "").strip().upper()
    if gold_letter not in allowed_letters:
        gold_letter = allowed_letters[0]

    max_chars = _longcodeqa_max_prompt_chars()
    empty_prompt = _longcodeqa_answer_contract(
        f"{prompt_goal}\nRepository:\n\n{question}",
        allowed_letters,
    )
    context_budget = max(0, max_chars - len(empty_prompt) - 32)
    context = _longcodeqa_context_excerpt(repo_text, question, max_chars=context_budget)
    query = _longcodeqa_answer_contract(
        f"{prompt_goal}\nRepository:\n{context}\n\n{question}",
        allowed_letters,
    )
    if len(query) > max_chars:
        overflow = len(query) - max_chars
        context = _middle_truncate_text(context, max(0, len(context) - overflow - 64))
        query = _longcodeqa_answer_contract(
            f"{prompt_goal}\nRepository:\n{context}\n\n{question}",
            allowed_letters,
        )

    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {letter}" for letter in allowed_letters],
        gold_index=allowed_letters.index(gold_letter),
        specific={
            "repo": line.get("repo"),
            "context_chars": len(repo_text),
            "prompt_chars": len(query),
            "correct_letter": gold_letter,
            "is_hard": line.get("is_hard"),
        },
    )


def longbench_prompt(line: dict, task_name: str | None = None) -> Doc:
    question = _normalize_newlines(line.get("input")).strip()
    context = _normalize_newlines(line.get("context"))
    answers = _coerce_string_list(line.get("answers"))
    if not answers:
        answers = [""]
    classes = _coerce_string_list(line.get("all_classes"))
    language = str(line.get("language") or "").lower()

    max_chars = _longcodeqa_max_prompt_chars()
    lines = [
        "You are evaluating a long-context reading task.",
        "Answer the question using only the provided context.",
        "Return only the concise final answer. Do not include analysis, markdown, citations, or extra text.",
    ]
    if classes:
        lines.append("If labels/classes are provided, answer with exactly one allowed label.")
        lines.append("Allowed labels/classes: " + ", ".join(classes))
    if language.startswith("zh"):
        lines.append("If the question is Chinese, answer in Chinese.")
    instruction = "\n".join(lines)
    empty_prompt = f"{instruction}\n\nContext:\n\nQuestion:\n{question}\n\nAnswer:"
    context_budget = max(0, max_chars - len(empty_prompt) - 32)
    rendered_context = _longcodeqa_context_excerpt(context, question, max_chars=context_budget)
    query = f"{instruction}\n\nContext:\n{rendered_context}\n\nQuestion:\n{question}\n\nAnswer:"
    if len(query) > max_chars:
        overflow = len(query) - max_chars
        rendered_context = _middle_truncate_text(rendered_context, max(0, len(rendered_context) - overflow - 64))
        query = f"{instruction}\n\nContext:\n{rendered_context}\n\nQuestion:\n{question}\n\nAnswer:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=answers,
        gold_index=0,
        specific={
            "id": line.get("_id") or line.get("task_id"),
            "dataset": line.get("dataset"),
            "language": line.get("language"),
            "length": line.get("length"),
            "context_chars": len(context),
            "prompt_chars": len(query),
        },
    )


def answer_judge_prompt(line: dict, task_name: str | None = None) -> Doc | None:
    mean_score = _mean_annotation_score(line.get("annotations"))
    if mean_score is None:
        return None
    expected_judgement = _judgement_text(mean_score > 0.5)
    query = (
        "Problem:\n"
        f"{line.get('question') or ''}\n\n"
        "Expected answer:\n"
        f"{line.get('gt_answer') or ''}\n\n"
        "Predicted answer:\n"
        f"{line.get('gen_answer') or ''}\n\n"
        "Decide whether the predicted answer matches the expected answer. "
        "Return exactly `Judgement: Yes` or `Judgement: No`.\n"
        "Judgement:"
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=[expected_judgement],
        gold_index=0,
        specific={
            "item_name": line.get("item_name"),
            "dataset_name": line.get("dataset_name"),
            "mean_score": mean_score,
        },
    )


def svamp_prompt(line: dict, task_name: str | None = None) -> Doc:
    body = str(line.get("Body") or "").strip()
    question = str(line.get("Question") or "").strip()
    if body and question:
        full_question = f"{body.rstrip('.')}. {question}"
    else:
        full_question = body or question
    return Doc(
        task_name=task_name,
        query=f"Question: {full_question}\nAnswer:",
        choices=[_format_answer(line.get("Answer", ""))],
        gold_index=0,
        specific={
            "id": line.get("ID"),
            "type": line.get("Type"),
            "equation": line.get("Equation"),
        },
    )


def _hf_answer_judge_task() -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name="rwkv_skills:answer_judge",
        prompt_function=answer_judge_prompt,
        hf_repo="nvidia/judges-verdict",
        hf_subset=None,
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=8,
        metrics=[JUDGEMENT_MATCH],
        stop_sequence=["\n"],
        version=0,
    )


def _hf_math_task(name: str, task: dict[str, object]) -> LightevalTaskConfig:
    split = str(task["split"])
    return LightevalTaskConfig(
        name=f"rwkv_skills:{name}",
        prompt_function=qwen_math_prompt,
        sample_fields=record_to_sample,
        solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
        scorer=math_scorer(),
        hf_repo=str(task["repo"]),
        hf_subset=task.get("subset"),
        hf_avail_splits=[split],
        evaluation_splits=[split],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=512,
        metrics=[Metrics.expr_gold_metric],
        stop_sequence=["Question:"],
        version=0,
    )


def _hf_svamp_task() -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name="rwkv_skills:svamp",
        prompt_function=svamp_prompt,
        hf_repo="tongyx361/svamp",
        hf_subset=None,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=512,
        metrics=[Metrics.expr_gold_metric],
        stop_sequence=["Question:"],
        version=0,
    )


def _hf_polymath_task(language: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"rwkv_skills:polymath_{language}",
        prompt_function=qwen_math_prompt,
        hf_repo="Qwen/PolyMath",
        hf_subset=language,
        hf_avail_splits=["top", "high", "medium", "low"],
        evaluation_splits=["top", "high", "medium", "low"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=512,
        metrics=[Metrics.expr_gold_metric],
        stop_sequence=["Question:"],
        version=0,
    )


def _hf_multiple_choice_task(name: str, task: dict[str, str]) -> LightevalTaskConfig:
    split = task["split"]
    return LightevalTaskConfig(
        name=f"rwkv_skills:{name}",
        prompt_function=supergpqa_prompt,
        hf_repo=task["repo"],
        hf_subset=None,
        hf_avail_splits=[split],
        evaluation_splits=[split],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=5,
        metrics=[MULTIPLE_CHOICE_LETTER_MATCH],
        stop_sequence=["\n"],
        version=0,
    )


def _hf_mmmlu_task(name: str, subset: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"rwkv_skills:{name}",
        prompt_function=mmmlu_prompt,
        hf_repo="openai/MMMLU",
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=5,
        metrics=[MULTIPLE_CHOICE_LETTER_MATCH],
        stop_sequence=["\n"],
        version=0,
    )


def _hf_wmt24pp_task(target_language: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"rwkv_skills:wmt24pp_{target_language}",
        prompt_function=wmt24pp_prompt,
        hf_repo="google/wmt24pp",
        hf_subset=f"en-{target_language}",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=None,
        metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
        stop_sequence=["\n"],
        version=0,
    )


def _hf_human_eval_task(
    name: str,
    repo: str,
    *,
    subset: str | None = None,
    prompt_function=human_eval_prompt,
) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"rwkv_skills:{name}",
        prompt_function=prompt_function,
        hf_repo=repo,
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=1024,
        metrics=[HUMANEVAL_PASS_AT_1],
        stop_sequence=[],
        version=0,
    )


def _hf_mbpp_task(name: str, metric: SampleLevelMetric) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"rwkv_skills:{name}",
        prompt_function=mbpp_prompt,
        hf_repo="evalplus/mbppplus",
        hf_subset=None,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=1024,
        metrics=[metric],
        stop_sequence=[],
        version=0,
    )


def _hf_longcodeqa_task() -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name="rwkv_skills:longcodeqa",
        prompt_function=longcodeqa_prompt,
        hf_repo=LONGCODEQA_LIGHTEVAL_DATASET,
        hf_subset=None,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=16,
        metrics=[LONGCODEQA_LETTER_MATCH],
        stop_sequence=["\n"],
        version=1,
    )


def _hf_longbench_task(dataset: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"rwkv_skills:longbench_{_longbench_task_suffix(dataset)}",
        prompt_function=longbench_prompt,
        hf_repo=LONG_BENCH_HF_REPO,
        hf_subset=dataset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=256,
        metrics=[LONG_BENCH_EXACT_MATCH, LONG_BENCH_F1],
        stop_sequence=["\nUser:", "\nSystem:", "\nAssistant:"],
        version=1,
    )


def _hf_bfcl_ast_task(name: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"rwkv_skills:{name}",
        prompt_function=bfcl_ast_prompt,
        hf_repo=BFCL_LIGHTEVAL_DATASET,
        hf_subset=name,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=768,
        metrics=[BFCL_AST_ACCURACY],
        stop_sequence=["\nUser:", "\nSystem:", "\nAssistant:"],
        version=1,
    )


def _hf_bfcl_exec_task(name: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"rwkv_skills:{name}",
        prompt_function=bfcl_exec_prompt,
        hf_repo=BFCL_EXEC_LIGHTEVAL_DATASET,
        hf_subset=name,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=768,
        metrics=[BFCL_EXEC_ACCURACY],
        stop_sequence=["\nUser:", "\nSystem:", "\nAssistant:"],
        version=0,
    )


def _hf_apibank_task(name: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"rwkv_skills:{name}",
        prompt_function=apibank_prompt,
        hf_repo=APIBANK_LIGHTEVAL_DATASET,
        hf_subset=name,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=512,
        metrics=[APIBANK_ACCURACY],
        stop_sequence=["\nUser:", "\nSystem:", "\nAssistant:"],
        version=0,
    )


_install_longcodeqa_data_files_patch()


TASKS_TABLE = [
    _hf_human_eval_task("human_eval", "openai/openai_humaneval"),
    _hf_human_eval_task("human_eval_cn", str(LOCAL_DATA_ROOT / "human_eval_cn")),
    _hf_human_eval_task(
        "human_eval_fix",
        "bigcode/humanevalpack",
        subset="python",
        prompt_function=human_eval_fix_prompt,
    ),
    _hf_human_eval_task("human_eval_plus", "evalplus/humanevalplus"),
    _hf_mbpp_task("mbpp", MBPP_PASS_AT_1),
    _hf_mbpp_task("mbpp_plus", MBPP_PLUS_PASS_AT_1),
    _hf_longcodeqa_task(),
    *[_hf_longbench_task(dataset) for dataset in LONG_BENCH_DATASETS],
    *[_hf_bfcl_ast_task(name) for name in BFCL_AST_TASKS],
    *[_hf_bfcl_exec_task(name) for name in BFCL_EXEC_TASKS],
    *[_hf_apibank_task(name) for name in APIBANK_TASKS],
    *[_hf_math_task(name, task) for name, task in HF_MATH_TASKS.items()],
    _hf_answer_judge_task(),
    _hf_svamp_task(),
    *[_hf_polymath_task(language) for language in HF_POLYMATH_LANGUAGES],
    *[_hf_multiple_choice_task(name, task) for name, task in HF_MULTIPLE_CHOICE_TASKS.items()],
    *[_hf_mmmlu_task(name, subset) for name, subset in HF_MMMLU_TASKS.items()],
    *[_hf_wmt24pp_task(target_language) for target_language in WMT24PP_TARGET_LANGUAGES],
]
