from __future__ import annotations

import contextlib
import faulthandler
import io
import linecache
import multiprocessing
import os
import platform
import re
import signal
import sys
import tempfile
import traceback
from pathlib import Path
from string import ascii_uppercase
from typing import Any

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
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_FENCED_CODE_RE = re.compile(
    r"```[ \t]*(?:python|py)?[^\S\r\n]*\r?\n(?P<code>.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_DEF_RE = re.compile(r"def\s+(?P<name>[\w_]+)\s*\(")
_LEADING_END_THINK_RE = re.compile(r"^[\s\r\n]*</think>[ \t]*\r?\n?", re.IGNORECASE)
_STANDALONE_CODE_FENCE_RE = re.compile(r"^[ \t]*```[ \t]*(?:python|py)?[ \t]*$", re.IGNORECASE)

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
    *[_hf_math_task(name, task) for name, task in HF_MATH_TASKS.items()],
    _hf_answer_judge_task(),
    _hf_svamp_task(),
    *[_hf_polymath_task(language) for language in HF_POLYMATH_LANGUAGES],
    *[_hf_multiple_choice_task(name, task) for name, task in HF_MULTIPLE_CHOICE_TASKS.items()],
    *[_hf_mmmlu_task(name, subset) for name, subset in HF_MMMLU_TASKS.items()],
    *[_hf_wmt24pp_task(target_language) for target_language in WMT24PP_TARGET_LANGUAGES],
]
