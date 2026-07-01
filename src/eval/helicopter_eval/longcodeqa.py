from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import re
import string
from typing import Any, Mapping, Sequence
import zipfile

from .openai_client import chat_completion
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


LONGCODEBENCH_HF_REPO = "Steefano/LCB"
LONGCODEQA_ARCHIVE = "LongCodeQA.zip"
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_ANSWER_PREFIX_RE = re.compile(
    r"(?:final\s+answer|correct\s+answer|answer|答案|最终答案)\s*(?:is|为)?\s*[:：]?\s*(?:\*\*)?\s*\(?([A-Z])\)?(?:\*\*)?\b",
    re.IGNORECASE,
)
_CHOICE_LINE_RE = re.compile(r"^\s*([A-Z])\)", re.MULTILINE)
_FENCED_BLOCK_RE = re.compile(r"^\s*```(?:json|text)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)
_INLINE_BOLD_CHOICE_RE = re.compile(r"\*\*\s*\(?([A-Z])\)?\s*[\).:]", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class LongCodeQASample:
    sample_index: int
    task_id: str
    prompt: str
    repo_text: str
    question: str
    correct_letter: str
    repo: str = ""
    context_bucket: str = ""
    context_size: int = 0


@dataclass(frozen=True, slots=True)
class LongCodeQAResult:
    sample_index: int
    task_id: str
    prompt: str
    completion: str
    answer: str
    reference_answer: str
    is_passed: bool
    fail_reason: str

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
class LongCodeQARunConfig:
    base_url: str
    model: str
    benchmark: str = "longcodeqa"
    limit: int | None = None
    split: str = "test"
    source_path: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 64
    timeout_s: float = 600.0
    scoreboard_dataset: str | None = None
    job_name: str = "function_longcodebench"
    job_id: str | None = None
    runner: str = "helicopter_eval.longcodeqa"
    cot_mode: str = "CoT"


def normalize_newlines(value: str) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n")


def source_path(config: LongCodeQARunConfig) -> Path:
    if config.source_path:
        return Path(config.source_path).expanduser().resolve()
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in integration environments
        raise SystemExit("longcodeqa requires `huggingface_hub`; install the rwkv dependency group.") from exc
    return Path(
        hf_hub_download(
            repo_id=LONGCODEBENCH_HF_REPO,
            filename=LONGCODEQA_ARCHIVE,
            repo_type="dataset",
        )
    )


def _context_bucket_chars(bucket: str) -> int:
    text = str(bucket or "").strip().upper()
    match = re.fullmatch(r"(\d+)\s*([KMG])?", text)
    if not match:
        return 0
    value = int(match.group(1))
    unit = match.group(2) or ""
    if unit == "K":
        return value * 1024
    if unit == "M":
        return value * 1024 * 1024
    if unit == "G":
        return value * 1024 * 1024 * 1024
    return value


def _source_sort_key(value: str | Path) -> tuple[int, str]:
    path = Path(str(value))
    bucket_value = _context_bucket_chars(path.stem)
    return (bucket_value if bucket_value > 0 else 10**12, str(value))


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _rows_from_zip(path: Path):
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".json")]
        for name in sorted(names, key=_source_sort_key):
            payload = json.loads(archive.read(name).decode("utf-8"))
            if not isinstance(payload, list):
                continue
            context_bucket = Path(name).stem
            for index, row in enumerate(payload):
                if not isinstance(row, Mapping):
                    continue
                yield {
                    **dict(row),
                    "context_bucket": context_bucket,
                    "context_size": _context_bucket_chars(context_bucket),
                    "_source_path": f"{path}:{name}",
                    "_source_index": index,
                }


def _rows_from_json(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else [payload]
    for index, row in enumerate(rows):
        if isinstance(row, Mapping):
            yield {**dict(row), "_source_path": str(path), "_source_index": index}


def _iter_source_rows(path: Path):
    if path.is_file() and path.suffix.lower() == ".zip":
        yield from _rows_from_zip(path)
        return
    if path.is_file() and path.suffix.lower() == ".json":
        yield from _rows_from_json(path)
        return
    if path.is_dir():
        for child in sorted(path.rglob("*.zip"), key=_source_sort_key):
            yield from _rows_from_zip(child)
        for child in sorted(path.rglob("*.json"), key=_source_sort_key):
            yield from _rows_from_json(child)
        return
    raise FileNotFoundError(path)


def _sample_from_row(index: int, row: Mapping[str, Any]) -> LongCodeQASample:
    context_bucket = str(row.get("context_bucket") or row.get("bucket") or "").strip()
    task_id = str(row.get("task_id") or row.get("id") or "").strip()
    if not task_id:
        task_id = f"longcodeqa_{(context_bucket or 'dataset').lower()}_{index:05d}"
    question = normalize_newlines(str(row.get("question") or ""))
    repo_text = normalize_newlines(str(row.get("repo_text") or row.get("repository") or row.get("context") or ""))
    prompt = normalize_newlines(str(row.get("prompt") or ""))
    if not prompt:
        prompt_goal = str(row.get("prompt_goal") or "").strip() or (
            "You are going to be provided the content of a repository and a question about it. "
            "Provide the answer to the question by stating ONLY the letter associated to the question."
        )
        prompt = f"{prompt_goal}\nRepository: {repo_text}\n{question}\nAnswer:"
    return LongCodeQASample(
        sample_index=index,
        task_id=task_id,
        prompt=prompt,
        repo_text=repo_text,
        question=question,
        correct_letter=str(row.get("correct_letter") or row.get("answer") or "").strip().upper()[:1],
        repo=str(row.get("repo") or ""),
        context_bucket=context_bucket,
        context_size=_coerce_int(row.get("context_size"), default=_context_bucket_chars(context_bucket)),
    )


def load_samples(config: LongCodeQARunConfig) -> list[LongCodeQASample]:
    if config.limit is not None and int(config.limit) < 0:
        raise ValueError("limit must be non-negative")
    limit = None if config.limit is None else int(config.limit)
    samples: list[LongCodeQASample] = []
    for row in _iter_source_rows(source_path(config)):
        if limit is not None and len(samples) >= limit:
            break
        samples.append(_sample_from_row(len(samples), row))
    return samples


def choice_letters(question: str) -> tuple[str, ...]:
    matches = [match.group(1).upper() for match in _CHOICE_LINE_RE.finditer(str(question or ""))]
    seen: set[str] = set()
    ordered: list[str] = []
    for letter in matches:
        if letter not in seen:
            seen.add(letter)
            ordered.append(letter)
    return tuple(ordered or list(string.ascii_uppercase[:4]))


def build_prompt(sample: LongCodeQASample) -> str:
    letters = ", ".join(choice_letters(sample.question))
    instruction = (
        f"{sample.prompt.rstrip()}\n\n"
        "Return the final answer as exactly one option letter. "
        f"Allowed letters: {letters}. "
        'You may answer as plain text like `A` or JSON like {"answer":"A"}.'
    )
    return f"User: {instruction}\n\nAssistant:"


def _clean_answer_text(text: str) -> str:
    body = normalize_newlines(str(text or "")).strip()
    if body.startswith("Assistant:"):
        body = body[len("Assistant:") :].strip()
    body = _THINK_BLOCK_RE.sub("", body).strip()
    fence_match = _FENCED_BLOCK_RE.match(body)
    if fence_match:
        body = fence_match.group(1).strip()
    if body.startswith("```"):
        lines = body.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json", "```text"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        body = "\n".join(lines).strip()
    if body.endswith("```"):
        body = body[: -len("```")].strip()
    return body


def _extract_letter_from_value(value: Any, *, allowed: set[str]) -> str:
    if isinstance(value, str):
        candidate = value.strip().upper()
        return candidate if len(candidate) == 1 and candidate in allowed else ""
    if isinstance(value, Mapping):
        for key in ("answer", "choice", "letter", "prediction", "final_answer"):
            if key in value:
                letter = _extract_letter_from_value(value.get(key), allowed=allowed)
                if letter:
                    return letter
        arguments = value.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        if isinstance(arguments, Mapping):
            letter = _extract_letter_from_value(arguments, allowed=allowed)
            if letter:
                return letter
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            letter = _extract_letter_from_value(item, allowed=allowed)
            if letter:
                return letter
    return ""


def normalize_answer(text: str, *, allowed_letters: Sequence[str]) -> str:
    body = _clean_answer_text(text)
    allowed = {letter.upper() for letter in allowed_letters if str(letter).strip()} or set(string.ascii_uppercase[:8])
    if body.startswith(("{", "[")):
        try:
            letter = _extract_letter_from_value(json.loads(body), allowed=allowed)
        except json.JSONDecodeError:
            letter = ""
        if letter:
            return letter
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    for line in reversed(lines):
        match = _ANSWER_PREFIX_RE.search(line)
        if match and match.group(1).upper() in allowed:
            return match.group(1).upper()
    for line in reversed(lines):
        match = re.fullmatch(r"\(?([A-Z])\)?[\.\)]?", line.strip().strip("`"), flags=re.IGNORECASE)
        if match and match.group(1).upper() in allowed:
            return match.group(1).upper()
    match = re.match(r"\(?([A-Z])\)?[\.\):,\s]", body.lstrip(), flags=re.IGNORECASE)
    if match and match.group(1).upper() in allowed:
        return match.group(1).upper()
    bold_letters = [
        match.group(1).upper()
        for match in _INLINE_BOLD_CHOICE_RE.finditer(body)
        if match.group(1).upper() in allowed
    ]
    if bold_letters and len(set(bold_letters)) == 1:
        return bold_letters[0]
    return ""


def evaluate_completion(sample: LongCodeQASample, completion: str) -> tuple[str, bool]:
    prediction = normalize_answer(completion, allowed_letters=choice_letters(sample.question))
    expected = sample.correct_letter
    return prediction, bool(expected and prediction == expected)


def generate_completion(sample: LongCodeQASample, config: LongCodeQARunConfig) -> LongCodeQAResult:
    prompt = build_prompt(sample)
    completion = chat_completion(
        base_url=config.base_url,
        model=config.model,
        prompt=prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
    )
    prediction, is_passed = evaluate_completion(sample, completion)
    return LongCodeQAResult(
        sample_index=sample.sample_index,
        task_id=sample.task_id,
        prompt=prompt,
        completion=completion,
        answer=prediction,
        reference_answer=sample.correct_letter,
        is_passed=is_passed,
        fail_reason="" if is_passed else "wrong_letter",
    )


def evaluate_samples(samples: Sequence[LongCodeQASample], config: LongCodeQARunConfig) -> list[LongCodeQAResult]:
    return [generate_completion(sample, config) for sample in samples]


def scoreboard_dataset_name(config: LongCodeQARunConfig) -> str:
    dataset = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    return dataset


def job_id(config: LongCodeQARunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def completion_sampling_config(config: LongCodeQARunConfig) -> dict[str, Any]:
    return {
        "answer": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_tokens,
        }
    }


def task_sampling_config(config: LongCodeQARunConfig) -> dict[str, Any]:
    return {
        "avg_k": 1,
        "pass_ks": [1],
        "prompt_profile": "helicopter",
        "sampling_config": completion_sampling_config(config),
    }


def write_results(results: Sequence[LongCodeQAResult], *, config: LongCodeQARunConfig, repo_root: Path) -> int:
    parse_rate = sum(1 for result in results if result.answer) / len(results) if results else 0.0
    accuracy = sum(1 for result in results if result.is_passed) / len(results) if results else 0.0
    return int(
        asyncio.run(
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
                    extra_metrics={
                        "avg@1": accuracy,
                        "success_rate": accuracy,
                        "longcodeqa_accuracy": accuracy,
                        "longcodeqa_answer_parse_rate": parse_rate,
                    },
                ),
                repo_root=repo_root,
            )
        )
    )


def run_longcodeqa(config: LongCodeQARunConfig, *, repo_root: Path) -> dict[str, Any]:
    samples = load_samples(config)
    results = evaluate_samples(samples, config)
    task_id = write_results(results, config=config, repo_root=repo_root)
    passed = sum(1 for result in results if result.is_passed)
    return {
        "task_id": task_id,
        "benchmark": config.benchmark,
        "dataset": scoreboard_dataset_name(config),
        "model": config.model,
        "total": len(results),
        "passed": passed,
        "accuracy": passed / len(results) if results else 0.0,
    }


def dry_run_summary(config: LongCodeQARunConfig) -> dict[str, Any]:
    return {
        "benchmark": config.benchmark,
        "source": config.source_path or f"hf://{LONGCODEBENCH_HF_REPO}/{LONGCODEQA_ARCHIVE}",
        "split": config.split,
        "limit": config.limit,
        "base_url": config.base_url,
        "model": config.model,
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": job_id(config),
    }
