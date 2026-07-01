from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
import string
from typing import Any, Mapping, Sequence
import unicodedata

from .openai_client import chat_completion
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


LONG_BENCH_SOURCE = "THUDM/LongBench"
LONG_BENCH_DATASETS = frozenset(
    {
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
    }
)
LONG_BENCH_QA_DATASETS = frozenset(
    {
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "multifieldqa_zh",
        "hotpotqa",
        "2wikimqa",
        "musique",
        "dureader",
        "triviaqa",
    }
)
_LONG_BENCH_ZH_DATASETS = frozenset({"multifieldqa_zh", "dureader", "vcsum", "lsht", "passage_retrieval_zh"})
_WORD_OR_CJK_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
_ANSWER_PREFIX_RE = re.compile(r"^\s*(?:final\s+answer|answer|答案|最终答案)\s*[:：]\s*", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True, slots=True)
class LongBenchSample:
    sample_index: int
    task_id: str
    dataset: str
    question: str
    context: str
    answers: tuple[str, ...]
    all_classes: tuple[str, ...] = ()
    language: str = "en"
    length: int = 0
    category: str = ""


@dataclass(frozen=True, slots=True)
class LongBenchResult:
    sample_index: int
    task_id: str
    prompt: str
    completion: str
    answer: str
    reference_answer: str
    is_passed: bool
    fail_reason: str
    f1: float
    exact_match: bool

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
class LongBenchRunConfig:
    base_url: str
    model: str
    benchmark: str = "longbench"
    limit: int | None = None
    split: str = "test"
    source_root: str | None = None
    include_datasets: tuple[str, ...] = ()
    balance_by_dataset: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 128
    timeout_s: float = 600.0
    prompt_max_chars: int = 8192
    scoreboard_dataset: str | None = None
    job_name: str = "function_longbench"
    job_id: str | None = None
    runner: str = "helicopter_eval.longbench"
    cot_mode: str = "CoT"


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\r\n", "\n").replace("\r", "\n")).strip()


def _coerce_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items: list[str] = []
        for item in value:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                items.extend(_coerce_tuple(item))
                continue
            text = str(item).strip()
            if text:
                items.append(text)
        return tuple(items)
    text = str(value).strip()
    return (text,) if text else ()


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _infer_language(dataset: str, question: str, context: str) -> str:
    if dataset.lower() in _LONG_BENCH_ZH_DATASETS:
        return "zh"
    sample = f"{question}\n{context[:2000]}"
    cjk = sum(1 for ch in sample if "\u4e00" <= ch <= "\u9fff")
    return "zh" if cjk >= 20 else "en"


def _category(dataset: str) -> str:
    name = dataset.lower()
    if name in {"narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"}:
        return "single_doc_qa"
    if name in {"hotpotqa", "2wikimqa", "musique", "dureader"}:
        return "multi_doc_qa"
    if name in {"gov_report", "qmsum", "multi_news", "vcsum"}:
        return "summarization"
    if name in {"trec", "triviaqa", "samsum", "lsht"}:
        return "few_shot"
    if name in {"passage_count", "passage_retrieval_en", "passage_retrieval_zh"}:
        return "synthetic"
    if name in {"lcc", "repobench-p"}:
        return "code"
    return "unknown"


def _sample_from_payload(index: int, dataset: str, payload: Mapping[str, Any]) -> LongBenchSample:
    context = str(payload.get("context") or payload.get("document") or payload.get("passage") or "")
    question = str(payload.get("input") or payload.get("question") or payload.get("query") or payload.get("instruction") or "")
    task_id = str(payload.get("task_id") or payload.get("_id") or payload.get("id") or f"{dataset}_{index:05d}")
    language = str(payload.get("language") or _infer_language(dataset, question, context)).strip() or "en"
    return LongBenchSample(
        sample_index=index,
        task_id=task_id,
        dataset=dataset,
        question=question,
        context=context,
        answers=_coerce_tuple(payload.get("answers", payload.get("answer", payload.get("target")))),
        all_classes=_coerce_tuple(payload.get("all_classes") or payload.get("classes") or payload.get("choices")),
        language=language,
        length=_coerce_int(payload.get("length"), default=len(context)),
        category=str(payload.get("category") or _category(dataset)),
    )


def _iter_local_rows(root: Path, split: str, datasets: Sequence[str]):
    include = {item.lower() for item in datasets}
    search_roots = [root / split, root / "data" / split, root / "data", root]
    seen: set[Path] = set()
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in sorted(search_root.rglob("*.jsonl")):
            resolved = path.resolve()
            if resolved in seen or ".manifest" in resolved.name:
                continue
            seen.add(resolved)
            dataset = path.stem if path.stem.lower() not in {"test", "dev", "validation"} else path.parent.name
            if include and dataset.lower() not in include:
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield dataset, json.loads(line)


def _iter_hf_rows(split: str, datasets: Sequence[str]):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised in integration environments
        raise SystemExit("longbench requires the `datasets` package; install the rwkv dependency group.") from exc
    for dataset in datasets:
        for row in load_dataset(LONG_BENCH_SOURCE, dataset, split=split):
            yield dataset, dict(row)


def _round_robin(samples: Sequence[LongBenchSample]) -> list[LongBenchSample]:
    buckets: dict[str, list[LongBenchSample]] = {}
    for sample in samples:
        buckets.setdefault(sample.dataset, []).append(sample)
    ordered: list[LongBenchSample] = []
    names = sorted(buckets)
    index = 0
    while True:
        added = False
        for name in names:
            bucket = buckets[name]
            if index < len(bucket):
                original = bucket[index]
                ordered.append(LongBenchSample(sample_index=len(ordered), **{k: getattr(original, k) for k in original.__dataclass_fields__ if k != "sample_index"}))
                added = True
        if not added:
            return ordered
        index += 1


def load_samples(config: LongBenchRunConfig) -> list[LongBenchSample]:
    if config.limit is not None and int(config.limit) < 0:
        raise ValueError("limit must be non-negative")
    datasets = tuple(config.include_datasets or tuple(sorted(LONG_BENCH_DATASETS)))
    source_root = Path(config.source_root).expanduser().resolve() if config.source_root else None
    rows = (
        _iter_local_rows(source_root, config.split, datasets)
        if source_root is not None and source_root.exists()
        else _iter_hf_rows(config.split, datasets)
    )
    samples: list[LongBenchSample] = []
    if config.balance_by_dataset:
        all_samples = [_sample_from_payload(index, dataset, payload) for index, (dataset, payload) in enumerate(rows)]
        samples = _round_robin(all_samples)
        return samples[: int(config.limit)] if config.limit is not None else samples
    for dataset, payload in rows:
        if config.limit is not None and len(samples) >= int(config.limit):
            break
        samples.append(_sample_from_payload(len(samples), dataset, payload))
    return samples


def _middle_truncate(text: str, max_chars: int) -> str:
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


def build_prompt(sample: LongBenchSample, *, prompt_max_chars: int) -> str:
    context = normalize_text(sample.context)
    question = normalize_text(sample.question)
    lines = [
        "You are evaluating a long-context reading task.",
        "Answer the question using only the provided context.",
        "Return only the concise final answer. Do not include analysis, markdown, citations, or extra text.",
    ]
    if sample.all_classes:
        lines.append("If labels/classes are provided, answer with exactly one allowed label.")
        lines.append("Allowed labels/classes: " + ", ".join(sample.all_classes))
    if sample.language.lower().startswith("zh"):
        lines.append("If the question is Chinese, answer in Chinese.")
    lines.extend(["", "Context:", context, "", "Question:", question])
    instruction = "\n".join(lines)
    prompt = f"User: {instruction}\n\nAssistant:"
    if len(prompt) > prompt_max_chars:
        overhead = len(f"User: {' '.join(lines[:3])}\n\nContext:\n\nQuestion:\n{question}\n\nAssistant:")
        context_budget = max(256, prompt_max_chars - overhead - 16)
        prompt = f"User: {' '.join(lines[:3])}\n\nContext:\n{_middle_truncate(context, context_budget)}\n\nQuestion:\n{question}\n\nAssistant:"
    return prompt


def normalize_answer(text: str) -> str:
    body = _THINK_BLOCK_RE.sub("", str(text or "")).strip()
    if not body:
        return ""
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    for line in reversed(lines):
        match = _ANSWER_PREFIX_RE.match(line)
        if match:
            return line[match.end() :].strip().strip("`")
    return _ANSWER_PREFIX_RE.sub("", lines[-1]).strip().strip("`") if lines else ""


def _normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def exact_match(prediction: str, reference: str) -> bool:
    return _normalize_for_match(prediction) == _normalize_for_match(reference)


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _WORD_OR_CJK_RE.findall(_normalize_for_match(prediction))
    ref_tokens = _WORD_OR_CJK_RE.findall(_normalize_for_match(reference))
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


def score_answer(prediction: str, references: Sequence[str]) -> tuple[bool, float, str]:
    normalized_prediction = normalize_answer(prediction)
    best_exact = False
    best_f1 = 0.0
    best_ref = ""
    for reference in references:
        ref = str(reference or "").strip()
        if not ref:
            continue
        is_exact = exact_match(normalized_prediction, ref)
        f1 = token_f1(normalized_prediction, ref)
        if is_exact or f1 > best_f1:
            best_exact = is_exact
            best_f1 = f1
            best_ref = ref
        if is_exact:
            break
    return best_exact, best_f1, best_ref


def generate_completion(sample: LongBenchSample, config: LongBenchRunConfig) -> LongBenchResult:
    prompt = build_prompt(sample, prompt_max_chars=config.prompt_max_chars)
    completion = chat_completion(
        base_url=config.base_url,
        model=config.model,
        prompt=prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
    )
    answer = normalize_answer(completion)
    is_exact, f1, best_reference = score_answer(answer, sample.answers)
    passed = is_exact or f1 >= 0.8
    return LongBenchResult(
        sample_index=sample.sample_index,
        task_id=sample.task_id,
        prompt=prompt,
        completion=completion,
        answer=answer,
        reference_answer=best_reference or (sample.answers[0] if sample.answers else ""),
        is_passed=passed,
        fail_reason="" if passed else f"f1={f1:.4f}",
        f1=f1,
        exact_match=is_exact,
    )


def evaluate_samples(samples: Sequence[LongBenchSample], config: LongBenchRunConfig) -> list[LongBenchResult]:
    return [generate_completion(sample, config) for sample in samples]


def scoreboard_dataset_name(config: LongBenchRunConfig) -> str:
    dataset = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    return dataset


def job_id(config: LongBenchRunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def completion_sampling_config(config: LongBenchRunConfig) -> dict[str, Any]:
    return {"answer": {"temperature": config.temperature, "top_p": config.top_p, "max_new_tokens": config.max_tokens}}


def task_sampling_config(config: LongBenchRunConfig) -> dict[str, Any]:
    return {"avg_k": 1, "pass_ks": [1], "prompt_profile": "helicopter", "sampling_config": completion_sampling_config(config)}


def write_results(results: Sequence[LongBenchResult], *, config: LongBenchRunConfig, repo_root: Path) -> int:
    total = len(results)
    avg_f1 = sum(result.f1 for result in results) / total if total else 0.0
    exact_rate = sum(1 for result in results if result.exact_match) / total if total else 0.0
    pass_rate = sum(1 for result in results if result.is_passed) / total if total else 0.0
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
                        "avg@1": pass_rate,
                        "success_rate": pass_rate,
                        "longbench_avg_f1": avg_f1,
                        "longbench_exact_match_rate": exact_rate,
                    },
                ),
                repo_root=repo_root,
            )
        )
    )


def run_longbench(config: LongBenchRunConfig, *, repo_root: Path) -> dict[str, Any]:
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
        "avg_f1": sum(result.f1 for result in results) / len(results) if results else 0.0,
    }


def dry_run_summary(config: LongBenchRunConfig) -> dict[str, Any]:
    return {
        "benchmark": config.benchmark,
        "source": config.source_root or f"hf://{LONG_BENCH_SOURCE}",
        "split": config.split,
        "limit": config.limit,
        "include_datasets": list(config.include_datasets or tuple(sorted(LONG_BENCH_DATASETS))),
        "balance_by_dataset": config.balance_by_dataset,
        "base_url": config.base_url,
        "model": config.model,
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": job_id(config),
    }
