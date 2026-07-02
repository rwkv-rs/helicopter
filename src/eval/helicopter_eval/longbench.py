from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import random
import re
import string
from typing import Any, Mapping, Sequence
import unicodedata

from .openai_client import chat_completion, text_completion
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


LONG_BENCH_SOURCE = "THUDM/LongBench"
LONGBENCH_STOP_SUFFIXES = ("\nUser:", "\nSystem:", "\nAssistant:")
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
    metadata: dict[str, Any] | None = None


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
    metadata: dict[str, Any] | None = None

    def to_scoreboard(self) -> ScoreboardEvalResult:
        return ScoreboardEvalResult(
            sample_index=self.sample_index,
            prompt=self.prompt,
            completion=self.completion,
            answer=self.answer,
            reference_answer=self.reference_answer,
            is_passed=self.is_passed,
            fail_reason=self.fail_reason,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class LongBenchLoadResult:
    samples: list[LongBenchSample]
    source_count: int
    requested_sample_size: int | None
    sample_applied: bool


@dataclass(frozen=True, slots=True)
class LongBenchRunConfig:
    base_url: str
    model: str
    benchmark: str = "longbench"
    limit: int | None = None
    sample_size: int | None = None
    sample_seed: int = 42
    split: str = "test"
    source_root: str | None = None
    source_path: str | None = None
    include_datasets: tuple[str, ...] = ()
    balance_by_dataset: bool = False
    infer_protocol: str = "chat"
    temperature: float = 0.0
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed_requests: bool = False
    stop_suffixes: tuple[str, ...] = ()
    max_tokens: int = 128
    timeout_s: float = 600.0
    prompt_max_chars: int = 8192
    scoreboard_dataset: str | None = None
    job_name: str = "function_longbench"
    job_id: str | None = None
    runner: str = "helicopter_eval.longbench"
    cot_mode: str = "CoT"


def normalize_text(value: str) -> str:
    normalized = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = normalized.strip()
    return re.sub(r"\n{2,}", "\n", normalized)


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


def _sample_metadata(
    payload: Mapping[str, Any],
    *,
    original_sample_index: int,
    dataset_sample_index: int,
    dataset: str,
    task_id: str,
    config: LongBenchRunConfig,
) -> dict[str, Any]:
    source_id = str(payload.get("source_id") or task_id)
    try:
        original_index = int(payload.get("original_sample_index", original_sample_index))
    except (TypeError, ValueError):
        original_index = original_sample_index
    try:
        dataset_index = int(payload.get("dataset_sample_index", dataset_sample_index))
    except (TypeError, ValueError):
        dataset_index = dataset_sample_index
    metadata = {
        "benchmark": config.benchmark,
        "split": config.split,
        "original_sample_index": original_index,
        "dataset_sample_index": dataset_index,
        "source_id": source_id,
        "longbench_dataset": dataset,
    }
    for source_key, target_key in (
        ("length", "length"),
        ("language", "language"),
        ("category", "category"),
        ("all_classes", "all_classes"),
    ):
        value = payload.get(source_key)
        if value is not None:
            metadata[target_key] = value
    return metadata


def sample_to_manifest_row(sample: LongBenchSample) -> dict[str, Any]:
    metadata = dict(sample.metadata or {})
    row: dict[str, Any] = {
        "task_id": sample.task_id,
        "dataset": sample.dataset,
        "input": sample.question,
        "context": sample.context,
        "answers": list(sample.answers),
        "all_classes": list(sample.all_classes),
        "language": sample.language,
        "length": sample.length,
        "category": sample.category,
        "manifest_sample_index": sample.sample_index,
    }
    for key in ("original_sample_index", "dataset_sample_index", "source_id"):
        if key in metadata:
            row[key] = metadata[key]
    return row


def _samples_identity_sha256(samples: Sequence[LongBenchSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        metadata = sample.metadata or {}
        original_index = metadata.get("original_sample_index", sample.sample_index)
        source_id = metadata.get("source_id", sample.task_id)
        digest.update(f"{sample.sample_index}\t{sample.dataset}\t{source_id}\t{original_index}\n".encode("utf-8"))
    return digest.hexdigest()


def write_manifest(samples: Sequence[LongBenchSample], path: str | Path) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample_to_manifest_row(sample), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return target.resolve()


def _sample_from_payload(
    index: int,
    dataset: str,
    payload: Mapping[str, Any],
    *,
    config: LongBenchRunConfig,
    dataset_sample_index: int,
) -> LongBenchSample:
    context = str(payload.get("context") or payload.get("document") or payload.get("passage") or "")
    question = str(payload.get("input") or payload.get("question") or payload.get("query") or payload.get("instruction") or "")
    task_id = str(payload.get("task_id") or payload.get("_id") or payload.get("id") or f"{dataset}_{index:05d}")
    language = str(payload.get("language") or _infer_language(dataset, question, context)).strip() or "en"
    category = str(payload.get("category") or _category(dataset))
    length = _coerce_int(payload.get("length"), default=len(context))
    return LongBenchSample(
        sample_index=index,
        task_id=task_id,
        dataset=dataset,
        question=question,
        context=context,
        answers=_coerce_tuple(payload.get("answers", payload.get("answer", payload.get("target")))),
        all_classes=_coerce_tuple(payload.get("all_classes") or payload.get("classes") or payload.get("choices")),
        language=language,
        length=length,
        category=category,
        metadata={
            **_sample_metadata(
                {**dict(payload), "language": language, "category": category, "length": length},
                original_sample_index=index,
                dataset_sample_index=dataset_sample_index,
                dataset=dataset,
                task_id=task_id,
                config=config,
            )
        },
    )


def _infer_local_dataset(path: Path) -> str:
    return path.stem if path.stem.lower() not in {"test", "dev", "validation"} else path.parent.name


def _iter_jsonl_path(path: Path, datasets: Sequence[str]):
    include = {item.lower() for item in datasets}
    default_dataset = _infer_local_dataset(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            dataset = str(payload.get("dataset") or payload.get("subset") or default_dataset).strip() or default_dataset
            if include and dataset.lower() not in include:
                continue
            yield dataset, payload


def _iter_local_rows(root: Path, split: str, datasets: Sequence[str]):
    if root.is_file():
        yield from _iter_jsonl_path(root, datasets)
        return
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
            yield from _iter_jsonl_path(path, datasets)


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
                ordered.append(replace(original, sample_index=len(ordered)))
                added = True
        if not added:
            return ordered
        index += 1


def _load_samples(config: LongBenchRunConfig) -> LongBenchLoadResult:
    if config.limit is not None and int(config.limit) < 0:
        raise ValueError("limit must be non-negative")
    if config.sample_size is not None and int(config.sample_size) < 0:
        raise ValueError("sample_size must be non-negative")
    if config.limit is not None and config.sample_size is not None:
        raise ValueError("limit and sample_size are mutually exclusive")
    datasets = tuple(config.include_datasets or tuple(sorted(LONG_BENCH_DATASETS)))
    source_path = Path(config.source_path).expanduser().resolve() if config.source_path else None
    source_root = Path(config.source_root).expanduser().resolve() if config.source_root else None
    if source_path is not None:
        if not source_path.exists():
            raise FileNotFoundError(f"LongBench source_path does not exist: {source_path}")
        rows = _iter_local_rows(source_path, config.split, datasets)
    elif source_root is not None and source_root.exists():
        rows = _iter_local_rows(source_root, config.split, datasets)
    else:
        rows = _iter_hf_rows(config.split, datasets)
    requested_sample_size = None if config.sample_size is None else int(config.sample_size)
    dataset_indexes: dict[str, int] = {}
    all_samples: list[LongBenchSample] = []
    for dataset, payload in rows:
        if (
            config.limit is not None
            and requested_sample_size is None
            and not config.balance_by_dataset
            and len(all_samples) >= int(config.limit)
        ):
            break
        dataset_sample_index = dataset_indexes.get(dataset, 0)
        dataset_indexes[dataset] = dataset_sample_index + 1
        all_samples.append(
            _sample_from_payload(
                len(all_samples),
                dataset,
                payload,
                config=config,
                dataset_sample_index=dataset_sample_index,
            )
        )
    if config.balance_by_dataset:
        samples = _round_robin(all_samples)
    else:
        samples = all_samples
    if config.limit is not None:
        samples = samples[: int(config.limit)]
    source_count = len(samples)
    sample_applied = False
    if requested_sample_size is not None and requested_sample_size < len(samples):
        rng = random.Random(int(config.sample_seed))
        samples = sorted(rng.sample(samples, requested_sample_size), key=lambda item: item.sample_index)
        samples = [replace(sample, sample_index=index) for index, sample in enumerate(samples)]
        sample_applied = True
    return LongBenchLoadResult(
        samples=samples,
        source_count=source_count,
        requested_sample_size=requested_sample_size,
        sample_applied=sample_applied,
    )


def load_samples(config: LongBenchRunConfig) -> list[LongBenchSample]:
    return _load_samples(config).samples


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
    instruction = normalize_text("\n".join(lines))
    prompt = f"User: {instruction}\n\nAssistant:"
    if len(prompt) > prompt_max_chars:
        empty_lines = list(lines)
        context_index = len(empty_lines) - 4
        empty_lines[context_index] = ""
        overhead = len(f"User: {normalize_text(chr(10).join(empty_lines))}\n\nAssistant:")
        context_budget = max(256, prompt_max_chars - overhead - 16)
        fitted_context = _middle_truncate(context, context_budget)
        truncated_lines = list(lines)
        truncated_lines[context_index] = fitted_context
        prompt = f"User: {normalize_text(chr(10).join(truncated_lines))}\n\nAssistant:"
        if len(prompt) > prompt_max_chars:
            extra = len(prompt) - prompt_max_chars
            fitted_context = _middle_truncate(fitted_context, max(0, len(fitted_context) - extra - 32))
            truncated_lines[context_index] = fitted_context
            prompt = f"User: {normalize_text(chr(10).join(truncated_lines))}\n\nAssistant:"
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


def _sample_repeat_seed(sample_index: int, *, repeat_index: int = 0, pass_index: int = 0, stage: int = 1) -> int:
    sample = int(sample_index)
    repeat = int(repeat_index)
    passed = int(pass_index)
    stage_id = int(stage)
    if sample < 0 or repeat < 0 or passed < 0 or stage_id < 0:
        raise ValueError("sample_index/repeat_index/pass_index/stage must be non-negative")
    if sample >= (1 << 31) or repeat >= (1 << 24) or passed >= (1 << 16) or stage_id >= (1 << 8):
        raise ValueError("sample seed component exceeds the rwkv-skills compact seed range")
    base_seed = (sample << 32) | (repeat << 8) | stage_id
    if passed == 0:
        return base_seed
    return (base_seed ^ (((passed + 1) * 0x9E3779B97F4A7C15) & 0x7FFFFFFFFFFFFFFF)) & 0x7FFFFFFFFFFFFFFF


def generate_completion(sample: LongBenchSample, config: LongBenchRunConfig) -> LongBenchResult:
    prompt = build_prompt(sample, prompt_max_chars=config.prompt_max_chars)
    seed = _sample_repeat_seed(sample.sample_index, stage=1) if config.seed_requests else None
    stop = list(config.stop_suffixes) if config.stop_suffixes else None
    if config.infer_protocol == "completions":
        completion = text_completion(
            base_url=config.base_url,
            model=config.model,
            prompt=prompt,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            seed=seed,
            stop=stop,
        )
    else:
        completion = chat_completion(
            base_url=config.base_url,
            model=config.model,
            prompt=prompt,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            seed=seed,
            stop=stop,
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
        metadata=sample.metadata,
    )


def evaluate_samples(samples: Sequence[LongBenchSample], config: LongBenchRunConfig) -> list[LongBenchResult]:
    return [generate_completion(sample, config) for sample in samples]


def scoreboard_dataset_name(config: LongBenchRunConfig, *, sample_applied: bool | None = None) -> str:
    dataset = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    if config.sample_size is not None and (sample_applied is not False or config.source_path is not None):
        dataset = f"{dataset}_sample{int(config.sample_size)}_seed{int(config.sample_seed)}"
    return dataset


def job_id(config: LongBenchRunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def completion_sampling_config(config: LongBenchRunConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_new_tokens": config.max_tokens,
        "infer_protocol": config.infer_protocol,
    }
    if config.presence_penalty:
        payload["presence_penalty"] = config.presence_penalty
    if config.frequency_penalty:
        payload["frequency_penalty"] = config.frequency_penalty
    if config.seed_requests:
        payload["seed_policy"] = "rwkv_skills_sample_repeat_seed_stage1"
    if config.stop_suffixes:
        payload["stop"] = list(config.stop_suffixes)
    return {"answer": payload}


def task_sampling_config(config: LongBenchRunConfig) -> dict[str, Any]:
    return {"avg_k": 1, "pass_ks": [1], "prompt_profile": "helicopter", "sampling_config": completion_sampling_config(config)}


def write_results(
    results: Sequence[LongBenchResult],
    *,
    config: LongBenchRunConfig,
    repo_root: Path,
    dataset_name: str | None = None,
) -> int:
    total = len(results)
    avg_f1 = sum(result.f1 for result in results) / total if total else 0.0
    exact_rate = sum(1 for result in results if result.exact_match) / total if total else 0.0
    pass_rate = sum(1 for result in results if result.is_passed) / total if total else 0.0
    return int(
        asyncio.run(
            write_scoreboard_results(
                [result.to_scoreboard() for result in results],
                config=ScoreboardWriteConfig(
                    dataset=dataset_name or scoreboard_dataset_name(config),
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
    loaded = _load_samples(config)
    samples = loaded.samples
    results = evaluate_samples(samples, config)
    dataset_name = scoreboard_dataset_name(config, sample_applied=loaded.sample_applied)
    task_id = write_results(results, config=config, repo_root=repo_root, dataset_name=dataset_name)
    passed = sum(1 for result in results if result.is_passed)
    return {
        "task_id": task_id,
        "benchmark": config.benchmark,
        "dataset": dataset_name,
        "model": config.model,
        "total": len(results),
        "passed": passed,
        "avg_f1": sum(result.f1 for result in results) / len(results) if results else 0.0,
        "source_total": loaded.source_count,
        "sample_requested": loaded.requested_sample_size,
        "sample_applied": loaded.sample_applied,
    }


def export_sample_manifest(config: LongBenchRunConfig, path: str | Path) -> dict[str, Any]:
    loaded = _load_samples(config)
    target = write_manifest(loaded.samples, path)
    dataset_name = scoreboard_dataset_name(config, sample_applied=loaded.sample_applied)
    return {
        "benchmark": config.benchmark,
        "dataset": dataset_name,
        "manifest_path": str(target),
        "total": len(loaded.samples),
        "source_total": loaded.source_count,
        "sample_requested": loaded.requested_sample_size,
        "sample_applied": loaded.sample_applied,
        "sample_identity_sha256": _samples_identity_sha256(loaded.samples),
    }


def dry_run_summary(config: LongBenchRunConfig) -> dict[str, Any]:
    payload = {
        "benchmark": config.benchmark,
        "source": config.source_path or config.source_root or f"hf://{LONG_BENCH_SOURCE}",
        "split": config.split,
        "limit": config.limit,
        "include_datasets": list(config.include_datasets or tuple(sorted(LONG_BENCH_DATASETS))),
        "balance_by_dataset": config.balance_by_dataset,
        "infer_protocol": config.infer_protocol,
        "base_url": config.base_url,
        "model": config.model,
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": job_id(config),
    }
    if config.sample_size is not None:
        payload["sample_size"] = config.sample_size
        payload["sample_seed"] = config.sample_seed
    if config.presence_penalty:
        payload["presence_penalty"] = config.presence_penalty
    if config.frequency_penalty:
        payload["frequency_penalty"] = config.frequency_penalty
    if config.seed_requests:
        payload["seed_requests"] = True
    if config.stop_suffixes:
        payload["stop_suffixes"] = list(config.stop_suffixes)
    return payload
