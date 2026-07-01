from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

from .openai_client import chat_completion
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


FLORES_DATASET_ID = "openlanguagedata/flores_plus"
WMT24PP_DATASET_ID = "google/wmt24pp"
FLORES_LANGUAGE_CONFIGS = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
}
LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
}
DEFAULT_FLORES_SOURCE_LANGUAGES = ("en", "de", "es", "fr", "it", "ja")
DEFAULT_FLORES_TARGET_LANGUAGES = ("en", "de", "es", "fr", "it", "ja")
DEFAULT_WMT24PP_TARGET_LANGUAGES = ("de_DE", "es_MX", "fr_FR", "it_IT", "ja_JP")


@dataclass(frozen=True, slots=True)
class TranslationSample:
    sample_index: int
    task_id: str
    source_text: str
    reference_translation: str
    source_language: str
    target_language: str
    source_language_name: str
    target_language_name: str


@dataclass(frozen=True, slots=True)
class TranslationResult:
    sample_index: int
    task_id: str
    prompt: str
    completion: str
    answer: str
    reference_answer: str
    score: float
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
class TranslationRunConfig:
    base_url: str
    model: str
    benchmark: str
    source_type: str
    dataset_name: str
    limit: int | None = None
    split: str = "test"
    source_languages: tuple[str, ...] = DEFAULT_FLORES_SOURCE_LANGUAGES
    target_languages: tuple[str, ...] = DEFAULT_FLORES_TARGET_LANGUAGES
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    timeout_s: float = 600.0
    pass_threshold: float = 0.5
    scoreboard_dataset: str | None = None
    job_name: str = "translation_chrf"
    job_id: str | None = None
    runner: str = "helicopter_eval.translation"
    cot_mode: str = "NoCoT"


def load_samples(config: TranslationRunConfig) -> list[TranslationSample]:
    if config.limit is not None and int(config.limit) < 0:
        raise ValueError("limit must be non-negative")
    if config.source_type == "hf_flores200":
        return _load_flores_samples(config)
    if config.source_type == "hf_wmt24pp":
        return _load_wmt24pp_samples(config)
    raise ValueError(f"unsupported translation source_type: {config.source_type}")


def _load_flores_samples(config: TranslationRunConfig) -> list[TranslationSample]:
    if config.split not in {"dev", "devtest"}:
        raise ValueError("flores200 only supports dev or devtest split")
    rows: list[TranslationSample] = []
    cache: dict[str, list[str]] = {}
    for src_lang in config.source_languages:
        if src_lang not in FLORES_LANGUAGE_CONFIGS:
            raise ValueError(f"unsupported flores200 source language: {src_lang}")
        for tgt_lang in config.target_languages:
            if tgt_lang not in FLORES_LANGUAGE_CONFIGS:
                raise ValueError(f"unsupported flores200 target language: {tgt_lang}")
            if src_lang == tgt_lang:
                continue
            src_texts = cache.setdefault(src_lang, _load_flores_language(src_lang, config.split))
            tgt_texts = cache.setdefault(tgt_lang, _load_flores_language(tgt_lang, config.split))
            for row_index, (source_text, target_text) in enumerate(zip(src_texts, tgt_texts, strict=True)):
                rows.append(
                    TranslationSample(
                        sample_index=len(rows),
                        task_id=f"flores200__{src_lang}_{tgt_lang}_{row_index:05d}",
                        source_text=source_text,
                        reference_translation=target_text,
                        source_language=src_lang,
                        target_language=tgt_lang,
                        source_language_name=_lang_display(src_lang),
                        target_language_name=_lang_display(tgt_lang),
                    )
                )
                if config.limit is not None and len(rows) >= int(config.limit):
                    return rows
    return rows


def _load_wmt24pp_samples(config: TranslationRunConfig) -> list[TranslationSample]:
    if config.split != "test":
        raise ValueError("wmt24pp only provides test split")
    rows: list[TranslationSample] = []
    for target_language in config.target_languages:
        for row_index, (source_text, target_text) in enumerate(_load_wmt24pp_pair(target_language)):
            rows.append(
                TranslationSample(
                    sample_index=len(rows),
                    task_id=f"wmt24pp__en_{target_language}_{row_index:05d}",
                    source_text=source_text,
                    reference_translation=target_text,
                    source_language="en",
                    target_language=target_language,
                    source_language_name=_lang_display("en"),
                    target_language_name=_lang_display(target_language),
                )
            )
            if config.limit is not None and len(rows) >= int(config.limit):
                return rows
    return rows


def _load_flores_language(lang: str, split: str) -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset(FLORES_DATASET_ID, FLORES_LANGUAGE_CONFIGS[lang], split=split)
    return [str(example["text"]) for example in dataset]


def _load_wmt24pp_pair(target_language: str) -> list[tuple[str, str]]:
    from datasets import load_dataset

    dataset = load_dataset(WMT24PP_DATASET_ID, f"en-{target_language}", split="train")
    return [(str(example["source"]), str(example["target"])) for example in dataset]


def build_prompt(sample: TranslationSample) -> str:
    return (
        f"Translate the following text from {sample.source_language_name} to {sample.target_language_name}.\n"
        "Return only the translation, with no explanation, no markdown, and no surrounding quotes.\n\n"
        f"Text:\n{sample.source_text}\n\n"
        "Translation:"
    )


def score_completion(completion: str, reference: str, *, threshold: float = 0.5) -> tuple[float, bool, str]:
    answer = normalize_translation(completion)
    target = normalize_translation(reference)
    score = chrf_score(answer, target)
    passed = score >= float(threshold)
    reason = "" if passed else f"chrf_below_threshold:{score:.4f}<{float(threshold):.4f}"
    return score, passed, reason


def generate_completion(sample: TranslationSample, config: TranslationRunConfig) -> TranslationResult:
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
    score, passed, fail_reason = score_completion(
        completion,
        sample.reference_translation,
        threshold=config.pass_threshold,
    )
    return TranslationResult(
        sample_index=sample.sample_index,
        task_id=sample.task_id,
        prompt=prompt,
        completion=completion,
        answer=normalize_translation(completion),
        reference_answer=sample.reference_translation,
        score=score,
        is_passed=passed,
        fail_reason=fail_reason,
    )


def evaluate_samples(samples: Sequence[TranslationSample], config: TranslationRunConfig) -> list[TranslationResult]:
    return [generate_completion(sample, config) for sample in samples]


def scoreboard_dataset_name(config: TranslationRunConfig) -> str:
    dataset = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    return dataset


def job_id(config: TranslationRunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def completion_sampling_config(config: TranslationRunConfig) -> dict[str, Any]:
    return {
        "translation": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_tokens,
        }
    }


def task_sampling_config(config: TranslationRunConfig) -> dict[str, Any]:
    return {
        "avg_k": 1,
        "pass_ks": [1],
        "prompt_profile": "helicopter_translation_direct",
        "metric": "chrf",
        "pass_threshold": config.pass_threshold,
        "sampling_config": completion_sampling_config(config),
    }


def write_results(results: Sequence[TranslationResult], *, config: TranslationRunConfig, repo_root: Path) -> int:
    avg_chrf = sum(result.score for result in results) / len(results) if results else 0.0
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
                extra_metrics={"chrf": avg_chrf},
            ),
            repo_root=repo_root,
        )
    )
    return int(task_id)


def run_translation(config: TranslationRunConfig, *, repo_root: Path) -> dict[str, Any]:
    samples = load_samples(config)
    results = evaluate_samples(samples, config)
    task_id = write_results(results, config=config, repo_root=repo_root)
    passed = sum(1 for result in results if result.is_passed)
    avg_chrf = sum(result.score for result in results) / len(results) if results else 0.0
    return {
        "task_id": task_id,
        "benchmark": config.benchmark,
        "dataset": scoreboard_dataset_name(config),
        "model": config.model,
        "total": len(results),
        "passed": passed,
        "accuracy": passed / len(results) if results else 0.0,
        "chrf": avg_chrf,
    }


def dry_run_summary(config: TranslationRunConfig) -> dict[str, Any]:
    return {
        "benchmark": config.benchmark,
        "hf_dataset": config.dataset_name,
        "source_type": config.source_type,
        "split": config.split,
        "limit": config.limit,
        "source_languages": list(config.source_languages),
        "target_languages": list(config.target_languages),
        "base_url": config.base_url,
        "model": config.model,
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": job_id(config),
        "metric": "chrf",
        "pass_threshold": config.pass_threshold,
    }


def normalize_translation(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"(?is)^```(?:\w+)?\s*(.*?)\s*```$", r"\1", value).strip()
    value = re.sub(r"(?i)^(translation|answer)\s*:\s*", "", value).strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1].strip()
    return " ".join(value.split())


def chrf_score(prediction: str, reference: str, *, max_order: int = 6, beta: float = 2.0) -> float:
    pred = normalize_translation(prediction)
    ref = normalize_translation(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    precisions: list[float] = []
    recalls: list[float] = []
    for order in range(1, max(1, int(max_order)) + 1):
        pred_counts = _char_ngram_counts(pred, order)
        ref_counts = _char_ngram_counts(ref, order)
        overlap = sum((pred_counts & ref_counts).values())
        pred_total = sum(pred_counts.values())
        ref_total = sum(ref_counts.values())
        precisions.append(overlap / pred_total if pred_total else 0.0)
        recalls.append(overlap / ref_total if ref_total else 0.0)
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    beta_sq = float(beta) ** 2
    return (1 + beta_sq) * precision * recall / ((beta_sq * precision) + recall)


def _char_ngram_counts(text: str, order: int) -> Counter[str]:
    normalized = " ".join(str(text).split())
    if len(normalized) < order:
        return Counter()
    return Counter(normalized[index : index + order] for index in range(len(normalized) - order + 1))


def _lang_display(lang: str) -> str:
    base = str(lang).split("_", 1)[0]
    return LANGUAGE_NAMES.get(base, str(lang))


__all__ = [
    "TranslationRunConfig",
    "TranslationSample",
    "build_prompt",
    "chrf_score",
    "dry_run_summary",
    "load_samples",
    "normalize_translation",
    "run_translation",
    "score_completion",
]
