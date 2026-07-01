from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Sequence

from .openai_client import chat_completion
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


DEFAULT_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True, slots=True)
class ChoiceSet:
    labels: tuple[str, ...]
    texts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MultipleChoiceSample:
    sample_index: int
    question: str
    choices: ChoiceSet
    reference_answer: str


@dataclass(frozen=True, slots=True)
class MultipleChoiceResult:
    sample_index: int
    question: str
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
class MultipleChoiceRunConfig:
    base_url: str
    model: str
    benchmark: str
    dataset_name: str
    question_field: str
    choices_field: str
    answer_field: str
    limit: int | None = None
    dataset_config: str | None = None
    choice_fields: tuple[str, ...] = ()
    split: str = "test"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 32
    timeout_s: float = 600.0
    choice_labels: str = DEFAULT_LABELS
    prompt_template: str = "{question}\n\n{choices}\n\nAnswer with the letter only."
    scoreboard_dataset: str | None = None
    job_name: str = "multi_choice_plain"
    job_id: str | None = None
    runner: str = "helicopter_eval.multiple_choice"
    cot_mode: str = "NoCoT"


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().split()).lower()


def normalize_choices(raw_choices: Any, *, fallback_labels: str = DEFAULT_LABELS) -> ChoiceSet:
    labels: Sequence[Any] | None = None
    texts: Sequence[Any]
    if isinstance(raw_choices, dict):
        if "text" in raw_choices:
            texts = raw_choices["text"]
            labels = raw_choices.get("label")
        elif "choices" in raw_choices:
            texts = raw_choices["choices"]
            labels = raw_choices.get("labels")
        else:
            labels = tuple(raw_choices.keys())
            texts = tuple(raw_choices.values())
    elif isinstance(raw_choices, (list, tuple)):
        texts = raw_choices
    else:
        raise ValueError(f"unsupported choices field: {type(raw_choices).__name__}")

    text_values = tuple(str(item) for item in texts)
    if labels is None:
        label_values = tuple(fallback_labels[index] for index in range(len(text_values)))
    else:
        label_values = tuple(str(item).strip().upper() for item in labels)
    if len(label_values) != len(text_values):
        raise ValueError("choice labels and texts have different lengths")
    return ChoiceSet(labels=label_values, texts=text_values)


def reference_answer(raw_answer: Any, choices: ChoiceSet) -> str:
    if isinstance(raw_answer, int):
        if 0 <= raw_answer < len(choices.labels):
            return choices.labels[raw_answer]
        raise ValueError(f"answer index out of range: {raw_answer!r}")
    text = str(raw_answer).strip()
    upper = text.upper()
    if upper in choices.labels:
        return upper
    if text.isdigit():
        index = int(text)
        if 0 <= index < len(choices.labels):
            return choices.labels[index]
    normalized_answer = normalize_text(text)
    for label, choice_text in zip(choices.labels, choices.texts):
        if normalize_text(choice_text) == normalized_answer:
            return label
    raise ValueError(f"answer does not match any choice: {raw_answer!r}")


def build_prompt(question: str, choices: ChoiceSet, *, template: str) -> str:
    rendered_choices = "\n".join(f"{label}. {text}" for label, text in zip(choices.labels, choices.texts))
    return template.format(question=question.strip(), choices=rendered_choices)


def completion_answer(completion: str, labels: Sequence[str]) -> str:
    if not labels:
        return ""
    label_pattern = "|".join(re.escape(label.upper()) for label in sorted(labels, key=len, reverse=True))
    matches = re.findall(rf"(?<!\w)({label_pattern})(?!\w)", completion.upper())
    return matches[-1] if matches else ""


def scoreboard_dataset_name(config: MultipleChoiceRunConfig) -> str:
    dataset = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    return dataset


def job_id(config: MultipleChoiceRunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def completion_sampling_config(config: MultipleChoiceRunConfig) -> dict[str, Any]:
    return {
        "answer": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_tokens,
        }
    }


def task_sampling_config(config: MultipleChoiceRunConfig) -> dict[str, Any]:
    return {
        "avg_k": 1,
        "pass_ks": [1],
        "prompt_profile": "helicopter",
        "sampling_config": completion_sampling_config(config),
    }


def _iter_hf_rows(config: MultipleChoiceRunConfig):
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as exc:  # pragma: no cover - exercised in integration environments
        raise SystemExit("multiple-choice eval requires the `datasets` package; install the rwkv dependency group.") from exc

    if config.dataset_config == "*":
        config_names = tuple(name for name in get_dataset_config_names(config.dataset_name) if name and name != "default")
    else:
        config_names = (config.dataset_config,)
    for config_name in config_names:
        if config_name is None:
            dataset = load_dataset(config.dataset_name, split=config.split)
        else:
            dataset = load_dataset(config.dataset_name, config_name, split=config.split)
        for item in dataset:
            yield item


def _row_choices(item: Any, config: MultipleChoiceRunConfig) -> ChoiceSet:
    if config.choice_fields:
        return normalize_choices([item[field] for field in config.choice_fields], fallback_labels=config.choice_labels)
    return normalize_choices(item[config.choices_field], fallback_labels=config.choice_labels)


def load_samples(config: MultipleChoiceRunConfig) -> list[MultipleChoiceSample]:
    if config.limit is not None and int(config.limit) < 0:
        raise ValueError("limit must be non-negative")
    limit = None if config.limit is None else int(config.limit)
    samples: list[MultipleChoiceSample] = []
    for index, item in enumerate(_iter_hf_rows(config)):
        if limit is not None and index >= limit:
            break
        choices = _row_choices(item, config)
        samples.append(
            MultipleChoiceSample(
                sample_index=index,
                question=str(item[config.question_field]),
                choices=choices,
                reference_answer=reference_answer(item[config.answer_field], choices),
            )
        )
    return samples


def generate_completion(sample: MultipleChoiceSample, config: MultipleChoiceRunConfig) -> MultipleChoiceResult:
    prompt = build_prompt(sample.question, sample.choices, template=config.prompt_template)
    completion = chat_completion(
        base_url=config.base_url,
        model=config.model,
        prompt=prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
    )
    answer = completion_answer(completion, sample.choices.labels)
    is_passed = answer == sample.reference_answer
    return MultipleChoiceResult(
        sample_index=sample.sample_index,
        question=sample.question,
        prompt=prompt,
        completion=completion,
        answer=answer,
        reference_answer=sample.reference_answer,
        is_passed=is_passed,
        fail_reason="" if is_passed else f"expected {sample.reference_answer}, got {answer}",
    )


def evaluate_samples(
    samples: Sequence[MultipleChoiceSample],
    config: MultipleChoiceRunConfig,
) -> list[MultipleChoiceResult]:
    return [generate_completion(sample, config) for sample in samples]


def write_results(results: Sequence[MultipleChoiceResult], *, config: MultipleChoiceRunConfig, repo_root: Path) -> int:
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
                ),
                repo_root=repo_root,
            )
        )
    )


def run_multiple_choice(config: MultipleChoiceRunConfig, *, repo_root: Path) -> dict[str, Any]:
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


def dry_run_summary(config: MultipleChoiceRunConfig) -> dict[str, Any]:
    return {
        "benchmark": config.benchmark,
        "hf_dataset": config.dataset_name,
        "hf_config": config.dataset_config,
        "split": config.split,
        "limit": config.limit,
        "base_url": config.base_url,
        "model": config.model,
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": job_id(config),
    }
