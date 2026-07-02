from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
import io
import json
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence
import urllib.request
import xml.etree.ElementTree as ET

from .openai_client import chat_completion
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_POLYMATH_CONFIG_NAMES = (
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


@dataclass(frozen=True, slots=True)
class FreeResponseSample:
    sample_index: int
    question: str
    reference_answer: str
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class FreeResponseResult:
    sample_index: int
    question: str
    prompt: str
    completion: str
    answer: str
    reference_answer: str
    is_passed: bool
    fail_reason: str
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
class FreeResponseRunConfig:
    base_url: str
    model: str
    benchmark: str
    dataset_name: str
    question_field: str
    answer_field: str
    limit: int | None = None
    dataset_config: str | None = None
    source_type: str = "hf"
    source_url: str | None = None
    source_urls: tuple[str, ...] = ()
    source_path: str | None = None
    row_adapter: str | None = None
    split: str = "test"
    sample_size: int | None = None
    sample_seed: int = 42
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    timeout_s: float = 600.0
    answer_marker: str | None = "####"
    reference_answer_overrides: Mapping[str, str] | None = None
    prompt_template: str = "Question: {question}\nAnswer:"
    scoreboard_dataset: str | None = None
    job_name: str = "free_response_judge"
    job_id: str | None = None
    runner: str = "helicopter_eval.free_response"
    cot_mode: str = "CoT"


def normalize_number(value: str) -> str:
    matches = _NUMBER_RE.findall(value.replace(",", ""))
    if not matches:
        return value.strip()
    number = matches[-1].replace(",", "").strip()
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return number


def extract_marked_answer(text: str, marker: str | None) -> str:
    if marker and marker in text:
        text = text.rsplit(marker, 1)[1]
    return normalize_number(text)


def build_prompt(question: str, *, template: str = "Question: {question}\nAnswer:") -> str:
    return template.format(question=question.strip())


def completion_answer(completion: str, *, marker: str | None = "####") -> str:
    return extract_marked_answer(completion, marker)


def scoreboard_dataset_name(config: FreeResponseRunConfig) -> str:
    dataset = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    if config.sample_size is not None:
        dataset = f"{dataset}_sample{int(config.sample_size)}_seed{int(config.sample_seed)}"
    return dataset


def job_id(config: FreeResponseRunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def task_sampling_config(config: FreeResponseRunConfig) -> dict[str, Any]:
    return {
        "avg_k": 1,
        "pass_ks": [1],
        "prompt_profile": "helicopter",
        "sampling_config": completion_sampling_config(config),
    }


def completion_sampling_config(config: FreeResponseRunConfig) -> dict[str, Any]:
    return {
        "answer": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_tokens,
        }
    }


def _extract_answer_from_solution(text: str) -> str | None:
    matches = _BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip()
    match = re.search(r"The final answer is (.+)$", text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    for line in reversed(text.splitlines()):
        value = line.strip()
        if value:
            return value
    return None


def _iter_qwen_math_rows(config: FreeResponseRunConfig):
    url = (
        "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/refs/heads/main/"
        f"evaluation/data/{config.dataset_name}/{config.split}.jsonl"
    )
    with urllib.request.urlopen(url, timeout=config.timeout_s) as response:
        for raw_line in response:
            payload = json.loads(raw_line.decode("utf-8"))
            if "answer" in payload:
                payload["expected_answer"] = payload.pop("answer")
            if "problem" not in payload and "question" in payload:
                payload["problem"] = payload.pop("question")
            if config.dataset_name == "olympiadbench" and "final_answer" in payload:
                answers = payload.pop("final_answer")
                if isinstance(answers, (list, tuple)) and answers:
                    payload["expected_answer"] = str(answers[0]).strip("$")
            if config.dataset_name == "minerva_math" and "solution" in payload:
                extracted = _extract_answer_from_solution(str(payload["solution"]))
                if extracted is not None:
                    payload["expected_answer"] = extracted
            yield payload


def _source_urls(config: FreeResponseRunConfig) -> tuple[str, ...]:
    if config.source_urls:
        return config.source_urls
    if config.source_url:
        return (config.source_url,)
    return ()


def _read_url_text(url: str, *, timeout_s: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return response.read().decode("utf-8")


def _iter_url_jsonl_rows(config: FreeResponseRunConfig):
    urls = _source_urls(config)
    if not urls:
        raise ValueError("url_jsonl source requires source_url")
    for url in urls:
        with urllib.request.urlopen(url, timeout=config.timeout_s) as response:
            for source_index, raw_line in enumerate(response):
                line = raw_line.decode("utf-8").strip()
                if line:
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        payload.setdefault("_source_index", source_index)
                        payload.setdefault("_source_url", url)
                    yield payload


def _iter_url_json_rows(config: FreeResponseRunConfig):
    if not config.source_url:
        raise ValueError("url_json source requires source_url")
    with urllib.request.urlopen(config.source_url, timeout=config.timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("url_json source must return a list")
    yield from payload


def _iter_url_csv_rows(config: FreeResponseRunConfig):
    if not config.source_url:
        raise ValueError("url_csv source requires source_url")
    reader = csv.DictReader(io.StringIO(_read_url_text(config.source_url, timeout_s=config.timeout_s)))
    yield from reader


def _iter_url_xml_rows(config: FreeResponseRunConfig):
    if not config.source_url:
        raise ValueError("url_xml source requires source_url")
    root = ET.fromstring(_read_url_text(config.source_url, timeout_s=config.timeout_s))
    yield from root.iter("Problem")


def _iter_package_jsonl_rows(config: FreeResponseRunConfig):
    if not config.source_path:
        raise ValueError("package_jsonl source requires source_path")
    resource = resources.files("helicopter_eval").joinpath(config.source_path)
    with resource.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _iter_hf_rows(config: FreeResponseRunConfig):
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as exc:  # pragma: no cover - exercised in integration environments
        raise SystemExit("free-response eval requires the `datasets` package; install the rwkv dependency group.") from exc

    if config.dataset_config == "*":
        def _rows():
            for config_name in sorted(
                name for name in get_dataset_config_names(config.dataset_name) if name and name != "default"
            ):
                yield from load_dataset(config.dataset_name, config_name, split=config.split)

        return _rows()
    return iter(load_dataset(config.dataset_name, config.dataset_config, split=config.split))


def _polymath_source_splits(split: str) -> tuple[str, ...]:
    if split == "all":
        return ("top", "high", "medium", "low")
    if split in {"top", "high", "medium", "low"}:
        return (split,)
    raise ValueError("polymath only supports all/top/high/medium/low split")


def _iter_polymath_rows(config: FreeResponseRunConfig):
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised in integration environments
        raise SystemExit(
            "polymath eval requires the `huggingface_hub` and `pyarrow` packages; "
            "install the rwkv dependency group."
        ) from exc

    for config_name in _POLYMATH_CONFIG_NAMES:
        for source_split in _polymath_source_splits(config.split):
            path = hf_hub_download(
                repo_id=config.dataset_name,
                filename=f"{config_name}/{source_split}.parquet",
                repo_type="dataset",
            )
            for index, row in enumerate(pq.read_table(path).to_pylist()):
                payload = dict(row)
                payload["_polymath_language"] = config_name
                payload["_polymath_difficulty"] = source_split
                payload["_polymath_index"] = index
                yield payload


def _iter_rows(config: FreeResponseRunConfig):
    if config.source_type == "hf":
        return _iter_hf_rows(config)
    if config.source_type == "qwen_math":
        return _iter_qwen_math_rows(config)
    if config.source_type == "url_jsonl":
        return _iter_url_jsonl_rows(config)
    if config.source_type == "url_json":
        return _iter_url_json_rows(config)
    if config.source_type == "url_csv":
        return _iter_url_csv_rows(config)
    if config.source_type == "url_xml":
        return _iter_url_xml_rows(config)
    if config.source_type == "package_jsonl":
        return _iter_package_jsonl_rows(config)
    if config.source_type == "polymath":
        return _iter_polymath_rows(config)
    raise ValueError(f"unsupported free-response source_type: {config.source_type}")


def _strip_math_odyssey_problem(text: str) -> str:
    text = text.replace("\\underline{\\hspace{2cm}}", "")
    parts = text.split("\\end{problem}")
    return parts[0].strip() if parts else text.strip()


def _normalize_math_odyssey_answer(answer: str) -> str:
    endings = (
        "\\\n\\noindent",
        "\\\n\n\\noindent",
        "\\\n\t\\noindent",
        ".\n\n\\noindent",
        "\n\n\\noindent",
        "\\\n\n  \n\t\\noindent",
        "\\\\ \n\t\\noindent",
        "\\\n\n\t\\noindent",
    )
    for ending in endings:
        if answer.endswith(ending):
            answer = answer[: -len(ending)]
            break
    answer = answer.strip().strip("\\")
    if answer.endswith("."):
        answer = answer[:-1]
    return answer.replace("$", "").strip()


def _intish(value: Any) -> Any:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if int(numeric) == numeric:
        return int(numeric)
    return numeric


def _element_text(element: ET.Element, name: str) -> str:
    child = element.find(name)
    if child is None or child.text is None:
        return ""
    return child.text.strip()


@lru_cache(maxsize=None)
def _gsm_plus_cleaned_indexes(cleaning: str = "light") -> frozenset[int]:
    resource = resources.files("helicopter_eval").joinpath("data/free_response/gsm_plus_cleaned_indexes.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if cleaning == "none":
        return frozenset()
    if cleaning not in payload:
        raise ValueError(f"unknown gsm_plus cleaning level: {cleaning}")
    return frozenset(int(item) for item in payload[cleaning])


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
    if not scores:
        return None
    return sum(scores) / len(scores)


def adapt_free_response_row(item: Any, config: FreeResponseRunConfig) -> dict[str, Any] | None:
    if config.row_adapter == "answer_solution":
        payload = dict(item)
        if "answer" in payload:
            payload["expected_answer"] = payload.pop("answer")
        if "solution" in payload:
            payload["reference_solution"] = payload.pop("solution")
        return payload

    if config.row_adapter == "math_odyssey":
        key, payload = next(iter(dict(item).items()))
        return {
            "problem": _strip_math_odyssey_problem(str(payload["question"])),
            "expected_answer": _normalize_math_odyssey_answer(str(payload["answer"])),
            "original_answer": payload["answer"],
            "reference_solution": payload["reasoning"],
            "label": payload["label"],
            "level": payload["level"],
            "id": key,
        }

    if config.row_adapter == "answer_to_expected":
        payload = dict(item)
        if "answer" in payload:
            payload["expected_answer"] = payload.pop("answer")
        return payload

    if config.row_adapter == "svamp":
        answer = item["Answer"]
        if isinstance(answer, (int, float)) and int(answer) == answer:
            answer = int(answer)
        return {
            "problem": str(item["Body"]).rstrip(".") + ". " + str(item["Question"]),
            "expected_answer": answer,
            "reference_equation": item["Equation"],
        }

    if config.row_adapter == "algebra222":
        return {
            "problem": item["question"],
            "expected_answer": _intish(item["final_answer"]),
        }

    if config.row_adapter == "asdiv_xml":
        answer = _element_text(item, "Answer").split("(")[0].strip()
        return {
            "problem": f"{_element_text(item, 'Body')} {_element_text(item, 'Question')}".strip(),
            "expected_answer": str(_intish(answer)),
            "type": _element_text(item, "Solution-Type"),
        }

    if config.row_adapter == "mawps":
        return {
            "problem": item["input"],
            "expected_answer": _intish(item["target"]),
            "type": item.get("_source_name"),
        }

    if config.row_adapter == "gsm_plus":
        valid_indices = _gsm_plus_cleaned_indexes("light")
        source_index = int(item.get("_source_index", -1))
        if valid_indices and source_index not in valid_indices:
            return None
        expected_answer = item.get("answer") or item.get("expected_answer")
        if expected_answer == "None":
            expected_answer = "insufficient"
        payload = dict(item)
        payload["problem"] = item["question"]
        payload["expected_answer"] = expected_answer
        payload["reference_solution"] = item.get("solution") or item.get("reference_solution")
        payload["perturbation_type"] = str(item.get("perturbation_type", "")).replace(" ", "_")
        return payload

    if config.row_adapter == "hle":
        if item.get("image"):
            return None
        return {
            "id": item["id"],
            "problem": item["question"],
            "expected_answer": item["answer"],
            "answer_type": item.get("answer_type"),
            "reference_solution": item.get("rationale"),
            "raw_subject": item.get("raw_subject"),
            "category": item.get("category"),
            "author_name": item.get("author_name"),
            "canary": item.get("canary"),
        }

    if config.row_adapter == "answer_judge":
        mean_score = _mean_annotation_score(item.get("annotations"))
        if mean_score is None:
            return None
        expected_judgement = "Judgement: Yes" if mean_score > 0.5 else "Judgement: No"
        problem = str(item.get("question", "") or "")
        expected_answer = str(item.get("gt_answer", "") or "")
        predicted_answer = str(item.get("gen_answer", "") or "")
        return {
            "problem": (
                "Problem:\n"
                f"{problem}\n\n"
                "Expected answer:\n"
                f"{expected_answer}\n\n"
                "Predicted answer:\n"
                f"{predicted_answer}\n\n"
                "Decide whether the predicted answer matches the expected answer. "
                "Return exactly `Judgement: Yes` or `Judgement: No`."
            ),
            "expected_answer": expected_answer,
            "predicted_answer": predicted_answer,
            "expected_judgement": expected_judgement,
            "comment": f"judges-verdict mean_score={mean_score:.3f}",
            "source": item.get("dataset_name") or "judges-verdict",
            "source_id": item.get("item_name"),
        }

    if config.row_adapter == "simpleqa_verified":
        return {
            "id": item.get("original_index"),
            "question": item["problem"],
            "expected_answer": item["answer"],
            "metadata": dict(item),
        }

    if config.row_adapter == "polymath":
        payload: dict[str, Any] = {
            "id": item.get("id") or (
                f"{item.get('_polymath_language')}_{item.get('_polymath_difficulty')}_{item.get('_polymath_index')}"
            ),
            "problem": str(item.get("question") or item.get("problem") or ""),
            "expected_answer": str(item.get("answer") or item.get("expected_answer") or ""),
            "language": item.get("_polymath_language"),
            "difficulty": item.get("_polymath_difficulty"),
            "source_index": item.get("_polymath_index"),
            "source": "polymath",
        }
        for key in ("solution", "explanation", "subject", "topic"):
            value = item.get(key)
            if value is not None:
                payload[key] = value
        return payload

    return dict(item)


def _sample_metadata(item: Mapping[str, Any], *, original_sample_index: int, config: FreeResponseRunConfig) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "benchmark": config.benchmark,
        "dataset_name": config.dataset_name,
        "split": config.split,
        "original_sample_index": original_sample_index,
    }
    for source_key, target_key in (
        ("id", "source_id"),
        ("source", "source"),
        ("language", "language"),
        ("difficulty", "difficulty"),
        ("source_index", "source_index"),
        ("subject", "subject"),
        ("topic", "topic"),
    ):
        value = item.get(source_key)
        if value is not None:
            metadata[target_key] = value
    return metadata


def _renumber_samples(samples: Sequence[FreeResponseSample]) -> list[FreeResponseSample]:
    return [
        FreeResponseSample(
            sample_index=index,
            question=sample.question,
            reference_answer=sample.reference_answer,
            metadata=sample.metadata,
        )
        for index, sample in enumerate(samples)
    ]


def load_samples(config: FreeResponseRunConfig) -> list[FreeResponseSample]:
    if config.limit is not None and int(config.limit) < 0:
        raise ValueError("limit must be non-negative")
    if config.sample_size is not None and int(config.sample_size) < 0:
        raise ValueError("sample_size must be non-negative")
    if config.limit is not None and config.sample_size is not None:
        raise ValueError("limit and sample_size are mutually exclusive")
    limit = None if config.limit is None else int(config.limit)
    samples: list[FreeResponseSample] = []
    for raw_item in _iter_rows(config):
        if limit is not None and len(samples) >= limit:
            break
        item = adapt_free_response_row(raw_item, config)
        if item is None:
            continue
        original_sample_index = len(samples)
        question = str(item[config.question_field])
        reference = extract_marked_answer(str(item[config.answer_field]), config.answer_marker)
        if config.reference_answer_overrides and question in config.reference_answer_overrides:
            reference = str(config.reference_answer_overrides[question])
        samples.append(
            FreeResponseSample(
                sample_index=len(samples),
                question=question,
                reference_answer=reference,
                metadata=_sample_metadata(item, original_sample_index=original_sample_index, config=config),
            )
        )
    if config.sample_size is not None and int(config.sample_size) < len(samples):
        rng = random.Random(int(config.sample_seed))
        samples = sorted(rng.sample(samples, int(config.sample_size)), key=lambda item: item.sample_index)
        samples = _renumber_samples(samples)
    return samples


def generate_completion(sample: FreeResponseSample, config: FreeResponseRunConfig) -> FreeResponseResult:
    prompt = build_prompt(sample.question, template=config.prompt_template)
    completion = chat_completion(
        base_url=config.base_url,
        model=config.model,
        prompt=prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
    )
    answer = completion_answer(completion, marker=config.answer_marker)
    is_passed = answer == sample.reference_answer
    return FreeResponseResult(
        sample_index=sample.sample_index,
        question=sample.question,
        prompt=prompt,
        completion=completion,
        answer=answer,
        reference_answer=sample.reference_answer,
        is_passed=is_passed,
        fail_reason="" if is_passed else f"expected {sample.reference_answer}, got {answer}",
        metadata=sample.metadata,
    )


def evaluate_samples(samples: Sequence[FreeResponseSample], config: FreeResponseRunConfig) -> list[FreeResponseResult]:
    return [generate_completion(sample, config) for sample in samples]


def write_results(results: Sequence[FreeResponseResult], *, config: FreeResponseRunConfig, repo_root: Path) -> int:
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
            ),
            repo_root=repo_root,
        )
    )
    return int(task_id)


def run_free_response(config: FreeResponseRunConfig, *, repo_root: Path) -> dict[str, Any]:
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


def dry_run_summary(config: FreeResponseRunConfig) -> dict[str, Any]:
    payload = {
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
    if config.sample_size is not None:
        payload["sample_size"] = config.sample_size
        payload["sample_seed"] = config.sample_seed
    return payload
