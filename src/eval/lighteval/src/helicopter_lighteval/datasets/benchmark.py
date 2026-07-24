"""Benchmark registry types and shared LightEval document preparation."""

from __future__ import annotations

import hashlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from string import ascii_uppercase
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class BenchmarkName:
    value: str


class Field(StrEnum):
    MATHS = "maths"
    KNOWLEDGE = "knowledge"
    CODING = "coding"
    INSTRUCTION_FOLLOWING = "instruction_following"


@dataclass(frozen=True, slots=True)
class BenchmarkInfo:
    name: BenchmarkName
    field: Field
    display_name: str
    lighteval_task_name: str
    create: type[Benchmark]


class Benchmark(ABC):
    @abstractmethod
    def get_query(self, row: Mapping[str, Any], document: Any) -> str:
        """Return the raw user query for one dataset row."""

    def prepare_documents(self, *, task: Any, documents: Sequence[Any]) -> None:
        """Replace LightEval prompt wrappers without changing scoring metadata."""

        if task.dataset is None:
            raise ValueError("LightEval task dataset is unavailable")
        splits = tuple(task.config.evaluation_splits)
        if len(splits) != 1:
            raise ValueError("benchmark preparation requires one evaluation split")
        rows = task.dataset[splits[0]]
        for document in documents:
            if document.fewshot_samples:
                raise ValueError("benchmark preparation requires zero-shot documents")
            try:
                row = rows[int(document.id)]
            except (IndexError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"cannot recover dataset row for LightEval document {document.id}"
                ) from error
            query = self.get_query(row, document)
            document.query = query
            document.original_query = query
            document.instruction = None

    def query_revision(self) -> str:
        """Identify the selected benchmark query code and shared preparation code."""

        source = "\n".join(
            (
                inspect.getsource(type(self).get_query),
                inspect.getsource(Benchmark.prepare_documents),
                inspect.getsource(get_string),
                inspect.getsource(get_strings),
                inspect.getsource(render_choices),
            )
        )
        return hashlib.sha256(source.encode()).hexdigest()


def get_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"dataset field must be non-empty text: {field}")
    return value.strip()


def get_strings(row: Mapping[str, Any], field: str) -> list[str]:
    value = row.get(field)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"dataset field must be a text sequence: {field}")
    return [get_string({field: item}, field) for item in value]


def render_choices(question: str, choices: Sequence[str]) -> str:
    if not choices or len(choices) > len(ascii_uppercase):
        raise ValueError("benchmark has an unsupported choice count")
    rendered = [question, ""]
    rendered.extend(
        f"{letter}. {choice}"
        for letter, choice in zip(ascii_uppercase[: len(choices)], choices, strict=True)
    )
    return "\n".join(rendered)
