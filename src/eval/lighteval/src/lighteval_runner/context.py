from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any


@dataclass(frozen=True, slots=True)
class ContextSection:
    name: str
    role: str
    content: str
    priority: int
    can_drop: bool = False
    drop_group: str | None = None

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported context role: {self.role}")
        if self.drop_group is not None and not self.can_drop:
            raise ValueError("drop_group requires can_drop")


@dataclass(frozen=True, slots=True)
class ContextDocument:
    sections: tuple[ContextSection, ...]


@dataclass(frozen=True, slots=True)
class ContextBudgetPolicy:
    max_prompt_tokens: int


@dataclass(frozen=True, slots=True)
class BudgetedContext:
    sections: tuple[ContextSection, ...]
    dropped_sections: tuple[str, ...]
    prompt_token_count: int


def from_lighteval_doc(doc: Any) -> ContextDocument:
    """Mirror pinned LightEval PromptManager API message semantics exactly."""

    sections: list[ContextSection] = []
    instruction_used = False
    fewshots = getattr(doc, "fewshot_samples", ()) or ()
    for index, fewshot in enumerate(fewshots):
        query = _without_instruction(fewshot.query, fewshot.instruction)
        if index == 0 and getattr(doc, "instruction", None) is not None:
            instruction_used = True
            query = doc.instruction + query
        group = f"fewshot-{index}"
        sections.append(
            ContextSection(f"{group}-query", "user", query, 10, True, group)
        )
        sections.append(
            ContextSection(
                f"{group}-answer",
                "assistant",
                fewshot.get_golds()[0],
                10,
                True,
                group,
            )
        )
    query = getattr(doc, "query", None)
    if not isinstance(query, str) or not query:
        raise ValueError("LightEval formatter did not produce a non-empty query")
    instruction = getattr(doc, "instruction", None)
    if instruction is not None and not isinstance(instruction, str):
        raise ValueError("LightEval formatter instruction must be a string")
    query = _without_instruction(query, instruction)
    if instruction is not None and not instruction_used:
        query = instruction + query
    sections.append(ContextSection("query", "user", query, 100))
    return ContextDocument(tuple(sections))


def _without_instruction(query: str, instruction: str | None) -> str:
    if instruction is not None and query.startswith(instruction):
        return query[len(instruction) :].strip()
    return query


def apply_context_budget(
    document: ContextDocument,
    policy: ContextBudgetPolicy,
    token_count: Callable[[tuple[ContextSection, ...]], int],
) -> BudgetedContext:
    if policy.max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    retained = list(document.sections)
    dropped: list[str] = []
    while token_count(tuple(retained)) > policy.max_prompt_tokens:
        candidates = [section for section in retained if section.can_drop]
        if not candidates:
            raise ValueError(
                "context exceeds token budget and no whole section can be dropped"
            )
        candidate = min(
            candidates, key=lambda section: (section.priority, section.name)
        )
        group = candidate.drop_group
        removed = [
            section
            for section in retained
            if section is candidate
            or (group is not None and section.drop_group == group)
        ]
        for section in removed:
            retained.remove(section)
            dropped.append(section.name)
    return BudgetedContext(
        sections=tuple(retained),
        dropped_sections=tuple(dropped),
        prompt_token_count=token_count(tuple(retained)),
    )
