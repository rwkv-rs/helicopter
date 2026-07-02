from __future__ import annotations

from collections.abc import Callable, Sequence
import random
from typing import TypeVar


T = TypeVar("T")


def apply_limit_or_sample(
    items: Sequence[T],
    *,
    limit: int | None,
    sample_size: int | None,
    sample_seed: int,
    sort_key: Callable[[T], object] | None = None,
) -> list[T]:
    validate_limit_or_sample(limit=limit, sample_size=sample_size)
    rows = list(items)
    if sample_size is not None:
        requested = int(sample_size)
        if requested < len(rows):
            indexed_rows = list(enumerate(rows))
            selected = random.Random(int(sample_seed)).sample(indexed_rows, requested)
            if sort_key is not None:
                return [item for _index, item in sorted(selected, key=lambda row: sort_key(row[1]))]
            return [item for _index, item in sorted(selected, key=lambda row: row[0])]
        return rows
    if limit is not None:
        return rows[: int(limit)]
    return rows


def validate_limit_or_sample(*, limit: int | None, sample_size: int | None) -> None:
    if limit is not None and int(limit) < 0:
        raise ValueError("limit must be non-negative")
    if sample_size is not None and int(sample_size) < 0:
        raise ValueError("sample_size must be non-negative")
    if limit is not None and sample_size is not None:
        raise ValueError("limit and sample_size are mutually exclusive")


def dataset_sample_suffix(*, sample_size: int | None, sample_seed: int) -> str:
    if sample_size is None:
        return ""
    return f"_sample{int(sample_size)}_seed{int(sample_seed)}"
