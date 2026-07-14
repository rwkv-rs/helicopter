"""Deterministic, offline dataset preparation for Any2RWKV distillation."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import re
import tempfile
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence


SPLIT_NAMES = (
    "distill_train",
    "nvfp4_calibration",
    "validation",
    "ruler",
    "downstream",
    "smoke",
)
QUALITY_GATE_SPLITS = frozenset(("validation", "ruler", "downstream", "smoke"))
DEFAULT_SPLIT_RATIOS: Mapping[str, str] = {
    "distill_train": "0.94",
    "nvfp4_calibration": "0.01",
    "validation": "0.02",
    "ruler": "0.01",
    "downstream": "0.01",
    "smoke": "0.01",
}


class Tokenizer(Protocol):
    """The narrow tokenizer interface used by the offline preparer."""

    eos_token_id: int | None
    chat_template: str | None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]: ...


class DataPreparationError(ValueError):
    """Raised when local input violates the frozen data contract."""


class DuplicateSampleError(DataPreparationError):
    """Raised when exact duplicate handling is configured as strict."""

    def __init__(self, report: Mapping[str, Any]):
        kind = report.get("kind", "exact_duplicate")
        count = report["exact_duplicate_pair_count"]
        if kind == "near_duplicate":
            message = (
                f"found {count} near-duplicate pair(s); "
                "use near_duplicate_policy='report' to retain them with an audit report"
            )
        else:
            message = (
                f"found {count} exact duplicate pair(s); "
                "use exact_duplicate_policy='drop' to retain one canonical sample"
            )
        super().__init__(message)
        self.report = dict(report)


@dataclass(frozen=True)
class DataPreparationConfig:
    """Frozen parameters that determine every prepared token row."""

    burn_in_tokens: int
    supervised_tokens: int
    seed: int = 20260714
    split_ratios: Mapping[str, str | float | Decimal] = field(
        default_factory=lambda: dict(DEFAULT_SPLIT_RATIOS)
    )
    exact_duplicate_policy: str = "drop"
    near_duplicate_policy: str = "report"
    near_duplicate_threshold: float = 0.8
    near_duplicate_ngram: int = 3
    minhash_permutations: int = 32
    minhash_bands: int = 8
    max_lsh_bucket_size: int = 256
    id_field: str = "sample_id"
    text_field: str = "text"

    def __post_init__(self) -> None:
        if self.burn_in_tokens < 0:
            raise DataPreparationError("burn_in_tokens must be non-negative")
        if self.supervised_tokens <= 0:
            raise DataPreparationError("supervised_tokens must be positive")
        if self.exact_duplicate_policy not in {"drop", "reject"}:
            raise DataPreparationError("exact_duplicate_policy must be 'drop' or 'reject'")
        if self.near_duplicate_policy not in {"report", "reject"}:
            raise DataPreparationError("near_duplicate_policy must be 'report' or 'reject'")
        if not 0 < self.near_duplicate_threshold < 1:
            raise DataPreparationError("near_duplicate_threshold must be between zero and one")
        if self.near_duplicate_ngram <= 0:
            raise DataPreparationError("near_duplicate_ngram must be positive")
        if self.minhash_permutations <= 0:
            raise DataPreparationError("minhash_permutations must be positive")
        if self.minhash_bands <= 0 or self.minhash_permutations % self.minhash_bands:
            raise DataPreparationError("minhash_bands must evenly divide minhash_permutations")
        if self.max_lsh_bucket_size < 2:
            raise DataPreparationError("max_lsh_bucket_size must be at least two")
        _decimal_ratios(self.split_ratios)

    @property
    def packed_tokens(self) -> int:
        return self.burn_in_tokens + self.supervised_tokens


@dataclass(frozen=True)
class PreparedSample:
    sample_id: str
    normalized_text: str
    content_sha256: str
    split: str
    input_ids: tuple[int, ...]


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    """Hash a local tokenizer tree without following symlinks."""

    path = path.resolve()
    if not path.is_dir():
        raise DataPreparationError(f"tokenizer path is not a local directory: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file() and not item.is_symlink())
    if not files:
        raise DataPreparationError(f"tokenizer directory contains no regular files: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise DataPreparationError("sample text must be a string")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def iter_jsonl(paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        if not path.is_file():
            raise DataPreparationError(f"local JSONL input does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DataPreparationError(f"{path}:{line_number}: invalid JSON: {error}") from error
                if not isinstance(row, dict):
                    raise DataPreparationError(f"{path}:{line_number}: each row must be a JSON object")
                yield row


def stable_split(
    sample_id: str,
    *,
    seed: int,
    split_ratios: Mapping[str, str | float | Decimal] = DEFAULT_SPLIT_RATIOS,
) -> str:
    ratios = _decimal_ratios(split_ratios)
    digest = hashlib.sha256(f"any2rwkv-split-v1\0{seed}\0{sample_id}".encode("utf-8")).digest()
    point = Decimal(int.from_bytes(digest, "big")) / Decimal(1 << 256)
    cumulative = Decimal(0)
    for split in SPLIT_NAMES:
        cumulative += ratios[split]
        if point < cumulative:
            return split
    return SPLIT_NAMES[-1]


def prepare_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    tokenizer: Tokenizer,
    config: DataPreparationConfig,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Deduplicate, split and pack local rows without any network access."""

    raw_rows = list(rows)
    canonical_rows, exact_report = _deduplicate_rows(raw_rows, config=config)
    if exact_report["exact_duplicate_pair_count"] and config.exact_duplicate_policy == "reject":
        raise DuplicateSampleError(exact_report)

    samples = [
        _prepare_sample(row, tokenizer=tokenizer, config=config)
        for row in sorted(canonical_rows, key=lambda item: str(item[config.id_field]))
    ]
    near_report = _near_duplicate_report(samples, config=config)
    if near_report["pair_count"] and config.near_duplicate_policy == "reject":
        raise DuplicateSampleError(
            {
                "exact_duplicate_pair_count": near_report["pair_count"],
                "pairs": near_report["pairs"],
                "kind": "near_duplicate",
            }
        )

    samples_by_split: dict[str, list[PreparedSample]] = {name: [] for name in SPLIT_NAMES}
    for sample in samples:
        samples_by_split[sample.split].append(sample)

    packed_by_split: dict[str, list[dict[str, Any]]] = {}
    packing: dict[str, Any] = {}
    for split in SPLIT_NAMES:
        packed, split_report = _pack_split(
            samples_by_split[split],
            split=split,
            row_tokens=config.packed_tokens,
            burn_in_tokens=config.burn_in_tokens,
            supervised_tokens=config.supervised_tokens,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        packed_by_split[split] = packed
        packing[split] = split_report

    split_ids = {split: [sample.sample_id for sample in values] for split, values in samples_by_split.items()}
    _assert_mutually_exclusive(split_ids)
    report = {
        "schema_version": 1,
        "input_row_count": len(raw_rows),
        "accepted_sample_count": len(samples),
        "exact_duplicates": exact_report,
        "near_duplicates": near_report,
        "split_sample_ids": split_ids,
        "split_sample_id_sha256": {
            split: canonical_json_sha256(ids) for split, ids in split_ids.items()
        },
        "packing": packing,
        "invariants": {
            "sample_ids_mutually_exclusive": True,
            "calibration_quality_gate_overlap": [],
            "packed_row_tokens": config.packed_tokens,
        },
    }
    return packed_by_split, report


def prepare_jsonl_dataset(
    input_paths: Sequence[Path],
    *,
    output_dir: Path,
    tokenizer: Tokenizer,
    tokenizer_path: Path,
    dataset_repository: str,
    dataset_revision: str,
    tokenizer_repository: str,
    tokenizer_revision: str,
    config: DataPreparationConfig,
) -> dict[str, Any]:
    """Prepare local JSONL files and atomically write split data plus manifest."""

    resolved_inputs = tuple(path.resolve() for path in input_paths)
    if not resolved_inputs:
        raise DataPreparationError("at least one local JSONL input is required")
    input_files = [
        {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
        for path in resolved_inputs
    ]
    packed, report = prepare_rows(iter_jsonl(resolved_inputs), tokenizer=tokenizer, config=config)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_metadata: dict[str, Any] = {}
    for split, rows in packed.items():
        path = output_dir / f"{split}.jsonl"
        _write_jsonl_atomic(path, rows)
        split_metadata[split] = {
            "path": path.name,
            "sha256": file_sha256(path),
            "row_count": len(rows),
            "token_count": sum(len(row["input_ids"]) for row in rows),
            "source_sample_count": len(report["split_sample_ids"][split]),
            "source_sample_ids_sha256": report["split_sample_id_sha256"][split],
            "quality_gate": split in QUALITY_GATE_SPLITS,
        }

    dedupe_path = output_dir / "deduplication-report.json"
    _write_json_atomic(
        dedupe_path,
        {"exact_duplicates": report["exact_duplicates"], "near_duplicates": report["near_duplicates"]},
    )
    template = getattr(tokenizer, "chat_template", None) or ""
    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "generator": "any2rwkv.data/v1",
        "dataset": {
            "repository": dataset_repository,
            "revision": dataset_revision,
            "local_input_files": input_files,
            "combined_input_sha256": canonical_json_sha256(
                [{"sha256": item["sha256"], "bytes": item["bytes"]} for item in input_files]
            ),
        },
        "tokenizer": {
            "repository": tokenizer_repository,
            "revision": tokenizer_revision,
            "local_tree_sha256": directory_sha256(tokenizer_path),
            "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "chat_template_present": bool(template),
        },
        "seed": config.seed,
        "packing": {
            "burn_in_tokens": config.burn_in_tokens,
            "supervised_tokens": config.supervised_tokens,
            "packed_row_tokens": config.packed_tokens,
            "eos_separator_token_id": getattr(tokenizer, "eos_token_id", None),
            "drop_incomplete_tail": True,
        },
        "split_assignment": {
            "algorithm": "sha256(any2rwkv-split-v1\\0seed\\0sample_id)",
            "ratios": {name: str(value) for name, value in _decimal_ratios(config.split_ratios).items()},
            "sample_ids_mutually_exclusive": True,
            "calibration_forbidden_from_quality_gates": True,
            "quality_gate_splits": sorted(QUALITY_GATE_SPLITS),
        },
        "deduplication": {
            "report_path": dedupe_path.name,
            "report_sha256": file_sha256(dedupe_path),
            "exact_policy": config.exact_duplicate_policy,
            "near_policy": config.near_duplicate_policy,
            "exact_duplicate_pair_count": report["exact_duplicates"]["exact_duplicate_pair_count"],
            "near_duplicate_pair_count": report["near_duplicates"]["pair_count"],
        },
        "splits": split_metadata,
        "audit": {
            "input_row_count": report["input_row_count"],
            "accepted_sample_count": report["accepted_sample_count"],
            "packing": report["packing"],
            "calibration_quality_gate_overlap": [],
        },
    }
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    _write_json_atomic(output_dir / "data-splits.json", manifest)
    return manifest


def _decimal_ratios(
    ratios: Mapping[str, str | float | Decimal],
) -> dict[str, Decimal]:
    if set(ratios) != set(SPLIT_NAMES):
        missing = sorted(set(SPLIT_NAMES) - set(ratios))
        extra = sorted(set(ratios) - set(SPLIT_NAMES))
        raise DataPreparationError(f"split ratios mismatch; missing={missing}, extra={extra}")
    converted = {name: Decimal(str(ratios[name])) for name in SPLIT_NAMES}
    if any(value <= 0 for value in converted.values()):
        raise DataPreparationError("all split ratios must be positive")
    if sum(converted.values()) != Decimal(1):
        raise DataPreparationError("split ratios must sum exactly to 1")
    return converted


def _deduplicate_rows(
    rows: Sequence[Mapping[str, Any]], *, config: DataPreparationConfig
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    indexed: list[tuple[str, str, Mapping[str, Any]]] = []
    sample_ids: set[str] = set()
    for index, row in enumerate(rows):
        if config.id_field not in row:
            raise DataPreparationError(f"input row {index} is missing id field {config.id_field!r}")
        sample_id = str(row[config.id_field])
        if not sample_id:
            raise DataPreparationError(f"input row {index} has an empty sample id")
        if sample_id in sample_ids:
            raise DataPreparationError(
                f"sample id {sample_id!r} occurs more than once; ids must be globally unique"
            )
        sample_ids.add(sample_id)
        normalized = _normalized_row_content(row, config=config)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        indexed.append((sample_id, digest, row))

    by_digest: dict[str, str] = {}
    accepted: list[Mapping[str, Any]] = []
    pairs: list[dict[str, str]] = []
    for sample_id, digest, row in sorted(indexed, key=lambda item: item[0]):
        canonical_id = by_digest.get(digest)
        if canonical_id is not None:
            pairs.append(
                {
                    "canonical_sample_id": canonical_id,
                    "duplicate_sample_id": sample_id,
                    "content_sha256": digest,
                }
            )
            continue
        by_digest[digest] = sample_id
        accepted.append(row)
    return accepted, {
        "algorithm": "sha256(NFKC-and-whitespace-normalized-content)",
        "policy": config.exact_duplicate_policy,
        "exact_duplicate_pair_count": len(pairs),
        "pairs": sorted(pairs, key=lambda pair: (pair["canonical_sample_id"], pair["duplicate_sample_id"])),
    }


def _normalized_row_content(row: Mapping[str, Any], *, config: DataPreparationConfig) -> str:
    if "messages" in row:
        messages = row["messages"]
        if not isinstance(messages, list) or not messages:
            raise DataPreparationError("messages must be a non-empty list")
        normalized_messages = []
        for message in messages:
            if not isinstance(message, Mapping) or "role" not in message or "content" not in message:
                raise DataPreparationError("each message must contain role and content")
            normalized_messages.append(
                {"role": str(message["role"]), "content": normalize_text(message["content"])}
            )
        return json.dumps(normalized_messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if config.text_field not in row:
        raise DataPreparationError(
            f"sample must contain either 'messages' or text field {config.text_field!r}"
        )
    return normalize_text(row[config.text_field])


def _prepare_sample(
    row: Mapping[str, Any], *, tokenizer: Tokenizer, config: DataPreparationConfig
) -> PreparedSample:
    sample_id = str(row[config.id_field])
    normalized = _normalized_row_content(row, config=config)
    if "messages" in row:
        input_ids = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=False
        )
    else:
        input_ids = tokenizer.encode(normalize_text(row[config.text_field]), add_special_tokens=False)
    if not isinstance(input_ids, list) or not all(isinstance(token, int) for token in input_ids):
        raise DataPreparationError(f"tokenizer returned invalid ids for sample {sample_id!r}")
    if not input_ids:
        raise DataPreparationError(f"sample {sample_id!r} tokenized to an empty sequence")
    return PreparedSample(
        sample_id=sample_id,
        normalized_text=normalized,
        content_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        split=stable_split(sample_id, seed=config.seed, split_ratios=config.split_ratios),
        input_ids=tuple(input_ids),
    )


def _pack_split(
    samples: Sequence[PreparedSample],
    *,
    split: str,
    row_tokens: int,
    burn_in_tokens: int,
    supervised_tokens: int,
    eos_token_id: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    token_buffer: list[int] = []
    owner_buffer: list[str] = []
    total_stream_tokens = 0
    for sample in samples:
        tokens = list(sample.input_ids)
        if eos_token_id is not None and (not tokens or tokens[-1] != eos_token_id):
            tokens.append(eos_token_id)
        total_stream_tokens += len(tokens)
        offset = 0
        while offset < len(tokens):
            take = min(row_tokens - len(token_buffer), len(tokens) - offset)
            token_buffer.extend(tokens[offset : offset + take])
            owner_buffer.extend([sample.sample_id] * take)
            offset += take
            if len(token_buffer) == row_tokens:
                spans = _owner_spans(owner_buffer)
                rows.append(
                    {
                        "row_id": f"{split}-{len(rows):08d}",
                        "split": split,
                        "input_ids": token_buffer,
                        "burn_in_tokens": burn_in_tokens,
                        "supervised_tokens": supervised_tokens,
                        "source_sample_ids": list(dict.fromkeys(owner_buffer)),
                        "source_token_spans": spans,
                    }
                )
                token_buffer = []
                owner_buffer = []
    return rows, {
        "source_sample_count": len(samples),
        "stream_token_count": total_stream_tokens,
        "packed_row_count": len(rows),
        "packed_token_count": len(rows) * row_tokens,
        "dropped_tail_tokens": len(token_buffer),
    }


def _owner_spans(owners: Sequence[str]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    start = 0
    while start < len(owners):
        end = start + 1
        while end < len(owners) and owners[end] == owners[start]:
            end += 1
        spans.append({"sample_id": owners[start], "start": start, "end": end})
        start = end
    return spans


def _near_duplicate_report(
    samples: Sequence[PreparedSample], *, config: DataPreparationConfig
) -> dict[str, Any]:
    shingles = {
        sample.sample_id: _word_shingles(sample.normalized_text, config.near_duplicate_ngram)
        for sample in samples
    }
    signatures = {
        sample_id: _minhash_signature(values, config.minhash_permutations)
        for sample_id, values in shingles.items()
    }
    rows_per_band = config.minhash_permutations // config.minhash_bands
    buckets: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
    for sample_id in sorted(signatures):
        signature = signatures[sample_id]
        for band in range(config.minhash_bands):
            start = band * rows_per_band
            buckets[(band, signature[start : start + rows_per_band])].append(sample_id)
    candidates: set[tuple[str, str]] = set()
    oversized_buckets: list[dict[str, Any]] = []
    for (band, signature), members in sorted(buckets.items(), key=lambda item: repr(item[0])):
        if len(members) > config.max_lsh_bucket_size:
            oversized_buckets.append(
                {
                    "band": band,
                    "signature_sha256": canonical_json_sha256(signature),
                    "member_count": len(members),
                }
            )
            continue
        for left_index, left in enumerate(members):
            for right in members[left_index + 1 :]:
                candidates.add((left, right) if left < right else (right, left))
    split_by_id = {sample.sample_id: sample.split for sample in samples}
    pairs = []
    for left, right in sorted(candidates):
        union = shingles[left] | shingles[right]
        score = len(shingles[left] & shingles[right]) / len(union) if union else 1.0
        if score >= config.near_duplicate_threshold:
            pairs.append(
                {
                    "left_sample_id": left,
                    "right_sample_id": right,
                    "left_split": split_by_id[left],
                    "right_split": split_by_id[right],
                    "cross_split": split_by_id[left] != split_by_id[right],
                    "jaccard": score,
                }
            )
    return {
        "algorithm": "word-ngram-bottom-k-minhash-lsh-candidates+exact-jaccard-v1",
        "policy": config.near_duplicate_policy,
        "threshold": config.near_duplicate_threshold,
        "word_ngram": config.near_duplicate_ngram,
        "minhash_permutations": config.minhash_permutations,
        "minhash_bands": config.minhash_bands,
        "max_lsh_bucket_size": config.max_lsh_bucket_size,
        "candidate_pair_count": len(candidates),
        "pair_count": len(pairs),
        "cross_split_pair_count": sum(pair["cross_split"] for pair in pairs),
        "pairs": pairs,
        "oversized_buckets": oversized_buckets,
        "candidate_search_complete": not oversized_buckets,
    }


def _word_shingles(text: str, ngram: int) -> frozenset[str]:
    words = normalize_text(text).casefold().split()
    if len(words) < ngram:
        return frozenset((" ".join(words),))
    return frozenset(" ".join(words[index : index + ngram]) for index in range(len(words) - ngram + 1))


def _minhash_signature(shingles: frozenset[str], permutations: int) -> tuple[int, ...]:
    if not shingles:
        return tuple(0 for _ in range(permutations))
    minima = heapq.nsmallest(
        permutations,
        (
            int.from_bytes(
                hashlib.sha256(f"any2rwkv-minhash-v1\0{shingle}".encode("utf-8")).digest()[:8],
                "big",
            )
            for shingle in shingles
        ),
    )
    return tuple(minima + [2**64 - 1] * (permutations - len(minima)))


def _assert_mutually_exclusive(split_ids: Mapping[str, Sequence[str]]) -> None:
    owner: dict[str, str] = {}
    for split in SPLIT_NAMES:
        for sample_id in split_ids[split]:
            previous = owner.setdefault(sample_id, split)
            if previous != split:
                raise AssertionError(f"sample {sample_id!r} appears in both {previous} and {split}")
    calibration = set(split_ids["nvfp4_calibration"])
    quality = set().union(*(set(split_ids[split]) for split in QUALITY_GATE_SPLITS))
    overlap = calibration & quality
    if overlap:
        raise AssertionError(f"calibration overlaps quality gates: {sorted(overlap)}")


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    _write_text_atomic(path, payload)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _write_text_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
