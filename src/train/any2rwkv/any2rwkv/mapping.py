from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable

from .errors import CoverageError


class TargetProvenance(StrEnum):
    COPIED = "copied"
    ALGEBRAIC = "algebraic"
    FITTED = "fitted"
    INITIALIZED = "initialized"


class SourceDisposition(StrEnum):
    CONSUMED = "consumed"
    PRESERVED = "preserved"
    INTENTIONALLY_UNMAPPED = "intentionally-unmapped"
    REJECTED = "rejected"


@dataclass(frozen=True)
class TargetEntry:
    target: str
    provenance: TargetProvenance
    sources: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    evidence: str
    source_hashes: tuple[str, ...]


@dataclass(frozen=True)
class SourceEntry:
    source: str
    disposition: SourceDisposition
    targets: tuple[str, ...]
    reason: str


def tensor_sha256(value: object) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


class MappingLedger:
    def __init__(self) -> None:
        self.targets: dict[str, TargetEntry] = {}
        self.sources: dict[str, SourceEntry] = {}

    def add_target(self, entry: TargetEntry) -> None:
        if entry.target in self.targets:
            raise CoverageError(f"duplicate target mapping: {entry.target}")
        if entry.provenance in {TargetProvenance.COPIED, TargetProvenance.ALGEBRAIC} and not entry.sources:
            raise CoverageError(f"{entry.provenance} target requires at least one source: {entry.target}")
        if entry.provenance == TargetProvenance.INITIALIZED and entry.sources:
            raise CoverageError(
                f"initialized target cannot claim source provenance: {entry.target}"
            )
        unresolved_copy_metadata = (
            entry.provenance == TargetProvenance.COPIED
            and entry.dtype == "source"
            and not entry.shape
        )
        if (not entry.shape and not unresolved_copy_metadata) or any(
            dimension <= 0 for dimension in entry.shape
        ):
            raise CoverageError(f"target mapping has invalid shape: {entry.target}")
        if not entry.dtype:
            raise CoverageError(f"target mapping lacks dtype: {entry.target}")
        if not entry.evidence:
            raise CoverageError(f"target mapping lacks formula/solver evidence: {entry.target}")
        self.targets[entry.target] = entry

    def add_source(self, entry: SourceEntry) -> None:
        if entry.source in self.sources:
            raise CoverageError(f"duplicate source disposition: {entry.source}")
        if not entry.reason:
            raise CoverageError(f"source disposition lacks reason: {entry.source}")
        self.sources[entry.source] = entry

    def validate(self, source_names: Iterable[str], target_names: Iterable[str]) -> dict[str, object]:
        expected_sources = set(source_names)
        expected_targets = set(target_names)
        missing_sources = sorted(expected_sources - self.sources.keys())
        missing_targets = sorted(expected_targets - self.targets.keys())
        extra_sources = sorted(self.sources.keys() - expected_sources)
        extra_targets = sorted(self.targets.keys() - expected_targets)
        broken_edges: list[str] = []
        for name, entry in self.targets.items():
            broken_edges.extend(f"target:{name}->source:{source}" for source in entry.sources if source not in self.sources)
            broken_edges.extend(
                f"target:{name}->source:{source}:missing-reverse-edge"
                for source in entry.sources
                if source in self.sources and name not in self.sources[source].targets
            )
        for name, entry in self.sources.items():
            broken_edges.extend(f"source:{name}->target:{target}" for target in entry.targets if target not in self.targets)
            broken_edges.extend(
                f"source:{name}->target:{target}:missing-reverse-edge"
                for target in entry.targets
                if target in self.targets and name not in self.targets[target].sources
            )
            if entry.disposition in {
                SourceDisposition.CONSUMED,
                SourceDisposition.PRESERVED,
            } and not entry.targets:
                broken_edges.append(
                    f"source:{name}:{entry.disposition.value}-without-target"
                )
            if entry.disposition in {
                SourceDisposition.INTENTIONALLY_UNMAPPED,
                SourceDisposition.REJECTED,
            } and entry.targets:
                broken_edges.append(
                    f"source:{name}:{entry.disposition.value}-with-target"
                )
        if missing_sources or missing_targets or extra_sources or extra_targets or broken_edges:
            raise CoverageError(
                f"incomplete mapping coverage missing_sources={missing_sources} missing_targets={missing_targets} "
                f"extra_sources={extra_sources} extra_targets={extra_targets} broken_edges={broken_edges}"
            )
        return {
            "source_total": len(expected_sources),
            "target_total": len(expected_targets),
            "source_coverage": 1.0,
            "target_coverage": 1.0,
            "provenance": {kind.value: sum(e.provenance == kind for e in self.targets.values()) for kind in TargetProvenance},
            "disposition": {kind.value: sum(e.disposition == kind for e in self.sources.values()) for kind in SourceDisposition},
        }

    def write(self, path: Path) -> None:
        payload = {
            "schema_version": 1,
            "targets": [asdict(self.targets[name]) for name in sorted(self.targets)],
            "sources": [asdict(self.sources[name]) for name in sorted(self.sources)],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=list) + "\n", encoding="utf-8")


def finalize_fitted_mapping(
    run_dir: Path,
    *,
    student_sha256: str,
    trace_sha256: str,
) -> dict[str, object]:
    """Promote only actually trained mixer tensors to fitted provenance."""
    mapping_path = run_dir / "mapping.json"
    plan_path = run_dir / "warm-start-plan.json"
    coverage_path = run_dir / "mapping-coverage.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    trainable = {
        str(entry["target"])
        for entry in plan.get("entries", [])
        if entry.get("provenance") in {"fitted", "initialized"}
        or entry.get("is_semantically_lossless") is False
    }
    targets = mapping.get("targets")
    if not isinstance(targets, list):
        raise CoverageError("mapping.json has no target rows")
    found = set()
    for entry in targets:
        target = str(entry.get("target", ""))
        if target not in trainable:
            continue
        found.add(target)
        entry["provenance"] = TargetProvenance.FITTED.value
        entry["evidence"] = (
            str(entry.get("evidence", ""))
            + f"; fitted by layerwise distillation student_sha256={student_sha256} active_layer_trace_sha256={trace_sha256}"
        )
    if found != trainable:
        raise CoverageError(
            f"trained warm-start targets differ from mapping rows: missing={sorted(trainable - found)}"
        )
    mapping_path.write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    coverage["provenance"] = {
        kind.value: sum(entry.get("provenance") == kind.value for entry in targets)
        for kind in TargetProvenance
    }
    coverage["fitted_student_sha256"] = student_sha256
    coverage["active_layer_trace_sha256"] = trace_sha256
    coverage_path.write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return coverage
