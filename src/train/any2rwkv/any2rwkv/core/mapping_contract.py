from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

from ..errors import ContractError, CoverageError
from ..mapping import SourceDisposition, TargetProvenance


def _require_digest(value: str, label: str) -> None:
    if (
        len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{label} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class CalibrationDevelopmentFinalSplit:
    calibration: tuple[str, ...]
    development: tuple[str, ...]
    final: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = {
            "calibration": self.calibration,
            "development": self.development,
            "final": self.final,
        }
        for name, rows in groups.items():
            if not rows or any(not row for row in rows) or len(set(rows)) != len(rows):
                raise ContractError(f"{name} split must contain unique non-empty IDs")
        overlap = {
            "calibration-development": sorted(
                set(self.calibration) & set(self.development)
            ),
            "calibration-final": sorted(set(self.calibration) & set(self.final)),
            "development-final": sorted(set(self.development) & set(self.final)),
        }
        if any(overlap.values()):
            raise ContractError(f"mapping splits overlap: {overlap}")


class CandidateSelection:
    """Freeze selection before the final split can be observed."""

    def __init__(self, split: CalibrationDevelopmentFinalSplit) -> None:
        self.split = split
        self.selected_candidate: str | None = None
        self.final_verification: object | None = None

    def select(
        self,
        candidate_scores: Mapping[str, Mapping[str, float]],
        *,
        minimize: bool = True,
    ) -> str:
        if self.selected_candidate is not None:
            raise ContractError("mapping candidate selection is already frozen")
        if not candidate_scores:
            raise ContractError("mapping candidate selection requires candidates")
        for candidate, scores in candidate_scores.items():
            if set(scores) != {"calibration", "development"}:
                raise ContractError(
                    f"candidate {candidate} selection may use only calibration/development"
                )
        key = lambda item: (item[1]["development"], item[1]["calibration"], item[0])
        self.selected_candidate = (min if minimize else max)(
            candidate_scores.items(), key=key
        )[0]
        return self.selected_candidate

    def verify_final(self, *, candidate: str, evidence: object) -> None:
        if self.selected_candidate is None:
            raise ContractError(
                "final split cannot be observed before selection is frozen"
            )
        if candidate != self.selected_candidate:
            raise ContractError(
                "final split may verify only the frozen selected candidate"
            )
        if self.final_verification is not None:
            raise ContractError("final split verification is already recorded")
        self.final_verification = evidence


@dataclass(frozen=True)
class MaterializedTarget:
    target: str
    provenance: TargetProvenance
    primary_sources: tuple[str, ...]
    auxiliary_sources: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    source_hashes: Mapping[str, str]
    formula_or_solver_version: str
    materialized_sha256: str
    perturbed_materialized_sha256: str | None = None


@dataclass(frozen=True)
class SourceConsumption:
    source: str
    disposition: SourceDisposition
    targets: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    source_sha256: str
    reason: str


class StrictMappingLedger:
    """Two-axis, exact-coverage ledger for materialized conversion outputs."""

    def __init__(self) -> None:
        self.targets: dict[str, MaterializedTarget] = {}
        self.sources: dict[str, SourceConsumption] = {}

    def add_target(self, entry: MaterializedTarget) -> None:
        if not entry.target or entry.target in self.targets:
            raise CoverageError(
                f"duplicate or empty materialized target: {entry.target}"
            )
        sources = (*entry.primary_sources, *entry.auxiliary_sources)
        if len(set(sources)) != len(sources):
            raise CoverageError(
                f"target consumes a source more than once: {entry.target}"
            )
        if entry.provenance == TargetProvenance.INITIALIZED:
            if sources:
                raise CoverageError(
                    "initialized target cannot claim source consumption"
                )
        elif not entry.primary_sources:
            raise CoverageError(
                f"mapped target requires a primary source: {entry.target}"
            )
        if set(entry.source_hashes) != set(sources):
            raise CoverageError(
                f"target source hashes do not cover its inputs: {entry.target}"
            )
        for source, digest in entry.source_hashes.items():
            _require_digest(digest, f"source hash for {source}")
        if not entry.shape or any(
            type(size) is not int or size <= 0 for size in entry.shape
        ):
            raise CoverageError(
                f"materialized target has invalid shape: {entry.target}"
            )
        if not entry.dtype or not entry.formula_or_solver_version:
            raise CoverageError(
                f"materialized target lacks dtype/formula/solver: {entry.target}"
            )
        _require_digest(entry.materialized_sha256, "materialized target hash")
        if entry.provenance == TargetProvenance.FITTED:
            if entry.perturbed_materialized_sha256 is None:
                raise CoverageError(
                    f"fitted target lacks input influence evidence: {entry.target}"
                )
            _require_digest(
                entry.perturbed_materialized_sha256, "perturbed target hash"
            )
            if entry.perturbed_materialized_sha256 == entry.materialized_sha256:
                raise CoverageError(
                    f"input perturbation did not affect fitted target: {entry.target}"
                )
        self.targets[entry.target] = entry

    def add_source(self, entry: SourceConsumption) -> None:
        if not entry.source or entry.source in self.sources:
            raise CoverageError(
                f"duplicate or empty source consumption: {entry.source}"
            )
        if not entry.shape or any(
            type(size) is not int or size <= 0 for size in entry.shape
        ):
            raise CoverageError(f"source consumption has invalid shape: {entry.source}")
        if not entry.dtype or not entry.reason:
            raise CoverageError(
                f"source consumption lacks dtype/reason: {entry.source}"
            )
        _require_digest(entry.source_sha256, "source tensor hash")
        mapped = entry.disposition in {
            SourceDisposition.CONSUMED,
            SourceDisposition.PRESERVED,
        }
        if mapped != bool(entry.targets):
            raise CoverageError(
                f"source disposition and target edges disagree: {entry.source}"
            )
        if len(entry.targets) > 1 or len(set(entry.targets)) != len(entry.targets):
            raise CoverageError(
                f"source tensor is consumed more than once: {entry.source}"
            )
        self.sources[entry.source] = entry

    def validate(
        self, *, source_names: Iterable[str], target_names: Iterable[str]
    ) -> dict[str, object]:
        expected_sources, expected_targets = set(source_names), set(target_names)
        if (
            set(self.sources) != expected_sources
            or set(self.targets) != expected_targets
        ):
            raise CoverageError("source or materialized target coverage is incomplete")
        for target, entry in self.targets.items():
            for source in (*entry.primary_sources, *entry.auxiliary_sources):
                source_entry = self.sources.get(source)
                if source_entry is None or source_entry.targets != (target,):
                    raise CoverageError(
                        f"mapping reverse edge is missing: {target}<->{source}"
                    )
                if entry.source_hashes[source] != source_entry.source_sha256:
                    raise CoverageError(
                        f"mapping source hash differs across axes: {source}"
                    )
        for source, entry in self.sources.items():
            for target in entry.targets:
                target_entry = self.targets.get(target)
                if target_entry is None or source not in (
                    *target_entry.primary_sources,
                    *target_entry.auxiliary_sources,
                ):
                    raise CoverageError(
                        f"mapping forward edge is missing: {source}<->{target}"
                    )
        return {
            "source_coverage": 1.0,
            "target_coverage": 1.0,
            "sources": [asdict(self.sources[name]) for name in sorted(self.sources)],
            "targets": [asdict(self.targets[name]) for name in sorted(self.targets)],
        }
