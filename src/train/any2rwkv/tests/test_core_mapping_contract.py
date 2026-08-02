from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from any2rwkv.core import (
    CalibrationDevelopmentFinalSplit,
    CandidateSelection,
    MaterializedTarget,
    SourceConsumption,
    StrictMappingLedger,
    canonical_digest,
)
from any2rwkv.errors import ContractError, CoverageError
from any2rwkv.mapping import SourceDisposition, TargetProvenance


class CoreMappingContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.split = CalibrationDevelopmentFinalSplit(
            calibration=("c0", "c1"),
            development=("d0", "d1"),
            final=("f0", "f1"),
        )

    def test_observable_output_fit_selects_without_observing_final(self) -> None:
        source = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        observable = 2.0 * source + 1.0
        rows = {
            "calibration": torch.tensor([0, 1]),
            "development": torch.tensor([2, 3]),
            "final": torch.tensor([4, 5]),
        }
        calibration_design = torch.cat(
            (source[rows["calibration"]], torch.ones(2, 1)), dim=1
        )
        fitted = torch.linalg.lstsq(
            calibration_design, observable[rows["calibration"]]
        ).solution.squeeze(1)
        candidates = {
            "identity-v1": (torch.tensor(1.0), torch.tensor(0.0)),
            "affine-solver-v1": (fitted[0], fitted[1]),
        }

        scores = {}
        for name, (weight, bias) in candidates.items():
            scores[name] = {
                role: float(
                    torch.nn.functional.mse_loss(
                        source[index] * weight + bias, observable[index]
                    )
                )
                for role, index in rows.items()
                if role != "final"
            }
        selection = CandidateSelection(self.split)
        selected = selection.select(scores)
        self.assertEqual(selected, "affine-solver-v1")

        final_index = rows["final"]
        weight, bias = candidates[selected]
        final_error = float(
            torch.nn.functional.mse_loss(
                source[final_index] * weight + bias, observable[final_index]
            )
        )
        selection.verify_final(candidate=selected, evidence={"output_mse": final_error})
        self.assertAlmostEqual(selection.final_verification["output_mse"], 0.0)

        source_hash = canonical_digest(source.tolist())
        auxiliary_hash = canonical_digest(observable.tolist())
        target = torch.stack((weight, bias))
        perturbed_source = source.clone()
        perturbed_source[0].add_(0.25)
        perturbed_design = torch.cat(
            (perturbed_source[rows["calibration"]], torch.ones(2, 1)), dim=1
        )
        perturbed_target = torch.linalg.lstsq(
            perturbed_design, observable[rows["calibration"]]
        ).solution.squeeze(1)
        ledger = StrictMappingLedger()
        ledger.add_target(
            MaterializedTarget(
                target="student.output.weight",
                provenance=TargetProvenance.FITTED,
                primary_sources=("teacher.output.weight",),
                auxiliary_sources=("teacher.output.bias",),
                shape=(2,),
                dtype="float32",
                source_hashes={
                    "teacher.output.weight": source_hash,
                    "teacher.output.bias": auxiliary_hash,
                },
                formula_or_solver_version="observable-affine-solver-v1",
                materialized_sha256=canonical_digest(target.tolist()),
                perturbed_materialized_sha256=canonical_digest(
                    perturbed_target.tolist()
                ),
            )
        )
        ledger.add_source(
            SourceConsumption(
                source="teacher.output.weight",
                disposition=SourceDisposition.CONSUMED,
                targets=("student.output.weight",),
                shape=tuple(source.shape),
                dtype="float32",
                source_sha256=source_hash,
                reason="observable affine fit",
            )
        )
        ledger.add_source(
            SourceConsumption(
                source="teacher.output.bias",
                disposition=SourceDisposition.CONSUMED,
                targets=("student.output.weight",),
                shape=tuple(observable.shape),
                dtype="float32",
                source_sha256=auxiliary_hash,
                reason="auxiliary affine offset fit",
            )
        )
        report = ledger.validate(
            source_names=("teacher.output.weight", "teacher.output.bias"),
            target_names=("student.output.weight",),
        )
        self.assertEqual(report["source_coverage"], 1.0)
        self.assertEqual(report["target_coverage"], 1.0)

    def test_splits_are_disjoint_and_final_is_verification_only(self) -> None:
        with self.assertRaisesRegex(ContractError, "overlap"):
            CalibrationDevelopmentFinalSplit(("same",), ("same",), ("final",))
        selection = CandidateSelection(self.split)
        with self.assertRaisesRegex(ContractError, "before selection"):
            selection.verify_final(candidate="candidate", evidence=0.0)
        with self.assertRaisesRegex(ContractError, "only calibration/development"):
            selection.select(
                {"candidate": {"calibration": 1.0, "development": 1.0, "final": 0.0}}
            )

    def test_ledger_fails_closed_for_coverage_consumption_and_influence(self) -> None:
        digest = "a" * 64
        fitted = MaterializedTarget(
            target="target",
            provenance=TargetProvenance.FITTED,
            primary_sources=("source",),
            auxiliary_sources=(),
            shape=(1,),
            dtype="float32",
            source_hashes={"source": digest},
            formula_or_solver_version="ridge-v1",
            materialized_sha256=digest,
            perturbed_materialized_sha256="b" * 64,
        )
        ledger = StrictMappingLedger()
        ledger.add_target(fitted)
        ledger.add_source(
            SourceConsumption(
                "source",
                SourceDisposition.CONSUMED,
                ("target",),
                (1,),
                "float32",
                digest,
                "fit input",
            )
        )
        with self.assertRaisesRegex(CoverageError, "coverage is incomplete"):
            ledger.validate(
                source_names=("source", "missing"), target_names=("target",)
            )
        with self.assertRaisesRegex(CoverageError, "consumed more than once"):
            StrictMappingLedger().add_source(
                SourceConsumption(
                    "source",
                    SourceDisposition.CONSUMED,
                    ("target-a", "target-b"),
                    (1,),
                    "float32",
                    digest,
                    "ambiguous reuse",
                )
            )
        unchanged = replace(
            fitted, perturbed_materialized_sha256=fitted.materialized_sha256
        )
        with self.assertRaisesRegex(CoverageError, "did not affect fitted target"):
            StrictMappingLedger().add_target(unchanged)


if __name__ == "__main__":
    unittest.main()
