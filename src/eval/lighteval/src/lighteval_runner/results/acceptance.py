from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .artifacts import verify_manifest


def verify_acceptance_runs(
    runs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    verification: dict[str, dict[str, Any]] = {}
    for label, record in runs.items():
        manifest_path = Path(record["manifests"][0])
        manifest = verify_manifest(manifest_path)
        samples = json.loads(
            (manifest_path.parent / "samples.json").read_text(encoding="utf-8")
        )
        summary = json.loads(
            (manifest_path.parent / "summary.json").read_text(encoding="utf-8")
        )
        verification[label] = {
            "status": manifest.status.value,
            "accounting": dict(manifest.accounting),
            "identities": manifest.identities,
            "sample_statuses": [sample["status"] for sample in samples],
            "terminal_reasons": [
                sample.get("generation", {}).get("terminal_reason")
                for sample in samples
            ],
            "repair_actions": [
                sample.get("scoring", {}).get("repair_action") for sample in samples
            ],
            "summary": summary,
        }
        if manifest.status.value != "completed":
            raise AssertionError(f"{label} did not complete")
    return verification
