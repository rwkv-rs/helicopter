from __future__ import annotations

import hashlib
import json

from lighteval_runner.execution import RunStatus, SampleAccounting
from lighteval_runner.results.acceptance import verify_acceptance_runs
from lighteval_runner.results.artifacts import RunArtifacts


def test_remote_acceptance_verifies_deserialized_accounting(tmp_path) -> None:
    identity = {"task": "test"}
    identity_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifacts = RunArtifacts(tmp_path, run_id="accepted-run")
    artifacts.write_json(
        "samples.json",
        [
            {
                "sample_id": "sample-1",
                "status": "scored",
                "generation": {"terminal_reason": "stop"},
                "scoring": {"repair_action": "none"},
            }
        ],
    )
    artifacts.write_json("summary.json", {"metrics": {"score": 1.0}})
    manifest_path = artifacts.finalize(
        status=RunStatus.COMPLETED,
        identity_digest=identity_digest,
        identities={"run": identity},
        accounting=SampleAccounting(1, 1, 0, 1, 1, 0, 1),
    )

    verification = verify_acceptance_runs(
        {"math-A": {"manifests": [str(manifest_path)]}}
    )

    assert verification["math-A"]["status"] == "completed"
    assert verification["math-A"]["accounting"]["scored"] == 1
