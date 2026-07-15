from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from lighteval_runner.execution import RunStatus, SampleAccounting
from lighteval_runner.results.artifacts import (
    RunArtifacts,
    record_publication_attempt,
    verify_manifest,
)


RUN_IDENTITY = {"task": {"name": "test"}}
IDENTITY = hashlib.sha256(
    json.dumps(RUN_IDENTITY, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
IDENTITIES = {"run": RUN_IDENTITY}
ACCOUNTING = SampleAccounting(1, 1, 0, 1, 1, 0, 1)


def test_run_directory_is_exclusive_and_manifest_commits_checksums(tmp_path):
    writer = RunArtifacts(tmp_path, run_id="run")
    entry = writer.write_json("samples/part-0.json", {"answer": 42})
    manifest_path = writer.finalize(
        status=RunStatus.COMPLETED,
        identity_digest=IDENTITY,
        identities=IDENTITIES,
        accounting=ACCOUNTING,
    )
    manifest = json.loads(manifest_path.read_text())

    assert (
        hashlib.sha256((writer.run_dir / entry.relative_path).read_bytes()).hexdigest()
        == entry.sha256
    )
    assert manifest["artifacts"] == [
        {
            "relative_path": entry.relative_path,
            "sha256": entry.sha256,
            "size_bytes": entry.size_bytes,
        }
    ]
    assert not list(writer.run_dir.rglob("*.tmp"))
    assert verify_manifest(manifest_path).identity_digest == IDENTITY
    with pytest.raises(FileExistsError):
        RunArtifacts(tmp_path, run_id="run")


def test_artifacts_cannot_be_overwritten_or_written_after_finalize(tmp_path):
    writer = RunArtifacts(tmp_path)
    writer.write_json("details.json", {"attempt": 1})
    with pytest.raises(FileExistsError):
        writer.write_json("details.json", {"attempt": 2})
    writer.finalize(
        status=RunStatus.FAILED,
        identity_digest=IDENTITY,
        identities=IDENTITIES,
        accounting=ACCOUNTING,
    )
    with pytest.raises(RuntimeError):
        writer.write_json("late.json", {})
    with pytest.raises(RuntimeError):
        writer.finalize(
            status=RunStatus.FAILED,
            identity_digest=IDENTITY,
            identities=IDENTITIES,
            accounting=ACCOUNTING,
        )


@pytest.mark.parametrize(
    "path",
    ["/absolute.json", "../escape.json", "nested/../escape.json", "manifest.json"],
)
def test_artifact_paths_cannot_escape_or_replace_manifest(tmp_path, path):
    writer = RunArtifacts(tmp_path)
    with pytest.raises((ValueError, FileExistsError)):
        writer.write_json(path, {})


@pytest.mark.parametrize("run_id", ["../escape", "/absolute", "", "a/b"])
def test_external_run_id_cannot_escape_output_root(tmp_path, run_id):
    with pytest.raises(ValueError):
        RunArtifacts(tmp_path, run_id=run_id)


def test_manifest_verification_detects_corrupt_artifact_without_directory_discovery(
    tmp_path,
):
    writer = RunArtifacts(tmp_path, run_id="run")
    writer.write_json("samples/evidence.json", {"answer": 42})
    manifest_path = writer.finalize(
        status=RunStatus.COMPLETED,
        identity_digest=IDENTITY,
        identities=IDENTITIES,
        accounting=ACCOUNTING,
    )
    (writer.run_dir / "unrelated-newer-file.json").write_text("ignored")
    assert verify_manifest(manifest_path).run_id == "run"
    (writer.run_dir / "samples/evidence.json").write_text("corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_manifest(manifest_path)


def test_legacy_v1_manifest_remains_verifiable(tmp_path):
    writer = RunArtifacts(tmp_path, run_id="legacy-run")
    writer.write_json("samples.json", [])
    manifest_path = writer.finalize(
        status=RunStatus.COMPLETED,
        identity_digest=IDENTITY,
        identities=IDENTITIES,
        accounting=ACCOUNTING,
    )
    payload = json.loads(manifest_path.read_text())
    payload["schema_version"] = 1
    payload.pop("completed_at")
    manifest_path.write_text(json.dumps(payload))

    manifest = verify_manifest(manifest_path)

    assert manifest.schema_version == 1
    assert manifest.completed_at is None


def test_concurrent_run_creation_has_exactly_one_owner(tmp_path):
    def create():
        try:
            return RunArtifacts(tmp_path, run_id="same-run")
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: create(), range(2)))
    assert sum(result is not None for result in results) == 1


def test_failed_atomic_replace_cleans_temporary_file(tmp_path, monkeypatch):
    writer = RunArtifacts(tmp_path, run_id="replace-failure")

    def fail_replace(_source, _target):
        raise OSError("replace failed")

    monkeypatch.setattr("lighteval_runner.results.artifacts.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        writer.write_json("details.json", {"attempt": 1})
    assert not list(writer.run_dir.glob("*.tmp"))


def test_publication_receipt_is_atomic_and_single_assignment(tmp_path):
    writer = RunArtifacts(tmp_path, run_id="published")
    writer.finalize(
        status=RunStatus.COMPLETED,
        identity_digest=IDENTITY,
        identities=IDENTITIES,
        accounting=ACCOUNTING,
    )
    receipt = record_publication_attempt(
        writer.run_dir / "manifest.json",
        {"status": "failed", "retry_identity": "ingest:digest"},
    )
    assert (
        json.loads(receipt.read_text())["attempts"][0]["retry_identity"]
        == "ingest:digest"
    )
    record_publication_attempt(
        writer.run_dir / "manifest.json",
        {"status": "published", "retry_identity": "ingest:digest"},
    )
    assert len(json.loads(receipt.read_text())["attempts"]) == 2
    with pytest.raises(ValueError, match="identity changed"):
        record_publication_attempt(
            writer.run_dir / "manifest.json",
            {"status": "failed", "retry_identity": "ingest:other"},
        )
