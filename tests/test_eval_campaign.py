from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from helicopter_eval import campaign
from helicopter_eval.config import (
    EvaluationConfig,
    EvaluationEnvironment,
    WeightIdentity,
)
from helicopter_eval.manifest import (
    ManifestError,
    ManifestStore,
    campaign_directory,
    remove_acknowledged_shard,
    remove_campaign_child_directory,
)
from helicopter_eval.plan import build_plan
from helicopter_eval.registry import RegistrySnapshot, RegistryTask


def _plan(tmp_path: Path):
    weight_path = tmp_path / "weight.pth"
    weight_path.write_bytes(b"weight")
    weight = WeightIdentity(
        configured_path="weight.pth",
        path=weight_path,
        display_name="weight.pth",
        sha256=hashlib.sha256(b"weight").hexdigest(),
    )
    task = RegistryTask(
        identity="gsm8k|0",
        name="gsm8k",
        version="0",
        module_family="gsm8k",
        module="lighteval.tasks.tasks.gsm8k",
        dataset="openai/gsm8k",
        subset="main",
        evaluation_splits=("test",),
        languages=("english",),
        upstream_tags=("math",),
        primary_domain="math",
    )
    registry = RegistrySnapshot(
        lighteval_version="0.13.0",
        tasks=(task,),
        module_count=1,
        digest="a" * 64,
        domain_rules_version="test",
        domain_rules_digest="b" * 64,
        unknown_domain_modules=(),
    )
    return build_plan(
        EvaluationConfig(schema_version=1, weights=("weight.pth",)),
        (weight,),
        registry,
    )


def _environment(tmp_path: Path) -> EvaluationEnvironment:
    return EvaluationEnvironment(
        weight_root=tmp_path,
        scoreboard_url="https://scoreboard.test",
        scoreboard_token="secret",
        staging_root=tmp_path / "staging",
    )


def test_resume_key_includes_resolved_weight_digest(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    replacement_weight = replace(
        plan.units[0].weight,
        sha256="f" * 64,
    )
    replacement_units = tuple(
        replace(unit, weight=replacement_weight) for unit in plan.units
    )
    changed_weight_plan = replace(plan, units=replacement_units)

    assert changed_weight_plan.config_digest == plan.config_digest
    assert campaign._resume_key(changed_weight_plan) != campaign._resume_key(plan)


def test_cleanup_only_removes_exact_campaign_child(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    campaign_id = "campaign"
    directory = campaign_directory(staging, campaign_id)
    shard = directory / "weight" / "fp16" / "shard"
    shard.mkdir(parents=True)
    (shard / "results.json").write_text("content", encoding="utf-8")
    sibling = staging / "do-not-delete"
    sibling.mkdir(parents=True)

    remove_acknowledged_shard(
        staging_root=staging,
        campaign_id=campaign_id,
        shard_path="weight/fp16/shard",
    )
    assert not shard.exists()
    assert sibling.exists()

    with pytest.raises(ManifestError, match="not normalized"):
        remove_acknowledged_shard(
            staging_root=staging,
            campaign_id=campaign_id,
            shard_path=".",
        )
    with pytest.raises(ManifestError, match="normalized"):
        remove_acknowledged_shard(
            staging_root=staging,
            campaign_id=campaign_id,
            shard_path="../../do-not-delete",
        )


def test_symlink_shard_is_never_deleted(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    directory = campaign_directory(staging, "campaign")
    directory.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = directory / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ManifestError, match="escapes|symlink"):
        remove_acknowledged_shard(
            staging_root=staging,
            campaign_id="campaign",
            shard_path="link",
        )
    assert outside.exists()


def test_resume_removes_only_registered_interrupted_attempt(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    environment = _environment(tmp_path)
    resume_key = campaign._resume_key(plan)
    manifest = campaign._new_manifest(
        plan,
        resume_key,
        "11111111-1111-1111-1111-111111111111",
    )
    unit = plan.units[0]
    shard = unit.shards[0]
    key = f"{unit.weight.sha256}:{unit.wkv_mode}:{shard.shard_id}"
    directory = campaign_directory(
        environment.staging_root,
        manifest.campaign_id,
    )
    attempt = directory / "weight" / "fp16" / "attempt-interrupted"
    attempt.mkdir(parents=True)
    (attempt / "partial").write_text("incomplete", encoding="utf-8")
    sibling = directory / "unregistered"
    sibling.mkdir()
    (sibling / "preserve").write_text("unknown", encoding="utf-8")
    manifest.attempted_shard_paths[key] = str(attempt.relative_to(directory))

    campaign._cleanup_interrupted_attempts(
        manifest=manifest,
        environment=environment,
    )

    assert manifest.attempted_shard_paths == {}
    assert not attempt.exists()
    assert (sibling / "preserve").read_text(encoding="utf-8") == "unknown"


def test_failed_shard_persists_only_exception_type(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    environment = _environment(tmp_path)
    resume_key = campaign._resume_key(plan)
    manifest = campaign._new_manifest(
        plan,
        resume_key,
        "11111111-1111-1111-1111-111111111111",
    )
    store = ManifestStore(environment.staging_root, resume_key)
    unit = plan.units[0]
    shard = unit.shards[0]
    directory = campaign_directory(
        environment.staging_root,
        manifest.campaign_id,
    )
    attempt = directory / "weight" / unit.wkv_mode / "failed"
    attempt.mkdir(parents=True)
    model_execution = {"wkv_mode": unit.wkv_mode}
    campaign._record_shard_attempt(
        store=store,
        manifest=manifest,
        campaign_dir=directory,
        unit=unit,
        shard=shard,
        shard_dir=attempt,
        model_execution=model_execution,
    )

    campaign._record_shard_failure(
        store=store,
        manifest=manifest,
        campaign_dir=directory,
        unit=unit,
        failure=SimpleNamespace(
            shard=shard,
            path=attempt,
            error_type="builtins.RuntimeError",
            message="must-not-be-persisted",
        ),
    )

    key = f"{unit.weight.sha256}:{unit.wkv_mode}:{shard.shard_id}"
    persisted = store.load()
    assert persisted is not None
    assert key not in persisted.attempted_shard_paths
    assert persisted.failed_shard_errors[key] == "builtins.RuntimeError"
    assert "must-not-be-persisted" not in store.path.read_text(encoding="utf-8")


def test_resume_removes_only_registered_model_runtime(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    environment = _environment(tmp_path)
    resume_key = campaign._resume_key(plan)
    manifest = campaign._new_manifest(
        plan,
        resume_key,
        "11111111-1111-1111-1111-111111111111",
    )
    unit = plan.units[0]
    unit_key = f"{unit.weight.sha256}:{unit.wkv_mode}"
    directory = campaign_directory(
        environment.staging_root,
        manifest.campaign_id,
    )
    runtime = directory / "runtime" / unit.weight.sha256 / unit.wkv_mode
    runtime.mkdir(parents=True)
    (runtime / "cache").write_text("interrupted", encoding="utf-8")
    sibling = directory / "unregistered-runtime"
    sibling.mkdir()
    (sibling / "preserve").write_text("unknown", encoding="utf-8")
    manifest.runtime_paths[unit_key] = str(runtime.relative_to(directory))

    campaign._cleanup_interrupted_runtimes(
        manifest=manifest,
        environment=environment,
    )

    assert manifest.runtime_paths == {}
    assert not runtime.exists()
    assert (sibling / "preserve").read_text(encoding="utf-8") == "unknown"


def test_manifest_schema_rejects_unsafe_paths_and_digest_state_overlap() -> None:
    raw = {
        "version": 1,
        "resume_key": "1" * 64,
        "campaign_id": "11111111-1111-1111-1111-111111111111",
        "config_digest": "2" * 64,
        "registry_digest": "3" * 64,
        "domain_rules_digest": "4" * 64,
        "eval_contract_digest": "5" * 64,
        "weight_sha256": ["6" * 64],
        "registry_task_identities": ["task|0"],
        "shard_paths": {"shard": "../../outside"},
        "attempted_shard_paths": {},
        "failed_shard_paths": {},
        "failed_shard_errors": {},
        "runtime_paths": {},
        "model_executions": {},
        "pending_task_digests": {},
        "acknowledged_task_digests": {},
    }
    with pytest.raises(ManifestError, match="unsafe path"):
        campaign.CampaignManifest.from_json(raw)

    raw["shard_paths"] = {}
    raw["pending_task_digests"] = {"task": "a" * 64}
    raw["acknowledged_task_digests"] = {"task": "a" * 64}
    with pytest.raises(ManifestError, match="states overlap"):
        campaign.CampaignManifest.from_json(raw)


def test_manifest_store_wraps_atomic_persistence_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path)
    resume_key = campaign._resume_key(plan)
    store = ManifestStore(tmp_path / "staging", resume_key)
    manifest = campaign._new_manifest(
        plan,
        resume_key,
        "11111111-1111-1111-1111-111111111111",
    )

    def fail_write(*_args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(
        "helicopter_eval.manifest._write_json_atomic",
        fail_write,
    )

    with pytest.raises(ManifestError, match="cannot persist campaign manifest"):
        store.save(manifest)


def test_manifest_store_rejects_non_standard_json_constants(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    resume_key = campaign._resume_key(plan)
    store = ManifestStore(tmp_path / "staging", resume_key)
    manifest = campaign._new_manifest(
        plan,
        resume_key,
        "11111111-1111-1111-1111-111111111111",
    )
    store.save(manifest)
    raw = store.path.read_text(encoding="utf-8").replace(
        '"model_executions": {}',
        '"model_executions": {"unit": {"capacity": NaN}}',
    )
    store.path.write_text(raw, encoding="utf-8")

    with pytest.raises(ManifestError, match="non-standard JSON constant"):
        store.load()


def test_manifest_plan_rejects_runtime_path_for_a_different_unit(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    resume_key = campaign._resume_key(plan)
    manifest = campaign._new_manifest(
        plan,
        resume_key,
        "11111111-1111-1111-1111-111111111111",
    )
    unit = plan.units[0]
    unit_key = f"{unit.weight.sha256}:{unit.wkv_mode}"
    manifest.runtime_paths[unit_key] = f"runtime/{unit.weight.sha256}/fp32io16"

    with pytest.raises(
        ManifestError,
        match="runtime path does not match its unit",
    ):
        campaign._validate_manifest_plan(manifest, plan)


def test_manifest_plan_rejects_shard_state_without_model_execution(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    resume_key = campaign._resume_key(plan)
    manifest = campaign._new_manifest(
        plan,
        resume_key,
        "11111111-1111-1111-1111-111111111111",
    )
    unit = plan.units[0]
    shard = unit.shards[0]
    shard_key = f"{unit.weight.sha256}:{unit.wkv_mode}:{shard.shard_id}"
    manifest.shard_paths[shard_key] = "fp16/shard"

    with pytest.raises(
        ManifestError,
        match="lacks model execution metadata",
    ):
        campaign._validate_manifest_plan(manifest, plan)


def test_manifest_plan_requires_exact_weight_and_registry_snapshots(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    resume_key = campaign._resume_key(plan)
    manifest = campaign._new_manifest(
        plan,
        resume_key,
        "11111111-1111-1111-1111-111111111111",
    )
    manifest.registry_task_identities = ["different|0"]

    with pytest.raises(ManifestError, match="registry snapshot"):
        campaign._validate_manifest_plan(manifest, plan)


def test_backend_commit_before_local_ack_recovers_without_recompute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path)
    environment = _environment(tmp_path)
    backend_digests: dict[str, str] = {}
    evaluated: list[str] = []
    fail_after_commit = True
    completed = False
    campaign_id = "11111111-1111-1111-1111-111111111111"

    class FakeClient:
        def __init__(self, _environment):
            pass

        def create_campaign(self, payload, resume_key):
            return {
                "campaign_id": campaign_id,
                "disposition": "resumed" if backend_digests else "created",
                "status": "incomplete",
                "expected_task_count": plan.expected_task_count,
                "acknowledged_task_digests": dict(backend_digests),
            }

        def campaign_status(self, _campaign_id):
            expected = [
                campaign._task_identity(unit, task.identity)
                for unit in plan.units
                for task in plan.registry.tasks
            ]
            return {
                "campaign_id": campaign_id,
                "status": "complete" if completed else "incomplete",
                "expected_task_count": plan.expected_task_count,
                "acknowledged_task_digests": dict(backend_digests),
                "missing_task_identities": [
                    identity for identity in expected if identity not in backend_digests
                ],
            }

        def publish_task(
            self,
            *,
            campaign_id,
            task_identity,
            payload,
            digest,
        ):
            nonlocal fail_after_commit
            backend_digests[task_identity] = digest
            if fail_after_commit:
                fail_after_commit = False
                raise RuntimeError("simulated lost response")
            return {
                "task_identity": task_identity,
                "content_digest": digest,
                "disposition": "unchanged",
            }

        def finalize(self, _campaign_id):
            nonlocal completed
            completed = True
            return {
                "campaign_id": campaign_id,
                "status": "complete",
                "task_count": plan.expected_task_count,
            }

    def fake_evaluate_unit(
        *,
        unit,
        shards,
        campaign_dir,
        on_shard_started,
        on_shard_completed,
        on_shard_failed,
        on_runtime_started,
        on_runtime_finished,
    ):
        evaluated.append(unit.wkv_mode)
        runtime_dir = campaign_dir / "runtime" / unit.weight.sha256 / unit.wkv_mode
        on_runtime_started(runtime_dir)
        runtime_dir.mkdir(parents=True)
        outputs = []
        for shard in shards:
            shard_dir = campaign_dir / unit.wkv_mode / shard.shard_id.replace(":", "-")
            model_execution = {"wkv_mode": unit.wkv_mode}
            on_shard_started(shard, shard_dir, model_execution)
            shard_dir.mkdir(parents=True)
            (shard_dir / "standard-result").write_text(
                "preserve until ack", encoding="utf-8"
            )
            evaluation = (shard, shard_dir, model_execution)
            outputs.append(evaluation)
            on_shard_completed(evaluation)
        remove_campaign_child_directory(campaign_dir, runtime_dir)
        on_runtime_finished()
        return outputs, []

    def fake_publications(*, campaign_id, unit, shard, **_kwargs):
        identity = campaign._task_identity(unit, shard.tasks[0].identity)
        payload = {
            "campaign_id": campaign_id,
            "identity": identity,
            "mode": unit.wkv_mode,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        return [(identity, payload, digest)]

    monkeypatch.setattr(campaign, "ScoreboardClient", FakeClient)
    monkeypatch.setattr(campaign, "publications_from_shard", fake_publications)
    monkeypatch.setattr(
        "helicopter_eval.lighteval_adapter.evaluate_unit",
        fake_evaluate_unit,
    )

    assert campaign.run_campaign(plan=plan, environment=environment) == 1
    assert evaluated == ["fp16", "fp32io16"]
    assert list(environment.staging_root.glob("campaigns/*.json"))
    assert not list(environment.staging_root.glob("runs/**/standard-result"))

    evaluated.clear()
    assert campaign.run_campaign(plan=plan, environment=environment) == 0
    assert evaluated == []
    assert environment.staging_root.is_dir()
    control_files = list((environment.staging_root / "control").glob("*.json"))
    assert len(control_files) == 1
    control = json.loads(control_files[0].read_text(encoding="utf-8"))
    assert control["content_location"] == "scoreboard-database-only"
    assert "secret" not in json.dumps(control)
    assert not (environment.staging_root / "runs").exists()
    assert not (environment.staging_root / "campaigns").exists()


def test_publication_failure_reuses_persisted_artifact_without_recompute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path)
    environment = _environment(tmp_path)
    campaign_id = "11111111-1111-1111-1111-111111111111"
    backend_digests: dict[str, str] = {}
    evaluated: list[str] = []
    allow_publication = False
    completed = False
    campaign_created = False

    class FakeClient:
        def __init__(self, _environment):
            pass

        def create_campaign(self, payload, resume_key):
            nonlocal campaign_created
            disposition = "resumed" if campaign_created else "created"
            campaign_created = True
            return {
                "campaign_id": campaign_id,
                "disposition": disposition,
                "status": "incomplete",
                "expected_task_count": plan.expected_task_count,
                "acknowledged_task_digests": dict(backend_digests),
            }

        def campaign_status(self, _campaign_id):
            expected = [
                campaign._task_identity(unit, task.identity)
                for unit in plan.units
                for task in plan.registry.tasks
            ]
            return {
                "campaign_id": campaign_id,
                "status": "complete" if completed else "incomplete",
                "expected_task_count": plan.expected_task_count,
                "acknowledged_task_digests": dict(backend_digests),
                "missing_task_identities": [
                    identity for identity in expected if identity not in backend_digests
                ],
            }

        def publish_task(
            self,
            *,
            campaign_id,
            task_identity,
            payload,
            digest,
        ):
            if not allow_publication:
                raise RuntimeError("scoreboard unavailable")
            backend_digests[task_identity] = digest
            return {
                "task_identity": task_identity,
                "content_digest": digest,
                "disposition": "created",
            }

        def finalize(self, _campaign_id):
            nonlocal completed
            completed = True
            return {
                "campaign_id": campaign_id,
                "status": "complete",
                "task_count": plan.expected_task_count,
            }

    def fake_evaluate_unit(
        *,
        unit,
        shards,
        campaign_dir,
        on_shard_started,
        on_shard_completed,
        on_shard_failed,
        on_runtime_started,
        on_runtime_finished,
    ):
        evaluated.append(unit.wkv_mode)
        runtime_dir = campaign_dir / "runtime" / unit.weight.sha256 / unit.wkv_mode
        on_runtime_started(runtime_dir)
        runtime_dir.mkdir(parents=True)
        outputs = []
        for shard in shards:
            shard_dir = campaign_dir / unit.wkv_mode / shard.shard_id.replace(":", "-")
            model_execution = {"wkv_mode": unit.wkv_mode}
            on_shard_started(shard, shard_dir, model_execution)
            shard_dir.mkdir(parents=True)
            (shard_dir / "standard-result").write_text(
                "reuse after failed publication",
                encoding="utf-8",
            )
            evaluation = (shard, shard_dir, model_execution)
            outputs.append(evaluation)
            on_shard_completed(evaluation)
        remove_campaign_child_directory(campaign_dir, runtime_dir)
        on_runtime_finished()
        return outputs, []

    def fake_publications(*, campaign_id, unit, shard, **_kwargs):
        identity = campaign._task_identity(unit, shard.tasks[0].identity)
        payload = {
            "campaign_id": campaign_id,
            "identity": identity,
            "mode": unit.wkv_mode,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        return [(identity, payload, digest)]

    monkeypatch.setattr(campaign, "ScoreboardClient", FakeClient)
    monkeypatch.setattr(campaign, "publications_from_shard", fake_publications)
    monkeypatch.setattr(
        "helicopter_eval.lighteval_adapter.evaluate_unit",
        fake_evaluate_unit,
    )

    assert campaign.run_campaign(plan=plan, environment=environment) == 1
    assert evaluated == ["fp16", "fp32io16"]
    assert len(list(environment.staging_root.glob("runs/**/standard-result"))) == 2

    allow_publication = True
    evaluated.clear()
    assert campaign.run_campaign(plan=plan, environment=environment) == 0
    assert evaluated == []
    assert not (environment.staging_root / "runs").exists()
    assert not (environment.staging_root / "campaigns").exists()


def test_completed_backend_recovery_cleans_locally_without_starting_new_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path)
    environment = _environment(tmp_path)
    old_campaign_id = "11111111-1111-1111-1111-111111111111"
    resume_key = campaign._resume_key(plan)
    store = ManifestStore(environment.staging_root, resume_key)
    manifest = campaign._new_manifest(plan, resume_key, old_campaign_id)
    backend_digests: dict[str, str] = {}

    for unit in plan.units:
        shard = unit.shards[0]
        unit_key = f"{unit.weight.sha256}:{unit.wkv_mode}"
        key = f"{unit.weight.sha256}:{unit.wkv_mode}:{shard.shard_id}"
        identity = campaign._task_identity(unit, shard.tasks[0].identity)
        backend_digests[identity] = hashlib.sha256(identity.encode()).hexdigest()
        manifest.acknowledged_task_digests[identity] = backend_digests[identity]
        manifest.model_executions[unit_key] = {"wkv_mode": unit.wkv_mode}
        success = (
            campaign_directory(environment.staging_root, old_campaign_id)
            / unit.wkv_mode
            / "success"
        )
        failure = (
            campaign_directory(environment.staging_root, old_campaign_id)
            / unit.wkv_mode
            / "failure"
        )
        success.mkdir(parents=True)
        failure.mkdir()
        (success / "results.json").write_text("result", encoding="utf-8")
        (failure / "error.txt").write_text("failure", encoding="utf-8")
        manifest.shard_paths[key] = str(
            success.relative_to(
                campaign_directory(environment.staging_root, old_campaign_id)
            )
        )
        manifest.failed_shard_paths[key] = [
            str(
                failure.relative_to(
                    campaign_directory(
                        environment.staging_root,
                        old_campaign_id,
                    )
                )
            )
        ]
        manifest.failed_shard_errors[key] = "builtins.RuntimeError"
    store.save(manifest)

    class FakeClient:
        def __init__(self, _environment):
            pass

        def campaign_status(self, campaign_id):
            assert campaign_id == old_campaign_id
            return {
                "campaign_id": old_campaign_id,
                "status": "complete",
                "expected_task_count": plan.expected_task_count,
                "acknowledged_task_digests": backend_digests,
                "missing_task_identities": [],
            }

        def create_campaign(self, _payload, _resume_key):
            raise AssertionError("completed recovery must not start a new campaign")

    monkeypatch.setattr(campaign, "ScoreboardClient", FakeClient)

    assert campaign.run_campaign(plan=plan, environment=environment) == 0
    assert not (environment.staging_root / "runs").exists()
    assert not (environment.staging_root / "campaigns").exists()
    assert (environment.staging_root / "control" / f"{old_campaign_id}.json").is_file()


def test_mismatched_local_manifest_is_quarantined_without_touching_its_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path)
    environment = _environment(tmp_path)
    resume_key = campaign._resume_key(plan)
    store = ManifestStore(environment.staging_root, resume_key)
    manifest = campaign._new_manifest(
        plan,
        resume_key,
        "11111111-1111-1111-1111-111111111111",
    )
    manifest.config_digest = "f" * 64
    store.save(manifest)
    old_run = (
        campaign_directory(environment.staging_root, manifest.campaign_id) / "unknown"
    )
    old_run.mkdir(parents=True)
    (old_run / "preserve.txt").write_text("unknown content", encoding="utf-8")

    class CurrentCampaignRequested(RuntimeError):
        pass

    class FakeClient:
        def __init__(self, _environment):
            pass

        def create_campaign(self, payload, requested_resume_key):
            assert requested_resume_key == resume_key
            assert old_run.is_dir()
            assert list(
                (environment.staging_root / "campaigns" / "quarantine").glob(
                    f"{resume_key}.*.json"
                )
            )
            raise CurrentCampaignRequested

    monkeypatch.setattr(campaign, "ScoreboardClient", FakeClient)

    with pytest.raises(CurrentCampaignRequested):
        campaign.run_campaign(plan=plan, environment=environment)
    assert (old_run / "preserve.txt").read_text(encoding="utf-8") == "unknown content"


def test_quarantined_manifest_never_reuses_unregistered_run_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path)
    environment = _environment(tmp_path)
    resume_key = campaign._resume_key(plan)
    campaign_id = "11111111-1111-1111-1111-111111111111"
    store = ManifestStore(environment.staging_root, resume_key)
    manifest = campaign._new_manifest(plan, resume_key, campaign_id)
    manifest.registry_task_identities = ["corrupt|0"]
    store.save(manifest)
    old_run = campaign_directory(environment.staging_root, campaign_id) / "unknown"
    old_run.mkdir(parents=True)
    evidence = old_run / "preserve.txt"
    evidence.write_text("unknown content", encoding="utf-8")

    class FakeClient:
        def __init__(self, _environment):
            pass

        def create_campaign(self, _payload, requested_resume_key):
            assert requested_resume_key == resume_key
            return {
                "campaign_id": campaign_id,
                "disposition": "resumed",
                "status": "incomplete",
                "expected_task_count": plan.expected_task_count,
                "acknowledged_task_digests": {},
            }

    monkeypatch.setattr(campaign, "ScoreboardClient", FakeClient)

    with pytest.raises(
        ManifestError,
        match="without a matching manifest",
    ):
        campaign.run_campaign(plan=plan, environment=environment)

    assert evidence.read_text(encoding="utf-8") == "unknown content"
