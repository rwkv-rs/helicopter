from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

import helicopter_eval
from helicopter_cli import __main__ as helicopter_main
from helicopter_eval import runner as evaluation_runner
from helicopter_eval.config import (
    EvaluationConfigurationError,
    load_evaluation_config,
    load_evaluation_environment,
    resolve_weights,
    verify_weight_identity,
)
from helicopter_eval.domains import assign_domain
from helicopter_eval.plan import WKV_MODES, build_plan, build_shards
from helicopter_eval.registry import (
    RegistrySnapshot,
    RegistryTask,
    _snapshot_registry,
    load_default_registry,
)
from helicopter_eval.runner import run


def _environment(tmp_path: Path, token: str = "do-not-print") -> dict[str, str]:
    weight_root = tmp_path / "weights"
    weight_root.mkdir()
    (tmp_path / "staging").mkdir(mode=0o700)
    return {
        "WEIGHT_PATH": str(weight_root),
        "HELICOPTER_SCOREBOARD_URL": "https://scoreboard.example.test",
        "HELICOPTER_SCOREBOARD_TOKEN": token,
        "HELICOPTER_EVAL_STAGING_ROOT": str(tmp_path / "staging"),
    }


def _config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "lighteval.toml"
    path.write_text(body, encoding="utf-8")
    return path


def _task(name: str, module: str = "module") -> RegistryTask:
    return RegistryTask(
        identity=f"{name}|0",
        name=name,
        version="0",
        module_family=module,
        module=f"lighteval.tasks.tasks.{module}",
        dataset=f"dataset/{name}",
        subset="default",
        evaluation_splits=("test",),
        languages=("english",),
        upstream_tags=("knowledge",),
        primary_domain="knowledge",
    )


def _registry(tasks: tuple[RegistryTask, ...]) -> RegistrySnapshot:
    return RegistrySnapshot(
        lighteval_version="0.13.0",
        tasks=tasks,
        module_count=len({task.module_family for task in tasks}),
        digest="a" * 64,
        domain_rules_version="test",
        domain_rules_digest="b" * 64,
        unknown_domain_modules=(),
    )


def test_config_accepts_only_schema_and_unique_weights(tmp_path: Path) -> None:
    path = _config(
        tmp_path,
        'schema_version = 1\nweights = ["a.pth", "nested/b.pth"]\n',
    )
    assert load_evaluation_config(path).weights == ("a.pth", "nested/b.pth")

    for field in ("benchmarks", "tasks", "exclude", "max_samples", "wkv_mode"):
        path.write_text(
            f'schema_version = 1\nweights = ["a.pth"]\n{field} = []\n',
            encoding="utf-8",
        )
        with pytest.raises(EvaluationConfigurationError, match="unknown"):
            load_evaluation_config(path)


@pytest.mark.parametrize(
    "body, message",
    [
        ("schema_version = 2\nweights = ['a.pth']\n", "schema_version"),
        ("schema_version = 1\nweights = []\n", "non-empty"),
        (
            "schema_version = 1\nweights = [' a.pth']\n",
            "trimmed",
        ),
        (
            "schema_version = 1\nweights = ['a.pth', 'a.pth']\n",
            "duplicate",
        ),
    ],
)
def test_config_rejects_invalid_public_contract(
    tmp_path: Path, body: str, message: str
) -> None:
    with pytest.raises(EvaluationConfigurationError, match=message):
        load_evaluation_config(_config(tmp_path, body))


def test_weight_resolution_is_bounded_and_content_addressed(tmp_path: Path) -> None:
    env = _environment(tmp_path)
    root = Path(env["WEIGHT_PATH"])
    (root / "a.pth").write_bytes(b"first")
    (root / "nested").mkdir()
    (root / "nested/b.pth").write_bytes(b"second")
    config = load_evaluation_config(
        _config(
            tmp_path,
            'schema_version = 1\nweights = ["a.pth", "nested/b.pth"]\n',
        )
    )
    resolved = resolve_weights(config, load_evaluation_environment(env))
    assert [weight.display_name for weight in resolved] == ["a.pth", "b.pth"]
    assert all(len(weight.sha256) == 64 for weight in resolved)

    outside = tmp_path / "outside.pth"
    outside.write_bytes(b"outside")
    escaped = load_evaluation_config(
        _config(tmp_path, 'schema_version = 1\nweights = ["../outside.pth"]\n')
    )
    with pytest.raises(EvaluationConfigurationError, match="normalized"):
        resolve_weights(escaped, load_evaluation_environment(env))

    for unnormalized in ("nested/./b.pth", "nested//b.pth"):
        invalid = load_evaluation_config(
            _config(
                tmp_path,
                f'schema_version = 1\nweights = ["{unnormalized}"]\n',
            )
        )
        with pytest.raises(EvaluationConfigurationError, match="normalized"):
            resolve_weights(invalid, load_evaluation_environment(env))

    link = root / "linked.pth"
    link.symlink_to(root / "a.pth")
    linked = load_evaluation_config(
        _config(tmp_path, 'schema_version = 1\nweights = ["linked.pth"]\n')
    )
    with pytest.raises(EvaluationConfigurationError, match="symlinks"):
        resolve_weights(linked, load_evaluation_environment(env))

    (root / "a.pth").write_bytes(b"replaced after preflight")
    with pytest.raises(
        EvaluationConfigurationError,
        match="content changed after preflight",
    ):
        verify_weight_identity(resolved[0])


def test_environment_rejects_overlapping_roots_and_url_credentials(
    tmp_path: Path,
) -> None:
    env = _environment(tmp_path)
    env["HELICOPTER_EVAL_STAGING_ROOT"] = str(Path(env["WEIGHT_PATH"]) / "staging")
    with pytest.raises(EvaluationConfigurationError, match="must not overlap"):
        load_evaluation_environment(env)

    other = tmp_path / "other"
    other.mkdir()
    env = _environment(other)
    env["HELICOPTER_SCOREBOARD_URL"] = "https://token@example.test"
    with pytest.raises(EvaluationConfigurationError, match="without credentials"):
        load_evaluation_environment(env)


def test_environment_never_changes_permissions_of_existing_staging_root(
    tmp_path: Path,
) -> None:
    weight_root = tmp_path / "weights"
    weight_root.mkdir()
    staging_root = tmp_path / "shared-staging"
    staging_root.mkdir(mode=0o755)
    staging_root.chmod(0o755)
    env = {
        "WEIGHT_PATH": str(weight_root),
        "HELICOPTER_SCOREBOARD_URL": "https://scoreboard.example.test",
        "HELICOPTER_SCOREBOARD_TOKEN": "do-not-print",
        "HELICOPTER_EVAL_STAGING_ROOT": str(staging_root),
    }

    with pytest.raises(
        EvaluationConfigurationError,
        match="owned by the current user and have mode 0700",
    ):
        load_evaluation_environment(env)

    assert stat.S_IMODE(staging_root.stat().st_mode) == 0o755


def test_environment_rejects_product_root_as_staging(
    tmp_path: Path,
) -> None:
    env = _environment(tmp_path)
    env["HELICOPTER_EVAL_STAGING_ROOT"] = str(
        Path(helicopter_eval.__file__).resolve().parents[3]
    )

    with pytest.raises(
        EvaluationConfigurationError,
        match="must be a dedicated child directory",
    ):
        load_evaluation_environment(env)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("WEIGHT_PATH", "relative/weights", "WEIGHT_PATH must be an absolute"),
        (
            "HELICOPTER_EVAL_STAGING_ROOT",
            "relative/staging",
            "STAGING_ROOT must be an absolute",
        ),
        (
            "HELICOPTER_SCOREBOARD_URL",
            "https://example.test/path?token=secret",
            "without credentials, query, or fragment",
        ),
        (
            "HELICOPTER_SCOREBOARD_URL",
            "https://example.test/path#secret",
            "without credentials, query, or fragment",
        ),
        (
            "HELICOPTER_SCOREBOARD_TOKEN",
            "token with spaces",
            "visible ASCII",
        ),
        (
            "HELICOPTER_SCOREBOARD_TOKEN",
            "token-\u00e9",
            "visible ASCII",
        ),
    ],
)
def test_environment_rejects_relative_roots_and_unsafe_urls(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    env = _environment(tmp_path)
    env[field] = value
    with pytest.raises(EvaluationConfigurationError, match=message):
        load_evaluation_environment(env)


def test_eval_cli_rejects_readable_private_environment_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env_file = tmp_path / ".env.local"
    env_file.write_text("HELICOPTER_SCOREBOARD_TOKEN=secret\n", encoding="utf-8")
    env_file.chmod(0o640)
    monkeypatch.setattr(helicopter_main, "find_root", lambda: tmp_path)
    monkeypatch.setattr(
        helicopter_main,
        "find_env_path",
        lambda _root, _path, **_kwargs: env_file,
    )
    monkeypatch.setattr(
        helicopter_main,
        "load_env",
        lambda _root, _path, **_kwargs: ({}, env_file),
    )

    with pytest.raises(SystemExit) as raised:
        helicopter_main.main(["eval", "--config", "lighteval.toml", "--dry-run"])

    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert "owned by the current user" in error
    assert "have mode 0600" in error


def test_eval_cli_rejects_private_environment_owned_by_another_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env_file = tmp_path / ".env.local"
    env_file.write_text("HELICOPTER_SCOREBOARD_TOKEN=secret\n", encoding="utf-8")
    env_file.chmod(0o600)
    monkeypatch.setattr(helicopter_main, "find_root", lambda: tmp_path)
    monkeypatch.setattr(
        helicopter_main,
        "find_env_path",
        lambda _root, _path, **_kwargs: env_file,
    )
    actual_uid = os.geteuid()
    monkeypatch.setattr(
        helicopter_main.os,
        "geteuid",
        lambda: actual_uid + 1,
    )
    loaded = False

    def unexpected_load(_root, _path, **_kwargs):
        nonlocal loaded
        loaded = True
        return {}, env_file

    monkeypatch.setattr(helicopter_main, "load_env", unexpected_load)

    with pytest.raises(SystemExit) as raised:
        helicopter_main.main(["eval", "--config", "lighteval.toml", "--dry-run"])

    assert raised.value.code == 2
    assert "owned by the current user" in capsys.readouterr().err
    assert loaded is False


def test_eval_cli_rejects_symlink_before_loading_private_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "secret.env"
    target.write_text("HELICOPTER_SCOREBOARD_TOKEN=secret\n", encoding="utf-8")
    target.chmod(0o600)
    link = tmp_path / ".env.local"
    link.symlink_to(target)
    loaded = False

    def unexpected_load(_root, _path, **_kwargs):
        nonlocal loaded
        loaded = True
        return {}, link

    monkeypatch.setattr(helicopter_main, "find_root", lambda: tmp_path)
    monkeypatch.setattr(helicopter_main, "load_env", unexpected_load)

    with pytest.raises(SystemExit) as raised:
        helicopter_main.main(["eval", "--config", "lighteval.toml", "--dry-run"])

    assert raised.value.code == 2
    assert "regular non-symlink" in capsys.readouterr().err
    assert loaded is False


def test_eval_cli_rejects_broken_private_environment_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    link = tmp_path / ".env.local"
    link.symlink_to(tmp_path / "missing.env")
    loaded = False

    def unexpected_load(_root, _path, **_kwargs):
        nonlocal loaded
        loaded = True
        return {}, link

    monkeypatch.setattr(helicopter_main, "find_root", lambda: tmp_path)
    monkeypatch.setattr(helicopter_main, "load_env", unexpected_load)

    with pytest.raises(SystemExit) as raised:
        helicopter_main.main(["eval", "--config", "lighteval.toml", "--dry-run"])

    assert raised.value.code == 2
    assert "regular non-symlink" in capsys.readouterr().err
    assert loaded is False


def test_eval_cli_resolves_config_relative_to_invocation_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    invocation = tmp_path / "invocation"
    repository.mkdir()
    invocation.mkdir()
    config = invocation / "lighteval.toml"
    config.write_text(
        'schema_version = 1\nweights = ["model.pth"]\n',
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    monkeypatch.chdir(invocation)
    monkeypatch.setattr(helicopter_main, "find_root", lambda: repository)
    monkeypatch.setattr(
        helicopter_main,
        "find_env_path",
        lambda _root, _path, **_kwargs: None,
    )
    monkeypatch.setattr(
        helicopter_main,
        "load_env",
        lambda _root, _path, **_kwargs: ({"PRIVATE": "value"}, None),
    )
    monkeypatch.setattr(
        helicopter_eval,
        "run",
        lambda **kwargs: captured.update(kwargs) or 0,
    )

    assert (
        helicopter_main.main(["eval", "--config", "./lighteval.toml", "--dry-run"]) == 0
    )
    assert captured == {
        "config_path": config,
        "env": {"PRIVATE": "value"},
        "dry_run": True,
    }


def test_plan_is_weight_ordered_paired_and_deterministic(tmp_path: Path) -> None:
    env = _environment(tmp_path)
    root = Path(env["WEIGHT_PATH"])
    (root / "a.pth").write_bytes(b"a")
    (root / "b.pth").write_bytes(b"b")
    config = load_evaluation_config(
        _config(
            tmp_path,
            'schema_version = 1\nweights = ["a.pth", "b.pth"]\n',
        )
    )
    weights = resolve_weights(config, load_evaluation_environment(env))
    registry = _registry(tuple(_task(f"task-{index}") for index in range(25)))
    plan = build_plan(config, weights, registry)

    assert [(unit.weight.display_name, unit.wkv_mode) for unit in plan.units] == [
        ("a.pth", WKV_MODES[0]),
        ("a.pth", WKV_MODES[1]),
        ("b.pth", WKV_MODES[0]),
        ("b.pth", WKV_MODES[1]),
    ]
    assert len(build_shards(registry)) == 25
    assert plan.expected_task_count == 100
    assert build_plan(config, weights, registry) == plan


def test_domain_rules_use_tags_override_and_other() -> None:
    assert assign_domain("ordinary", ["math"]).primary_domain == "math"
    assert (
        assign_domain("wikitext", ["language-modeling"]).primary_domain == "knowledge"
    )
    assert assign_domain("aa_omniscience", []).primary_domain == "knowledge"
    assert assign_domain("new_upstream", ["unmapped"]).primary_domain == "other"


def test_dry_run_redacts_token_and_does_not_create_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    token = "private-scoreboard-token"
    env = _environment(tmp_path, token=token)
    root = Path(env["WEIGHT_PATH"])
    (root / "a.pth").write_bytes(b"a")
    staging = Path(env["HELICOPTER_EVAL_STAGING_ROOT"])
    staging.rmdir()
    config = _config(tmp_path, 'schema_version = 1\nweights = ["a.pth"]\n')
    monkeypatch.setattr(
        "helicopter_eval.runner.load_default_registry",
        lambda: _registry((_task("one"), _task("two"))),
    )
    monkeypatch.setattr(
        "helicopter_eval.runner.run_preflight",
        lambda environment: {
            "scoreboard": {
                "url": environment.scoreboard_url,
                "status": "ready",
            }
        },
    )

    assert run(config_path=config, env=env, dry_run=True) == 0
    output = capsys.readouterr().out
    assert token not in output
    assert "[REDACTED]" in output
    assert not staging.exists()
    payload = json.loads(output)
    assert payload["plan"]["execution_unit_count"] == 2
    assert payload["plan"]["expected_task_count"] == 4
    assert [task["identity"] for task in payload["plan"]["registry"]["tasks"]] == [
        "one|0",
        "two|0",
    ]
    assert payload["plan"]["shards"][0]["task_identities"] == ["one|0"]
    assert payload["plan"]["shards"][1]["task_identities"] == ["two|0"]


def test_eval_process_environment_reaches_native_dependencies_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HELICOPTER_EXISTING_TEST_VALUE", "before")
    monkeypatch.delenv("HELICOPTER_TASK_CREDENTIAL", raising=False)

    with evaluation_runner._process_environment(
        {
            "HELICOPTER_EXISTING_TEST_VALUE": "during",
            "HELICOPTER_TASK_CREDENTIAL": "private",
        }
    ):
        assert os.environ["HELICOPTER_EXISTING_TEST_VALUE"] == "during"
        assert os.environ["HELICOPTER_TASK_CREDENTIAL"] == "private"

    assert os.environ["HELICOPTER_EXISTING_TEST_VALUE"] == "before"
    assert "HELICOPTER_TASK_CREDENTIAL" not in os.environ


def test_default_registry_snapshot_is_complete_and_non_multilingual() -> None:
    snapshot = load_default_registry()
    assert snapshot.lighteval_version == "0.13.0"
    assert len(snapshot.tasks) == 646
    assert snapshot.module_count == 96
    assert len({task.identity for task in snapshot.tasks}) == 646
    assert all(".multilingual." not in task.module for task in snapshot.tasks)
    assert all(task.evaluation_splits for task in snapshot.tasks)
    assert any(task.module_family == "aa_omniscience" for task in snapshot.tasks)
    assert (
        next(
            task for task in snapshot.tasks if task.module_family == "aa_omniscience"
        ).primary_domain
        == "knowledge"
    )


def test_registry_snapshot_automatically_includes_new_default_tasks() -> None:
    def config(name: str):
        return SimpleNamespace(
            name=name,
            full_name=f"{name}|0",
            version=0,
            hf_repo=f"dataset/{name}",
            hf_subset="default",
            evaluation_splits=("test",),
        )

    class FakeRegistry:
        def __init__(self, names: tuple[str, ...]):
            self.names = names

        def load_tasks(self):
            return {
                f"{name}|0": SimpleNamespace(config=config(name)) for name in self.names
            }

        def get_tasks_dump(self):
            return [
                {
                    "module": "lighteval.tasks.tasks.fixture",
                    "docstring": {
                        "languages": ["english"],
                        "tags": ["knowledge"],
                    },
                    "tasks": [{"name": name} for name in self.names],
                }
            ]

    before = _snapshot_registry(FakeRegistry(("first",)), "0.13.0")
    after = _snapshot_registry(FakeRegistry(("first", "new")), "0.13.0")

    assert [task.name for task in before.tasks] == ["first"]
    assert [task.name for task in after.tasks] == ["first", "new"]
    assert before.digest != after.digest


def test_registry_snapshot_rejects_task_without_evaluation_split() -> None:
    config = SimpleNamespace(
        name="empty-split",
        full_name="empty-split|0",
        version=0,
        hf_repo="dataset/empty-split",
        hf_subset="default",
        evaluation_splits=(),
    )

    class FakeRegistry:
        def load_tasks(self):
            return {"empty-split": SimpleNamespace(config=config)}

        def get_tasks_dump(self):
            return [
                {
                    "module": "lighteval.tasks.tasks.synthetic",
                    "docstring": {
                        "tags": ["knowledge"],
                        "languages": ["english"],
                    },
                    "tasks": [{"name": "empty-split"}],
                }
            ]

    with pytest.raises(EvaluationConfigurationError, match="no evaluation split"):
        _snapshot_registry(FakeRegistry(), "0.13.0")


@pytest.mark.parametrize(
    ("splits", "message"),
    [
        ((" test",), "invalid evaluation splits"),
        (("test", "test"), "duplicate evaluation splits"),
    ],
)
def test_registry_snapshot_rejects_unsafe_evaluation_splits(
    splits: tuple[str, ...],
    message: str,
) -> None:
    config = SimpleNamespace(
        name="invalid-split",
        full_name="invalid-split|0",
        version=0,
        hf_repo="dataset/invalid-split",
        hf_subset="default",
        evaluation_splits=splits,
    )

    class FakeRegistry:
        def load_tasks(self):
            return {"invalid-split": SimpleNamespace(config=config)}

        def get_tasks_dump(self):
            return [
                {
                    "module": "lighteval.tasks.tasks.synthetic",
                    "docstring": {
                        "tags": ["knowledge"],
                        "languages": ["english", "english"],
                    },
                    "tasks": [{"name": "invalid-split"}],
                }
            ]

    with pytest.raises(EvaluationConfigurationError, match=message):
        _snapshot_registry(FakeRegistry(), "0.13.0")
