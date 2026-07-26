from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

import helicopter_lighteval
from helicopter_cli import __main__ as helicopter_main
from helicopter_lighteval import runner as evaluation_runner
from helicopter_lighteval.config import (
    EvaluationConfigurationError,
    load_evaluation_config,
    load_evaluation_environment,
    resolve_weights,
    verify_weight_identity,
)
from helicopter_lighteval.plan import WKV_MODES, build_plan, build_shards
from helicopter_lighteval.registry import (
    RegistrySnapshot,
    RegistryTask,
    _inventory,
    _resolve_selectors,
    _snapshot_registry,
    load_configured_registry,
)
from helicopter_lighteval.runner import run


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
        selector=name,
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
    )


def _registry(tasks: tuple[RegistryTask, ...]) -> RegistrySnapshot:
    selectors = tuple(dict.fromkeys(task.selector for task in tasks))
    return RegistrySnapshot(
        lighteval_version="0.13.0",
        configured_selectors=selectors,
        resolved_selectors=selectors,
        skipped_selectors=(),
        tasks=tasks,
        module_count=len({task.module_family for task in tasks}),
        digest="a" * 64,
    )


def test_config_accepts_schema_weights_and_benchmarks(tmp_path: Path) -> None:
    path = _config(
        tmp_path,
        'schema_version = 1\nweights = ["a.pth", "nested/b.pth"]\n'
        'benchmarks = ["mmlu", "gsm8k"]\n',
    )
    config = load_evaluation_config(path)
    assert config.weights == ("a.pth", "nested/b.pth")
    assert config.benchmarks == ("mmlu", "gsm8k")
    assert config.prompt_template == "bot"

    for field in ("tasks", "exclude", "max_samples", "wkv_mode"):
        path.write_text(
            'schema_version = 1\nweights = ["a.pth"]\n'
            f'benchmarks = ["gsm8k"]\n{field} = []\n',
            encoding="utf-8",
        )
        with pytest.raises(EvaluationConfigurationError, match="unknown"):
            load_evaluation_config(path)


@pytest.mark.parametrize(
    "prompt_template",
    ["bot", "assistant", "function_calling"],
)
def test_config_accepts_vllm_rwkv_prompt_templates(
    tmp_path: Path,
    prompt_template: str,
) -> None:
    config = load_evaluation_config(
        _config(
            tmp_path,
            f'prompt_template = "{prompt_template}"\n'
            'schema_version = 1\nweights = ["a.pth"]\n'
            'benchmarks = ["gsm8k"]\n',
        )
    )

    assert config.prompt_template == prompt_template


@pytest.mark.parametrize(
    "body, message",
    [
        (
            "schema_version = 1\nweights = ['a.pth']\n",
            "missing eval config fields: benchmarks",
        ),
        (
            "schema_version = 2\nweights = ['a.pth']\nbenchmarks = ['gsm8k']\n",
            "schema_version",
        ),
        (
            "schema_version = 1\nweights = []\nbenchmarks = ['gsm8k']\n",
            "non-empty",
        ),
        (
            "schema_version = 1\nweights = [' a.pth']\nbenchmarks = ['gsm8k']\n",
            "trimmed",
        ),
        (
            "schema_version = 1\nweights = ['a.pth', 'a.pth']\n"
            "benchmarks = ['gsm8k']\n",
            "duplicate",
        ),
        (
            "schema_version = 1\nweights = ['a.pth']\nbenchmarks = []\n",
            "benchmarks",
        ),
        (
            "schema_version = 1\nweights = ['a.pth']\n"
            "benchmarks = ['gsm8k', 'gsm8k']\n",
            "duplicate benchmarks",
        ),
        (
            "schema_version = 1\nprompt_template = 'unknown'\n"
            "weights = ['a.pth']\nbenchmarks = ['gsm8k']\n",
            "prompt_template",
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
            'schema_version = 1\nweights = ["a.pth", "nested/b.pth"]\n'
            'benchmarks = ["fixture"]\n',
        )
    )
    resolved = resolve_weights(config, load_evaluation_environment(env))
    assert [weight.display_name for weight in resolved] == ["a.pth", "b.pth"]
    assert all(len(weight.sha256) == 64 for weight in resolved)

    outside = tmp_path / "outside.pth"
    outside.write_bytes(b"outside")
    escaped = load_evaluation_config(
        _config(
            tmp_path,
            'schema_version = 1\nweights = ["../outside.pth"]\n'
            'benchmarks = ["fixture"]\n',
        )
    )
    with pytest.raises(EvaluationConfigurationError, match="normalized"):
        resolve_weights(escaped, load_evaluation_environment(env))

    for unnormalized in ("nested/./b.pth", "nested//b.pth"):
        invalid = load_evaluation_config(
            _config(
                tmp_path,
                f'schema_version = 1\nweights = ["{unnormalized}"]\n'
                'benchmarks = ["fixture"]\n',
            )
        )
        with pytest.raises(EvaluationConfigurationError, match="normalized"):
            resolve_weights(invalid, load_evaluation_environment(env))

    link = root / "linked.pth"
    link.symlink_to(root / "a.pth")
    linked = load_evaluation_config(
        _config(
            tmp_path,
            'schema_version = 1\nweights = ["linked.pth"]\nbenchmarks = ["fixture"]\n',
        )
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
        Path(__file__).resolve().parents[1]
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
        'schema_version = 1\nweights = ["model.pth"]\nbenchmarks = ["fixture"]\n',
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
        helicopter_lighteval,
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
            'schema_version = 1\nweights = ["a.pth", "b.pth"]\n'
            'benchmarks = ["fixture"]\n',
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


def test_selectors_expand_exact_and_superset_and_skip_unknown() -> None:
    inventory = _inventory(
        [
            {
                "module": "lighteval.tasks.tasks.fixture",
                "docstring": {"tags": ["knowledge"]},
                "tasks": [
                    {"name": "single"},
                    {"name": "suite:first"},
                    {"name": "suite:second"},
                ],
            }
        ]
    )
    by_task, resolved, skipped = _resolve_selectors(
        ("single", "suite", "missing"),
        inventory,
    )
    assert by_task == {
        "single": "single",
        "suite:first": "suite",
        "suite:second": "suite",
    }
    assert resolved == ("single", "suite")
    assert skipped == ("missing",)


def test_registry_inventory_deduplicates_same_module_but_rejects_ambiguity() -> None:
    repeated = {
        "module": "lighteval.tasks.multilingual.tasks.xnli",
        "tasks": [{"name": "xnli_eng_mcf"}],
    }
    assert list(_inventory([repeated, repeated])) == ["xnli_eng_mcf"]
    with pytest.raises(EvaluationConfigurationError, match="more than once"):
        _inventory(
            [
                repeated,
                {
                    "module": "lighteval.tasks.tasks.other",
                    "tasks": [{"name": "xnli_eng_mcf"}],
                },
            ]
        )
    with pytest.raises(EvaluationConfigurationError, match="more than once"):
        _inventory(
            [
                repeated,
                {
                    "module": repeated["module"],
                    "docstring": {"tags": ["different"]},
                    "tasks": [{"name": "xnli_eng_mcf"}],
                },
            ]
        )


def test_overlapping_selectors_are_rejected() -> None:
    inventory = _inventory(
        [
            {
                "module": "lighteval.tasks.tasks.fixture",
                "tasks": [
                    {"name": "suite:first"},
                    {"name": "suite:second"},
                ],
            }
        ]
    )
    with pytest.raises(EvaluationConfigurationError, match="overlap"):
        _resolve_selectors(("suite", "suite:first"), inventory)
    with pytest.raises(EvaluationConfigurationError, match="none"):
        _resolve_selectors(("missing",), inventory)


def test_multilingual_task_is_included_only_when_selected() -> None:
    inventory = _inventory(
        [
            {
                "module": ("lighteval.tasks.multilingual.tasks.ceval_zho_mcf.main"),
                "docstring": {
                    "languages": ["chinese"],
                    "tags": ["knowledge", "multilingual"],
                },
                "tasks": [
                    {"name": "ceval_zho_mcf:accountant"},
                    {"name": "ceval_zho_mcf:advanced_mathematics"},
                ],
            }
        ]
    )
    by_task, resolved, skipped = _resolve_selectors(
        ("ceval_zho_mcf:accountant", "not-selected"),
        inventory,
    )
    assert by_task == {"ceval_zho_mcf:accountant": "ceval_zho_mcf:accountant"}
    assert resolved == ("ceval_zho_mcf:accountant",)
    assert skipped == ("not-selected",)
    assert inventory["ceval_zho_mcf:accountant"].module_family == "ceval_zho_mcf"


def test_dry_run_redacts_token_and_does_not_create_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    token = "private-scoreboard-token"
    env = _environment(tmp_path, token=token)
    root = Path(env["WEIGHT_PATH"])
    (root / "a.pth").write_bytes(b"a")
    staging = Path(env["HELICOPTER_EVAL_STAGING_ROOT"])
    staging.rmdir()
    config = _config(
        tmp_path,
        'schema_version = 1\nweights = ["a.pth"]\n'
        'benchmarks = ["fixture", "missing"]\n',
    )
    registry = _registry(
        (
            replace(_task("one"), selector="fixture"),
            replace(_task("two"), selector="fixture"),
        )
    )
    registry = replace(
        registry,
        configured_selectors=("fixture", "missing"),
        resolved_selectors=("fixture",),
        skipped_selectors=("missing",),
    )
    monkeypatch.setattr(
        "helicopter_lighteval.runner.load_configured_registry",
        lambda selectors: registry,
    )
    monkeypatch.setattr(
        "helicopter_lighteval.runner.run_preflight",
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
    assert payload["plan"]["registry"]["configured_selectors"] == [
        "fixture",
        "missing",
    ]
    assert payload["plan"]["registry"]["resolved_selectors"] == ["fixture"]
    assert payload["plan"]["registry"]["skipped_selectors"] == ["missing"]
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


def test_configured_registry_expands_available_and_reports_skipped() -> None:
    snapshot = load_configured_registry(("mmlu_pro", "not_in_this_release"))
    assert snapshot.lighteval_version == "0.13.0"
    assert snapshot.configured_selectors == ("mmlu_pro", "not_in_this_release")
    assert snapshot.resolved_selectors == ("mmlu_pro",)
    assert snapshot.skipped_selectors == ("not_in_this_release",)
    assert snapshot.tasks
    assert {task.selector for task in snapshot.tasks} == {"mmlu_pro"}
    assert all(task.evaluation_splits for task in snapshot.tasks)


def test_registry_snapshot_changes_when_selected_tasks_change() -> None:
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
