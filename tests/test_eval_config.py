from __future__ import annotations

import json
from pathlib import Path

import pytest

from helicopter_lighteval import evaluate
from helicopter_lighteval.config import ConfigError, LightEvalConfig


def _environment(tmp_path: Path) -> dict[str, str]:
    weight_root = tmp_path / "weights"
    weight_root.mkdir()
    (weight_root / "a.pth").write_bytes(b"a")
    (weight_root / "b.pth").write_bytes(b"b")
    return {
        "WEIGHT_PATH": str(weight_root),
        "HELICOPTER_SCOREBOARD_URL": "https://scoreboard.example.test",
        "HELICOPTER_SCOREBOARD_TOKEN": "secret-token",
        "HELICOPTER_EVAL_STAGING_ROOT": str(tmp_path / "staging"),
    }


def _write_config(path: Path, body: str | None = None) -> Path:
    path.write_text(
        body
        or """
schema_version = 1
prompt_template = "assistant"
weights = ["a.pth", "b.pth"]
benchmarks = ["mmlu", "gsm8k"]
""",
        encoding="utf-8",
    )
    return path


def test_read_minimal_multi_weight_config(tmp_path: Path) -> None:
    config = LightEvalConfig.read(
        _write_config(tmp_path / "eval.toml"),
        _environment(tmp_path),
    )

    assert [path.name for path in config.weights] == ["a.pth", "b.pth"]
    assert len(set(config.weight_hashes)) == 2
    assert config.benchmarks == ("mmlu", "gsm8k")
    assert config.prompt == ("\n\nAssistant: ", "\nUser:")
    assert config.public()["scoreboard_token"] == "[REDACTED]"
    assert "secret-token" not in json.dumps(config.public())


def test_local_config_expands_invocation_paths_without_scoreboard(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    environment.pop("HELICOPTER_SCOREBOARD_URL")
    environment.pop("HELICOPTER_SCOREBOARD_TOKEN")
    environment.pop("HELICOPTER_EVAL_STAGING_ROOT")
    environment["MAXRL_EVAL_WEIGHT"] = "a.pth"
    environment["MAXRL_EVAL_RESULT_PATH"] = str(tmp_path / "metrics.json")
    config = LightEvalConfig.read(
        _write_config(
            tmp_path / "eval.toml",
            """
schema_version = 1
publish = false
result_path = "${MAXRL_EVAL_RESULT_PATH}"
weights = ["${MAXRL_EVAL_WEIGHT}"]
wkv_modes = ["fp32io16"]
benchmarks = ["aime25", "gsm8k", "asdiv", "math_500"]
""",
        ),
        environment,
    )

    assert config.publish is False
    assert config.scoreboard_url is None
    assert config.result_path == tmp_path / "metrics.json"
    assert config.staging_root == tmp_path / ".lighteval-staging"
    assert config.wkv_modes == ("fp32io16",)


def test_http_backend_requires_runtime_pool_manifest(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    environment.pop("HELICOPTER_SCOREBOARD_URL")
    environment.pop("HELICOPTER_SCOREBOARD_TOKEN")
    environment.pop("HELICOPTER_EVAL_STAGING_ROOT")
    environment["MAXRL_EVAL_WEIGHT"] = "a.pth"
    environment["MAXRL_EVAL_RESULT_PATH"] = str(tmp_path / "metrics.json")
    config_path = _write_config(
        tmp_path / "eval.toml",
        """
schema_version = 1
backend = "vllm_http"
publish = false
result_path = "${MAXRL_EVAL_RESULT_PATH}"
weights = ["${MAXRL_EVAL_WEIGHT}"]
wkv_modes = ["fp32io16"]
benchmarks = ["aime25"]
""",
    )

    with pytest.raises(ConfigError, match="HELICOPTER_VLLM_POOL_MANIFEST"):
        LightEvalConfig.read(config_path, environment)

    manifest = tmp_path / "pool.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "global_step": 7,
                "wkv_mode": "fp32io16",
                "vllm_version": "0.23.1.dev0",
                "max_model_len": 10240,
                "replicas": [
                    {
                        "base_url": "http://10.0.0.1:8000",
                        "max_concurrency": 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    environment["HELICOPTER_VLLM_POOL_MANIFEST"] = str(manifest)
    environment["MAXRL_EVAL_STEP"] = "7"
    config = LightEvalConfig.read(config_path, environment)

    assert config.backend == "vllm_http"
    assert config.vllm_pool_manifest == manifest


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            'schema_version = 2\nweights = ["a.pth"]\nbenchmarks = ["mmlu"]\n',
            "schema_version must be 1",
        ),
        (
            'schema_version = 1\nweights = ["a.pth"]\n'
            'benchmarks = ["mmlu"]\nmax_samples = 10\n',
            "unknown eval config fields",
        ),
        (
            'schema_version = 1\nweights = ["a.pth"]\nbenchmarks = ["mmlu", "mmlu"]\n',
            "duplicate benchmarks",
        ),
        (
            'schema_version = 1\nweights = ["a.pth"]\n'
            'benchmarks = ["mmlu"]\nprompt_template = "unknown"\n',
            "prompt_template must be one of",
        ),
    ],
)
def test_read_rejects_invalid_public_contract(
    tmp_path: Path,
    body: str,
    message: str,
) -> None:
    with pytest.raises(ConfigError, match=message):
        LightEvalConfig.read(
            _write_config(tmp_path / "eval.toml", body),
            _environment(tmp_path),
        )


def test_read_rejects_weight_escape_and_symlink(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    outside = tmp_path / "outside.pth"
    outside.write_bytes(b"outside")
    (Path(environment["WEIGHT_PATH"]) / "link.pth").symlink_to(outside)

    for configured in ("../outside.pth", "link.pth"):
        body = (
            f'schema_version = 1\nweights = ["{configured}"]\nbenchmarks = ["mmlu"]\n'
        )
        with pytest.raises(ConfigError):
            LightEvalConfig.read(
                _write_config(tmp_path / "eval.toml", body),
                environment,
            )


def test_read_rejects_duplicate_weight_content(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    (Path(environment["WEIGHT_PATH"]) / "copy.pth").write_bytes(b"a")
    body = (
        'schema_version = 1\nweights = ["a.pth", "copy.pth"]\nbenchmarks = ["mmlu"]\n'
    )

    with pytest.raises(ConfigError, match="duplicate weight content"):
        LightEvalConfig.read(
            _write_config(tmp_path / "eval.toml", body),
            environment,
        )


def test_dry_run_resolves_without_creating_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    environment = _environment(tmp_path)
    config_path = _write_config(
        tmp_path / "eval.toml",
        'schema_version = 1\nweights = ["a.pth"]\nbenchmarks = ["mmlu", "missing"]\n',
    )
    task = {
        "selector": "mmlu",
        "task_name": "mmlu|0",
        "task_version": "0",
        "module_family": "mmlu",
        "module": "lighteval.tasks.tasks.mmlu",
        "dataset": "cais/mmlu",
        "subset": "all",
        "evaluation_splits": ["test"],
        "languages": ["english"],
        "upstream_tags": ["knowledge"],
    }

    class Client:
        def __init__(self, *_args) -> None:
            pass

        def preflight(self) -> dict[str, str]:
            return {"status": "ready"}

        def create_campaign(self, *_args):
            raise AssertionError("dry-run must not create a campaign")

    monkeypatch.setattr(
        evaluate,
        "_resolve_benchmarks",
        lambda _selectors: ([task], ["missing"], "0.13.0"),
    )
    monkeypatch.setattr(evaluate, "ScoreboardClient", Client)

    assert evaluate.run(config_path=config_path, env=environment, dry_run=True) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["skipped_benchmarks"] == ["missing"]
    assert output["execution_units"] == 2
    assert output["expected_task_count"] == 2
    assert "secret-token" not in json.dumps(output)


def test_local_run_never_constructs_scoreboard_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    environment.pop("HELICOPTER_SCOREBOARD_URL")
    environment.pop("HELICOPTER_SCOREBOARD_TOKEN")
    environment.pop("HELICOPTER_EVAL_STAGING_ROOT")
    environment["MAXRL_EVAL_WEIGHT"] = "a.pth"
    environment["MAXRL_EVAL_RESULT_PATH"] = str(tmp_path / "metrics.json")
    config_path = _write_config(
        tmp_path / "eval.toml",
        """
schema_version = 1
publish = false
result_path = "${MAXRL_EVAL_RESULT_PATH}"
weights = ["${MAXRL_EVAL_WEIGHT}"]
wkv_modes = ["fp32io16"]
benchmarks = ["aime25", "gsm8k", "asdiv", "math_500"]
""",
    )
    task = {
        "selector": "gsm8k",
        "task_name": "gsm8k|0",
    }
    monkeypatch.setattr(
        evaluate,
        "_resolve_benchmarks",
        lambda _selectors: ([task], [], "0.13.0"),
    )
    monkeypatch.setattr(
        evaluate,
        "ScoreboardClient",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("local evaluation must not access Scoreboard")
        ),
    )
    monkeypatch.setattr(evaluate, "_run_local", lambda **_kwargs: 0)

    assert evaluate.run(config_path=config_path, env=environment, dry_run=False) == 0
