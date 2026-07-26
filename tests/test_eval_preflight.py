from pathlib import Path

import pytest

from helicopter_eval.config import (
    EvaluationConfigurationError,
    EvaluationEnvironment,
)
from helicopter_eval import preflight


def _environment(tmp_path: Path) -> EvaluationEnvironment:
    weight_root = tmp_path / "weights"
    weight_root.mkdir()
    return EvaluationEnvironment(
        weight_root=weight_root,
        scoreboard_url="https://scoreboard.example.test",
        scoreboard_token="private",
        staging_root=tmp_path / "staging",
    )


def test_preflight_proves_release_editable_source_and_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(preflight.__file__).resolve().parents[3]

    def dependency(name: str):
        if name == "lighteval":
            return {"version": "0.13.0", "direct_url": {}}
        return {
            "version": "0.13.0.dev0",
            "direct_url": {
                "url": (repository / "src/infer/vllm-rwkv").as_uri(),
                "dir_info": {"editable": True},
            },
        }

    class Client:
        def __init__(self, _environment):
            pass

        def preflight(self):
            return {
                "status": "ready",
                "schema_version": "lighteval-campaign-v2",
                "lighteval_version": "0.13.0",
                "publisher_principal": "eval-worker",
            }

    monkeypatch.setattr(preflight, "_dependency_source", dependency)
    monkeypatch.setattr(preflight, "ScoreboardClient", Client)
    result = preflight.run_preflight(_environment(tmp_path))
    assert result["dependencies"]["lighteval"]["version"] == "0.13.0"
    assert result["scoreboard"]["status"] == "ready"
    assert result["scoreboard"]["publisher_principal"] == "eval-worker"
    assert not (tmp_path / "staging").exists()


@pytest.mark.parametrize(
    ("lighteval_version", "lighteval_direct_url", "editable", "message"),
    [
        ("0.12.0", {}, True, "lighteval must be exactly"),
        (
            "0.13.0",
            {"url": "file:///tmp/lighteval"},
            True,
            "locked registry release",
        ),
        ("0.13.0", {}, False, "editable source mismatch"),
    ],
)
def test_preflight_fails_closed_on_dependency_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lighteval_version: str,
    lighteval_direct_url: dict,
    editable: bool,
    message: str,
) -> None:
    repository = Path(preflight.__file__).resolve().parents[3]

    def dependency(name: str):
        if name == "lighteval":
            return {
                "version": lighteval_version,
                "direct_url": lighteval_direct_url,
            }
        return {
            "version": "0.13.0.dev0",
            "direct_url": {
                "url": (repository / "src/infer/vllm-rwkv").as_uri(),
                "dir_info": {"editable": editable},
            },
        }

    monkeypatch.setattr(preflight, "_dependency_source", dependency)
    with pytest.raises(EvaluationConfigurationError, match=message):
        preflight.run_preflight(_environment(tmp_path))


def test_preflight_rejects_shared_writable_staging_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    tmp_path.chmod(0o777)
    monkeypatch.setattr(
        preflight,
        "_dependency_source",
        lambda name: (
            {"version": "0.13.0", "direct_url": {}}
            if name == "lighteval"
            else {
                "version": "0.13.0.dev0",
                "direct_url": {
                    "url": (
                        Path(preflight.__file__).resolve().parents[3]
                        / "src/infer/vllm-rwkv"
                    ).as_uri(),
                    "dir_info": {"editable": True},
                },
            }
        ),
    )

    with pytest.raises(
        EvaluationConfigurationError,
        match="not group/world writable",
    ):
        preflight.run_preflight(environment)
