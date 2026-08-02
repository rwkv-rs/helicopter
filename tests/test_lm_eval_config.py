from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from helicopter_lm_eval.config import ConfigError, LMEvalConfig


def _manifest(tmp_path: Path, max_model_len: int = 16) -> Path:
    path = tmp_path / "pool.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "global_step": 9,
                "wkv_mode": "fp32io16",
                "vllm_version": "0.23.1.dev0",
                "max_model_len": max_model_len,
                "replicas": [
                    {
                        "base_url": "http://127.0.0.1:8000",
                        "max_concurrency": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _config(
    tmp_path: Path,
    *,
    batch_size: int = 2,
    eot_token_id: int = 0,
    max_gen_toks: int = 8,
    extra: str = "",
) -> Path:
    path = tmp_path / "eval.toml"
    path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'backend = "vllm_http"',
                'tasks = ["wikitext"]',
                f'output_dir = "{tmp_path / "results"}"',
                f"batch_size = {batch_size}",
                f"eot_token_id = {eot_token_id}",
                f"max_gen_toks = {max_gen_toks}",
                "log_samples = false",
                extra,
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_config_loads_ppl_settings_and_manifest(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    config = LMEvalConfig.read(
        _config(tmp_path),
        {"HELICOPTER_VLLM_POOL_MANIFEST": str(manifest)},
    )

    assert config.tasks == ("wikitext",)
    assert config.batch_size == 2
    assert config.eot_token_id == 0
    assert config.max_gen_toks == 8
    assert config.limit is None
    assert config.manifest.global_step == 9


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"extra": "unknown = true"}, "unknown lm-eval config fields"),
        ({"eot_token_id": -1}, "eot_token_id must be a non-negative integer"),
        ({"batch_size": 0}, "batch_size must be a positive integer"),
        ({"max_gen_toks": 0}, "max_gen_toks must be a positive integer"),
        ({"extra": "limit = 0"}, "limit must be a positive integer"),
    ],
)
def test_config_rejects_invalid_fields(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    manifest = _manifest(tmp_path)
    with pytest.raises(ConfigError, match=message):
        LMEvalConfig.read(
            _config(tmp_path, **overrides),
            {"HELICOPTER_VLLM_POOL_MANIFEST": str(manifest)},
        )


def test_config_requires_a_regular_manifest(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be a regular file"):
        LMEvalConfig.read(
            _config(tmp_path),
            {"HELICOPTER_VLLM_POOL_MANIFEST": str(tmp_path / "missing.json")},
        )


def test_config_rejects_output_path_that_is_not_a_directory(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    (tmp_path / "results").write_text("not a directory", encoding="utf-8")

    with pytest.raises(ConfigError, match="output_dir must be a regular directory"):
        LMEvalConfig.read(
            _config(tmp_path),
            {"HELICOPTER_VLLM_POOL_MANIFEST": str(manifest)},
        )


def test_published_config_builds_weight_and_wkv_execution_matrix(tmp_path: Path) -> None:
    weight_root = tmp_path / "weights"
    weight_root.mkdir()
    weight = weight_root / "model.pth"
    weight.write_bytes(b"checkpoint")
    digest = hashlib.sha256(b"checkpoint").hexdigest()
    manifests = []
    for mode in ("fp16", "fp32io16"):
        path = tmp_path / f"{mode}.json"
        mode_dir = tmp_path / mode
        mode_dir.mkdir()
        payload = json.loads(_manifest(mode_dir).read_text(encoding="utf-8"))
        payload.update(
            wkv_mode=mode,
            weight_sha256=digest,
            weight_display_name=weight.name,
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
        manifests.append(path)
    config_path = tmp_path / "campaign.toml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'backend = "vllm_http"',
                "publish = true",
                'tasks = ["wikitext"]',
                f'output_dir = "{tmp_path / "results"}"',
                'weights = ["model.pth"]',
                'wkv_modes = ["fp16", "fp32io16"]',
                "max_gen_toks = 8",
                "pool_manifests = [",
                *(f'  "{path}",' for path in manifests),
                "]",
            ]
        ),
        encoding="utf-8",
    )

    config = LMEvalConfig.read(
        config_path,
        {
            "WEIGHT_PATH": str(weight_root),
            "HELICOPTER_SCOREBOARD_URL": "https://scoreboard.example",
            "HELICOPTER_SCOREBOARD_TOKEN": "secret",
        },
    )

    assert [unit.wkv_mode for unit in config.execution_units] == [
        "fp16",
        "fp32io16",
    ]
    assert all(unit.weight_sha256 == digest for unit in config.execution_units)
    assert config.public()["scoreboard_token"] == "[REDACTED]"


def test_published_config_rejects_smoke_limit(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    with pytest.raises(ConfigError, match="must not set limit"):
        LMEvalConfig.read(
            _config(tmp_path, extra="publish = true\nlimit = 1"),
            {"HELICOPTER_VLLM_POOL_MANIFEST": str(manifest)},
        )
