from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from any2rwkv.core.experiment_tracking import (
    ExperimentTracker,
    write_experiment_report,
)
from any2rwkv.errors import ContractError


def _plan(**overrides):
    values = {
        "wandb_mode": "disabled",
        "wandb_project": None,
        "wandb_entity": None,
        "wandb_group": None,
        "wandb_tags": (),
        "wandb_job_types": (
            ("profile", "performance-profile"),
            ("distill", "layerwise-distillation"),
            ("corrective", "fully-recurrent-corrective"),
        ),
        "evidence_tier": "fixture",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_non_fixture_run_requires_online_wandb_before_training(
    tmp_path: Path,
) -> None:
    tracker = ExperimentTracker.from_plan(
        run_dir=tmp_path,
        plan=_plan(evidence_tier="exploratory"),
    )
    with pytest.raises(ContractError, match="require online W&B"):
        tracker.start(config={})


def test_online_wandb_binds_identity_config_and_finish_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    class FakeRun:
        id = "v42-layer0"
        project = "any2rwkv"
        group = "adamw-baseline"
        job_type = "layerwise-distillation"
        url = "https://wandb.invalid/any2rwkv/v42-layer0"

        def define_metric(self, *args, **kwargs):
            calls.setdefault("metrics", []).append((args, kwargs))

        def log(self, metrics):
            calls.setdefault("logs", []).append(metrics)

        def finish(self, *, exit_code):
            calls["exit_code"] = exit_code

    def init(**kwargs):
        calls["init"] = kwargs
        return FakeRun()

    fake_wandb = SimpleNamespace(
        init=init,
        Settings=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("HELICOPTER_RUN_ID", "v42-layer0")
    tracker = ExperimentTracker.from_plan(
        run_dir=tmp_path,
        plan=_plan(
            evidence_tier="exploratory",
            wandb_mode="online",
            wandb_project="any2rwkv",
            wandb_group="adamw-baseline",
        ),
    )
    config = {
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1e-6,
            "final_learning_rate": 1e-6,
            "warmup_steps": 10,
            "betas": [0.9, 0.99],
            "weight_decay": 0.1,
            "gradient_clip_norm": 1.0,
        },
        "batch": {"per_rank": 8, "world_size": 8, "accumulation_steps": 1},
    }

    tracker.start(config=config)
    tracker.log_metrics(
        {
            "selection": {
                "kernel_covered_wall_fraction": 0.95,
                "eligible": True,
            }
        }
    )
    tracker.finish(exit_code=0)

    init_call = calls["init"]
    assert init_call["id"] == "v42-layer0"
    assert init_call["job_type"] == "layerwise-distillation"
    assert init_call["resume"] == "never"
    assert init_call["config"] == config
    assert calls["exit_code"] == 0
    assert calls["logs"][-1] == {
        "selection/kernel_covered_wall_fraction": 0.95,
        "selection/eligible": 1.0,
    }
    tracking = json.loads(
        (tmp_path / "experiment-tracking.json").read_text(encoding="utf-8")
    )
    assert tracking["status"] == "completed"
    assert tracking["schema_version"] == 2
    assert len(tracking["attempt_id"]) == 32
    assert tracking["group"] == "adamw-baseline"
    assert tracking["tags"] == []
    assert tracking["url"] == FakeRun.url
    assert tracking["config"] == config


def test_wandb_init_failure_overwrites_stale_completed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "experiment-tracking.json").write_text(
        json.dumps({"status": "completed", "attempt_id": "stale"}),
        encoding="utf-8",
    )

    def init(**_kwargs):
        raise RuntimeError("offline")

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=init,
            Settings=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )
    monkeypatch.setenv("HELICOPTER_RUN_ID", "fresh-attempt")
    tracker = ExperimentTracker.from_plan(
        run_dir=tmp_path,
        plan=_plan(
            evidence_tier="exploratory",
            wandb_mode="online",
            wandb_project="any2rwkv",
        ),
    )

    with pytest.raises(ContractError, match="W&B initialization failed"):
        tracker.start(config={"plan": "fresh"})

    tracking = json.loads(
        (tmp_path / "experiment-tracking.json").read_text(encoding="utf-8")
    )
    assert tracking["status"] == "failed"
    assert tracking["run_id"] == "fresh-attempt"
    assert tracking["attempt_id"] != "stale"
    assert tracking["config"] == {"plan": "fresh"}


def test_non_primary_waits_for_same_launch_wandb_before_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = "a" * 32
    config = {"plan": "distributed"}
    config_sha = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (tmp_path / "experiment-tracking.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "backend": "wandb",
                "mode": "online",
                "status": "running",
                "run_id": "distributed-run",
                "attempt_id": "b" * 32,
                "launch_token_sha256": hashlib.sha256(
                    token.encode("ascii")
                ).hexdigest(),
                "coordinated_world_size": 8,
                "project": "any2rwkv",
                "entity": None,
                "group": "baseline",
                "tags": [],
                "job_type": "layerwise-distillation",
                "url": "https://wandb.invalid/distributed-run",
                "config": config,
                "config_sha256": config_sha,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HELICOPTER_RUN_ID", "distributed-run")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("ANY2RWKV_ATTEMPT_TOKEN", token)
    tracker = ExperimentTracker.from_plan(
        run_dir=tmp_path,
        plan=_plan(
            evidence_tier="exploratory",
            wandb_mode="online",
            wandb_project="any2rwkv",
            wandb_group="baseline",
        ),
    )

    tracker.start(config=config)

    assert tracker._run is None


def test_report_says_incomplete_run_is_not_a_finished_model(tmp_path: Path) -> None:
    (tmp_path / "training-telemetry").mkdir()
    (tmp_path / "layer-major-progress.json").write_text(
        json.dumps(
            {
                "active_layer": 13,
                "epoch_index": 15,
                "history": [
                    {"layer": 11, "converged": True},
                    {"layer": 12, "converged": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "training-telemetry" / "layer-013-epoch-015.json").write_text(
        json.dumps(
            {
                "train_wall_seconds": 12.0,
                "checkpoint_wall_seconds": 10.0,
                "peak_cuda_reserved_bytes": 6 * 1024**3,
            }
        ),
        encoding="utf-8",
    )

    report = write_experiment_report(
        tmp_path,
        status="interrupted",
        reason="用户主动停止了低利用率训练",
    ).read_text(encoding="utf-8")

    assert "当前层：`13`" in report
    assert "checkpoint 占训练墙钟比例范围：`83.3%..83.3%`" in report
    assert "不能据此声称模型已经训成" in report
    assert "只有性能预检通过，才允许重新启动真实训练" in report
