from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..errors import ContractError


@contextmanager
def _operation_deadline(seconds: int):
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def timeout_handler(_signum, _frame):
        raise TimeoutError(f"W&B operation exceeded {seconds} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite_metrics(prefix: str, value: object) -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, Mapping):
        for name, child in value.items():
            child_prefix = f"{prefix}/{name}" if prefix else str(name)
            result.update(_finite_metrics(child_prefix, child))
    elif isinstance(value, bool):
        result[prefix] = float(value)
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        result[prefix] = float(value)
    return result


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def experiment_report_identity(
    tracking: Mapping[str, object], *, status: str
) -> dict[str, object]:
    """Return the machine-readable identity embedded in every Markdown report."""
    return {
        "schema_version": 1,
        "run_id": tracking.get("run_id"),
        "attempt_id": tracking.get("attempt_id"),
        "status": status,
        "tracking_status": tracking.get("status"),
        "config_sha256": tracking.get("config_sha256"),
    }


def experiment_report_identity_line(
    tracking: Mapping[str, object], *, status: str
) -> str:
    identity = experiment_report_identity(tracking, status=status)
    return (
        "<!-- any2rwkv-report-identity: "
        + json.dumps(identity, sort_keys=True, separators=(",", ":"))
        + " -->"
    )


@dataclass
class ExperimentTracker:
    run_dir: Path
    run_id: str
    mode: str
    project: str | None
    entity: str | None
    group: str | None
    tags: tuple[str, ...]
    action: str
    job_type: str
    evidence_tier: str
    is_primary: bool
    _run: object | None = None
    _config: dict[str, object] = field(default_factory=dict)
    _attempt_id: str | None = None
    _launch_token_sha256: str | None = None

    @property
    def operation_timeout_seconds(self) -> int:
        try:
            value = int(os.environ.get("ANY2RWKV_WANDB_TIMEOUT_SECONDS", "60"))
        except ValueError as error:
            raise ContractError(
                "ANY2RWKV_WANDB_TIMEOUT_SECONDS must be an integer"
            ) from error
        if not 1 <= value <= 600:
            raise ContractError("ANY2RWKV_WANDB_TIMEOUT_SECONDS must be in 1..600")
        return value

    @classmethod
    def from_plan(
        cls, *, run_dir: Path, plan: object, action: str = "distill"
    ) -> "ExperimentTracker":
        rank = int(os.environ.get("RANK", "0"))
        raw_run_id = os.environ.get("HELICOPTER_RUN_ID", run_dir.name)
        run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw_run_id).strip("-")
        if not run_id:
            raise ContractError("W&B run id is empty after normalization")
        job_types = dict(getattr(plan, "wandb_job_types", ()))
        job_type = job_types.get(action)
        if not job_type:
            if str(getattr(plan, "evidence_tier", "")) != "fixture":
                raise ContractError(f"W&B job_type is missing for action: {action}")
            job_type = f"fixture-{action}"
        return cls(
            run_dir=run_dir,
            run_id=run_id,
            mode=str(getattr(plan, "wandb_mode", "disabled")),
            project=getattr(plan, "wandb_project", None),
            entity=getattr(plan, "wandb_entity", None),
            group=getattr(plan, "wandb_group", None),
            tags=tuple(getattr(plan, "wandb_tags", ())),
            action=action,
            job_type=job_type,
            evidence_tier=str(getattr(plan, "evidence_tier", "")),
            is_primary=rank == 0,
        )

    def start(
        self, *, config: Mapping[str, object], resume_existing: bool = False
    ) -> None:
        if self.evidence_tier != "fixture" and self.mode != "online":
            raise ContractError(
                "non-fixture experiments require online W&B tracking before GPU work"
            )
        self._config = dict(config)
        launch_token = os.environ.get("ANY2RWKV_ATTEMPT_TOKEN")
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if launch_token is None:
            if self.evidence_tier != "fixture" and world_size > 1:
                raise ContractError(
                    "distributed W&B startup requires ANY2RWKV_ATTEMPT_TOKEN"
                )
            launch_token = uuid.uuid4().hex
        if re.fullmatch(r"[0-9a-f]{32}", launch_token) is None:
            raise ContractError("ANY2RWKV_ATTEMPT_TOKEN must be 32 lowercase hex digits")
        self._launch_token_sha256 = hashlib.sha256(
            launch_token.encode("ascii")
        ).hexdigest()
        if self.mode == "disabled":
            return
        if not self.is_primary:
            self._wait_for_primary_startup()
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._attempt_id = uuid.uuid4().hex
        tracking_path = self.run_dir / "experiment-tracking.json"
        base_tracking = {
            "schema_version": 2,
            "backend": "wandb",
            "mode": self.mode,
            "run_id": self.run_id,
            "attempt_id": self._attempt_id,
            "launch_token_sha256": self._launch_token_sha256,
            "coordinated_world_size": world_size,
            "project": self.project,
            "entity": self.entity,
            "group": self.group,
            "tags": list(self.tags),
            "job_type": self.job_type,
            "url": None,
            "status": "initializing",
            "resume_existing": resume_existing,
            "started_unix": time.time(),
            "config": self._config,
            "config_sha256": _sha256_json(self._config),
        }
        _write_json(tracking_path, base_tracking)
        try:
            import wandb

            with _operation_deadline(self.operation_timeout_seconds):
                self._run = wandb.init(
                    project=self.project,
                    entity=self.entity,
                    group=self.group,
                    tags=list(self.tags),
                    job_type=self.job_type,
                    id=self.run_id,
                    name=self.run_id,
                    resume="must" if resume_existing else "never",
                    mode=self.mode,
                    dir=str(self.run_dir),
                    config=self._config,
                    settings=wandb.Settings(
                        init_timeout=self.operation_timeout_seconds
                    ),
                )
            if self._run is None:
                raise RuntimeError("wandb.init returned no run")
            if (
                str(getattr(self._run, "id", "")) != self.run_id
                or str(getattr(self._run, "project", "")) != str(self.project)
                or (
                    self.entity is not None
                    and str(getattr(self._run, "entity", "")) != self.entity
                )
                or str(getattr(self._run, "group", None) or "") != str(self.group or "")
                or str(getattr(self._run, "job_type", "")) != self.job_type
                or not str(getattr(self._run, "url", "") or "")
            ):
                raise RuntimeError("W&B returned an identity different from the plan")
            if self.action != "profile":
                self._run.define_metric("optimizer_step")
                self._run.define_metric("*", step_metric="optimizer_step")
            base_tracking.update(
                {
                    "url": getattr(self._run, "url", None),
                    "status": "running",
                    "initialized_unix": time.time(),
                }
            )
            _write_json(tracking_path, base_tracking)
        except BaseException as error:
            if self._run is not None:
                try:
                    with _operation_deadline(self.operation_timeout_seconds):
                        self._run.finish(exit_code=1)
                except BaseException:
                    pass
                finally:
                    self._run = None
            base_tracking.update(
                {
                    "status": "failed",
                    "exit_code": 1,
                    "finished_unix": time.time(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            _write_json(tracking_path, base_tracking)
            raise ContractError(f"W&B initialization failed: {error}") from error

    def _wait_for_primary_startup(self) -> None:
        """Keep non-primary ranks on CPU until rank 0 has a live online run."""
        tracking_path = self.run_dir / "experiment-tracking.json"
        config_sha256 = _sha256_json(self._config)
        deadline = time.monotonic() + self.operation_timeout_seconds
        while time.monotonic() < deadline:
            tracking = _read_json(tracking_path)
            if (
                tracking is None
                or tracking.get("launch_token_sha256")
                != self._launch_token_sha256
            ):
                time.sleep(0.05)
                continue
            if (
                tracking.get("run_id") != self.run_id
                or tracking.get("mode") != self.mode
                or tracking.get("project") != self.project
                or tracking.get("entity") != self.entity
                or tracking.get("group") != self.group
                or tracking.get("tags") != list(self.tags)
                or tracking.get("job_type") != self.job_type
                or tracking.get("config_sha256") != config_sha256
                or tracking.get("coordinated_world_size")
                != int(os.environ.get("WORLD_SIZE", "1"))
            ):
                raise ContractError(
                    "rank-0 W&B startup identity differs from this rank"
                )
            status = tracking.get("status")
            if status == "running" and tracking.get("url"):
                return
            if status == "failed":
                raise ContractError(
                    "rank-0 W&B initialization failed: "
                    + str(tracking.get("error", "unknown error"))
                )
            time.sleep(0.05)
        raise ContractError(
            "timed out waiting for rank 0 to initialize W&B before GPU work"
        )

    def log_progress(self, phase: str, progress_path: Path) -> None:
        if self._run is None:
            return
        progress = _read_json(progress_path)
        if progress is None:
            raise ContractError(f"cannot log malformed progress file: {progress_path}")
        history = progress.get("history")
        latest = history[-1] if isinstance(history, list) and history else {}
        active_steps = int(progress.get("active_optimizer_steps", 0) or 0)
        completed_steps = int(progress.get("completed_optimizer_steps", 0) or 0)
        raw_active_layer = progress.get("active_layer", -1)
        raw_epoch = progress.get("epoch_index", -1)
        metrics: dict[str, object] = {
            "optimizer_step": completed_steps + active_steps,
            "phase": phase,
            "active_layer": int(
                raw_active_layer if raw_active_layer is not None else -1
            ),
            "epoch": int(raw_epoch if raw_epoch is not None else -1),
        }
        metrics.update(_finite_metrics("validation", latest))
        metrics.update(_finite_metrics("train", progress.get("last_train_metrics", {})))
        telemetry_dir = self.run_dir / "training-telemetry"
        telemetry_files = sorted(telemetry_dir.glob("*.json"))
        if telemetry_files:
            telemetry = _read_json(telemetry_files[-1])
            if telemetry is not None:
                metrics.update(_finite_metrics("system", telemetry))
        with _operation_deadline(self.operation_timeout_seconds):
            self._run.log(metrics)

    def log_result(self, result: Mapping[str, object]) -> None:
        if self._run is None:
            return
        metrics: dict[str, object] = {"result/status": str(result.get("status", ""))}
        metrics.update(_finite_metrics("result", result))
        with _operation_deadline(self.operation_timeout_seconds):
            self._run.log(metrics)

    def log_metrics(self, metrics: Mapping[str, object]) -> None:
        """Log an explicit finite scalar tree without reading progress files."""
        if self._run is None:
            return
        values: dict[str, object] = _finite_metrics("", metrics)
        if not values:
            raise ContractError("W&B metric payload contains no finite scalars")
        with _operation_deadline(self.operation_timeout_seconds):
            self._run.log(values)

    def finish(self, *, exit_code: int) -> None:
        if not self.is_primary or self.mode == "disabled":
            return
        tracking_path = self.run_dir / "experiment-tracking.json"
        tracking = _read_json(tracking_path) or {}
        run_url = getattr(self._run, "url", None) if self._run is not None else None
        try:
            if self._run is not None:
                with _operation_deadline(self.operation_timeout_seconds):
                    self._run.finish(exit_code=exit_code)
            tracking.update(
                {
                    "status": "completed" if exit_code == 0 else "failed",
                    "exit_code": exit_code,
                    "url": run_url or tracking.get("url"),
                    "finished_unix": time.time(),
                    "config": self._config,
                    "config_sha256": _sha256_json(self._config),
                }
            )
            _write_json(tracking_path, tracking)
        except BaseException as error:
            tracking.update(
                {
                    "status": "failed",
                    "exit_code": 1,
                    "finished_unix": time.time(),
                    "error": f"{type(error).__name__}: {error}",
                    "config": self._config,
                    "config_sha256": _sha256_json(self._config),
                }
            )
            _write_json(tracking_path, tracking)
            raise
        finally:
            self._run = None

    def abort(self, *, reason: str) -> None:
        """Best-effort local failure finalization without distributed collectives."""
        try:
            self.finish(exit_code=1)
        finally:
            tracking_path = self.run_dir / "experiment-tracking.json"
            tracking = _read_json(tracking_path) or {}
            tracking.update(
                {
                    "status": "failed",
                    "exit_code": 1,
                    "finished_unix": time.time(),
                    "error": reason,
                }
            )
            _write_json(tracking_path, tracking)


def write_experiment_report(
    run_dir: Path,
    *,
    status: str,
    reason: str | None = None,
) -> Path:
    """Write a short, plain-Chinese report from durable run evidence."""
    layer_progress = _read_json(run_dir / "layer-major-progress.json")
    corrective_progress = _read_json(run_dir / "global-corrective-progress.json")
    progress = layer_progress or corrective_progress or {}
    is_corrective = layer_progress is None and corrective_progress is not None
    history = progress.get("history")
    rows = history if isinstance(history, list) else []
    completed_layers = sorted(
        {
            int(row["layer"])
            for row in rows
            if isinstance(row, dict)
            and isinstance(row.get("layer"), int)
            and bool(row.get("converged", row.get("convergence_observed", False)))
        }
    )
    active_layer = progress.get(
        "active_layer", progress.get("next_visit") if is_corrective else None
    )
    epoch = progress.get(
        "epoch_index", progress.get("sweep_index") if is_corrective else None
    )
    telemetry_rows = [
        value
        for path in sorted((run_dir / "training-telemetry").glob("*.json"))
        if (value := _read_json(path)) is not None
    ]
    tracking = _read_json(run_dir / "experiment-tracking.json") or {}
    experiment_config = tracking.get("config")
    experiment_config = experiment_config if isinstance(experiment_config, dict) else {}
    optimizer = experiment_config.get("optimizer")
    optimizer = optimizer if isinstance(optimizer, dict) else {}
    batch = experiment_config.get("batch")
    batch = batch if isinstance(batch, dict) else {}
    comparison = experiment_config.get("comparison")
    comparison = comparison if isinstance(comparison, dict) else {}
    only_changes = comparison.get("only_changes")
    only_changes = only_changes if isinstance(only_changes, list) else []
    checkpoint_fractions = []
    peak_reserved = []
    for row in telemetry_rows:
        train_seconds = float(row.get("train_wall_seconds", 0) or 0)
        checkpoint_seconds = float(row.get("checkpoint_wall_seconds", 0) or 0)
        if train_seconds > 0:
            checkpoint_fractions.append(checkpoint_seconds / train_seconds)
        reserved = row.get("peak_cuda_reserved_bytes")
        if isinstance(reserved, (int, float)):
            peak_reserved.append(float(reserved))
    lines = [
        experiment_report_identity_line(tracking, status=status),
        "",
        "# 实验结果",
        "",
        "## 这次想验证什么",
        "",
        (
            "验证全部层换成 RWKV7 后的端到端微调是否改善模型输出，同时确认 8 张 GPU 没有明显空等。"
            if is_corrective
            else "验证当前这一层从原始 Qwen 结构换成 RWKV7 后，能否稳定学回原始层的输出，同时确认 8 张 GPU 没有明显空等。"
        ),
        "",
        "## 实际跑了什么",
        "",
        f"- 运行状态：`{status}`",
        f"- 当前层：`{active_layer}`；当前 epoch：`{epoch}`",
        f"- 已看到收敛记录的层：`{completed_layers}`",
        f"- W&B：{tracking.get('url') or '未建立在线 run'}",
        (
            "- 优化器："
            f"`{optimizer.get('name', '未知')}`；"
            f"lr `{optimizer.get('learning_rate', '未知')}` → "
            f"`{optimizer.get('final_learning_rate', '未知')}`；"
            f"warmup `{optimizer.get('warmup_steps', '未知')}`；"
            f"betas `{optimizer.get('betas', '未知')}`；"
            f"weight decay `{optimizer.get('weight_decay', '未知')}`；"
            f"gradient clip `{optimizer.get('gradient_clip_norm', '未知')}`"
        ),
        (
            "- batch：每卡 "
            f"`{batch.get('per_rank', '未知')}`，8 卡全局 "
            f"`{int(batch.get('per_rank', 0) or 0) * int(batch.get('world_size', 0) or 0) or '未知'}`，"
            f"梯度累积 `{batch.get('accumulation_steps', '未知')}`"
        ),
        "",
        "## 与上一次相比只改了什么",
        "",
        f"- 对照 run：`{comparison.get('baseline_run_id') or '未声明'}`",
        *(
            [f"- {change}" for change in only_changes]
            if only_changes
            else ["- 没有可核验的差异说明；本次结果不能做单变量归因。"]
        ),
        "",
        "## 看到的结果",
        "",
    ]
    if checkpoint_fractions:
        lines.append(
            "- checkpoint 占训练墙钟比例范围："
            f"`{min(checkpoint_fractions):.1%}..{max(checkpoint_fractions):.1%}`"
        )
    if peak_reserved:
        lines.append(
            "- 记录到的最大 CUDA reserved memory："
            f"`{max(peak_reserved) / 1024**3:.2f} GiB`"
        )
    if reason:
        lines.append(f"- 停止或失败原因：{reason}")
    if not checkpoint_fractions and not peak_reserved and not reason:
        lines.append("- 当前产物没有足够的性能或失败细节，不能补写结论。")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            (
                "这次运行已完成，但仍需以最终质量文件判断是否通过验收。"
                if status in {"complete", "completed", "layerwise-local-complete"}
                else "这次运行没有完成全部验收，不能据此声称模型已经训成。"
            ),
            "",
            "## 下一步",
            "",
            "先解决报告中暴露的性能或正确性问题；只有性能预检通过，才允许重新启动真实训练。",
            "",
        ]
    )
    path = run_dir / "experiment-report.md"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    return path
