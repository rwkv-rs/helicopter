from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import resolve_model_path, table
from .env import env_value, pick
from .paths import resolve_path


WKV_MODES = ("fp16", "fp32io16")
EMB_DEVICES = ("cpu", "gpu")


@dataclass
class CommandPlan:
    command: list[str]
    cwd: Path
    shown_env: dict[str, str]
    env: dict[str, str]


def prepend_venv_path(env: dict[str, str], root: Path) -> None:
    venv = resolve_path(
        str(env_value(env, "HELICOPTER_VENV", "VENV", "REMOTE_VENV") or ".venv"),
        root=root,
        env=env,
    )
    bin_dir = venv / "bin"
    if bin_dir.exists():
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"


def python_executable(
    *,
    root: Path,
    env: dict[str, str],
    require_configured: bool = False,
) -> str:
    python_value = env_value(env, "HELICOPTER_PYTHON", "PYTHON")
    if python_value:
        python = resolve_path(str(python_value), root=root, env=env)
        if require_configured and not os.access(python, os.X_OK):
            raise SystemExit(f"Python executable not found: {python}")
        return str(python)

    venv = resolve_path(
        str(env_value(env, "HELICOPTER_VENV", "VENV", "REMOTE_VENV") or ".venv"),
        root=root,
        env=env,
    )
    python = venv / "bin/python"
    if python.exists():
        return str(python)
    if require_configured:
        raise SystemExit(
            f"Python executable not found: {python}; run scripts/install_local.sh "
            "or set HELICOPTER_PYTHON"
        )
    return str(Path(sys.executable))


def _binary_flag(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip()
    if text == "1":
        return True
    if text == "0":
        return False
    raise SystemExit(f"{name} must be 0 or 1, got {value!r}")


def _resolve_fp16_accumulation(
    configured_value: Any,
    *,
    wkv_mode: str | None,
    name: str,
) -> bool | None:
    if configured_value is not None:
        return _binary_flag(configured_value, name=name)
    if wkv_mode is not None:
        return wkv_mode == "fp16"
    return None


def _apply_rwkv_env(
    command_env: dict[str, str],
    *,
    wkv_mode: str | None,
    emb_device: str | None,
    allow_fp16_accumulation: bool | None,
) -> None:
    if emb_device not in (None, "gpu"):
        raise SystemExit(
            "RWKV7 Model Runner V2 keeps embeddings on GPU; CPU embedding is not supported"
        )
    expected_fp16_accumulation = wkv_mode == "fp16" if wkv_mode is not None else None
    if (
        allow_fp16_accumulation is not None
        and expected_fp16_accumulation is not None
        and allow_fp16_accumulation != expected_fp16_accumulation
    ):
        raise SystemExit(
            "RWKV7 Model Runner V2 derives GEMM accumulation from WKV mode: "
            f"{wkv_mode} requires allow_fp16_accumulation={expected_fp16_accumulation}"
        )
    command_env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    if wkv_mode is not None:
        command_env["VLLM_RWKV7_WKV_MODE"] = wkv_mode


def _strip_vllm_env(env: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in env.items() if not key.startswith("VLLM_")}


def build_infer_plan(
    args: Any,
    *,
    root: Path,
    env: dict[str, str],
    config: dict[str, Any],
) -> CommandPlan:
    model_path, model = resolve_model_path(config, args.model, root=root, env=env)
    infer = table(config, "infer")
    runtime = table(config, "runtime")
    gpu = table(config, "gpu")

    wkv_mode_value = pick(
        args.wkv_mode,
        env_value(env, "HELICOPTER_INFER_WKV_MODE", "VLLM_RWKV7_WKV_MODE"),
        infer.get("wkv_mode"),
    )
    wkv_mode = str(wkv_mode_value) if wkv_mode_value is not None else None
    emb_device_value = pick(
        args.emb_device,
        env_value(env, "HELICOPTER_INFER_EMB_DEVICE", "VLLM_RWKV7_EMB_DEVICE"),
        infer.get("emb_device"),
    )
    emb_device = str(emb_device_value) if emb_device_value is not None else None
    allow_fp16_accumulation = _resolve_fp16_accumulation(
        pick(
            args.allow_fp16_accumulation,
            env_value(
                env,
                "HELICOPTER_INFER_ALLOW_FP16_ACCUMULATION",
                "VLLM_RWKV7_ALLOW_FP16_ACCUMULATION",
            ),
            infer.get("allow_fp16_accumulation"),
        ),
        wkv_mode=wkv_mode,
        name="HELICOPTER_INFER_ALLOW_FP16_ACCUMULATION",
    )
    host = str(pick(args.host, runtime.get("host"), default="0.0.0.0"))
    port = str(pick(args.port, runtime.get("port"), default="8000"))
    served_model_name = str(
        pick(
            args.served_model_name,
            model.get("served_model_name"),
            model.get("requested_name"),
            args.model,
        )
    )

    if not args.dry_run and not model_path.is_file():
        raise SystemExit(f"RWKV checkpoint not found: {model_path}")

    command = [
        "vllm",
        "serve",
        str(model_path),
        "--host",
        host,
        "--port",
        port,
        "--tokenizer-mode",
        "rwkv",
        "--load-format",
        "auto",
        "--served-model-name",
        served_model_name,
    ]
    option_values = {
        "--tensor-parallel-size": pick(
            args.tensor_parallel_size,
            env_value(env, "HELICOPTER_TENSOR_PARALLEL_SIZE"),
            infer.get("tensor_parallel_size"),
            gpu.get("tensor_parallel_size"),
        ),
        "--gpu-memory-utilization": pick(
            args.gpu_memory_utilization,
            infer.get("gpu_memory_utilization"),
        ),
        "--max-model-len": pick(
            args.max_model_len,
            model.get("max_model_len"),
            infer.get("max_model_len"),
        ),
        "--max-num-seqs": pick(
            args.max_num_seqs,
            model.get("max_num_seqs"),
            infer.get("max_num_seqs"),
        ),
        "--max-num-batched-tokens": pick(
            args.max_num_batched_tokens,
            model.get("max_num_batched_tokens"),
            infer.get("max_num_batched_tokens"),
        ),
    }
    for option, value in option_values.items():
        if value is not None:
            command.extend([option, str(value)])

    auto_tool_choice = pick(
        args.enable_auto_tool_choice,
        env_value(env, "VLLM_ENABLE_AUTO_TOOL_CHOICE"),
        infer.get("enable_auto_tool_choice"),
        default=False,
    )
    if (
        auto_tool_choice
        if isinstance(auto_tool_choice, bool)
        else str(auto_tool_choice).strip().lower() in {"1", "true", "yes", "on"}
    ):
        command.append("--enable-auto-tool-choice")

    shown_env: dict[str, str] = {}
    _apply_rwkv_env(
        shown_env,
        wkv_mode=wkv_mode,
        emb_device=emb_device,
        allow_fp16_accumulation=allow_fp16_accumulation,
    )
    plan_env = _strip_vllm_env(env)
    plan_env.update(shown_env)
    return CommandPlan(command=command, cwd=root, shown_env=shown_env, env=plan_env)


def build_takeoff_plan(
    args: Any,
    *,
    root: Path,
    env: dict[str, str],
) -> CommandPlan:
    """Delegate the complete MaxRL training contract to Verl."""

    config_path = resolve_path(args.config, root=root, env=env)
    verl_path = resolve_path(
        str(
            env_value(env, "HELICOPTER_VERL_PATH", "VERL_PATH") or "src/train/verl-rwkv"
        ),
        root=root,
        env=env,
    )
    rwkv_lm_path = resolve_path(
        str(
            env_value(env, "RWKV_LM_PATH", "HELICOPTER_RWKV_LM_PATH")
            or "src/train/rwkv-lm"
        ),
        root=root,
        env=env,
    )
    vllm_rwkv_path = resolve_path(
        str(
            env_value(env, "HELICOPTER_VLLM_RWKV_PATH", "VLLM_RWKV_PATH")
            or "src/infer/vllm-rwkv"
        ),
        root=root,
        env=env,
    )
    python = python_executable(root=root, env=env, require_configured=True)
    for path, label in (
        (config_path, "MaxRL config"),
        (verl_path, "verl-rwkv checkout"),
        (rwkv_lm_path, "rwkv-lm checkout"),
        (vllm_rwkv_path, "vllm-rwkv checkout"),
    ):
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")

    command = [
        python,
        "-m",
        "verl.trainer.maxrl",
        "--config",
        str(config_path),
    ]
    for override in args.override or []:
        command.extend(["--override", override])
    if getattr(args, "dry_run", False):
        command.append("--dry-run")

    plan_env = dict(env)
    current_pythonpath = plan_env.get("PYTHONPATH")
    plan_env["PYTHONPATH"] = (
        f"{vllm_rwkv_path}{os.pathsep}{current_pythonpath}"
        if current_pythonpath
        else str(vllm_rwkv_path)
    )
    plan_env["RWKV_LM_PATH"] = str(rwkv_lm_path)
    plan_env["HELICOPTER_PRODUCT_ROOT"] = str(root)
    shown_env = {
        "PYTHON": python,
        "PYTHONPATH": plan_env["PYTHONPATH"],
        "RWKV_LM_PATH": str(rwkv_lm_path),
        "HELICOPTER_PRODUCT_ROOT": str(root),
    }
    return CommandPlan(
        command=command,
        cwd=verl_path,
        shown_env=shown_env,
        env=plan_env,
    )
