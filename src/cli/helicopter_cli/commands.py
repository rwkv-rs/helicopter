from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import dataset_root, resolve_model_path, table
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


def prepend_venv_path(env: dict[str, str], root: Path, config: dict[str, Any]) -> None:
    paths = table(config, "paths")
    venv_value = pick(
        paths.get("venv"),
        env_value(env, "HELICOPTER_VENV", "VENV", "REMOTE_VENV"),
    )
    if not venv_value:
        venv_value = ".venv"
    venv = resolve_path(str(venv_value), root=root, env=env)
    bin_dir = venv / "bin"
    if bin_dir.exists():
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"


def python_executable(
    config: dict[str, Any],
    *,
    root: Path,
    env: dict[str, str],
    require_configured: bool = False,
) -> str:
    paths = table(config, "paths")
    python_value = pick(paths.get("python"), env_value(env, "HELICOPTER_PYTHON", "PYTHON"))
    if python_value:
        python = resolve_path(str(python_value), root=root, env=env)
        if require_configured and not os.access(python, os.X_OK):
            raise SystemExit(f"Python executable not found: {python}")
        return str(python)

    venv_value = pick(
        paths.get("venv"),
        env_value(env, "HELICOPTER_VENV", "VENV", "REMOTE_VENV"),
        ".venv",
    )
    venv = resolve_path(str(venv_value), root=root, env=env)
    python = venv / "bin/python"
    if python.exists():
        return str(python)
    if require_configured:
        raise SystemExit(
            f"Python executable not found: {python}; run scripts/install_local.sh "
            "or set HELICOPTER_PYTHON / paths.python"
        )
    return str(Path(sys.executable))


def apply_rwkv_env(
    command_env: dict[str, str],
    config: dict[str, Any],
    env: dict[str, str],
    *,
    wkv_mode: str,
    emb_device: str,
) -> None:
    rwkv7 = table(config, "rwkv7")
    command_env["VLLM_RWKV7_WKV_MODE"] = wkv_mode
    command_env["VLLM_RWKV7_EMB_DEVICE"] = emb_device
    use_v2_runner = pick(env_value(env, "VLLM_USE_V2_MODEL_RUNNER"), rwkv7.get("use_v2_model_runner"), True)
    command_env["VLLM_USE_V2_MODEL_RUNNER"] = (
        "1" if use_v2_runner else "0"
    ) if isinstance(use_v2_runner, bool) else str(use_v2_runner)
    for config_key, env_key in (
        ("rkv_mode", "VLLM_RWKV7_RKV_MODE"),
        ("cmix_sparse", "VLLM_RWKV7_CMIX_SPARSE"),
        ("low_rank_weight", "VLLM_RWKV7_LOW_RANK_WEIGHT"),
    ):
        value = pick(env_value(env, env_key), rwkv7.get(config_key))
        if value is not None:
            command_env[env_key] = str(value)


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

    wkv_mode = str(
        pick(
            args.wkv_mode,
            env_value(env, "HELICOPTER_INFER_WKV_MODE", "VLLM_RWKV7_WKV_MODE"),
            infer.get("wkv_mode"),
            default="fp16",
        )
    )
    emb_device = str(
        pick(
            args.emb_device,
            env_value(env, "HELICOPTER_INFER_EMB_DEVICE", "VLLM_RWKV7_EMB_DEVICE"),
            infer.get("emb_device"),
            default="gpu",
        )
    )
    host = str(pick(args.host, env_value(env, "VLLM_HOST"), runtime.get("host"), default="0.0.0.0"))
    port = str(pick(args.port, env_value(env, "VLLM_PORT"), runtime.get("port"), default="8000"))
    served_model_name = str(
        pick(args.served_model_name, model.get("served_model_name"), model.get("requested_name"), args.model)
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
            env_value(env, "VLLM_TENSOR_PARALLEL_SIZE", "HELICOPTER_TENSOR_PARALLEL_SIZE"),
            infer.get("tensor_parallel_size"),
            gpu.get("tensor_parallel_size"),
        ),
        "--gpu-memory-utilization": pick(
            args.gpu_memory_utilization,
            env_value(env, "VLLM_GPU_MEMORY_UTILIZATION"),
            infer.get("gpu_memory_utilization"),
        ),
        "--max-model-len": pick(
            args.max_model_len,
            env_value(env, "VLLM_MAX_MODEL_LEN"),
            model.get("max_model_len"),
            infer.get("max_model_len"),
        ),
        "--max-num-seqs": pick(
            args.max_num_seqs,
            env_value(env, "VLLM_MAX_NUM_SEQS"),
            infer.get("max_num_seqs"),
        ),
        "--max-num-batched-tokens": pick(
            args.max_num_batched_tokens,
            env_value(env, "VLLM_MAX_NUM_BATCHED_TOKENS"),
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
    if auto_tool_choice if isinstance(auto_tool_choice, bool) else str(auto_tool_choice).strip().lower() in {"1", "true", "yes", "on"}:
        command.append("--enable-auto-tool-choice")

    shown_env: dict[str, str] = {}
    apply_rwkv_env(shown_env, config, env, wkv_mode=wkv_mode, emb_device=emb_device)
    plan_env = dict(env)
    plan_env.update(shown_env)
    return CommandPlan(command=command, cwd=root, shown_env=shown_env, env=plan_env)


def build_takeoff_plan(
    args: Any,
    *,
    root: Path,
    env: dict[str, str],
    config: dict[str, Any],
) -> CommandPlan:
    if args.algorithm != "grpo":
        raise SystemExit("only grpo takeoff is supported for RWKV right now")

    model_path, _ = resolve_model_path(config, args.model, root=root, env=env)
    data_root = dataset_root(config, args.dataset, root=root, env=env)

    paths = table(config, "paths")
    gpu = table(config, "gpu")
    takeoff_common = table(config, "takeoff")
    takeoff_algo_value = takeoff_common.get(args.algorithm, {})
    takeoff_algo = takeoff_algo_value if isinstance(takeoff_algo_value, dict) else {}
    takeoff = {**takeoff_common, **takeoff_algo}

    verl_path = resolve_path(
        str(pick(paths.get("verl_path"), env_value(env, "HELICOPTER_VERL_PATH", "VERL_PATH"), "src/train/verl-rwkv")),
        root=root,
        env=env,
    )
    rwkv_lm_path = resolve_path(
        str(pick(paths.get("rwkv_lm_path"), env_value(env, "RWKV_LM_PATH", "HELICOPTER_RWKV_LM_PATH"), "src/train/rwkv-lm")),
        root=root,
        env=env,
    )
    vllm_rwkv_path = resolve_path(
        str(pick(paths.get("vllm_rwkv_path"), env_value(env, "VLLM_RWKV_PATH", "HELICOPTER_VLLM_RWKV_PATH"), "src/infer/vllm-rwkv")),
        root=root,
        env=env,
    )
    script = verl_path / "examples/rwkv_trainer/run_rwkv7_grpo_vllm.sh"

    if not args.dry_run:
        for path, message in (
            (model_path, "RWKV checkpoint not found"),
            (data_root, "dataset root not found"),
            (rwkv_lm_path, "rwkv-lm repository not found"),
            (vllm_rwkv_path, "vllm-rwkv repository not found"),
            (script, "verl RWKV GRPO script not found"),
        ):
            exists = path.is_dir() if "repository" in message or "root" in message else path.is_file()
            if not exists:
                raise SystemExit(f"{message}: {path}")

    wkv_mode = str(
        pick(
            args.wkv_mode,
            env_value(env, "HELICOPTER_TAKEOFF_WKV_MODE", "VLLM_RWKV7_WKV_MODE"),
            takeoff.get("wkv_mode"),
            default="fp32io16",
        )
    )
    emb_device = str(
        pick(
            args.emb_device,
            env_value(env, "HELICOPTER_TAKEOFF_EMB_DEVICE", "VLLM_RWKV7_EMB_DEVICE"),
            takeoff.get("emb_device"),
            default="cpu",
        )
    )
    num_nodes = pick(
        args.num_nodes,
        env_value(env, "HELICOPTER_NUM_NODES", "NNODES"),
        gpu.get("num_nodes"),
        takeoff.get("num_nodes"),
        default=1,
    )
    num_devices = pick(
        args.num_devices,
        env_value(env, "HELICOPTER_NUM_DEVICES", "NGPUS_PER_NODE"),
        gpu.get("num_devices"),
        takeoff.get("num_devices"),
        default=8,
    )

    shown_env = {
        "DATA_ROOT": str(data_root),
        "NGPUS_PER_NODE": str(num_devices),
        "NNODES": str(num_nodes),
        "PYTHON": python_executable(config, root=root, env=env, require_configured=True),
        "RWKV_LM_PATH": str(rwkv_lm_path),
        "RWKV_MODEL_PATH": str(model_path),
        "VLLM_RWKV_PATH": str(vllm_rwkv_path),
    }
    apply_rwkv_env(shown_env, config, env, wkv_mode=wkv_mode, emb_device=emb_device)

    env_keys = {
        "train_batch_size": "TRAIN_BATCH_SIZE",
        "ppo_mini_batch_size": "PPO_MINI_BATCH_SIZE",
        "ppo_micro_batch_size": "PPO_MICRO_BATCH_SIZE",
        "max_prompt_length": "MAX_PROMPT_LENGTH",
        "max_response_length": "MAX_RESPONSE_LENGTH",
        "ppo_max_token_len_per_gpu": "PPO_MAX_TOKEN_LEN_PER_GPU",
        "actor_lr": "ACTOR_LR",
        "rollout_n": "ROLLOUT_N",
        "rollout_tensor_parallel_size": "ROLLOUT_TP",
        "rollout_gpu_memory_utilization": "ROLLOUT_GPU_MEM_UTIL",
        "rollout_max_num_seqs": "ROLLOUT_MAX_NUM_SEQS",
        "rollout_max_num_batched_tokens": "ROLLOUT_MAX_NUM_BATCHED_TOKENS",
        "total_epochs": "TOTAL_EPOCHS",
        "save_freq": "SAVE_FREQ",
        "test_freq": "TEST_FREQ",
        "project_name": "PROJECT_NAME",
        "experiment_name": "EXPERIMENT_NAME",
    }
    for config_key, env_key in env_keys.items():
        env_override = env_value(env, env_key)
        if env_override is not None:
            shown_env[env_key] = str(env_override)
        elif config_key in takeoff:
            shown_env[env_key] = str(takeoff[config_key])
    if env_value(env, "RWKV_USE_DYNAMIC_BSZ") is not None:
        shown_env["RWKV_USE_DYNAMIC_BSZ"] = str(env_value(env, "RWKV_USE_DYNAMIC_BSZ"))
    elif "rwkv_use_dynamic_bsz" in takeoff:
        dynamic_bsz = takeoff["rwkv_use_dynamic_bsz"]
        shown_env["RWKV_USE_DYNAMIC_BSZ"] = (
            "True" if dynamic_bsz else "False"
        ) if isinstance(dynamic_bsz, bool) else str(dynamic_bsz)

    command = ["bash", str(script)]
    for override in args.override or []:
        command.append(override)

    plan_env = dict(env)
    plan_env.update(shown_env)
    return CommandPlan(command=command, cwd=verl_path, shown_env=shown_env, env=plan_env)
