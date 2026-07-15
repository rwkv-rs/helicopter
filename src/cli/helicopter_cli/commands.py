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


def format_hydra_file_list(value: Any, *, root: Path, env: dict[str, str]) -> str:
    if isinstance(value, list):
        files = [str(resolve_path(str(path), root=root, env=env)) for path in value]
        return "[" + ",".join(f"'{path}'" for path in files) + "]"
    return str(value)


def format_hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


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
            f"RWKV7 Model Runner V2 derives GEMM accumulation from WKV mode: "
            f"{wkv_mode} requires allow_fp16_accumulation={expected_fp16_accumulation}"
        )
    if wkv_mode is not None:
        command_env["VLLM_RWKV7_WKV_MODE"] = wkv_mode


def resolve_fp16_accumulation(
    configured_value: Any,
    *,
    wkv_mode: str | None,
    name: str,
) -> bool | None:
    if configured_value is not None:
        return binary_flag(configured_value, name=name)
    if wkv_mode is not None:
        return wkv_mode == "fp16"
    return None


def strip_vllm_env(env: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in env.items() if not key.startswith("VLLM_")}


def takeoff_value(
    takeoff: dict[str, Any],
    env: dict[str, str],
    config_key: str,
    env_key: str,
    default: Any = None,
) -> Any:
    return pick(env_value(env, env_key), takeoff.get(config_key), default)


def append_hydra_override(overrides: list[str], key: str, value: Any, *, optional: bool = False) -> None:
    if optional and (value is None or str(value) == ""):
        return
    overrides.append(f"{key}={format_hydra_value(value)}")


def append_rwkv_lm_engine_override(
    overrides: list[str],
    key: str,
    value: Any,
    *,
    optional: bool = False,
) -> None:
    for prefix in ("actor_rollout_ref.actor.engine", "actor_rollout_ref.ref.engine"):
        append_hydra_override(overrides, f"{prefix}.{key}", value, optional=optional)


def hydra_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def binary_flag(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip()
    if text == "1":
        return True
    if text == "0":
        return False
    raise SystemExit(f"{name} must be 0 or 1, got {value!r}")


def strict_positive_int(value: Any, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise SystemExit(f"{name} must be positive, got {parsed}")
    return parsed


def hydra_override_map(overrides: list[str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        resolved[key.lstrip("+")] = value
    return resolved


def validate_strict_on_policy_overrides(overrides: list[str], *, env: dict[str, str]) -> None:
    resolved = hydra_override_map(overrides)
    required = {
        "trainer.v1.trainer_mode": "sync",
        "actor_rollout_ref.hybrid_engine": "True",
        "actor_rollout_ref.actor.ppo_epochs": "1",
        "actor_rollout_ref.rollout.checkpoint_engine.backend": "naive",
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes": "64",
        "ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V2_MODEL_RUNNER": '"1"',
        "ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL": '"INFO"',
        "ray_kwargs.ray_init.runtime_env.env_vars.VLLM_RWKV7_STRICT_STREAMING_WEIGHT_UPDATE": '"1"',
        "algorithm.rollout_correction.rollout_is": "token",
        "algorithm.rollout_correction.rollout_is_threshold": "2.0",
        "algorithm.rollout_correction.rollout_is_batch_normalize": "False",
        "algorithm.rollout_correction.rollout_rs": "null",
        "algorithm.rollout_correction.bypass_mode": "False",
        "data.dataloader_num_workers": "0",
        "trainer.nnodes": "1",
        "trainer.n_gpus_per_node": "8",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": "1",
        "actor_rollout_ref.rollout.data_parallel_size": "1",
        "actor_rollout_ref.rollout.pipeline_model_parallel_size": "1",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": "8192",
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": "8192",
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": "8192",
        "actor_rollout_ref.actor.engine.infctx": "True",
        "actor_rollout_ref.ref.engine.infctx": "True",
        "actor_rollout_ref.actor.engine.chunk_ctx": "2048",
        "actor_rollout_ref.ref.engine.chunk_ctx": "2048",
    }
    for key, expected in required.items():
        actual = resolved.get(key)
        if actual != expected:
            raise SystemExit(
                f"strict on-policy takeoff requires {key}={expected}, got {actual!r}"
            )
    train_batch_size = strict_positive_int(
        resolved.get("data.train_batch_size"), name="data.train_batch_size"
    )
    ppo_mini_batch_size = strict_positive_int(
        resolved.get("actor_rollout_ref.actor.ppo_mini_batch_size"),
        name="actor_rollout_ref.actor.ppo_mini_batch_size",
    )
    if ppo_mini_batch_size != train_batch_size:
        raise SystemExit(
            "strict on-policy takeoff requires "
            "actor_rollout_ref.actor.ppo_mini_batch_size == data.train_batch_size, "
            f"got {ppo_mini_batch_size} != {train_batch_size}"
        )

    forbidden = {
        "rollout.nnodes",
        "rollout.n_gpus_per_node",
    }
    present = sorted(forbidden.intersection(resolved))
    if present:
        raise SystemExit(
            "strict on-policy takeoff forbids separate rollout resource overrides: "
            + ", ".join(present)
        )


def reward_function_path(takeoff: dict[str, Any], env: dict[str, str], verl_path: Path) -> Path:
    configured_path = takeoff_value(takeoff, env, "reward_function_path", "REWARD_FUNCTION_PATH")
    if configured_path:
        return Path(str(configured_path))

    reward_function = str(
        takeoff_value(takeoff, env, "reward_function", "REWARD_FUNCTION", "math_verify")
    ).strip()
    if reward_function in {"math_verify", "math_verify_reward"}:
        return verl_path / "examples/rwkv_trainer/math_verify_reward.py"
    if reward_function in {"math_dapo", "math_dapo_reward", "dapo"}:
        return verl_path / "examples/rwkv_trainer/math_dapo_reward.py"
    if reward_function.endswith(".py") or "/" in reward_function:
        return Path(reward_function)
    raise SystemExit(
        "Unknown reward_function "
        f"{reward_function!r}; use math_verify, math_dapo, or reward_function_path"
    )


def build_grpo_hydra_overrides(
    *,
    model_path: Path,
    data_root: Path,
    dataset: dict[str, Any],
    takeoff: dict[str, Any],
    env: dict[str, str],
    root: Path,
    verl_path: Path,
    rwkv_lm_path: Path,
    num_nodes: Any,
    num_devices: Any,
) -> list[str]:
    train_files = env_value(env, "TRAIN_FILES")
    if train_files is None:
        if "train_files" in dataset:
            train_files = format_hydra_file_list(dataset["train_files"], root=root, env=env)
        else:
            train_files = f"['{data_root}/train.parquet']"

    val_files = env_value(env, "VAL_FILES")
    if val_files is None:
        if "val_files" in dataset:
            val_files = format_hydra_file_list(dataset["val_files"], root=root, env=env)
        else:
            val_files = f"['{data_root}/test.parquet']"

    dynamic_bsz = takeoff_value(takeoff, env, "rwkv_use_dynamic_bsz", "RWKV_USE_DYNAMIC_BSZ", False)
    ppo_micro_batch_size = takeoff_value(takeoff, env, "ppo_micro_batch_size", "PPO_MICRO_BATCH_SIZE", 8)
    ppo_max_token_len_per_gpu = takeoff_value(
        takeoff,
        env,
        "ppo_max_token_len_per_gpu",
        "PPO_MAX_TOKEN_LEN_PER_GPU",
        8192,
    )
    rollout_tensor_parallel_size = takeoff_value(
        takeoff,
        env,
        "rollout_tensor_parallel_size",
        "ROLLOUT_TP",
        1,
    )
    rollout_gpu_memory_utilization = takeoff_value(
        takeoff,
        env,
        "rollout_gpu_memory_utilization",
        "ROLLOUT_GPU_MEM_UTIL",
    )
    rollout_n = takeoff_value(takeoff, env, "rollout_n", "ROLLOUT_N", 8)
    rollout_top_p = takeoff_value(takeoff, env, "rollout_top_p", "ROLLOUT_TOP_P", 0.8)
    rollout_max_num_seqs = takeoff_value(takeoff, env, "rollout_max_num_seqs", "ROLLOUT_MAX_NUM_SEQS")
    rollout_max_num_batched_tokens = takeoff_value(
        takeoff,
        env,
        "rollout_max_num_batched_tokens",
        "ROLLOUT_MAX_NUM_BATCHED_TOKENS",
    )
    rollout_update_bucket_mb = takeoff_value(
        takeoff,
        env,
        "rollout_update_weights_bucket_megabytes",
        "ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES",
        64,
    )
    trainer_n_gpus_per_node = takeoff_value(
        takeoff,
        env,
        "trainer_n_gpus_per_node",
        "TRAIN_NGPUS_PER_NODE",
        num_devices,
    )
    train_batch_size = takeoff_value(
        takeoff,
        env,
        "train_batch_size",
        "TRAIN_BATCH_SIZE",
        56,
    )
    ppo_mini_batch_size = takeoff_value(
        takeoff,
        env,
        "ppo_mini_batch_size",
        "PPO_MINI_BATCH_SIZE",
        train_batch_size,
    )
    ppo_epochs = takeoff_value(takeoff, env, "ppo_epochs", "PPO_EPOCHS", 1)
    seed = takeoff_value(takeoff, env, "seed", "HELICOPTER_SEED", 42)
    rwkv_ctx_len = takeoff_value(takeoff, env, "ctx_len", "RWKV_CTX_LEN")
    wkv_mode = str(takeoff_value(takeoff, env, "wkv_mode", "HELICOPTER_TAKEOFF_WKV_MODE", "fp32io16"))
    rollout_io_dtype = "float16" if wkv_mode in {"fp32io16", "fp16"} else None
    rwkv_infctx = hydra_bool(takeoff_value(takeoff, env, "infctx", "RWKV_INFCTX", False))
    rwkv_chunk_ctx = takeoff_value(takeoff, env, "chunk_ctx", "RWKV_CHUNK_CTX")
    val_do_sample = takeoff_value(takeoff, env, "val_do_sample", "VAL_DO_SAMPLE", True)
    val_temperature = takeoff_value(takeoff, env, "val_temperature", "VAL_TEMPERATURE", 1)
    val_top_k = takeoff_value(takeoff, env, "val_top_k", "VAL_TOP_K", 32)
    val_top_p = takeoff_value(takeoff, env, "val_top_p", "VAL_TOP_P", 0.28)
    val_n = takeoff_value(takeoff, env, "val_n", "VAL_N", 4)
    rwkv_generation_prompt = takeoff_value(
        takeoff,
        env,
        "rwkv_generation_prompt",
        "RWKV_GENERATION_PROMPT",
    )
    val_rwkv_generation_prompt = takeoff_value(
        takeoff,
        env,
        "val_rwkv_generation_prompt",
        "VAL_RWKV_GENERATION_PROMPT",
    )
    if rwkv_infctx:
        try:
            rwkv_chunk_ctx = int(rwkv_chunk_ctx)
        except (TypeError, ValueError) as exc:
            raise SystemExit("infctx requires chunk_ctx > 0") from exc
        if rwkv_chunk_ctx <= 0:
            raise SystemExit("infctx requires chunk_ctx > 0")
        if rwkv_chunk_ctx % 16 != 0:
            raise SystemExit("infctx chunk_ctx must be divisible by RWKV CUDA chunk length 16")
        if rwkv_ctx_len is not None and str(rwkv_ctx_len).strip():
            try:
                rwkv_ctx_len = int(rwkv_ctx_len)
            except (TypeError, ValueError) as exc:
                raise SystemExit("infctx requires integer ctx_len") from exc
            if rwkv_chunk_ctx >= rwkv_ctx_len:
                raise SystemExit("infctx requires chunk_ctx < ctx_len")

    reward_path = reward_function_path(takeoff, env, verl_path)
    overrides = [
        f"algorithm.adv_estimator={format_hydra_value(takeoff_value(takeoff, env, 'adv_estimator', 'ADV_ESTIMATOR', 'grpo'))}",
        "algorithm.use_kl_in_reward=False",
        f"data.train_files={train_files}",
        f"data.val_files={val_files}",
        f"data.train_batch_size={format_hydra_value(train_batch_size)}",
        f"data.seed={format_hydra_value(seed)}",
        f"data.max_prompt_length={format_hydra_value(takeoff_value(takeoff, env, 'max_prompt_length', 'MAX_PROMPT_LENGTH', 512))}",
        f"data.max_response_length={format_hydra_value(takeoff_value(takeoff, env, 'max_response_length', 'MAX_RESPONSE_LENGTH', 7168))}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"reward.custom_reward_function.path={reward_path}",
        "reward.custom_reward_function.name=compute_score",
        f"reward.reward_manager.name={format_hydra_value(takeoff_value(takeoff, env, 'reward_manager', 'REWARD_MANAGER', 'naive'))}",
        "model@actor_rollout_ref.model=rwkv_native",
        f"actor_rollout_ref.model.path={model_path}",
        f"actor_rollout_ref.model.rwkv_lm_path={rwkv_lm_path}",
        "actor@actor_rollout_ref.actor=rwkv_lm",
        f"actor_rollout_ref.actor.engine.rwkv_lm_path={rwkv_lm_path}",
        f"actor_rollout_ref.actor.optim.lr={format_hydra_value(takeoff_value(takeoff, env, 'actor_lr', 'ACTOR_LR', '1e-5'))}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={format_hydra_value(ppo_mini_batch_size)}",
        f"actor_rollout_ref.actor.ppo_epochs={format_hydra_value(ppo_epochs)}",
        f"actor_rollout_ref.actor.data_loader_seed={format_hydra_value(seed)}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={format_hydra_value(ppo_micro_batch_size)}",
        f"actor_rollout_ref.actor.use_dynamic_bsz={format_hydra_value(dynamic_bsz)}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={format_hydra_value(ppo_max_token_len_per_gpu)}",
        f"actor_rollout_ref.actor.use_kl_loss={format_hydra_value(takeoff_value(takeoff, env, 'actor_use_kl_loss', 'ACTOR_USE_KL_LOSS', True))}",
        f"actor_rollout_ref.actor.kl_loss_coef={format_hydra_value(takeoff_value(takeoff, env, 'actor_kl_loss_coef', 'ACTOR_KL_LOSS_COEF', 0.0))}",
        f"actor_rollout_ref.actor.kl_loss_type={format_hydra_value(takeoff_value(takeoff, env, 'actor_kl_loss_type', 'ACTOR_KL_LOSS_TYPE', 'low_var_kl'))}",
    ]
    append_hydra_override(
        overrides,
        "actor_rollout_ref.actor.optim.lr_warmup_steps",
        takeoff_value(takeoff, env, "actor_lr_warmup_steps", "ACTOR_LR_WARMUP_STEPS"),
        optional=True,
    )
    append_hydra_override(
        overrides,
        "actor_rollout_ref.actor.optim.weight_decay",
        takeoff_value(takeoff, env, "actor_weight_decay", "ACTOR_WEIGHT_DECAY"),
        optional=True,
    )
    append_hydra_override(
        overrides,
        "actor_rollout_ref.actor.entropy_coeff",
        takeoff_value(takeoff, env, "actor_entropy_coeff", "ACTOR_ENTROPY_COEFF"),
        optional=True,
    )
    append_hydra_override(
        overrides,
        "actor_rollout_ref.actor.optim.clip_grad",
        takeoff_value(takeoff, env, "actor_grad_clip", "ACTOR_GRAD_CLIP"),
        optional=True,
    )
    append_hydra_override(
        overrides,
        "actor_rollout_ref.actor.clip_ratio_low",
        takeoff_value(takeoff, env, "clip_ratio_low", "CLIP_RATIO_LOW"),
        optional=True,
    )
    append_hydra_override(
        overrides,
        "actor_rollout_ref.actor.clip_ratio_high",
        takeoff_value(takeoff, env, "clip_ratio_high", "CLIP_RATIO_HIGH"),
        optional=True,
    )
    append_hydra_override(
        overrides,
        "actor_rollout_ref.actor.clip_ratio_c",
        takeoff_value(takeoff, env, "clip_ratio_c", "CLIP_RATIO_C"),
        optional=True,
    )

    overrides.extend(
        [
            "ref@actor_rollout_ref.ref=rwkv_lm",
            f"actor_rollout_ref.ref.engine.rwkv_lm_path={rwkv_lm_path}",
            f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={format_hydra_value(ppo_micro_batch_size)}",
            f"actor_rollout_ref.ref.log_prob_use_dynamic_bsz={format_hydra_value(dynamic_bsz)}",
            f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={format_hydra_value(ppo_max_token_len_per_gpu)}",
            "actor_rollout_ref.rollout.name=vllm",
            "actor_rollout_ref.rollout.load_format=auto",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={format_hydra_value(rollout_tensor_parallel_size)}",
            f"actor_rollout_ref.rollout.n={format_hydra_value(rollout_n)}",
            f"actor_rollout_ref.rollout.seed={format_hydra_value(seed)}",
            f"actor_rollout_ref.rollout.top_p={format_hydra_value(rollout_top_p)}",
            "actor_rollout_ref.rollout.enable_prefix_caching=False",
            f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={format_hydra_value(ppo_micro_batch_size)}",
            f"actor_rollout_ref.rollout.log_prob_use_dynamic_bsz={format_hydra_value(dynamic_bsz)}",
            f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={format_hydra_value(ppo_max_token_len_per_gpu)}",
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.tokenizer_mode=rwkv",
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend=uni",
            '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V2_MODEL_RUNNER="1"',
            '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL="INFO"',
            '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_RWKV7_STRICT_STREAMING_WEIGHT_UPDATE="1"',
            "actor_rollout_ref.hybrid_engine=True",
            "trainer.v1.trainer_mode=sync",
            "actor_rollout_ref.rollout.checkpoint_engine.backend=naive",
            "actor_rollout_ref.rollout.checkpoint_engine."
            f"update_weights_bucket_megabytes={format_hydra_value(rollout_update_bucket_mb)}",
            "algorithm.rollout_correction.rollout_is=token",
            "algorithm.rollout_correction.rollout_is_threshold=2.0",
            "algorithm.rollout_correction.rollout_is_batch_normalize=False",
            "algorithm.rollout_correction.rollout_rs=null",
            "algorithm.rollout_correction.bypass_mode=False",
            "data.dataloader_num_workers=0",
            f"actor_rollout_ref.rollout.val_kwargs.do_sample={format_hydra_value(val_do_sample)}",
            f"actor_rollout_ref.rollout.val_kwargs.temperature={format_hydra_value(val_temperature)}",
            f"actor_rollout_ref.rollout.val_kwargs.top_k={format_hydra_value(val_top_k)}",
            f"actor_rollout_ref.rollout.val_kwargs.top_p={format_hydra_value(val_top_p)}",
            f"actor_rollout_ref.rollout.val_kwargs.n={format_hydra_value(val_n)}",
        ]
    )
    if rollout_io_dtype is not None:
        # RWKV-LM's training CUDA extensions are BF16-only. Keep the native
        # actor/ref engine precision while making the vLLM FP16 boundary explicit.
        overrides.append(f"actor_rollout_ref.rollout.dtype={rollout_io_dtype}")
    append_hydra_override(
        overrides,
        "data.train_max_samples",
        takeoff_value(takeoff, env, "train_max_samples", "TRAIN_MAX_SAMPLES"),
        optional=True,
    )
    append_hydra_override(
        overrides,
        "data.val_max_samples",
        takeoff_value(takeoff, env, "val_max_samples", "VAL_MAX_SAMPLES"),
        optional=True,
    )
    append_hydra_override(
        overrides,
        "+data.apply_chat_template_kwargs.rwkv_generation_prompt",
        rwkv_generation_prompt,
        optional=True,
    )
    append_hydra_override(
        overrides,
        "+data.val_apply_chat_template_kwargs.rwkv_generation_prompt",
        val_rwkv_generation_prompt,
        optional=True,
    )
    append_rwkv_lm_engine_override(overrides, "ctx_len", rwkv_ctx_len, optional=True)
    append_rwkv_lm_engine_override(overrides, "infctx", rwkv_infctx)
    if rwkv_infctx:
        append_rwkv_lm_engine_override(overrides, "chunk_ctx", rwkv_chunk_ctx)
    append_hydra_override(
        overrides,
        "actor_rollout_ref.rollout.gpu_memory_utilization",
        rollout_gpu_memory_utilization,
        optional=True,
    )
    append_hydra_override(
        overrides,
        "actor_rollout_ref.rollout.max_num_seqs",
        rollout_max_num_seqs,
        optional=True,
    )
    append_hydra_override(
        overrides,
        "actor_rollout_ref.rollout.max_num_batched_tokens",
        rollout_max_num_batched_tokens,
        optional=True,
    )
    for config_key, env_key, hydra_key in (
        ("rollout_mode", "ROLLOUT_MODE", "actor_rollout_ref.rollout.mode"),
        ("rollout_data_parallel_size", "ROLLOUT_DP", "actor_rollout_ref.rollout.data_parallel_size"),
        (
            "rollout_pipeline_parallel_size",
            "ROLLOUT_PP",
            "actor_rollout_ref.rollout.pipeline_model_parallel_size",
        ),
    ):
        append_hydra_override(
            overrides,
            hydra_key,
            takeoff_value(
                takeoff,
                env,
                config_key,
                env_key,
                1 if config_key in {"rollout_data_parallel_size", "rollout_pipeline_parallel_size"} else None,
            ),
            optional=True,
        )

    overrides.extend(
        [
            "critic.enable=False",
            'trainer.logger=["console","file"]',
            f"trainer.project_name={format_hydra_value(takeoff_value(takeoff, env, 'project_name', 'PROJECT_NAME', 'verl_rwkv_grpo'))}",
            f"trainer.experiment_name={format_hydra_value(takeoff_value(takeoff, env, 'experiment_name', 'EXPERIMENT_NAME', 'rwkv7_grpo_vllm'))}",
            f"trainer.nnodes={format_hydra_value(num_nodes)}",
            f"trainer.n_gpus_per_node={format_hydra_value(trainer_n_gpus_per_node)}",
            f"trainer.save_freq={format_hydra_value(takeoff_value(takeoff, env, 'save_freq', 'SAVE_FREQ', 20))}",
            f"trainer.test_freq={format_hydra_value(takeoff_value(takeoff, env, 'test_freq', 'TEST_FREQ', -1))}",
            f"trainer.val_before_train={format_hydra_value(takeoff_value(takeoff, env, 'val_before_train', 'VAL_BEFORE_TRAIN', True))}",
            f"trainer.total_epochs={format_hydra_value(takeoff_value(takeoff, env, 'total_epochs', 'TOTAL_EPOCHS', 2))}",
        ]
    )
    append_hydra_override(
        overrides,
        "trainer.validation_data_dir",
        takeoff_value(takeoff, env, "validation_data_dir", "VALIDATION_DATA_DIR"),
        optional=True,
    )
    append_hydra_override(
        overrides,
        "trainer.total_training_steps",
        takeoff_value(takeoff, env, "total_training_steps", "TOTAL_TRAINING_STEPS"),
        optional=True,
    )
    profiler_tool = takeoff_value(takeoff, env, "profiler_tool", "PROFILER_TOOL")
    profiler_steps = takeoff_value(takeoff, env, "profiler_steps", "PROFILER_STEPS")
    if profiler_tool is not None or profiler_steps is not None:
        if profiler_tool != "nsys" or not isinstance(profiler_steps, list) or not profiler_steps:
            raise SystemExit("strict profiling requires profiler_tool='nsys' and a non-empty profiler_steps list")
        overrides.extend(
            [
                "global_profiler.tool=nsys",
                f"global_profiler.steps={format_hydra_value(profiler_steps)}",
                "global_profiler.profile_continuous_steps=False",
                "actor_rollout_ref.actor.profiler.enable=True",
                "actor_rollout_ref.actor.profiler.all_ranks=True",
                "actor_rollout_ref.ref.profiler.enable=True",
                "actor_rollout_ref.ref.profiler.all_ranks=True",
                "actor_rollout_ref.rollout.profiler.enable=True",
                "actor_rollout_ref.rollout.profiler.all_ranks=True",
            ]
        )
    return overrides


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
    allow_fp16_accumulation_value = pick(
        args.allow_fp16_accumulation,
        env_value(
            env,
            "HELICOPTER_INFER_ALLOW_FP16_ACCUMULATION",
            "VLLM_RWKV7_ALLOW_FP16_ACCUMULATION",
        ),
        infer.get("allow_fp16_accumulation"),
    )
    allow_fp16_accumulation = resolve_fp16_accumulation(
        allow_fp16_accumulation_value,
        wkv_mode=wkv_mode,
        name="HELICOPTER_INFER_ALLOW_FP16_ACCUMULATION",
    )
    host = str(pick(args.host, runtime.get("host"), default="0.0.0.0"))
    port = str(pick(args.port, runtime.get("port"), default="8000"))
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
            infer.get("max_num_seqs"),
        ),
        "--max-num-batched-tokens": pick(
            args.max_num_batched_tokens,
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
    apply_rwkv_env(
        shown_env,
        wkv_mode=wkv_mode,
        emb_device=emb_device,
        allow_fp16_accumulation=allow_fp16_accumulation,
    )
    plan_env = strip_vllm_env(env)
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
    datasets = table(config, "datasets")
    dataset_value = datasets.get(args.dataset, {})
    dataset = dataset_value if isinstance(dataset_value, dict) else {}

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
        str(pick(paths.get("vllm_rwkv_path"), env_value(env, "HELICOPTER_VLLM_RWKV_PATH", "VLLM_RWKV_PATH"), "src/infer/vllm-rwkv")),
        root=root,
        env=env,
    )

    has_train_files = "train_files" in dataset or env_value(env, "TRAIN_FILES") is not None
    has_val_files = "val_files" in dataset or env_value(env, "VAL_FILES") is not None
    dataset_uses_explicit_files = has_train_files and has_val_files
    if not args.dry_run:
        for path, message in (
            (model_path, "RWKV checkpoint not found"),
            (rwkv_lm_path, "rwkv-lm repository not found"),
            (vllm_rwkv_path, "vllm-rwkv repository not found"),
        ):
            exists = path.is_dir() if "repository" in message or "root" in message else path.is_file()
            if not exists:
                raise SystemExit(f"{message}: {path}")
        if not dataset_uses_explicit_files and not data_root.is_dir():
            raise SystemExit(f"dataset root not found: {data_root}")

    wkv_mode = str(
        pick(
            args.wkv_mode,
            env_value(env, "HELICOPTER_TAKEOFF_WKV_MODE", "VLLM_RWKV7_WKV_MODE"),
            takeoff.get("wkv_mode"),
            default="fp32io16",
        )
    )
    emb_device_value = pick(
        args.emb_device,
        env_value(env, "HELICOPTER_TAKEOFF_EMB_DEVICE"),
        takeoff.get("emb_device"),
        default="gpu",
    )
    emb_device = str(emb_device_value) if emb_device_value is not None else None
    allow_fp16_accumulation_value = pick(
        args.allow_fp16_accumulation,
        env_value(
            env,
            "HELICOPTER_TAKEOFF_ALLOW_FP16_ACCUMULATION",
            "VLLM_RWKV7_ALLOW_FP16_ACCUMULATION",
        ),
        takeoff.get("allow_fp16_accumulation"),
    )
    allow_fp16_accumulation = resolve_fp16_accumulation(
        allow_fp16_accumulation_value,
        wkv_mode=wkv_mode,
        name="HELICOPTER_TAKEOFF_ALLOW_FP16_ACCUMULATION",
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

    python = python_executable(config, root=root, env=env, require_configured=True)
    shown_env: dict[str, str] = {}
    apply_rwkv_env(
        shown_env,
        wkv_mode=wkv_mode,
        emb_device=emb_device,
        allow_fp16_accumulation=allow_fp16_accumulation,
    )
    shown_env["PYTHON"] = python
    shown_env["RWKV_MODEL_PATH"] = str(model_path)
    shown_env["RWKV_LM_PATH"] = str(rwkv_lm_path)
    plan_env = strip_vllm_env(env)
    plan_env.update(shown_env)
    current_pythonpath = plan_env.get("PYTHONPATH")
    plan_env["PYTHONPATH"] = (
        f"{vllm_rwkv_path}{os.pathsep}{current_pythonpath}" if current_pythonpath else str(vllm_rwkv_path)
    )
    shown_env["PYTHONPATH"] = plan_env["PYTHONPATH"]

    overrides = build_grpo_hydra_overrides(
        model_path=model_path,
        data_root=data_root,
        dataset=dataset,
        takeoff=takeoff,
        env=env,
        root=root,
        verl_path=verl_path,
        rwkv_lm_path=rwkv_lm_path,
        num_nodes=num_nodes,
        num_devices=num_devices,
    )
    overrides.extend(args.override or [])
    validate_strict_on_policy_overrides(overrides, env=env)

    command = [
        python,
        "-m",
        "verl.trainer.main_ppo",
        *overrides,
    ]
    return CommandPlan(command=command, cwd=verl_path, shown_env=shown_env, env=plan_env)
