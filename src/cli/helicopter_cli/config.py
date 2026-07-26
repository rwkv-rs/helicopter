from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from .env import env_value, pick
from .paths import resolve_path


DEFAULT_LOCAL_CONFIG_DIR = Path("configs/local")
DEFAULT_EXAMPLE_CONFIG = Path("configs/example.toml")
GROUPED_CONFIG_SECTIONS = {
    "experiment",
    "model",
    "data",
    "algorithm",
    "reward",
    "optimizer",
    "generation",
    "execution",
    "evaluation",
    "checkpoint",
    "logging",
}
LEGACY_CONFIG_SECTIONS = {
    "models",
    "datasets",
    "infer",
    "runtime",
    "gpu",
    "takeoff",
    "paths",
}
SELECTED_DATASET_KEY = "__selected__"
CONTEXT_SUFFIX_RE = re.compile(r"(?:^|[-_.])ctx(?P<tokens>[1-9]\d*)(?=[-_.]|$)")


def default_config_path(root: Path) -> Path:
    local_dir = root / DEFAULT_LOCAL_CONFIG_DIR
    if local_dir.exists():
        local_configs = sorted(
            path for path in local_dir.glob("*.toml") if path.is_file()
        )
        if local_configs:
            return local_configs[-1]
    return root / DEFAULT_EXAMPLE_CONFIG


def load_config(
    root: Path,
    config_path: str | None,
) -> tuple[dict[str, Any], Path]:
    path = Path(config_path) if config_path else default_config_path(root)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise SystemExit(f"config file not found: {path}")
    with path.open("rb") as file:
        return compile_config(tomllib.load(file)), path


def table(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def _required_table(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise SystemExit(f"grouped config requires [{name}]")
    return value


def _required_value(section: dict[str, Any], key: str, *, section_name: str) -> Any:
    value = section.get(key)
    if value is None or value == "":
        raise SystemExit(f"grouped config requires {section_name}.{key}")
    return value


def _put_if_present(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None and value != "":
        target[key] = value


def is_grouped_config(config: dict[str, Any]) -> bool:
    return bool(GROUPED_CONFIG_SECTIONS.intersection(config))


def context_tokens_from_checkpoint(checkpoint: Any) -> int:
    """Derive model context length from the checkpoint filename's ``ctxN`` suffix."""

    filename = str(checkpoint).replace("\\", "/").rsplit("/", 1)[-1]
    matches = [
        int(match.group("tokens")) for match in CONTEXT_SUFFIX_RE.finditer(filename)
    ]
    if len(matches) != 1:
        raise SystemExit(
            "model.checkpoint filename must contain exactly one context suffix such as "
            f"'ctx10240'; got {filename!r}"
        )
    if matches[0] < 2:
        raise SystemExit(
            "model checkpoint context must leave room for prompt and response tokens"
        )
    return matches[0]


def compile_config(config: dict[str, Any]) -> dict[str, Any]:
    """Compile the public grouped TOML schema into the existing command-plan shape.

    The compatibility tables are in-memory implementation details. Checked-in
    experiment files stay self-contained and expose only the grouped schema.
    """

    if not is_grouped_config(config):
        return deepcopy(config)

    mixed_sections = sorted(LEGACY_CONFIG_SECTIONS.intersection(config))
    if mixed_sections:
        raise SystemExit(
            "grouped config cannot mix legacy sections: " + ", ".join(mixed_sections)
        )

    compiled = deepcopy(config)
    experiment = _required_table(compiled, "experiment")
    model = _required_table(compiled, "model")
    data = _required_table(compiled, "data")
    data_train = _required_table(data, "train")
    data_validation = _required_table(data, "validation")
    algorithm = _required_table(compiled, "algorithm")
    reward = _required_table(compiled, "reward")
    optimizer = _required_table(compiled, "optimizer")
    generation = _required_table(compiled, "generation")
    generation_train = _required_table(generation, "train")
    generation_validation = _required_table(generation, "validation")
    execution = _required_table(compiled, "execution")
    execution_rollout = _required_table(execution, "rollout")
    evaluation = _required_table(compiled, "evaluation")
    checkpoint = _required_table(compiled, "checkpoint")
    logging = _required_table(compiled, "logging")

    removed_fields = (
        (model, "context_tokens", "model.context_tokens"),
        (data_train, "max_prompt_tokens", "data.train.max_prompt_tokens"),
        (
            generation_train,
            "max_response_tokens",
            "generation.train.max_response_tokens",
        ),
        (generation_train, "stop_on_eos", "generation.train.stop_on_eos"),
        (execution, "dynamic_microbatching", "execution.dynamic_microbatching"),
        (
            execution,
            "train_token_budget_per_gpu",
            "execution.train_token_budget_per_gpu",
        ),
    )
    configured_removed_fields = [
        qualified_name
        for section, key, qualified_name in removed_fields
        if key in section
    ]
    if configured_removed_fields:
        raise SystemExit(
            "grouped config contains removed derived/invariant fields: "
            + ", ".join(configured_removed_fields)
        )

    model_name = str(_required_value(model, "name", section_name="model"))
    model_checkpoint = _required_value(model, "checkpoint", section_name="model")
    context_tokens = context_tokens_from_checkpoint(model_checkpoint)
    model_entry = {
        "path": model_checkpoint,
        "served_model_name": model_name,
        "max_model_len": context_tokens,
    }
    compiled["models"] = {model_name: model_entry}

    train_files = _required_value(data_train, "files", section_name="data.train")
    if not isinstance(train_files, list) or not train_files:
        raise SystemExit("data.train.files must be a non-empty list")
    train_prompt_key = str(data_train.get("prompt_field", "prompt")).strip()
    if not train_prompt_key:
        raise SystemExit("data.train.prompt_field must not be empty")
    val_prompt_key = str(data_validation.get("prompt_field", "prompt")).strip()
    if not val_prompt_key:
        raise SystemExit("data.validation.prompt_field must not be empty")
    suites = _required_value(data_validation, "suites", section_name="data.validation")
    if not isinstance(suites, list) or not suites:
        raise SystemExit(
            "data.validation.suites must contain at least one [[data.validation.suites]]"
        )
    val_files: list[Any] = []
    for index, suite in enumerate(suites):
        if not isinstance(suite, dict) or not suite.get("file"):
            raise SystemExit(f"data.validation.suites[{index}] requires file")
        val_files.append(suite["file"])
    compiled["datasets"] = {
        SELECTED_DATASET_KEY: {
            "train_files": train_files,
            "val_files": val_files,
            "train_prompt_key": train_prompt_key,
            "val_prompt_key": val_prompt_key,
        }
    }

    algorithm_name = str(_required_value(algorithm, "name", section_name="algorithm"))
    if algorithm_name == "maxrl":
        if "optimizer_steps" in experiment:
            raise SystemExit(
                "MaxRL experiment.optimizer_steps was removed; use "
                "experiment.candidate_dataset_passes (0 means manual stop)"
            )
        candidate_dataset_passes = int(
            _required_value(
                experiment,
                "candidate_dataset_passes",
                section_name="experiment",
            )
        )
        if candidate_dataset_passes < 0:
            raise SystemExit("experiment.candidate_dataset_passes must be >= 0")
    else:
        candidate_dataset_passes = 1
    prompts_per_step = _required_value(
        algorithm, "prompts_per_step", section_name="algorithm"
    )
    kl_coefficient = algorithm.get("kl_coefficient", 0.0)
    context_mode = str(
        _required_value(execution, "context_mode", section_name="execution")
    )
    if context_mode != "state_passing":
        raise SystemExit(
            "strict grouped config requires execution.context_mode='state_passing'"
        )
    validation_strategy = str(
        _required_value(
            generation_validation,
            "strategy",
            section_name="generation.validation",
        )
    )
    if validation_strategy not in {"sample", "greedy"}:
        raise SystemExit(
            "generation.validation.strategy must be either 'sample' or 'greedy'"
        )
    nodes = int(_required_value(execution, "nodes", section_name="execution"))
    gpus_per_node = int(
        _required_value(execution, "gpus_per_node", section_name="execution")
    )
    rollout_replicas = int(
        _required_value(execution_rollout, "replicas", section_name="execution.rollout")
    )
    rollout_tp = int(
        _required_value(
            execution_rollout,
            "tensor_parallel_size_per_replica",
            section_name="execution.rollout",
        )
    )
    if rollout_replicas * rollout_tp != nodes * gpus_per_node:
        raise SystemExit(
            "execution.rollout.replicas * tensor_parallel_size_per_replica "
            "must consume every configured GPU"
        )

    takeoff: dict[str, Any] = {
        "project_name": _required_value(
            experiment, "project", section_name="experiment"
        ),
        "experiment_name": _required_value(
            experiment, "name", section_name="experiment"
        ),
        "seed": _required_value(experiment, "seed", section_name="experiment"),
        "train_batch_size": prompts_per_step,
        # Strict on-policy performs exactly one optimizer update per rollout group.
        "ppo_mini_batch_size": prompts_per_step,
        "ppo_epochs": 1,
        # One generated response occupies one fixed microbatch slot.
        "ppo_micro_batch_size": 1,
        "rollout_n": _required_value(
            algorithm, "responses_per_prompt", section_name="algorithm"
        ),
        "clip_ratio_low": _required_value(
            algorithm, "ppo_clip", section_name="algorithm"
        ),
        "clip_ratio_high": _required_value(
            algorithm, "ppo_clip", section_name="algorithm"
        ),
        "actor_entropy_coeff": algorithm.get("entropy_coefficient", 0.0),
        "actor_use_kl_loss": float(kl_coefficient) != 0.0,
        "actor_kl_loss_coef": kl_coefficient,
        "reward_manager": _required_value(reward, "manager", section_name="reward"),
        "reward_function": _required_value(reward, "scorer", section_name="reward"),
        "actor_lr": _required_value(
            optimizer, "learning_rate", section_name="optimizer"
        ),
        "actor_lr_warmup_steps": optimizer.get("warmup_steps", 0),
        "actor_weight_decay": optimizer.get("weight_decay", 0.0),
        "actor_grad_clip": _required_value(
            optimizer, "gradient_norm_limit", section_name="optimizer"
        ),
        "rollout_temperature": _required_value(
            generation_train, "temperature", section_name="generation.train"
        ),
        "rollout_top_k": _required_value(
            generation_train, "top_k", section_name="generation.train"
        ),
        "rollout_top_p": _required_value(
            generation_train, "top_p", section_name="generation.train"
        ),
        "val_do_sample": validation_strategy == "sample",
        "val_n": _required_value(
            generation_validation,
            "responses_per_prompt",
            section_name="generation.validation",
        ),
        "val_temperature": _required_value(
            generation_validation,
            "temperature",
            section_name="generation.validation",
        ),
        "val_top_k": _required_value(
            generation_validation, "top_k", section_name="generation.validation"
        ),
        "val_top_p": _required_value(
            generation_validation, "top_p", section_name="generation.validation"
        ),
        "val_presence_penalty": _required_value(
            generation_validation,
            "presence_penalty",
            section_name="generation.validation",
        ),
        "val_frequency_penalty": _required_value(
            generation_validation,
            "frequency_penalty",
            section_name="generation.validation",
        ),
        "val_penalty_decay": _required_value(
            generation_validation,
            "penalty_decay",
            section_name="generation.validation",
        ),
        "wkv_mode": _required_value(execution, "wkv_mode", section_name="execution"),
        "ctx_len": model_entry["max_model_len"],
        "infctx": True,
        "chunk_ctx": _required_value(
            execution, "state_chunk_tokens", section_name="execution"
        ),
        "rwkv_use_dynamic_bsz": False,
        "num_nodes": nodes,
        "trainer_n_gpus_per_node": gpus_per_node,
        "rollout_tensor_parallel_size": rollout_tp,
        "rollout_pipeline_parallel_size": _required_value(
            execution_rollout,
            "pipeline_parallel_size_per_replica",
            section_name="execution.rollout",
        ),
        "rollout_data_parallel_size": 1,
        "rollout_max_num_seqs": _required_value(
            execution_rollout,
            "max_concurrent_sequences_per_replica",
            section_name="execution.rollout",
        ),
        "rollout_max_num_batched_tokens": _required_value(
            execution_rollout,
            "generation_token_budget_per_replica",
            section_name="execution.rollout",
        ),
        "rollout_update_weights_bucket_megabytes": _required_value(
            execution_rollout,
            "weight_update_bucket_mib",
            section_name="execution.rollout",
        ),
        "val_before_train": _required_value(
            evaluation, "before_training", section_name="evaluation"
        ),
        "test_freq": _required_value(
            evaluation, "every_optimizer_steps", section_name="evaluation"
        ),
        "save_freq": _required_value(
            checkpoint, "every_optimizer_steps", section_name="checkpoint"
        ),
        "trainer_loggers": _required_value(logging, "backends", section_name="logging"),
        "total_epochs": candidate_dataset_passes,
    }
    if algorithm_name != "maxrl":
        takeoff["total_training_steps"] = _required_value(
            experiment,
            "optimizer_steps",
            section_name="experiment",
        )
    if algorithm_name != "grpo":
        takeoff["adv_estimator"] = algorithm_name
    _put_if_present(takeoff, "clip_ratio_c", algorithm.get("dual_clip"))
    _put_if_present(takeoff, "train_max_samples", data_train.get("max_records"))
    _put_if_present(
        takeoff,
        "val_max_samples",
        data_validation.get("max_records"),
    )
    _put_if_present(
        takeoff,
        "rwkv_generation_prompt",
        model.get("prompt_mode"),
    )
    _put_if_present(
        takeoff,
        "val_rwkv_generation_prompt",
        model.get("prompt_mode"),
    )
    _put_if_present(
        takeoff,
        "rwkv_prompt_template",
        model.get("prompt_template"),
    )
    _put_if_present(
        takeoff,
        "validation_data_dir",
        evaluation.get("output_directory"),
    )
    compiled["takeoff"] = {"grpo": takeoff}
    compiled["gpu"] = {
        "num_nodes": execution["nodes"],
        "num_devices": execution["gpus_per_node"],
    }
    compiled["infer"] = {
        "wkv_mode": execution["wkv_mode"],
        "max_model_len": model_entry["max_model_len"],
    }
    return compiled


def resolve_model_entry(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    models = table(config, "models")
    seen: list[str] = []
    current_name = model_name

    while True:
        entry = models.get(current_name)
        if not isinstance(entry, dict):
            raise SystemExit(f"model alias not found in config: {model_name}")
        if current_name in seen:
            chain = " -> ".join([*seen, current_name])
            raise SystemExit(f"cyclic model alias in config: {chain}")
        seen.append(current_name)
        alias = entry.get("alias")
        if not alias:
            resolved = dict(entry)
            resolved.setdefault("name", current_name)
            resolved.setdefault("requested_name", model_name)
            return resolved
        current_name = str(alias)


def resolve_model_path(
    config: dict[str, Any],
    model_name: str,
    *,
    root: Path,
    env: dict[str, str],
) -> tuple[Path, dict[str, Any]]:
    entry = resolve_model_entry(config, model_name)
    paths = table(config, "paths")

    strict_checkpoint_path = env_value(env, "HELICOPTER_CHECKPOINT_PATH")
    if strict_checkpoint_path:
        return resolve_path(strict_checkpoint_path, root=root, env=env), entry

    if "path" in entry:
        return resolve_path(str(entry["path"]), root=root, env=env), entry

    filename = entry.get("file")
    if not filename:
        raise SystemExit(f"model {model_name} needs either path or file in config")

    base_value = pick(
        entry.get("weight_path"),
        paths.get("weight_path"),
        env_value(env, "WEIGHT_PATH", "HELICOPTER_WEIGHT_PATH"),
    )
    if not base_value:
        raise SystemExit(
            "WEIGHT_PATH is not set and config paths.weight_path is missing"
        )

    base = resolve_path(str(base_value), root=root, env=env)
    base_dir = base.parent if base.suffix == ".pth" else base
    return base_dir / str(filename), entry


def dataset_root(
    config: dict[str, Any],
    dataset_name: str,
    *,
    root: Path,
    env: dict[str, str],
) -> Path:
    datasets = table(config, "datasets")
    entry = datasets.get(dataset_name)
    paths = table(config, "paths")

    if isinstance(entry, dict) and entry.get("root"):
        return resolve_path(str(entry["root"]), root=root, env=env)

    base_value = pick(
        paths.get("datasets_path"),
        env_value(env, "DATASETS_PATH", "HELICOPTER_DATASETS_PATH"),
        "/workspace/Datasets",
    )
    return resolve_path(str(base_value), root=root, env=env) / dataset_name
