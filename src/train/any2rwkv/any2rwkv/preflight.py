from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from .artifacts import file_sha256, git_sha
from .errors import ContractError

RWKV_HF_REVISION = "15cd7d7e896efe852f6994a22c34fd14cb60c2c6"
RWKV_HF_SOURCE_URL = "https://github.com/rwkv-rs/hf-adapter.git"
RWKV_HF_REQUIREMENT = f"rwkv7-hf-adapter @ git+{RWKV_HF_SOURCE_URL}@{RWKV_HF_REVISION}"
TRANSFORMERS_REVISION = "eb8248eb9083288e7769518077a1be9c0f7cf7b8"
TRANSFORMERS_SOURCE_URL = "https://github.com/rwkv-rs/transformers-rwkv.git"
TRANSFORMERS_REQUIREMENT = (
    f"transformers @ git+{TRANSFORMERS_SOURCE_URL}@{TRANSFORMERS_REVISION}"
)


def _distribution_binding(
    name: str,
    *,
    expected_url: str,
    expected_revision: str,
) -> dict[str, Any]:
    direct_url_error: str | None = None
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        distribution = None
        direct_url_error = "distribution is not installed"
    direct_url: dict[str, Any] = {}
    if distribution is not None:
        raw_direct_url = distribution.read_text("direct_url.json")
        if raw_direct_url:
            try:
                decoded = json.loads(raw_direct_url)
                if isinstance(decoded, dict):
                    direct_url = decoded
                else:
                    direct_url_error = "direct_url.json is not a JSON object"
            except json.JSONDecodeError as error:
                direct_url_error = str(error)
        else:
            direct_url_error = "direct_url.json is missing"
    vcs_info = direct_url.get("vcs_info")
    if not isinstance(vcs_info, dict):
        vcs_info = {}
    actual_url = direct_url.get("url")
    commit_id = vcs_info.get("commit_id")
    requested_revision = vcs_info.get("requested_revision")
    vcs = vcs_info.get("vcs")
    source_matches = actual_url == expected_url
    requested_revision_matches = requested_revision == expected_revision
    revision_matches = commit_id == expected_revision
    return {
        "name": name,
        "version": distribution.version if distribution is not None else None,
        "direct_url": actual_url,
        "direct_url_error": direct_url_error,
        "vcs": vcs,
        "requested_revision": requested_revision,
        "commit_id": commit_id,
        "expected_url": expected_url,
        "expected_revision": expected_revision,
        "source_matches": source_matches,
        "requested_revision_matches": requested_revision_matches,
        "revision_matches": revision_matches,
        "requirement_satisfied": bool(
            distribution is not None
            and direct_url_error is None
            and vcs == "git"
            and source_matches
            and requested_revision_matches
            and revision_matches
        ),
    }


def _git_sha_or_none(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        return git_sha(path)
    except (OSError, RuntimeError, ValueError):
        return None


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must contain a JSON object: {path}")
    return payload


def collect_preflight(
    product_root: Path,
    *,
    expected_rwkv_hf_sha: str,
    expected_rwkv_lm_sha: str,
) -> dict[str, Any]:
    transformers_distribution = _distribution_binding(
        "transformers",
        expected_url=TRANSFORMERS_SOURCE_URL,
        expected_revision=TRANSFORMERS_REVISION,
    )
    rwkv_hf_distribution = _distribution_binding(
        "rwkv7-hf-adapter",
        expected_url=RWKV_HF_SOURCE_URL,
        expected_revision=RWKV_HF_REVISION,
    )
    rwkv_lm_checkout = (product_root / "src/train/rwkv-lm").resolve()
    try:
        rwkv_hf = importlib.import_module("rwkv7_hf")
        rwkv_hf_file = getattr(rwkv_hf, "__file__", None)
        rwkv_hf_module = Path(rwkv_hf_file).resolve() if rwkv_hf_file else None
    except ImportError:
        rwkv_hf_module = None
    try:
        from transformers.models.rwkv7.configuration_rwkv7 import Rwkv7Config
        from transformers.models.rwkv7.modeling_rwkv7 import Rwkv7ForCausalLM

        transformers_public_interface = bool(
            Rwkv7Config.model_type == "rwkv7"
            and Rwkv7ForCausalLM.base_model_prefix == "model"
        )
    except ImportError:
        transformers_public_interface = False
    rwkv_lm_commit = _git_sha_or_none(rwkv_lm_checkout)
    kernel_loader = rwkv_lm_checkout / "src/infctx_kernel.py"
    kernel_source = rwkv_lm_checkout / "cuda/rwkv7_statepassing_clampw.cu"
    kernel_binding = rwkv_lm_checkout / "cuda/rwkv7_statepassing_pybind.cpp"
    rwkv_hf_commit_matches = bool(
        expected_rwkv_hf_sha == RWKV_HF_REVISION
        and rwkv_hf_distribution["requirement_satisfied"]
    )
    rwkv_lm_commit_matches = rwkv_lm_commit == expected_rwkv_lm_sha
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory": properties.total_memory,
                    "compute_capability": [properties.major, properties.minor],
                }
            )
    return {
        "schema_version": 1,
        "host": platform.node(),
        "python": {"version": sys.version, "executable": sys.executable},
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "cuda_devices": devices,
        "rwkv_hf": {
            "import_name": "rwkv7_hf",
            "module_file": str(rwkv_hf_module) if rwkv_hf_module else None,
            "distribution": rwkv_hf_distribution,
            "requirement": RWKV_HF_REQUIREMENT,
            "expected_commit": expected_rwkv_hf_sha,
            "commit_matches": rwkv_hf_commit_matches,
        },
        "rwkv_lm": {
            "checkout_commit": rwkv_lm_commit,
            "expected_commit": expected_rwkv_lm_sha,
            "commit_matches": rwkv_lm_commit_matches,
            "kernel_loader": str(kernel_loader),
            "kernel_loader_sha256": (
                file_sha256(kernel_loader) if kernel_loader.is_file() else None
            ),
            "kernel_source": str(kernel_source),
            "kernel_source_sha256": (
                file_sha256(kernel_source) if kernel_source.is_file() else None
            ),
            "kernel_binding": str(kernel_binding),
            "kernel_binding_sha256": (
                file_sha256(kernel_binding) if kernel_binding.is_file() else None
            ),
            "loader": "any2rwkv.kernel.load_rwkv_lm_kernel",
        },
        "transformers": {
            "distribution": transformers_distribution,
            "requirement": TRANSFORMERS_REQUIREMENT,
            "public_interface": transformers_public_interface,
            "requirement_satisfied": bool(
                transformers_distribution["requirement_satisfied"]
                and transformers_public_interface
            ),
            "loader": "AutoModelForCausalLM.from_pretrained",
            "trust_remote_code": False,
        },
        "passed": bool(
            torch.cuda.is_available()
            and rwkv_hf_module is not None
            and rwkv_hf_module.is_file()
            and rwkv_hf_commit_matches
            and rwkv_lm_commit_matches
            and kernel_loader.is_file()
            and kernel_source.is_file()
            and kernel_binding.is_file()
            and transformers_distribution["requirement_satisfied"]
            and transformers_public_interface
        ),
    }


def _validate_raw_data_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json_object(path, label="raw data manifest")
    output = Path(str(payload.get("output", ""))).expanduser().resolve()
    expected_sha256 = str(payload.get("output_sha256", ""))
    if payload.get("schema_version") != 1 or payload.get("status") != "complete":
        raise ValueError("raw data manifest must be schema_version=1 and complete")
    if not output.is_file():
        raise ValueError(f"raw data file is missing: {output}")
    if output.stat().st_size != int(payload.get("output_bytes", -1)):
        raise ValueError("raw data byte size differs from its manifest")
    actual_sha256 = file_sha256(output)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "raw data SHA-256 differs from its manifest: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    binding = payload.get("binding")
    if not isinstance(binding, dict):
        raise ContractError("raw data manifest has no immutable binding")
    return {
        "manifest": str(path.resolve()),
        "manifest_sha256": file_sha256(path),
        "path": str(output),
        "sha256": actual_sha256,
        "bytes": output.stat().st_size,
        "rows": int(payload.get("rows_written", -1)),
        "binding": binding,
    }


def _validate_prepared_data_manifest(
    path: Path,
    *,
    raw_data: dict[str, Any],
    plan: Any,
) -> dict[str, Any]:
    payload = _read_json_object(path, label="prepared data manifest")
    if payload.get("schema_version") != 1 or payload.get("status") != "prepared":
        raise ValueError(
            "prepared data manifest must use schema_version=1 and status=prepared"
        )
    packing = payload.get("packing")
    if not isinstance(packing, dict) or (
        int(packing.get("burn_in_tokens", -1)) != plan.burn_in_tokens
        or int(packing.get("supervised_tokens", -1)) != plan.supervised_tokens
    ):
        raise ValueError("prepared data packing differs from the distillation plan")
    raw_binding = raw_data["binding"]
    dataset = payload.get("dataset")
    tokenizer = payload.get("tokenizer")
    if not isinstance(dataset, dict) or (
        dataset.get("repository") != raw_binding.get("dataset_repository")
        or dataset.get("revision") != raw_binding.get("dataset_revision")
    ):
        raise ValueError("prepared data repository/revision differs from raw data")
    if not isinstance(tokenizer, dict) or (
        tokenizer.get("local_tree_sha256") != raw_binding.get("tokenizer_tree_sha256")
    ):
        raise ValueError(
            "prepared data tokenizer differs from raw data materialization"
        )
    splits = payload.get("splits")
    required_splits = {
        "distill_train",
        "validation",
        "ruler",
        "downstream",
        "smoke",
    }
    if not isinstance(splits, dict) or set(splits) != required_splits:
        raise ValueError("prepared data must bind all five disjoint splits")
    split_bindings: dict[str, Any] = {}
    for split in sorted(required_splits):
        entry = splits[split]
        if not isinstance(entry, dict):
            raise ContractError(f"prepared data split binding is malformed: {split}")
        split_path = Path(str(entry.get("path", "")))
        if not split_path.is_absolute():
            split_path = (path.parent / split_path).resolve()
        if not split_path.is_file():
            raise ValueError(f"prepared data split is missing: {split_path}")
        expected_sha256 = str(entry.get("sha256", ""))
        actual_sha256 = file_sha256(split_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"prepared data split SHA-256 differs: {split}")
        split_bindings[split] = {
            "path": str(split_path),
            "sha256": actual_sha256,
            "row_count": int(entry.get("row_count", -1)),
            "token_count": int(entry.get("token_count", -1)),
        }
    deduplication = payload.get("deduplication")
    if not isinstance(deduplication, dict):
        raise ContractError("prepared data has no deduplication binding")
    report_path = path.parent / str(deduplication.get("report_path", ""))
    if not report_path.is_file() or file_sha256(report_path) != deduplication.get(
        "report_sha256"
    ):
        raise ValueError("prepared data deduplication report differs")
    return {
        "manifest": str(path.resolve()),
        "manifest_sha256": file_sha256(path),
        "splits": split_bindings,
        "packing": packing,
        "deduplication_report_sha256": deduplication.get("report_sha256"),
    }


def collect_full_loop_preflight(
    product_root: Path,
    *,
    recipe_id: str,
    source_manifest_path: Path,
    source_path: Path,
    raw_data_manifest_path: Path,
    dataset_manifest_path: Path,
    training_config_path: Path,
    lighteval_config_path: Path,
    evalscope_config_path: Path,
    expected_rwkv_hf_sha: str,
    expected_rwkv_lm_sha: str,
    allow_proxy_layers: bool,
    precision: str,
) -> dict[str, Any]:
    """Collect every immutable input required before the real GPU loop."""
    from .distill_runner import read_distillation_plan
    from .recipes import resolve_recipe
    from .source import verify_source

    blockers: list[str] = []
    environment = collect_preflight(
        product_root,
        expected_rwkv_hf_sha=expected_rwkv_hf_sha,
        expected_rwkv_lm_sha=expected_rwkv_lm_sha,
    )
    if not environment["transformers"]["requirement_satisfied"]:
        blockers.append(
            "transformers distribution does not satisfy exact requirement "
            f"{TRANSFORMERS_REQUIREMENT}: "
            f"{environment['transformers']['distribution']}"
        )
    if not environment["rwkv_hf"]["commit_matches"]:
        blockers.append("rwkv-hf backend is missing or at the wrong commit")
    if not environment["rwkv_lm"]["commit_matches"]:
        blockers.append("rwkv-lm backend is uninitialized or at the wrong commit")
    if not environment["rwkv_lm"]["kernel_loader_sha256"]:
        blockers.append("rwkv-lm kernel loader is missing")

    source_result: dict[str, Any] | None = None
    try:
        source_manifest = _read_json_object(
            source_manifest_path, label="source manifest"
        )
        preferred_source = source_manifest.get("remote_read_only_path")
        source_result = {
            "manifest": str(source_manifest_path.resolve()),
            "manifest_sha256": file_sha256(source_manifest_path),
            "repository": source_manifest.get("repository"),
            "revision": source_manifest.get("revision"),
            "preferred_materialization_path": preferred_source,
            "verified": None,
            "inspection": None,
        }
        verified = verify_source(source_manifest_path, source_path)
        resolved = resolve_recipe(recipe_id)
        inspection = resolved.source.inspect_checkpoint(
            source_path,
            require_final_layout=not allow_proxy_layers,
        )
        resolved.recipe.validate_source(inspection)
        source_result.update(
            {
                "verified": verified,
                "inspection": {
                    "adapter_id": inspection.adapter_id,
                    "num_layers": inspection.num_layers,
                    "hidden_size": inspection.hidden_size,
                    "metadata": inspection.metadata,
                },
            }
        )
    except (OSError, ValueError, KeyError) as error:
        blockers.append(f"source: {error}")

    raw_data: dict[str, Any] | None = None
    try:
        raw_data = _validate_raw_data_manifest(raw_data_manifest_path)
    except (OSError, ValueError) as error:
        blockers.append(f"raw_data: {error}")

    plan = None
    training: dict[str, Any] | None = None
    try:
        plan = read_distillation_plan(training_config_path)
        if plan.activation_fit_rows <= 0:
            raise ValueError("real loop requires activation_fit_rows > 0")
        if plan.corrective_min_sweeps < 1:
            raise ValueError("real loop requires at least one corrective sweep")
        if plan.distributed_world_size <= 0:
            raise ValueError("distributed_world_size must be positive")
        training = {
            "config": str(training_config_path.resolve()),
            "config_sha256": file_sha256(training_config_path),
            "layer_major": {
                "world_size": plan.distributed_world_size,
                "micro_batch_size": plan.micro_batch_size,
                "layer_min_epochs": plan.layer_min_epochs,
                "layer_max_epochs": plan.layer_max_epochs,
            },
            "activation_fit": {
                "rows": plan.activation_fit_rows,
                "ridge": plan.activation_fit_ridge,
                "functional_steps": plan.activation_fit_functional_steps,
                "functional_learning_rate": (
                    plan.activation_fit_functional_learning_rate
                ),
            },
            "corrective": {
                "min_sweeps": plan.corrective_min_sweeps,
                "max_sweeps": plan.corrective_max_sweeps,
                "min_delta": plan.corrective_min_delta,
                "loss_weights": plan.global_loss_weights.__dict__,
            },
        }
    except (OSError, ValueError) as error:
        blockers.append(f"training_config: {error}")

    prepared_data: dict[str, Any] | None = None
    if raw_data is not None and plan is not None:
        try:
            prepared_data = _validate_prepared_data_manifest(
                dataset_manifest_path,
                raw_data=raw_data,
                plan=plan,
            )
        except (OSError, ValueError) as error:
            blockers.append(f"prepared_data: {error}")
    elif not dataset_manifest_path.is_file():
        blockers.append(f"prepared_data: manifest is missing: {dataset_manifest_path}")

    evaluation: dict[str, Any] = {
        "checkpoints": ("source", "zero-step", "distilled"),
        "precision": precision,
        "evaluators": {},
    }
    for name, path in (
        ("lighteval", lighteval_config_path),
        ("evalscope", evalscope_config_path),
    ):
        if not path.is_file():
            blockers.append(f"{name}: config is missing: {path}")
            evaluation["evaluators"][name] = None
        else:
            evaluation["evaluators"][name] = {
                "config": str(path.resolve()),
                "config_sha256": file_sha256(path),
            }

    resolved_assets = {
        "source": {
            "checkpoint": str(source_path.resolve()),
            "manifest": str(source_manifest_path.resolve()),
            "manifest_sha256": (
                file_sha256(source_manifest_path)
                if source_manifest_path.is_file()
                else None
            ),
        },
        "data": {
            "raw_manifest": str(raw_data_manifest_path.resolve()),
            "raw_manifest_sha256": (
                file_sha256(raw_data_manifest_path)
                if raw_data_manifest_path.is_file()
                else None
            ),
            "prepared_manifest": str(dataset_manifest_path.resolve()),
            "prepared_manifest_sha256": (
                file_sha256(dataset_manifest_path)
                if dataset_manifest_path.is_file()
                else None
            ),
        },
        "training": {
            "config": str(training_config_path.resolve()),
            "config_sha256": (
                file_sha256(training_config_path)
                if training_config_path.is_file()
                else None
            ),
        },
        "evaluation": {
            "lighteval_config": str(lighteval_config_path.resolve()),
            "lighteval_config_sha256": (
                file_sha256(lighteval_config_path)
                if lighteval_config_path.is_file()
                else None
            ),
            "evalscope_config": str(evalscope_config_path.resolve()),
            "evalscope_config_sha256": (
                file_sha256(evalscope_config_path)
                if evalscope_config_path.is_file()
                else None
            ),
        },
        "backends": {
            "rwkv_hf_commit": expected_rwkv_hf_sha,
            "rwkv_lm_commit": expected_rwkv_lm_sha,
        },
    }
    return {
        "schema_version": 1,
        "status": "ready" if not blockers else "blocked",
        "recipe": recipe_id,
        "precision": precision,
        "environment": environment,
        "source": source_result,
        "raw_data": raw_data,
        "prepared_data": prepared_data,
        "training": training,
        "evaluation": evaluation,
        "resolved_assets": resolved_assets,
        "blockers": blockers,
        "passed": not blockers,
    }
