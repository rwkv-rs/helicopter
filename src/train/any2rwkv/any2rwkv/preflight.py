from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from .artifacts import file_sha256, sha256_json
from .errors import ContractError

TRANSFORMERS_REVISION = "2696927df9363b5fa175076bb827ba4da2c4e581"
TRANSFORMERS_SOURCE_URL = "https://github.com/rwkv-rs/transformers-rwkv.git"
TRANSFORMERS_REQUIREMENT = (
    f"transformers @ git+{TRANSFORMERS_SOURCE_URL}@{TRANSFORMERS_REVISION}"
)
FLA_RWKV7_SOURCE_URL = "https://github.com/rwkv-rs/fla-rwkv.git"
FLA_RWKV7_REVISION = "a4a8aa98df6ec5322f194a80ec57363dd045adfc"
FLASH_RWKV_SOURCE_URL = "https://github.com/rwkv-rs/FlashRWKV.git"
FLASH_RWKV_REVISION = "866aafd2eed146b0eda1ce03444009ae030f89e3"


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


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must contain a JSON object: {path}")
    return payload


def _require_module_ownership(
    distribution_name: str,
    module_name: str,
) -> dict[str, Any]:
    """Require the imported package to be owned by the pinned distribution."""
    try:
        distribution = importlib.metadata.distribution(distribution_name)
        module = importlib.import_module(module_name)
        module_file = Path(module.__file__).resolve()
        expected_file = Path(
            distribution.locate_file(module_name.replace(".", "/") + "/__init__.py")
        ).resolve()
    except (AttributeError, ImportError, TypeError) as error:
        raise ContractError(
            f"cannot establish {module_name} module ownership: {error}"
        ) from error
    if module_file != expected_file:
        raise ContractError(
            f"{module_name} module ownership mismatch: "
            f"distribution={expected_file} imported={module_file}"
        )
    return {
        "distribution": distribution_name,
        "module": module_name,
        "verified": True,
    }


def require_rwkv7_runtime() -> dict[str, Any]:
    """Fail closed unless the independent model and recurrent operator chain is exact."""
    transformers_distribution = _distribution_binding(
        "transformers",
        expected_url=TRANSFORMERS_SOURCE_URL,
        expected_revision=TRANSFORMERS_REVISION,
    )
    if not transformers_distribution["requirement_satisfied"]:
        raise ContractError(
            "Transformers does not satisfy the exact Any-to-RWKV runtime "
            f"requirement: {TRANSFORMERS_REQUIREMENT}"
        )
    fla_distribution = _distribution_binding(
        "flash-linear-attention",
        expected_url=FLA_RWKV7_SOURCE_URL,
        expected_revision=FLA_RWKV7_REVISION,
    )
    flash_distribution = _distribution_binding(
        "flash-rwkv",
        expected_url=FLASH_RWKV_SOURCE_URL,
        expected_revision=FLASH_RWKV_REVISION,
    )
    if not fla_distribution["requirement_satisfied"]:
        raise ContractError(
            "fla-rwkv does not satisfy the exact recurrent operator pin"
        )
    if not flash_distribution["requirement_satisfied"]:
        raise ContractError("FlashRWKV does not satisfy the exact recurrent kernel pin")
    transformers_ownership = _require_module_ownership("transformers", "transformers")
    fla_ownership = _require_module_ownership("flash-linear-attention", "fla.ops.rwkv7")
    try:
        rwkv7 = importlib.import_module("fla.ops.rwkv7")
        recurrent = rwkv7.recurrent_rwkv7
        provider = rwkv7.get_last_rwkv7_provider
        validate_runtime = rwkv7.validate_flash_rwkv_installation
    except (AttributeError, ImportError) as error:
        raise ContractError(
            "fla-rwkv does not expose the public recurrent_rwkv7 contract"
        ) from error
    if not all(callable(value) for value in (recurrent, provider, validate_runtime)):
        raise ContractError(
            "fla-rwkv recurrent operator/provider/provenance interface is incomplete"
        )
    required_parameters = {
        "r",
        "w",
        "k",
        "v",
        "a",
        "b",
        "scale",
        "initial_state",
        "output_final_state",
        "cu_seqlens",
        "state_indices",
        "mode",
    }
    try:
        parameters = inspect.signature(recurrent).parameters
    except (TypeError, ValueError) as error:
        raise ContractError(
            "fla-rwkv recurrent_rwkv7 signature is not inspectable"
        ) from error
    missing = sorted(required_parameters - parameters.keys())
    if missing:
        raise ContractError(
            f"fla-rwkv recurrent_rwkv7 signature is incompatible: missing={missing}"
        )
    try:
        flash_provenance = validate_runtime()
    except (RuntimeError, TypeError, ValueError) as error:
        raise ContractError(
            f"RWKV-7 recurrent operator provenance rejected: {error}"
        ) from error
    operators = {
        "repository": FLA_RWKV7_SOURCE_URL,
        "revision": FLA_RWKV7_REVISION,
        "flash_rwkv_repository": getattr(
            flash_provenance,
            "repository",
            flash_provenance.get("repository")
            if isinstance(flash_provenance, dict)
            else None,
        ),
        "flash_rwkv_revision": getattr(
            flash_provenance,
            "revision",
            flash_provenance.get("revision")
            if isinstance(flash_provenance, dict)
            else None,
        ),
        "operation": "fla.ops.rwkv7.recurrent_rwkv7",
    }
    expected_flash = {
        "flash_rwkv_repository": FLASH_RWKV_SOURCE_URL,
        "flash_rwkv_revision": FLASH_RWKV_REVISION,
    }
    mismatches = {
        name: {"expected": expected, "actual": operators[name]}
        for name, expected in expected_flash.items()
        if operators[name] != expected
    }
    if mismatches:
        raise ContractError(
            f"RWKV-7 operator manifest differs from the self-owned chain: {mismatches}"
        )
    return {
        "schema_version": 1,
        "transformers": {
            "distribution": "transformers",
            "version": transformers_distribution["version"],
            "repository": TRANSFORMERS_SOURCE_URL,
            "revision": TRANSFORMERS_REVISION,
            "requirement": TRANSFORMERS_REQUIREMENT,
            "module_ownership": transformers_ownership,
        },
        "operators": {
            **operators,
            "fla_distribution": fla_distribution,
            "flash_distribution": flash_distribution,
            "fla_module_ownership": fla_ownership,
        },
    }


def runtime_binding(manifest: dict[str, Any]) -> dict[str, Any]:
    return {"manifest": manifest, "sha256": sha256_json(manifest)}


def require_matching_rwkv7_runtime(metadata: dict[str, Any]) -> dict[str, Any]:
    current = runtime_binding(require_rwkv7_runtime())
    if metadata.get("runtime") != current:
        raise ContractError(
            "current RWKV-7 runtime manifest differs from initialized run metadata"
        )
    return current


def collect_preflight() -> dict[str, Any]:
    transformers_distribution = _distribution_binding(
        "transformers",
        expected_url=TRANSFORMERS_SOURCE_URL,
        expected_revision=TRANSFORMERS_REVISION,
    )
    transformers_public_interface = False
    runtime_manifest: dict[str, Any] | None = None
    runtime_provenance: dict[str, str] | None = None
    runtime_provenance_error: str | None = None
    try:
        runtime_manifest = require_rwkv7_runtime()
        transformers_public_interface = True
        runtime_provenance = runtime_manifest["operators"]
    except (
        AttributeError,
        ContractError,
        ImportError,
        RuntimeError,
        TypeError,
    ) as error:
        runtime_provenance = None
        runtime_provenance_error = f"{type(error).__name__}: {error}"
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
        "transformers": {
            "distribution": transformers_distribution,
            "requirement": TRANSFORMERS_REQUIREMENT,
            "public_interface": transformers_public_interface,
            "runtime_provenance": runtime_provenance,
            "runtime_provenance_error": runtime_provenance_error,
            "runtime_manifest": runtime_manifest,
            "requirement_satisfied": bool(
                transformers_distribution["requirement_satisfied"]
                and transformers_public_interface
                and runtime_provenance is not None
            ),
            "loader": "transformers.models.rwkv7.validate_rwkv7_runtime_provenance",
            "trust_remote_code": False,
        },
        "passed": bool(
            torch.cuda.is_available()
            and transformers_distribution["requirement_satisfied"]
            and transformers_public_interface
            and runtime_provenance is not None
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
    *,
    recipe_id: str,
    source_manifest_path: Path,
    source_path: Path,
    raw_data_manifest_path: Path,
    dataset_manifest_path: Path,
    training_config_path: Path,
    lighteval_config_path: Path,
    evalscope_config_path: Path,
    allow_proxy_layers: bool,
    precision: str,
) -> dict[str, Any]:
    """Collect every immutable input required before the real GPU loop."""
    from .distill_runner import read_distillation_plan
    from .recipes import resolve_recipe
    from .source import verify_source

    blockers: list[str] = []
    environment = collect_preflight()
    if not environment["transformers"]["requirement_satisfied"]:
        blockers.append(
            "transformers distribution does not satisfy exact requirement "
            f"{TRANSFORMERS_REQUIREMENT}: "
            f"{environment['transformers']}"
        )
    if not environment["torch"]["cuda_available"]:
        blockers.append(
            "CUDA is unavailable for the Any-to-RWKV architecture conversion"
        )

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
        "runtime": environment["transformers"],
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
