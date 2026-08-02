from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts import checkpoint_sha256, file_sha256, write_json
from ..distill import MIGRATION_BASELINE_STAGES
from ..errors import ContractError


_BINDING_KEYS = {
    "teacher_sha256",
    "tokenizer_sha256",
    "dataset_sha256",
    "split",
    "seed",
    "burn_in_tokens",
    "precision",
    "token_budget",
    "sample_ids_sha256",
}
_GQA_STAGES = (
    "gqa_exact_hazard_oracle",
    "gqa_bounded_hazard",
    "gqa_observable_compressed",
    "gqa_native_projected_zero_step",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _resolve(owner: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (owner.parent / path).resolve()


def _artifact_sha256(path: Path, kind: str) -> str:
    if kind == "file":
        return file_sha256(path)
    if kind == "checkpoint":
        return checkpoint_sha256(path)
    raise ContractError(f"unknown migration baseline artifact kind: {kind}")


def _read_reference(
    reference: object, *, owner: Path, label: str
) -> tuple[Path, str, str]:
    if not isinstance(reference, Mapping):
        raise ContractError(f"migration baseline {label} reference is missing")
    path = _resolve(owner, reference.get("path"))
    kind = str(reference.get("kind", ""))
    expected = str(reference.get("sha256", ""))
    if not _SHA256.fullmatch(expected):
        raise ContractError(f"migration baseline {label} SHA-256 is invalid")
    if (kind == "file" and not path.is_file()) or (
        kind == "checkpoint" and not path.is_dir()
    ):
        raise ContractError(f"migration baseline {label} artifact is missing: {path}")
    actual = _artifact_sha256(path, kind)
    if actual != expected:
        raise ContractError(
            f"migration baseline {label} SHA-256 mismatch: "
            f"expected={expected} actual={actual}"
        )
    return path, kind, actual


def _read_sample_rows(
    path: Path, *, expected_sha256: str
) -> tuple[tuple[dict[str, Any], ...], tuple[tuple[str, int], ...]]:
    if file_sha256(path) != expected_sha256:
        raise ContractError("migration baseline sample metrics SHA-256 mismatch")
    rows: list[dict[str, Any]] = []
    identities: list[tuple[str, int]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ContractError(
                    f"migration baseline sample metrics line {line_number} is invalid"
                ) from error
            sample_id = str(row.get("sample_id", ""))
            token_count = row.get("token_count")
            token_kl_sum = row.get("token_kl_sum")
            if (
                not sample_id
                or sample_id in seen
                or type(token_count) is not int
                or token_count <= 0
                or isinstance(token_kl_sum, bool)
                or not isinstance(token_kl_sum, (int, float))
                or not math.isfinite(float(token_kl_sum))
                or float(token_kl_sum) < -1e-8
            ):
                raise ContractError(
                    f"migration baseline sample row is invalid: line={line_number}"
                )
            zero_step_nmse = row.get("end_to_end_zero_step_nmse")
            if zero_step_nmse is not None and (
                isinstance(zero_step_nmse, bool)
                or not isinstance(zero_step_nmse, (int, float))
                or not math.isfinite(float(zero_step_nmse))
                or float(zero_step_nmse) < 0
            ):
                raise ContractError(
                    "migration baseline sample zero-step NMSE is invalid"
                )
            incremental = row.get("incremental_nmse")
            if incremental is not None and (
                not isinstance(incremental, Mapping)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0
                    for value in incremental.values()
                )
            ):
                raise ContractError(
                    "migration baseline sample incremental NMSE is invalid"
                )
            seen.add(sample_id)
            rows.append(dict(row))
            identities.append((sample_id, token_count))
    if not rows:
        raise ContractError("migration baseline sample metrics are empty")
    return tuple(rows), tuple(identities)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _relative_reference(
    path: Path, *, owner: Path, kind: str
) -> dict[str, str]:
    resolved = path.resolve()
    try:
        rendered = str(resolved.relative_to(owner.parent.resolve()))
    except ValueError:
        rendered = str(resolved)
    return {
        "path": rendered,
        "kind": kind,
        "sha256": _artifact_sha256(resolved, kind),
    }


def _validate_evidence_references(
    evidence: Mapping[str, object], *, owner: Path
) -> None:
    artifacts = evidence.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ContractError("migration baseline evidence artifacts must be an object")
    for label, reference in artifacts.items():
        _read_reference(reference, owner=owner, label=f"evidence {label}")


def write_migration_baseline_stage(
    output: Path,
    *,
    stage: str,
    protocol: Mapping[str, object],
    sample_rows: Sequence[Mapping[str, object]],
    candidate_path: Path,
    candidate_kind: str,
    evidence: Mapping[str, object],
) -> dict[str, Any]:
    """Publish one raw, hash-bound stage without accepting aggregate floats."""
    if stage not in MIGRATION_BASELINE_STAGES and not re.fullmatch(
        r"corrective_sweep_[0-9]+", stage
    ):
        raise ContractError(f"unknown migration baseline stage: {stage}")
    candidate = _relative_reference(
        candidate_path, owner=output, kind=candidate_kind
    )
    _validate_evidence_references(evidence, owner=output)
    temporary_sample_path = output.with_name(
        f".{output.stem}.{os.getpid()}.samples.jsonl"
    )
    try:
        _write_jsonl_atomic(temporary_sample_path, sample_rows)
        sample_sha256 = file_sha256(temporary_sample_path)
        parsed_rows, identities = _read_sample_rows(
            temporary_sample_path,
            expected_sha256=sample_sha256,
        )
        binding = _validate_binding(
            {
                **dict(protocol),
                "token_budget": sum(count for _, count in identities),
                "sample_ids_sha256": _sha256_json(
                    [sample_id for sample_id, _ in identities]
                ),
            }
        )
        content_sample_path = output.with_name(
            f"{output.stem}.samples.{sample_sha256}.jsonl"
        )
        os.replace(temporary_sample_path, content_sample_path)
    finally:
        temporary_sample_path.unlink(missing_ok=True)
    payload = {
        "schema_version": 1,
        "stage": stage,
        "binding": binding,
        "candidate": candidate,
        "sample_metrics": _relative_reference(
            content_sample_path, owner=output, kind="file"
        ),
        "evidence": dict(evidence),
        "row_count": len(parsed_rows),
    }
    write_json(output, payload)
    read_migration_baseline_stage(output)
    return payload


def _validate_binding(binding: object) -> dict[str, object]:
    if not isinstance(binding, Mapping) or set(binding) != _BINDING_KEYS:
        raise ContractError(
            "migration baseline binding must contain exactly "
            f"{sorted(_BINDING_KEYS)}"
        )
    value = dict(binding)
    for name in (
        "teacher_sha256",
        "tokenizer_sha256",
        "dataset_sha256",
        "sample_ids_sha256",
    ):
        if not _SHA256.fullmatch(str(value[name])):
            raise ContractError(f"migration baseline binding {name} is invalid")
    if (
        value["split"] != "validation"
        or type(value["seed"]) is not int
        or type(value["burn_in_tokens"]) is not int
        or int(value["burn_in_tokens"]) < 0
        or type(value["token_budget"]) is not int
        or int(value["token_budget"]) <= 0
        or not isinstance(value["precision"], str)
        or not value["precision"]
    ):
        raise ContractError("migration baseline shared protocol is invalid")
    return value


def read_migration_baseline_stage(
    path: Path,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], tuple[tuple[str, int], ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ContractError("migration baseline stage schema version is invalid")
    stage = str(payload.get("stage", ""))
    if stage not in MIGRATION_BASELINE_STAGES and not re.fullmatch(
        r"corrective_sweep_[0-9]+", stage
    ):
        raise ContractError(f"unknown migration baseline stage: {stage}")
    binding = _validate_binding(payload.get("binding"))
    candidate_path, candidate_kind, candidate_sha = _read_reference(
        payload.get("candidate"), owner=path, label="candidate"
    )
    sample_reference = payload.get("sample_metrics")
    sample_path, sample_kind, sample_sha = _read_reference(
        sample_reference, owner=path, label="sample metrics"
    )
    if sample_kind != "file":
        raise ContractError("migration baseline sample metrics must be a file")
    rows, identities = _read_sample_rows(sample_path, expected_sha256=sample_sha)
    if payload.get("row_count") != len(rows):
        raise ContractError("migration baseline stage row count differs")
    if sum(token_count for _, token_count in identities) != binding["token_budget"]:
        raise ContractError("migration baseline sample token budget differs")
    sample_ids_sha = _sha256_json([sample_id for sample_id, _ in identities])
    if sample_ids_sha != binding["sample_ids_sha256"]:
        raise ContractError("migration baseline sample ID digest differs")
    evidence = payload.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ContractError("migration baseline stage evidence is missing")
    _validate_evidence_references(evidence, owner=path)
    payload["binding"] = binding
    payload["candidate"] = {
        "path": str(candidate_path),
        "kind": candidate_kind,
        "sha256": candidate_sha,
    }
    payload["sample_metrics"] = {
        "path": str(sample_path),
        "kind": "file",
        "sha256": sample_sha,
    }
    payload["manifest_path"] = str(path.resolve())
    return payload, rows, identities


def build_migration_baseline_matrix(
    stage_manifests: Sequence[Path],
    *,
    student_checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    if not stage_manifests:
        raise ContractError("migration baseline matrix has no stage manifests")
    student_sha256 = checkpoint_sha256(student_checkpoint)
    stages: dict[str, tuple[dict[str, Any], tuple[dict[str, Any], ...]]] = {}
    manifest_sha256: dict[str, str] = {}
    shared_binding: dict[str, object] | None = None
    shared_identities: tuple[tuple[str, int], ...] | None = None
    for manifest_path in stage_manifests:
        payload, rows, identities = read_migration_baseline_stage(manifest_path)
        stage = str(payload["stage"])
        if stage in stages:
            raise ContractError(f"duplicate migration baseline stage: {stage}")
        binding = payload["binding"]
        if shared_binding is None:
            shared_binding = binding
            shared_identities = identities
        elif binding != shared_binding or identities != shared_identities:
            raise ContractError(
                "migration baseline stages do not share the same sample/token protocol"
            )
        stages[stage] = (payload, rows)
        manifest_sha256[stage] = file_sha256(manifest_path)
    missing = sorted(set(MIGRATION_BASELINE_STAGES) - stages.keys())
    if missing:
        raise ContractError(f"migration baseline matrix is incomplete: {missing}")
    corrective_indices = sorted(
        int(name.rsplit("_", 1)[1])
        for name in stages
        if name.startswith("corrective_sweep_")
    )
    if corrective_indices != list(range(max(corrective_indices, default=-1) + 1)):
        raise ContractError("migration baseline corrective sweep stages are not contiguous")
    assert shared_binding is not None
    assert shared_identities is not None

    gqa_trace_sha: str | None = None
    rows_by_stage: dict[str, dict[str, object]] = {}
    for stage, (payload, sample_rows) in stages.items():
        evidence = payload["evidence"]
        if stage in _GQA_STAGES:
            trace_sha = str(evidence.get("trace_sha256", ""))
            if not _SHA256.fullmatch(trace_sha):
                raise ContractError(f"{stage} lacks a frozen trace SHA-256")
            if gqa_trace_sha is None:
                gqa_trace_sha = trace_sha
            elif trace_sha != gqa_trace_sha:
                raise ContractError("GQA baseline stages use different frozen traces")
        if stage == "gqa_exact_hazard_oracle" and evidence.get(
            "counterfactual"
        ) != "oracle-signal":
            raise ContractError(
                "GQA exact baseline must come from the frozen oracle signal"
            )
        if stage == "activation_fitted":
            if evidence.get("solver_invoked") is not True:
                raise ContractError(
                    "activation-fitted baseline did not invoke a solver"
                )
            artifacts = evidence.get("artifacts")
            if not isinstance(artifacts, Mapping):
                raise ContractError("activation-fitted baseline has no artifacts")
            fit_path, fit_kind, fit_sha = _read_reference(
                artifacts.get("fit_report"),
                owner=Path(payload["manifest_path"]),
                label="fit report",
            )
            _, materialization_kind, materialization_sha = _read_reference(
                artifacts.get("materialization"),
                owner=Path(payload["manifest_path"]),
                label="materialization",
            )
            if fit_kind != "file" or materialization_kind != "checkpoint":
                raise ContractError(
                    "activation-fitted evidence requires a report file and checkpoint"
                )
            fit_payload = json.loads(fit_path.read_text(encoding="utf-8"))
            if (
                fit_payload.get("status") != "accepted"
                or not _SHA256.fullmatch(
                    str(fit_payload.get("selected_module_state_sha256", ""))
                )
            ):
                raise ContractError(
                    "activation-fitted report is not an accepted installed generation"
                )
        else:
            fit_sha = None
            materialization_sha = None
        if stage == "fully_recurrent" or stage.startswith("corrective_sweep_"):
            artifacts = evidence.get("artifacts")
            if not isinstance(artifacts, Mapping):
                raise ContractError(f"{stage} lacks a training curve artifact")
            _, curve_kind, _ = _read_reference(
                artifacts.get("training_curve"),
                owner=Path(payload["manifest_path"]),
                label=f"{stage} training curve",
            )
            if curve_kind != "file":
                raise ContractError(f"{stage} training curve must be a file")

        token_budget = sum(int(row["token_count"]) for row in sample_rows)
        token_kl_sum = sum(float(row["token_kl_sum"]) for row in sample_rows)
        zero_step_values = []
        incremental_keys: tuple[str, ...] | None = None
        incremental_sums: dict[str, float] = {}
        for row in sample_rows:
            if row.get("end_to_end_zero_step_nmse") is None:
                raise ContractError(
                    f"{stage} lacks per-sample end-to-end zero-step NMSE"
                )
            zero_step_values.append(
                (
                    float(row["end_to_end_zero_step_nmse"]),
                    int(row["token_count"]),
                )
            )
            incremental = row.get("incremental_nmse")
            if not isinstance(incremental, Mapping) or not incremental:
                raise ContractError(
                    f"{stage} lacks per-sample incremental zero-step NMSE"
                )
            current_keys = tuple(sorted(str(key) for key in incremental))
            if incremental_keys is None:
                incremental_keys = current_keys
                incremental_sums = {key: 0.0 for key in current_keys}
            elif current_keys != incremental_keys:
                raise ContractError(
                    f"{stage} uses inconsistent incremental NMSE components"
                )
            for key in current_keys:
                incremental_sums[key] += (
                    float(incremental[key]) * int(row["token_count"])
                )
        stage_row: dict[str, object] = {
            "mean_token_kl": max(0.0, token_kl_sum) / token_budget,
            "token_budget": token_budget,
            "candidate_sha256": payload["candidate"]["sha256"],
            "sample_metrics_sha256": payload["sample_metrics"]["sha256"],
            "sample_ids_sha256": shared_binding["sample_ids_sha256"],
            "trace_sha256": evidence.get("trace_sha256"),
            "end_to_end_zero_step_nmse": sum(
                value * count for value, count in zero_step_values
            )
            / token_budget,
            "incremental_zero_step_nmse": {
                key: value / token_budget
                for key, value in sorted(incremental_sums.items())
            },
        }
        if fit_sha is not None:
            stage_row.update(
                {
                    "solver_invoked": True,
                    "fit_report_sha256": fit_sha,
                    "materialization_sha256": materialization_sha,
                }
            )
        rows_by_stage[stage] = stage_row
    if float(rows_by_stage["teacher"]["mean_token_kl"]) > 1e-12:
        raise ContractError("teacher migration baseline token KL is not zero")

    matrix_binding = {
        "student_sha256": student_sha256,
        **shared_binding,
    }
    stage_references = {
        str(
            json.loads(path.read_text(encoding="utf-8"))["stage"]
        ): _relative_reference(path, owner=output, kind="file")
        for path in stage_manifests
    }
    result = {
        "schema_version": 3,
        "student_sha256": student_sha256,
        "binding": matrix_binding,
        "baselines": rows_by_stage,
        "stage_manifests": stage_references,
        "stage_manifest_sha256": manifest_sha256,
    }
    write_json(output, result)
    return result
