from __future__ import annotations

import json

from ...artifacts import write_json
from ...core import (
    ArchitectureInspection,
    DistillationExecutionRequest,
    GQAZeroStepValidationRequest,
    PerformanceProfileCacheRequest,
)
from ...errors import ContractError
from ...mapping import is_locally_trainable

_OBSERVABLE_COMPARISONS = (
    "logits",
    "block_output",
    "token_kl",
    "shifted_ce",
    "ppl",
    "long_context_tail",
    "input_gradients",
    "parameter_gradients",
)


def _publish_evaluation_policy(request: DistillationExecutionRequest) -> None:
    """Bind the corrective sweep to disjoint development data and observables."""
    calibration_ids = {
        sample_id
        for row in request.train_row_source_sample_ids
        for sample_id in row
    }
    development_ids = {
        sample_id
        for row in request.validation_row_source_sample_ids
        for sample_id in row
    }
    overlap = sorted(calibration_ids & development_ids)
    if overlap:
        raise ContractError(
            "calibration and development source samples must be disjoint: "
            + ", ".join(overlap[:8])
        )
    write_json(
        request.run_dir / "evaluation-policy.json",
        {
            "schema_version": 1,
            "calibration": {
                "role": "optimizer updates",
                "source_sample_count": len(calibration_ids),
            },
            "development": {
                "role": "candidate selection and rollback",
                "source_sample_count": len(development_ids),
            },
            "final": {
                "role": "held out from conversion and corrective selection",
                "consumed": False,
            },
            "allowed_cross_architecture_comparisons": list(
                _OBSERVABLE_COMPARISONS
            ),
            "forbidden_cross_architecture_comparisons": ["recurrent_state_mse"],
            "candidate_policy": {
                "frozen_baseline": "global-snapshots/pre-sweep",
                "acceptance_metric": "development.token_kl",
                "degradation_action": "restore-frozen-best-candidate",
            },
        },
    )


class Qwen35ToRWKV7Recipe:
    recipe_id = "qwen35_to_rwkv7"
    source_adapter_id = "qwen35"
    target_adapter_id = "rwkv7"

    def validate_source(self, inspection: ArchitectureInspection) -> None:
        if inspection.adapter_id != self.source_adapter_id:
            raise ContractError(
                f"recipe {self.recipe_id} requires source adapter "
                f"{self.source_adapter_id}, found {inspection.adapter_id}"
            )
        layer_types = set(inspection.metadata.get("layer_types", ()))
        if layer_types - {"linear_attention", "full_attention"}:
            raise ContractError(
                f"recipe {self.recipe_id} does not support layer types "
                f"{sorted(layer_types)}"
            )

    def run_layerwise_distillation(
        self, request: DistillationExecutionRequest
    ) -> dict[str, object]:
        from .global_corrective_runner import run_global_corrective
        from .layer_major_runner import run_suffix_free_layer_major

        _publish_evaluation_policy(request)
        layer_count = request.source_checkpoint.contract.num_hidden_layers
        local = run_suffix_free_layer_major(
            source_manifest=request.source_checkpoint,
            run_dir=request.run_dir,
            zero_step_dir=request.zero_step_dir,
            token_rows=request.token_rows,
            validation_rows=request.validation_rows,
            train_row_source_sample_ids=(
                request.train_row_source_sample_ids
            ),
            validation_row_source_sample_ids=(
                request.validation_row_source_sample_ids
            ),
            plan=request.plan,
            initial_trainable=self._initial_trainable_names(
                request.run_dir, layer_count
            ),
            training_config=request.training_config,
            dataset_manifest=request.dataset_manifest,
            resume=request.resume,
            progress_callback=request.progress_callback,
        )
        if local.get("status") == "exploratory-layer-calibration-complete":
            return local
        if local.get("status") != "layerwise-local-complete":
            raise ContractError("layerwise local stage did not produce a complete checkpoint")
        return run_global_corrective(
            source_manifest=request.source_checkpoint,
            run_dir=request.run_dir,
            zero_step_dir=request.zero_step_dir,
            token_rows=request.token_rows,
            validation_rows=request.validation_rows,
            plan=request.plan,
            training_config=request.training_config,
            dataset_manifest=request.dataset_manifest,
            progress_callback=request.progress_callback,
        )

    def run_corrective_distillation(
        self, request: DistillationExecutionRequest
    ) -> dict[str, object]:
        from .global_corrective_runner import run_global_corrective

        if getattr(request.plan, "exploratory_layer_limit", None) is not None:
            raise ContractError("corrective distillation forbids exploratory_layer_limit")
        return run_global_corrective(
            source_manifest=request.source_checkpoint,
            run_dir=request.run_dir,
            zero_step_dir=request.zero_step_dir,
            token_rows=request.token_rows,
            validation_rows=request.validation_rows,
            plan=request.plan,
            training_config=request.training_config,
            dataset_manifest=request.dataset_manifest,
            progress_callback=request.progress_callback,
        )

    def prepare_performance_profile_caches(
        self, request: PerformanceProfileCacheRequest
    ) -> dict[str, object]:
        from .layer_major_runner import prepare_performance_profile_caches

        layer_count = request.source_checkpoint.contract.num_hidden_layers
        return prepare_performance_profile_caches(
            source_manifest=request.source_checkpoint,
            run_dir=request.run_dir,
            zero_step_dir=request.zero_step_dir,
            token_rows=request.token_rows,
            validation_rows=request.validation_rows,
            plan=request.plan,
            initial_trainable=self._initial_trainable_names(
                request.run_dir, layer_count
            ),
            training_config=request.training_config,
            dataset_manifest=request.dataset_manifest,
            row_selection=request.row_selection,
        )

    def run_gqa_zero_step_validation(
        self, request: GQAZeroStepValidationRequest
    ) -> dict[str, object]:
        from .layer_major_runner import run_gqa_zero_step_validation

        layer_count = request.source_checkpoint.contract.num_hidden_layers
        return run_gqa_zero_step_validation(
            source_manifest=request.source_checkpoint,
            run_dir=request.run_dir,
            evidence_dir=request.evidence_dir,
            zero_step_dir=request.zero_step_dir,
            plan=request.plan,
            initial_trainable=self._initial_trainable_names(
                request.run_dir, layer_count
            ),
            training_config=request.training_config,
            dataset_manifest=request.dataset_manifest,
            layer_index=request.layer_index,
            train_row_source_sample_ids=(
                request.train_row_source_sample_ids
            ),
            validation_row_source_sample_ids=(
                request.validation_row_source_sample_ids
            ),
        )

    @staticmethod
    def _initial_trainable_names(run_dir, layer_count: int) -> list[set[str]]:
        payload = json.loads(
            (run_dir / "warm-start-plan.json").read_text(encoding="utf-8")
        )
        rows: list[set[str]] = [set() for _ in range(layer_count)]
        for entry in payload.get("entries", []):
            if not is_locally_trainable(entry):
                continue
            target = str(entry.get("target", ""))
            parts = target.split(".attn.", 1)
            if len(parts) != 2:
                continue
            layer_parts = parts[0].split(".")
            try:
                layer = int(layer_parts[layer_parts.index("layers") + 1])
            except (ValueError, IndexError):
                continue
            if not 0 <= layer < layer_count:
                raise ContractError(
                    f"warm-start trainable target has invalid layer: {target}"
                )
            rows[layer].add(parts[1])
        if any(not names for names in rows):
            missing = [index for index, names in enumerate(rows) if not names]
            raise ContractError(
                "warm-start plan leaves no fitted/initialized parameters for "
                f"layers: {missing}"
            )
        return rows
