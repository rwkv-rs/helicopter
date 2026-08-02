from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from ...checkpoint import CheckpointManifest, read_checkpoint
from ...contract import validate_source_config
from ...core import ArchitectureInspection
from ...errors import ContractError
from .geometry_contract import validate_qwen35_geometry


class Qwen35SourceAdapter:
    adapter_id = "qwen35"

    def inspect_checkpoint(
        self,
        checkpoint_dir: Path,
        *,
        require_final_layout: bool,
    ) -> ArchitectureInspection:
        config_path = checkpoint_dir.resolve() / "config.json"
        if not config_path.is_file():
            raise ContractError(f"HF config.json not found: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, Mapping):
            raise ContractError(f"HF config.json must contain an object: {config_path}")
        contract = validate_source_config(
            config,
            require_final_layers=require_final_layout,
            text_backbone_only=True,
        )
        geometry = validate_qwen35_geometry(config)
        return ArchitectureInspection(
            adapter_id=self.adapter_id,
            num_layers=contract.num_hidden_layers,
            hidden_size=contract.hidden_size,
            metadata={
                "model_type": contract.model_type,
                "architecture": contract.architecture,
                "layer_types": contract.layer_types,
                "has_moe": contract.has_moe,
                "mtp_num_hidden_layers": contract.mtp_num_hidden_layers,
                "extracted_text_backbone": contract.extracted_text_backbone,
                "model_id": geometry.model_id,
                "gdn_num_heads": geometry.gdn_num_heads,
                "gdn_head_size": geometry.gdn_head_size,
                "gqa_num_query_heads": geometry.gqa_num_query_heads,
                "gqa_head_size": geometry.gqa_head_size,
                "gqa_state_mappings": tuple(
                    {
                        "query_head": entry.query_head,
                        "matrix_state": entry.matrix_state,
                        "prefix_value_mean_state": entry.prefix_value_mean_state,
                    }
                    for entry in geometry.gqa_state_mappings
                ),
            },
        )

    def load_checkpoint(
        self,
        checkpoint_dir: Path,
        *,
        require_final_layout: bool,
    ) -> CheckpointManifest:
        return read_checkpoint(
            checkpoint_dir,
            require_final_layers=require_final_layout,
            text_backbone_only=True,
        )
