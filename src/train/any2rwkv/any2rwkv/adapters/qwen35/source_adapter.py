from __future__ import annotations

from pathlib import Path

from ...checkpoint import CheckpointManifest, read_checkpoint
from ...core import ArchitectureInspection


class Qwen35SourceAdapter:
    adapter_id = "qwen35"

    def inspect_checkpoint(
        self,
        checkpoint_dir: Path,
        *,
        require_final_layout: bool,
    ) -> ArchitectureInspection:
        checkpoint = self.load_checkpoint(
            checkpoint_dir,
            require_final_layout=require_final_layout,
        )
        contract = checkpoint.contract
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
