from __future__ import annotations

import os
from typing import Any

from ...checkpoint import CheckpointManifest
from ...contract import build_target_config
from ...errors import ContractError


class NativeRWKV7TargetAdapter:
    adapter_id = "rwkv7"
    BASE_TRAINING_ENVIRONMENT = {
        "RWKV_JIT_ON": "0",
        "RWKV_MY_TESTING": "x070",
        "RWKV_KERNEL": "",
        "RWKV_HEAD_L2WRAP_CE_CHUNK": "0",
        "RWKV_TRAIN_TYPE": "infctx",
        "RWKV_FLOAT_MODE": "bf16",
        "WKV_MODE": "fp32io16",
    }

    def build_target_config(
        self,
        source_checkpoint: object,
        *,
        require_final_layout: bool,
    ) -> dict[str, Any]:
        if not isinstance(source_checkpoint, CheckpointManifest):
            raise ContractError(
                "rwkv7 target adapter received an unsupported source checkpoint object"
            )
        return build_target_config(
            source_checkpoint.config,
            require_final_layers=require_final_layout,
        )

    def validate_training_environment(self, *, head_size: int) -> dict[str, str]:
        if head_size <= 0:
            raise ContractError("native RWKV7 head_size must be positive")
        expected_environment = {
            **self.BASE_TRAINING_ENVIRONMENT,
            "RWKV_HEAD_SIZE": str(head_size),
        }
        mismatches = {
            name: {"expected": expected, "actual": os.environ.get(name)}
            for name, expected in expected_environment.items()
            if os.environ.get(name) != expected
        }
        if mismatches:
            raise ContractError(
                "native RWKV7 training environment mismatch: "
                + ", ".join(
                    f"{name}=expected:{row['expected']!r}/actual:{row['actual']!r}"
                    for name, row in sorted(mismatches.items())
                )
            )
        return expected_environment
