from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """Stable application input; internal execution objects are intentionally absent."""

    model: str
    task: str
    output_root: Path
    snapshot_path: Path | None
    snapshot_manifest_path: Path | None
    snapshot_sha256: str | None
    endpoint_url: str
    checkpoint_sha256: str
    tokenizer_revision: str
    chat_template_revision: str
    expected_server_revision: str
    wkv_mode: str
    precision: str
    gemm_policy: str
    launch_contract: str
    product_revision: str
    product_dirty: bool = False
    cot_mode: str = "none"
    math_repair_strategy: str = "A"
    generation_limit_override: int | None = None
    generation_limit_override_source: str | None = None
    config_digest: str = ""
    config_evidence: Mapping[str, Any] | None = None
    max_samples: int | None = None
    publish_to_scoreboard: bool = False
    scoreboard_url: str | None = None
    scoreboard_token: str | None = None
    endpoint_api_key: str | None = None
    allow_non_comparable: bool = False

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.task.strip():
            raise ValueError("task must be a non-empty canonical identity")
        if not self.endpoint_url.strip():
            raise ValueError("endpoint_url must not be empty")
        for name, value in (("checkpoint_sha256", self.checkpoint_sha256),):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.snapshot_sha256 is not None and (
            len(self.snapshot_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.snapshot_sha256
            )
        ):
            raise ValueError("snapshot_sha256 must be a lowercase SHA-256 digest")
        if self.math_repair_strategy not in {"A", "B", "C"}:
            raise ValueError("math_repair_strategy must be A, B, or C")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if self.cot_mode not in {"none", "cot"}:
            raise ValueError("cot_mode must be none or cot")
        if self.generation_limit_override is not None:
            if self.generation_limit_override <= 0:
                raise ValueError("generation_limit_override must be positive")
            if not self.generation_limit_override_source:
                raise ValueError("generation limit override requires provenance")
        if self.publish_to_scoreboard and (
            not self.scoreboard_url or not self.scoreboard_token
        ):
            raise ValueError("scoreboard publication requires URL and token")
        if len(self.product_revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.product_revision
        ):
            raise ValueError("product_revision must be a lowercase Git commit")
        if self.config_digest and (
            len(self.config_digest) != 64
            or any(
                character not in "0123456789abcdef" for character in self.config_digest
            )
        ):
            raise ValueError("config_digest must be a lowercase SHA-256 digest")
        if self.config_evidence is not None and (
            self.config_evidence.get("effective_config_digest") != self.config_digest
        ):
            raise ValueError(
                "config evidence does not match the effective config digest"
            )


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """Stable application output referencing, rather than embedding, run evidence."""

    run_id: str
    run_status: str
    manifest_path: Path
    publication_status: str
    publication_error: str | None = None
    publication_retry_identity: str | None = None
    publication_task_id: int | None = None
    summary: dict[str, int | float | str | None] | None = None

    @property
    def is_success(self) -> bool:
        return self.run_status == "completed" and self.publication_status in {
            "not_requested",
            "published",
        }
