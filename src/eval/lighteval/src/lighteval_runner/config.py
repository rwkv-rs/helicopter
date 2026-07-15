from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ResolvedField:
    name: str
    value: Any
    source: str
    secret: bool = False


@dataclass(frozen=True, slots=True)
class ResolvedEvaluationConfig:
    fields: tuple[ResolvedField, ...]
    config_path: Path | None = None
    config_digest: str | None = None

    def get(self, name: str, default: Any = None) -> Any:
        return next(
            (field.value for field in self.fields if field.name == name), default
        )

    def provenance(self, name: str) -> str | None:
        return next((field.source for field in self.fields if field.name == name), None)

    def redacted_payload(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        sources: dict[str, str] = {}
        for field in self.fields:
            values[field.name] = (
                "<redacted>"
                if field.secret and field.value is not None
                else _evidence_value(field.value)
            )
            sources[field.name] = field.source
        return {
            "values": values,
            "sources": sources,
            "config_path": str(self.config_path) if self.config_path else None,
            "source_file_digest": self.config_digest,
            "effective_config_digest": self.identity_digest(),
        }

    def identity_digest(self) -> str:
        canonical = json.dumps(
            {
                "values": {
                    field.name: field.value for field in self.fields if not field.secret
                },
                "sources": {
                    field.name: field.source
                    for field in self.fields
                    if not field.secret
                },
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        return hashlib.sha256(canonical).hexdigest()


def resolve_evaluation_config(
    *,
    allowed_fields: frozenset[str],
    secret_fields: frozenset[str] = frozenset(),
    defaults: Mapping[str, Any],
    file_values: Mapping[str, Any] | None = None,
    environment_values: Mapping[str, Any] | None = None,
    cli_values: Mapping[str, Any] | None = None,
    config_path: Path | None = None,
) -> ResolvedEvaluationConfig:
    """Resolve values with explicit precedence and reject unaudited fields."""

    layers = (
        ("default", defaults),
        ("config", file_values or {}),
        ("environment", environment_values or {}),
        ("cli", cli_values or {}),
    )
    undeclared_secrets = sorted(secret_fields - allowed_fields)
    if undeclared_secrets:
        raise ValueError(
            f"secret fields are not allowed config fields: {', '.join(undeclared_secrets)}"
        )
    unknown = sorted(
        {key for _, layer in layers for key in layer if key not in allowed_fields}
    )
    if unknown:
        raise ValueError(f"unknown evaluation config fields: {', '.join(unknown)}")
    secret_outside_environment = sorted(
        name
        for source, layer in layers
        if source != "environment"
        for name, value in layer.items()
        if name in secret_fields and value is not None
    )
    if secret_outside_environment:
        raise ValueError(
            "evaluation secrets may only come from the environment: "
            + ", ".join(secret_outside_environment)
        )

    resolved: dict[str, ResolvedField] = {}
    for source, layer in layers:
        for name, value in layer.items():
            if value is not None:
                resolved[name] = ResolvedField(
                    name=name,
                    value=value,
                    source=source,
                    secret=name in secret_fields,
                )

    digest: str | None = None
    if config_path is not None:
        digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return ResolvedEvaluationConfig(
        fields=tuple(resolved[name] for name in sorted(resolved)),
        config_path=config_path.resolve() if config_path else None,
        config_digest=digest,
    )


def _evidence_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _evidence_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_evidence_value(item) for item in value]
    return value
