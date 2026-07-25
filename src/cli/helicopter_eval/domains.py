from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable


DOMAIN_RULES_VERSION = "2026-07-25"
TAG_TO_DOMAIN = {
    "code": "code",
    "code-generation": "code",
    "coding": "code",
    "instruction-following": "instruction-following",
    "language-modeling": "knowledge",
    "legal": "legal",
    "law": "legal",
    "math": "math",
    "mathematics": "math",
    "medical": "medical",
    "medicine": "medical",
    "knowledge": "knowledge",
    "general-knowledge": "knowledge",
    "reasoning": "reasoning",
    "safety": "safety",
}
MODULE_OVERRIDES = {
    "aa_omniscience": "knowledge",
}
DOMAIN_PRECEDENCE = (
    "safety",
    "legal",
    "medical",
    "code",
    "math",
    "instruction-following",
    "reasoning",
    "knowledge",
)


@dataclass(frozen=True)
class DomainAssignment:
    upstream_tags: tuple[str, ...]
    primary_domain: str


def rules_digest() -> str:
    payload = {
        "version": DOMAIN_RULES_VERSION,
        "tags": TAG_TO_DOMAIN,
        "overrides": MODULE_OVERRIDES,
        "precedence": DOMAIN_PRECEDENCE,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def assign_domain(module_family: str, tags: Iterable[str]) -> DomainAssignment:
    normalized = tuple(
        sorted(
            {
                str(tag).strip().lower()
                for tag in tags
                if isinstance(tag, str) and str(tag).strip()
            }
        )
    )
    override = MODULE_OVERRIDES.get(module_family)
    if override is not None:
        return DomainAssignment(normalized, override)
    mapped = {TAG_TO_DOMAIN[tag] for tag in normalized if tag in TAG_TO_DOMAIN}
    primary = next(
        (domain for domain in DOMAIN_PRECEDENCE if domain in mapped),
        "other",
    )
    return DomainAssignment(normalized, primary)
