from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import httpx

from ..execution import ModelIdentity, ProviderIdentity


class Comparability(StrEnum):
    OFFICIAL = "official"
    NON_COMPARABLE = "non_comparable"


@dataclass(frozen=True, slots=True)
class ProviderAttestation:
    model: ModelIdentity
    provider: ProviderIdentity
    capabilities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AttestationDecision:
    comparability: Comparability
    mismatches: tuple[str, ...]


def validate_attestation(
    expected: ProviderAttestation,
    actual: ProviderAttestation | None,
    *,
    official: bool,
    allow_non_comparable: bool,
) -> AttestationDecision:
    mismatches = _attestation_mismatches(expected, actual)
    if not mismatches:
        return AttestationDecision(
            Comparability.OFFICIAL if official else Comparability.NON_COMPARABLE,
            (),
        )
    if official or not allow_non_comparable:
        raise ValueError(f"provider attestation mismatch: {', '.join(mismatches)}")
    return AttestationDecision(Comparability.NON_COMPARABLE, mismatches)


def _attestation_mismatches(
    expected: ProviderAttestation, actual: ProviderAttestation | None
) -> tuple[str, ...]:
    if actual is None:
        return ("missing_attestation",)
    expected_payload = asdict(expected)
    actual_payload = asdict(actual)
    mismatches: list[str] = []
    for section in ("model", "provider"):
        for name, value in expected_payload[section].items():
            if actual_payload[section].get(name) != value:
                mismatches.append(f"{section}.{name}")
    if frozenset(actual.capabilities) != frozenset(expected.capabilities):
        mismatches.append("capabilities")
    return tuple(mismatches)


def fetch_provider_attestation(
    *, base_url: str, client: httpx.Client
) -> ProviderAttestation | None:
    try:
        response = client.get(f"{base_url.rstrip('/')}/helicopter/attestation")
        response.raise_for_status()
        payload = response.json()
        model = ModelIdentity(**payload["model"])
        provider = ProviderIdentity(**payload["provider"])
        capabilities = payload["capabilities"]
        if not isinstance(capabilities, list) or any(
            not isinstance(item, str) for item in capabilities
        ):
            raise TypeError("capabilities must be a string array")
        return ProviderAttestation(model, provider, tuple(capabilities))
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return None
