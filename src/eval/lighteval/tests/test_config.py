from __future__ import annotations

import json
from pathlib import Path

import pytest

from lighteval_runner.config import resolve_evaluation_config


def test_config_precedence_and_provenance_are_per_field():
    config = resolve_evaluation_config(
        allowed_fields=frozenset({"max_tokens", "api_key"}),
        secret_fields=frozenset({"api_key"}),
        defaults={"max_tokens": 2048},
        file_values={"max_tokens": 1024},
        environment_values={"api_key": "environment"},
        cli_values={"max_tokens": 512},
    )

    assert config.get("max_tokens") == 512
    assert config.provenance("max_tokens") == "cli"
    assert config.get("api_key") == "environment"
    assert config.provenance("api_key") == "environment"
    assert config.redacted_payload()["values"] == {
        "api_key": "<redacted>",
        "max_tokens": 512,
    }


def test_non_secret_token_fields_remain_in_identity():
    first = resolve_evaluation_config(
        allowed_fields=frozenset({"max_tokens", "stop_token_id", "tokenizer_revision"}),
        defaults={"max_tokens": 512, "stop_token_id": 0, "tokenizer_revision": "a"},
    )
    second = resolve_evaluation_config(
        allowed_fields=frozenset({"max_tokens", "stop_token_id", "tokenizer_revision"}),
        defaults={"max_tokens": 1024, "stop_token_id": 0, "tokenizer_revision": "a"},
    )

    assert first.redacted_payload()["values"]["max_tokens"] == 512
    assert first.identity_digest() != second.identity_digest()


def test_unknown_fields_and_misdeclared_secrets_fail_fast():
    with pytest.raises(ValueError, match="unknown"):
        resolve_evaluation_config(
            allowed_fields=frozenset({"known"}), defaults={"unknown": 1}
        )
    with pytest.raises(ValueError, match="secret fields"):
        resolve_evaluation_config(
            allowed_fields=frozenset({"known"}),
            secret_fields=frozenset({"api_key"}),
            defaults={"known": 1},
        )
    with pytest.raises(ValueError, match="only come from the environment"):
        resolve_evaluation_config(
            allowed_fields=frozenset({"api_key"}),
            secret_fields=frozenset({"api_key"}),
            defaults={},
            file_values={"api_key": "forbidden"},
        )


def test_secret_presence_and_value_do_not_change_workload_identity():
    first = resolve_evaluation_config(
        allowed_fields=frozenset({"max_tokens", "api_key"}),
        secret_fields=frozenset({"api_key"}),
        defaults={"max_tokens": 512},
        environment_values={"api_key": "first"},
    )
    second = resolve_evaluation_config(
        allowed_fields=frozenset({"max_tokens", "api_key"}),
        secret_fields=frozenset({"api_key"}),
        defaults={"max_tokens": 512},
        environment_values={"api_key": "second"},
    )
    assert first.identity_digest() == second.identity_digest()


def test_config_evidence_normalizes_paths_at_the_serialization_boundary():
    config = resolve_evaluation_config(
        allowed_fields=frozenset({"snapshot"}),
        defaults={"snapshot": Path("datasets/test.parquet")},
    )

    payload = config.redacted_payload()
    assert payload["values"]["snapshot"] == "datasets/test.parquet"
    json.dumps(payload)
