from __future__ import annotations

import base64
import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def encode_cursor(completed_at: str, run_id: str) -> str:
    return (
        base64.urlsafe_b64encode(canonical_json([completed_at, run_id]).encode())
        .decode()
        .rstrip("=")
    )


def decode_cursor(cursor: str) -> tuple[str, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded).decode())
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(not isinstance(item, str) for item in value)
        ):
            raise ValueError
        return value[0], value[1]
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid cursor") from error
