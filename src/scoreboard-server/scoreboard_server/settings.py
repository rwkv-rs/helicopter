from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AuthPrincipal:
    subject: str
    roles: frozenset[str]


@dataclass(frozen=True, slots=True)
class ScoreboardSettings:
    cors_origins: tuple[str, ...]
    tokens: tuple[tuple[str, AuthPrincipal], ...]

    @classmethod
    def from_env(cls) -> "ScoreboardSettings":
        origins = tuple(
            origin.strip()
            for origin in os.environ.get(
                "SCOREBOARD_CORS_ORIGINS", "http://127.0.0.1:3000"
            ).split(",")
            if origin.strip()
        )
        if not origins or "*" in origins:
            raise ValueError(
                "SCOREBOARD_CORS_ORIGINS must be a non-empty explicit allowlist"
            )
        raw_tokens = json.loads(os.environ.get("SCOREBOARD_AUTH_TOKENS", "{}"))
        if not isinstance(raw_tokens, dict):
            raise ValueError("SCOREBOARD_AUTH_TOKENS must be a JSON object")
        tokens: list[tuple[str, AuthPrincipal]] = []
        for token, value in raw_tokens.items():
            if not isinstance(token, str) or not token or not isinstance(value, dict):
                raise ValueError(
                    "scoreboard token entries must map non-empty tokens to objects"
                )
            subject = value.get("subject")
            roles = value.get("roles")
            if (
                not isinstance(subject, str)
                or not subject
                or not isinstance(roles, list)
                or any(
                    role not in {"publisher", "evidence_reader", "admin"}
                    for role in roles
                )
            ):
                raise ValueError(
                    "scoreboard principal must declare subject and known roles"
                )
            tokens.append((token, AuthPrincipal(subject, frozenset(roles))))
        return cls(cors_origins=origins, tokens=tuple(tokens))
