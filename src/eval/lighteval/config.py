from __future__ import annotations

import hashlib
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit


PROMPT_TEMPLATES = {
    "bot": ("\nBot✿", "✿"),
    "assistant": ("\n\nAssistant: ", "\nUser:"),
    "function_calling": ("\n### Assistant", "\n### User"),
}
_CONFIG_FIELDS = {"schema_version", "prompt_template", "weights", "benchmarks"}
_ENV_FIELDS = (
    "WEIGHT_PATH",
    "HELICOPTER_SCOREBOARD_URL",
    "HELICOPTER_SCOREBOARD_TOKEN",
    "HELICOPTER_EVAL_STAGING_ROOT",
)


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LightEvalConfig:
    prompt_template: str
    weights: tuple[Path, ...]
    weight_hashes: tuple[str, ...]
    benchmarks: tuple[str, ...]
    scoreboard_url: str
    scoreboard_token: str
    staging_root: Path

    @classmethod
    def read(
        cls,
        path: Path,
        env: Mapping[str, str] = os.environ,
    ) -> LightEvalConfig:
        try:
            with path.open("rb") as stream:
                raw = tomllib.load(stream)
        except FileNotFoundError as error:
            raise ConfigError(f"eval config not found: {path}") from error
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"invalid eval TOML: {error}") from error

        unknown = sorted(set(raw) - _CONFIG_FIELDS)
        if unknown:
            raise ConfigError("unknown eval config fields: " + ", ".join(unknown))
        missing = sorted({"schema_version", "weights", "benchmarks"} - set(raw))
        if missing:
            raise ConfigError("missing eval config fields: " + ", ".join(missing))
        if isinstance(raw["schema_version"], bool) or raw["schema_version"] != 1:
            raise ConfigError("schema_version must be 1")

        prompt_template = raw.get("prompt_template", "bot")
        if prompt_template not in PROMPT_TEMPLATES:
            raise ConfigError(
                "prompt_template must be one of: " + ", ".join(PROMPT_TEMPLATES)
            )
        configured_weights = cls._strings(raw["weights"], "weights")
        benchmarks = cls._strings(raw["benchmarks"], "benchmarks")

        missing_env = [name for name in _ENV_FIELDS if not env.get(name)]
        if missing_env:
            raise ConfigError(
                "missing private eval environment: " + ", ".join(missing_env)
            )
        weight_root = cls._absolute_path(env["WEIGHT_PATH"], "WEIGHT_PATH")
        if not weight_root.is_dir() or weight_root.is_symlink():
            raise ConfigError("WEIGHT_PATH must be a regular directory")
        weight_root = weight_root.resolve()

        staging_root = cls._absolute_path(
            env["HELICOPTER_EVAL_STAGING_ROOT"],
            "HELICOPTER_EVAL_STAGING_ROOT",
        )
        if staging_root.exists() and (
            not staging_root.is_dir() or staging_root.is_symlink()
        ):
            raise ConfigError(
                "HELICOPTER_EVAL_STAGING_ROOT must be a regular directory"
            )
        if staging_root.resolve(strict=False) == weight_root:
            raise ConfigError("staging root must differ from WEIGHT_PATH")

        scoreboard_url = env["HELICOPTER_SCOREBOARD_URL"]
        parsed_url = urlsplit(scoreboard_url)
        if (
            scoreboard_url != scoreboard_url.strip()
            or parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ConfigError("HELICOPTER_SCOREBOARD_URL must be an HTTP(S) base URL")

        weights: list[Path] = []
        weight_hashes: list[str] = []
        for configured in configured_weights:
            relative = Path(configured)
            if relative.is_absolute() or ".." in relative.parts:
                raise ConfigError(
                    f"weight path must stay below WEIGHT_PATH: {configured}"
                )
            candidate = weight_root
            for part in relative.parts:
                candidate /= part
                if candidate.is_symlink():
                    raise ConfigError(
                        f"weight path must not use symlinks: {configured}"
                    )
            try:
                candidate = candidate.resolve(strict=True)
                candidate.relative_to(weight_root)
            except (FileNotFoundError, ValueError) as error:
                raise ConfigError(
                    f"weight path is missing or outside WEIGHT_PATH: {configured}"
                ) from error
            if not candidate.is_file():
                raise ConfigError(f"weight path is not a regular file: {configured}")
            with candidate.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            weights.append(candidate)
            weight_hashes.append(digest)

        duplicate_hashes = sorted(
            digest for digest in set(weight_hashes) if weight_hashes.count(digest) > 1
        )
        if duplicate_hashes:
            raise ConfigError("duplicate weight content is not allowed")

        return cls(
            prompt_template=prompt_template,
            weights=tuple(weights),
            weight_hashes=tuple(weight_hashes),
            benchmarks=benchmarks,
            scoreboard_url=scoreboard_url.rstrip("/"),
            scoreboard_token=env["HELICOPTER_SCOREBOARD_TOKEN"],
            staging_root=staging_root,
        )

    @staticmethod
    def _strings(value: object, name: str) -> tuple[str, ...]:
        if (
            not isinstance(value, list)
            or not value
            or any(
                not isinstance(item, str) or not item or item != item.strip()
                for item in value
            )
        ):
            raise ConfigError(
                f"{name} must be a non-empty array of non-empty trimmed strings"
            )
        values = tuple(value)
        duplicates = sorted(item for item in set(values) if values.count(item) > 1)
        if duplicates:
            raise ConfigError(f"duplicate {name}: " + ", ".join(duplicates))
        return values

    @staticmethod
    def _absolute_path(raw: str, name: str) -> Path:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise ConfigError(f"{name} must be an absolute path")
        return path

    @property
    def prompt(self) -> tuple[str, str]:
        return PROMPT_TEMPLATES[self.prompt_template]

    def public(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "prompt_template": self.prompt_template,
            "weights": [
                {"name": path.name, "sha256": digest}
                for path, digest in zip(
                    self.weights,
                    self.weight_hashes,
                    strict=True,
                )
            ],
            "benchmarks": list(self.benchmarks),
            "scoreboard_url": self.scoreboard_url,
            "scoreboard_token": "[REDACTED]",
            "staging_root": str(self.staging_root),
        }
