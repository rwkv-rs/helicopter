from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import tomllib
from typing import Literal, Mapping, cast
from urllib.parse import urlsplit


PromptTemplate = Literal["bot", "assistant", "function_calling"]
PROMPT_TEMPLATE_STOPS: dict[PromptTemplate, str] = {
    "bot": "✿",
    "assistant": "\nUser:",
    "function_calling": "\n### User",
}
CONFIG_KEYS = frozenset({"schema_version", "prompt_template", "weights", "benchmarks"})
REQUIRED_CONFIG_KEYS = frozenset({"schema_version", "weights", "benchmarks"})
SCHEMA_VERSION = 1


class EvaluationConfigurationError(ValueError):
    """The public eval config or private environment is invalid."""


@dataclass(frozen=True)
class EvaluationConfig:
    schema_version: int
    weights: tuple[str, ...]
    benchmarks: tuple[str, ...]
    prompt_template: PromptTemplate = "bot"


@dataclass(frozen=True)
class WeightIdentity:
    configured_path: str
    path: Path
    display_name: str
    sha256: str


@dataclass(frozen=True)
class EvaluationEnvironment:
    weight_root: Path
    scoreboard_url: str
    scoreboard_token: str
    staging_root: Path


def load_evaluation_config(path: Path) -> EvaluationConfig:
    try:
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
    except FileNotFoundError as error:
        raise EvaluationConfigurationError(f"eval config not found: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise EvaluationConfigurationError(f"invalid eval TOML: {error}") from error

    unknown = sorted(set(raw) - CONFIG_KEYS)
    if unknown:
        raise EvaluationConfigurationError(
            "unknown eval config fields: " + ", ".join(unknown)
        )
    missing = sorted(REQUIRED_CONFIG_KEYS - set(raw))
    if missing:
        raise EvaluationConfigurationError(
            "missing eval config fields: " + ", ".join(missing)
        )
    version = raw["schema_version"]
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        raise EvaluationConfigurationError(f"schema_version must be {SCHEMA_VERSION}")
    prompt_template = raw.get("prompt_template", "bot")
    if (
        not isinstance(prompt_template, str)
        or prompt_template not in PROMPT_TEMPLATE_STOPS
    ):
        raise EvaluationConfigurationError(
            "prompt_template must be one of: " + ", ".join(PROMPT_TEMPLATE_STOPS)
        )
    weights = _string_array(raw["weights"], name="weights")
    benchmarks = _string_array(raw["benchmarks"], name="benchmarks")
    return EvaluationConfig(
        schema_version=SCHEMA_VERSION,
        prompt_template=cast(PromptTemplate, prompt_template),
        weights=weights,
        benchmarks=benchmarks,
    )


def _string_array(value: object, *, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, str) or not item or item != item.strip()
            for item in value
        )
    ):
        raise EvaluationConfigurationError(
            f"{name} must be a non-empty array of non-empty trimmed strings"
        )
    normalized = tuple(value)
    duplicates = sorted({item for item in normalized if normalized.count(item) > 1})
    if duplicates:
        raise EvaluationConfigurationError(
            f"duplicate {name} are not allowed: " + ", ".join(duplicates)
        )
    return normalized


def load_evaluation_environment(env: Mapping[str, str]) -> EvaluationEnvironment:
    names = {
        "WEIGHT_PATH": env.get("WEIGHT_PATH"),
        "HELICOPTER_SCOREBOARD_URL": env.get("HELICOPTER_SCOREBOARD_URL"),
        "HELICOPTER_SCOREBOARD_TOKEN": env.get("HELICOPTER_SCOREBOARD_TOKEN"),
        "HELICOPTER_EVAL_STAGING_ROOT": env.get("HELICOPTER_EVAL_STAGING_ROOT"),
    }
    missing = sorted(name for name, value in names.items() if not value)
    if missing:
        raise EvaluationConfigurationError(
            "missing private eval environment: " + ", ".join(missing)
        )
    raw_weight_root = Path(str(names["WEIGHT_PATH"])).expanduser()
    raw_staging_root = Path(str(names["HELICOPTER_EVAL_STAGING_ROOT"])).expanduser()
    if not raw_weight_root.is_absolute():
        raise EvaluationConfigurationError("WEIGHT_PATH must be an absolute path")
    if not raw_staging_root.is_absolute():
        raise EvaluationConfigurationError(
            "HELICOPTER_EVAL_STAGING_ROOT must be an absolute path"
        )
    if raw_weight_root.is_symlink():
        raise EvaluationConfigurationError("WEIGHT_PATH must not be a symlink")
    if raw_staging_root.is_symlink():
        raise EvaluationConfigurationError(
            "HELICOPTER_EVAL_STAGING_ROOT must not be a symlink"
        )
    weight_root = raw_weight_root.resolve()
    staging_root = raw_staging_root.resolve()
    product_root = Path(__file__).resolve().parents[5]
    if not weight_root.is_dir():
        raise EvaluationConfigurationError(
            f"WEIGHT_PATH is not a directory: {weight_root}"
        )
    if staging_root.exists() and not staging_root.is_dir():
        raise EvaluationConfigurationError(
            "HELICOPTER_EVAL_STAGING_ROOT must be a directory"
        )
    if staging_root in {Path("/"), Path.home().resolve(), product_root}:
        raise EvaluationConfigurationError(
            "HELICOPTER_EVAL_STAGING_ROOT must be a dedicated child directory, "
            "not a filesystem, home, or product root"
        )
    if staging_root.exists():
        staging_status = staging_root.stat()
        if (
            staging_status.st_uid != os.geteuid()
            or stat.S_IMODE(staging_status.st_mode) != 0o700
        ):
            raise EvaluationConfigurationError(
                "an existing HELICOPTER_EVAL_STAGING_ROOT must be owned by "
                "the current user and have mode 0700"
            )
    if (
        staging_root == weight_root
        or staging_root.is_relative_to(weight_root)
        or weight_root.is_relative_to(staging_root)
    ):
        raise EvaluationConfigurationError(
            "HELICOPTER_EVAL_STAGING_ROOT and WEIGHT_PATH must not overlap"
        )
    scoreboard_url = str(names["HELICOPTER_SCOREBOARD_URL"]).rstrip("/")
    parsed_url = urlsplit(scoreboard_url)
    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.hostname
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise EvaluationConfigurationError(
            "HELICOPTER_SCOREBOARD_URL must be an http(s) origin/base path "
            "without credentials, query, or fragment"
        )
    scoreboard_token = str(names["HELICOPTER_SCOREBOARD_TOKEN"])
    if any(
        ord(character) < 0x21 or ord(character) > 0x7E for character in scoreboard_token
    ):
        raise EvaluationConfigurationError(
            "HELICOPTER_SCOREBOARD_TOKEN must contain only visible ASCII characters"
        )
    return EvaluationEnvironment(
        weight_root=weight_root,
        scoreboard_url=scoreboard_url,
        scoreboard_token=scoreboard_token,
        staging_root=staging_root,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_weights(
    config: EvaluationConfig, environment: EvaluationEnvironment
) -> tuple[WeightIdentity, ...]:
    root = environment.weight_root
    identities: list[WeightIdentity] = []
    seen_paths: set[Path] = set()
    seen_digests: set[str] = set()
    for configured in config.weights:
        relative = Path(configured)
        if (
            relative.is_absolute()
            or not relative.parts
            or "." in relative.parts
            or ".." in relative.parts
            or relative.as_posix() != configured
        ):
            raise EvaluationConfigurationError(
                "weight path must be a normalized relative child of "
                f"WEIGHT_PATH: {configured}"
            )
        unresolved = root / relative
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise EvaluationConfigurationError(
                    f"weight path must not contain symlinks: {configured}"
                )
        candidate = unresolved.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise EvaluationConfigurationError(
                f"weight path escapes WEIGHT_PATH: {configured}"
            ) from error
        if not candidate.is_file():
            raise EvaluationConfigurationError(f"weight file not found: {configured}")
        if candidate in seen_paths:
            raise EvaluationConfigurationError(
                f"duplicate resolved weight path: {configured}"
            )
        digest = _sha256(candidate)
        if digest in seen_digests:
            raise EvaluationConfigurationError(
                f"duplicate weight content is not allowed: {configured}"
            )
        seen_paths.add(candidate)
        seen_digests.add(digest)
        identities.append(
            WeightIdentity(
                configured_path=configured,
                path=candidate,
                display_name=candidate.name,
                sha256=digest,
            )
        )
    return tuple(identities)


def verify_weight_identity(weight: WeightIdentity) -> None:
    if (
        weight.path.is_symlink()
        or not weight.path.is_file()
        or weight.path.resolve() != weight.path
    ):
        raise EvaluationConfigurationError(
            f"evaluation weight path changed after preflight: {weight.configured_path}"
        )
    if _sha256(weight.path) != weight.sha256:
        raise EvaluationConfigurationError(
            f"evaluation weight content changed after preflight: "
            f"{weight.configured_path}"
        )


def public_environment(environment: EvaluationEnvironment) -> dict[str, str]:
    return {
        "weight_root": str(environment.weight_root),
        "scoreboard_url": environment.scoreboard_url,
        "scoreboard_token": "[REDACTED]",
        "staging_root": str(environment.staging_root),
    }
