from __future__ import annotations

import hashlib
import os
import re
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
WKV_MODES = ("fp16", "fp32io16")
_CONFIG_FIELDS = {
    "schema_version",
    "backend",
    "prompt_template",
    "publish",
    "result_path",
    "weights",
    "benchmarks",
    "wkv_modes",
}
_ENV_REFERENCE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LightEvalConfig:
    prompt_template: str
    publish: bool
    result_path: Path | None
    weights: tuple[Path, ...]
    weight_hashes: tuple[str, ...]
    benchmarks: tuple[str, ...]
    wkv_modes: tuple[str, ...]
    scoreboard_url: str | None
    scoreboard_token: str | None
    staging_root: Path
    backend: str = "local"
    vllm_pool_manifest: Path | None = None

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

        backend = raw.get("backend", "local")
        if backend not in {"local", "vllm_http"}:
            raise ConfigError("backend must be one of: local, vllm_http")
        prompt_template = raw.get("prompt_template", "bot")
        if prompt_template not in PROMPT_TEMPLATES:
            raise ConfigError(
                "prompt_template must be one of: " + ", ".join(PROMPT_TEMPLATES)
            )
        configured_weights = tuple(
            cls._expand_environment(value, env, "weights")
            for value in cls._strings(raw["weights"], "weights")
        )
        benchmarks = cls._strings(raw["benchmarks"], "benchmarks")
        publish = raw.get("publish", True)
        if not isinstance(publish, bool):
            raise ConfigError("publish must be a boolean")
        wkv_modes = cls._strings(raw.get("wkv_modes", list(WKV_MODES)), "wkv_modes")
        unsupported_modes = sorted(set(wkv_modes) - set(WKV_MODES))
        if unsupported_modes:
            raise ConfigError("unsupported wkv_modes: " + ", ".join(unsupported_modes))

        configured_result_path = raw.get("result_path")
        result_path: Path | None = None
        if configured_result_path is not None:
            if not isinstance(configured_result_path, str):
                raise ConfigError("result_path must be a string")
            expanded = cls._expand_environment(
                configured_result_path,
                env,
                "result_path",
            )
            result_path = cls._absolute_path(expanded, "result_path")
        if publish and result_path is not None:
            raise ConfigError("result_path is only valid when publish = false")
        if not publish:
            if result_path is None:
                raise ConfigError("result_path is required when publish = false")
            if len(configured_weights) != 1 or len(wkv_modes) != 1:
                raise ConfigError(
                    "publish = false requires exactly one weight and one wkv_mode"
                )
        if backend == "vllm_http" and (
            len(configured_weights) != 1 or len(wkv_modes) != 1
        ):
            raise ConfigError(
                "backend = vllm_http requires exactly one weight and one wkv_mode"
            )

        vllm_pool_manifest: Path | None = None
        if backend == "vllm_http":
            configured_manifest = env.get("HELICOPTER_VLLM_POOL_MANIFEST")
            if not configured_manifest:
                raise ConfigError(
                    "backend = vllm_http requires HELICOPTER_VLLM_POOL_MANIFEST"
                )
            vllm_pool_manifest = cls._absolute_path(
                configured_manifest,
                "HELICOPTER_VLLM_POOL_MANIFEST",
            )
            if not vllm_pool_manifest.is_file() or vllm_pool_manifest.is_symlink():
                raise ConfigError(
                    "HELICOPTER_VLLM_POOL_MANIFEST must be a regular file"
                )
            from .http_pool import PoolError, PoolManifest

            try:
                manifest = PoolManifest.read(vllm_pool_manifest)
            except PoolError as error:
                raise ConfigError(str(error)) from error
            if manifest.wkv_mode != wkv_modes[0]:
                raise ConfigError(
                    "vLLM pool WKV mode does not match the evaluation configuration"
                )
            configured_step = env.get("MAXRL_EVAL_STEP")
            if configured_step is not None:
                try:
                    expected_step = int(configured_step)
                except ValueError as error:
                    raise ConfigError("MAXRL_EVAL_STEP must be an integer") from error
                if expected_step < 0 or str(expected_step) != configured_step:
                    raise ConfigError(
                        "MAXRL_EVAL_STEP must be a canonical non-negative integer"
                    )
                if manifest.global_step != expected_step:
                    raise ConfigError(
                        "vLLM pool global_step does not match MAXRL_EVAL_STEP"
                    )

        missing_env = [name for name in ("WEIGHT_PATH",) if not env.get(name)]
        if publish:
            missing_env.extend(
                name
                for name in (
                    "HELICOPTER_EVAL_STAGING_ROOT",
                    "HELICOPTER_SCOREBOARD_URL",
                    "HELICOPTER_SCOREBOARD_TOKEN",
                )
                if not env.get(name)
            )
        if missing_env:
            raise ConfigError(
                "missing private eval environment: " + ", ".join(missing_env)
            )
        weight_root = cls._absolute_path(env["WEIGHT_PATH"], "WEIGHT_PATH")
        if not weight_root.is_dir() or weight_root.is_symlink():
            raise ConfigError("WEIGHT_PATH must be a regular directory")
        weight_root = weight_root.resolve()

        configured_staging_root = env.get("HELICOPTER_EVAL_STAGING_ROOT")
        if configured_staging_root:
            staging_root = cls._absolute_path(
                configured_staging_root,
                "HELICOPTER_EVAL_STAGING_ROOT",
            )
        else:
            if result_path is None:
                raise ConfigError("local evaluation requires result_path")
            staging_root = result_path.parent / ".lighteval-staging"
        if staging_root.exists() and (
            not staging_root.is_dir() or staging_root.is_symlink()
        ):
            raise ConfigError(
                "HELICOPTER_EVAL_STAGING_ROOT must be a regular directory"
            )
        if staging_root.resolve(strict=False) == weight_root:
            raise ConfigError("staging root must differ from WEIGHT_PATH")

        scoreboard_url: str | None = None
        scoreboard_token: str | None = None
        if publish:
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
                raise ConfigError(
                    "HELICOPTER_SCOREBOARD_URL must be an HTTP(S) base URL"
                )
            scoreboard_url = scoreboard_url.rstrip("/")
            scoreboard_token = env["HELICOPTER_SCOREBOARD_TOKEN"]

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
            publish=publish,
            result_path=result_path,
            weights=tuple(weights),
            weight_hashes=tuple(weight_hashes),
            benchmarks=benchmarks,
            wkv_modes=wkv_modes,
            scoreboard_url=scoreboard_url,
            scoreboard_token=scoreboard_token,
            staging_root=staging_root,
            backend=backend,
            vllm_pool_manifest=vllm_pool_manifest,
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

    @staticmethod
    def _expand_environment(
        value: str,
        env: Mapping[str, str],
        name: str,
    ) -> str:
        match = _ENV_REFERENCE.fullmatch(value)
        if match is None:
            return value
        variable = match.group(1)
        expanded = env.get(variable)
        if not expanded:
            raise ConfigError(
                f"{name} references missing environment variable {variable}"
            )
        return expanded

    @property
    def prompt(self) -> tuple[str, str]:
        return PROMPT_TEMPLATES[self.prompt_template]

    def public(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "backend": self.backend,
            "prompt_template": self.prompt_template,
            "publish": self.publish,
            "result_path": str(self.result_path) if self.result_path else None,
            "weights": [
                {"name": path.name, "sha256": digest}
                for path, digest in zip(
                    self.weights,
                    self.weight_hashes,
                    strict=True,
                )
            ],
            "benchmarks": list(self.benchmarks),
            "wkv_modes": list(self.wkv_modes),
            "scoreboard_url": self.scoreboard_url,
            "scoreboard_token": "[REDACTED]" if self.scoreboard_token else None,
            "staging_root": str(self.staging_root),
            "vllm_pool_manifest": (
                str(self.vllm_pool_manifest) if self.vllm_pool_manifest else None
            ),
        }
