from __future__ import annotations

import hashlib
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

from helicopter_lighteval.http_pool import PoolError, PoolManifest


_CONFIG_FIELDS = {
    "schema_version",
    "backend",
    "tasks",
    "output_dir",
    "batch_size",
    "eot_token_id",
    "max_gen_toks",
    "limit",
    "log_samples",
    "publish",
    "weights",
    "wkv_modes",
    "pool_manifests",
}
_ENV_REFERENCE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ExecutionUnit:
    weight: Path | None
    weight_sha256: str | None
    wkv_mode: str
    manifest_path: Path
    manifest: PoolManifest


@dataclass(frozen=True)
class LMEvalConfig:
    tasks: tuple[str, ...]
    output_dir: Path
    batch_size: int
    eot_token_id: int
    max_gen_toks: int
    limit: int | None
    log_samples: bool
    vllm_pool_manifest: Path
    manifest: PoolManifest
    publish: bool = False
    execution_units: tuple[ExecutionUnit, ...] = ()
    scoreboard_url: str | None = None
    scoreboard_token: str | None = None
    backend: str = "vllm_http"

    @classmethod
    def read(
        cls,
        path: Path,
        env: Mapping[str, str] = os.environ,
    ) -> LMEvalConfig:
        try:
            with path.open("rb") as stream:
                raw = tomllib.load(stream)
        except FileNotFoundError as error:
            raise ConfigError(f"lm-eval config not found: {path}") from error
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"invalid lm-eval TOML: {error}") from error

        unknown = sorted(set(raw) - _CONFIG_FIELDS)
        if unknown:
            raise ConfigError("unknown lm-eval config fields: " + ", ".join(unknown))
        missing = sorted(
            {"schema_version", "backend", "tasks", "output_dir"} - set(raw)
        )
        if missing:
            raise ConfigError("missing lm-eval config fields: " + ", ".join(missing))
        if isinstance(raw["schema_version"], bool) or raw["schema_version"] != 1:
            raise ConfigError("schema_version must be 1")
        if raw["backend"] != "vllm_http":
            raise ConfigError("backend must be vllm_http")

        tasks = cls._strings(raw["tasks"], "tasks")
        configured_output = raw["output_dir"]
        if (
            not isinstance(configured_output, str)
            or not configured_output
            or configured_output != configured_output.strip()
        ):
            raise ConfigError("output_dir must be a non-empty trimmed string")
        configured_output = cls._expand_environment(
            configured_output, env, "output_dir"
        )
        output_dir = Path(configured_output).expanduser()
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        if output_dir.exists() and (
            not output_dir.is_dir() or output_dir.is_symlink()
        ):
            raise ConfigError("output_dir must be a regular directory")

        batch_size = raw.get("batch_size", 1)
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ConfigError("batch_size must be a positive integer")
        eot_token_id = raw.get("eot_token_id", 0)
        if (
            not isinstance(eot_token_id, int)
            or isinstance(eot_token_id, bool)
            or eot_token_id < 0
        ):
            raise ConfigError("eot_token_id must be a non-negative integer")
        log_samples = raw.get("log_samples", False)
        if not isinstance(log_samples, bool):
            raise ConfigError("log_samples must be a boolean")
        max_gen_toks = raw.get("max_gen_toks", 256)
        if (
            not isinstance(max_gen_toks, int)
            or isinstance(max_gen_toks, bool)
            or max_gen_toks <= 0
        ):
            raise ConfigError("max_gen_toks must be a positive integer")
        limit = raw.get("limit")
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
        ):
            raise ConfigError("limit must be a positive integer")

        publish = raw.get("publish", False)
        if not isinstance(publish, bool):
            raise ConfigError("publish must be a boolean")
        if publish and limit is not None:
            raise ConfigError("published evaluation must not set limit")

        configured_manifests = raw.get("pool_manifests")
        if configured_manifests is None:
            configured_manifest = env.get("HELICOPTER_VLLM_POOL_MANIFEST")
            if not configured_manifest:
                raise ConfigError(
                    "backend = vllm_http requires pool_manifests or "
                    "HELICOPTER_VLLM_POOL_MANIFEST"
                )
            manifest_values = (configured_manifest,)
        else:
            manifest_values = cls._strings(configured_manifests, "pool_manifests")
        manifest_paths = tuple(
            cls._manifest_path(cls._expand_environment(value, env, "pool_manifests"))
            for value in manifest_values
        )
        manifests: list[PoolManifest] = []
        for manifest_path in manifest_paths:
            try:
                manifest = PoolManifest.read(manifest_path)
            except PoolError as error:
                raise ConfigError(str(error)) from error
            if manifest.max_model_len < 4:
                raise ConfigError("vLLM pool max_model_len must be at least 4 for scoring")
            if max_gen_toks >= manifest.max_model_len - 2:
                raise ConfigError(
                    "max_gen_toks must be smaller than the effective vLLM context length"
                )
            manifests.append(manifest)

        configured_weights = raw.get("weights")
        weights: tuple[tuple[Path, str], ...] = ()
        if configured_weights is not None:
            weight_values = cls._strings(configured_weights, "weights")
            weight_root_raw = env.get("WEIGHT_PATH")
            if not weight_root_raw:
                raise ConfigError("weights require WEIGHT_PATH")
            weight_root = Path(weight_root_raw).expanduser().resolve(strict=True)
            weights = tuple(
                cls._weight(
                    cls._expand_environment(value, env, "weights"), weight_root
                )
                for value in weight_values
            )
            hashes = [digest for _path, digest in weights]
            if len(hashes) != len(set(hashes)):
                raise ConfigError("duplicate weight content is not allowed")

        configured_modes = raw.get("wkv_modes")
        if configured_modes is None:
            wkv_modes = tuple(dict.fromkeys(item.wkv_mode for item in manifests))
        else:
            wkv_modes = cls._strings(configured_modes, "wkv_modes")
        if any(mode not in {"fp16", "fp32io16"} for mode in wkv_modes):
            raise ConfigError("wkv_modes contains an unsupported mode")

        units: list[ExecutionUnit] = []
        if weights:
            dimensions = [
                (weight, digest, mode)
                for weight, digest in weights
                for mode in wkv_modes
            ]
            if len(dimensions) != len(manifests):
                raise ConfigError(
                    "pool_manifests must contain one entry per weight and WKV mode"
                )
            for (weight, digest, mode), manifest_path, manifest in zip(
                dimensions, manifest_paths, manifests, strict=True
            ):
                if manifest.wkv_mode != mode:
                    raise ConfigError("pool manifest WKV mode does not match its matrix unit")
                if manifest.weight_sha256 != digest:
                    raise ConfigError(
                        "pool manifest weight_sha256 does not match its matrix unit"
                    )
                if manifest.weight_display_name != weight.name:
                    raise ConfigError(
                        "pool manifest weight_display_name does not match its matrix unit"
                    )
                units.append(
                    ExecutionUnit(weight, digest, mode, manifest_path, manifest)
                )
        else:
            if len(manifests) != 1:
                raise ConfigError("multiple pool manifests require configured weights")
            units.append(
                ExecutionUnit(
                    None,
                    manifests[0].weight_sha256,
                    manifests[0].wkv_mode,
                    manifest_paths[0],
                    manifests[0],
                )
            )
        if publish and (not weights or set(wkv_modes) != {"fp16", "fp32io16"}):
            raise ConfigError(
                "published evaluation requires weights and both WKV modes"
            )

        scoreboard_url: str | None = None
        scoreboard_token: str | None = None
        if publish:
            scoreboard_url = env.get("HELICOPTER_SCOREBOARD_URL")
            scoreboard_token = env.get("HELICOPTER_SCOREBOARD_TOKEN")
            if not scoreboard_url or not scoreboard_token:
                raise ConfigError(
                    "published evaluation requires Scoreboard URL and token"
                )
            parsed = urlsplit(scoreboard_url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ConfigError("HELICOPTER_SCOREBOARD_URL must be an HTTP(S) base URL")
            scoreboard_url = scoreboard_url.rstrip("/")

        vllm_pool_manifest = manifest_paths[0]
        manifest = manifests[0]

        return cls(
            tasks=tasks,
            output_dir=output_dir,
            batch_size=batch_size,
            eot_token_id=eot_token_id,
            max_gen_toks=max_gen_toks,
            limit=limit,
            log_samples=log_samples,
            vllm_pool_manifest=vllm_pool_manifest,
            manifest=manifest,
            publish=publish,
            execution_units=tuple(units),
            scoreboard_url=scoreboard_url,
            scoreboard_token=scoreboard_token,
        )

    @staticmethod
    def _manifest_path(raw: str) -> Path:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise ConfigError("pool manifest paths must be absolute")
        if not path.is_file() or path.is_symlink():
            raise ConfigError("pool manifest path must be a regular file")
        return path

    @staticmethod
    def _weight(raw: str, weight_root: Path) -> tuple[Path, str]:
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise ConfigError(f"weight path must stay below WEIGHT_PATH: {raw}")
        candidate = weight_root
        for part in relative.parts:
            candidate /= part
            if candidate.is_symlink():
                raise ConfigError(f"weight path must not use symlinks: {raw}")
        try:
            candidate = candidate.resolve(strict=True)
            candidate.relative_to(weight_root)
        except (FileNotFoundError, ValueError) as error:
            raise ConfigError(f"weight path is missing or outside WEIGHT_PATH: {raw}") from error
        if not candidate.is_file():
            raise ConfigError(f"weight path is not a regular file: {raw}")
        with candidate.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return candidate, digest

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

    def public(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "backend": self.backend,
            "tasks": list(self.tasks),
            "output_dir": str(self.output_dir),
            "batch_size": self.batch_size,
            "eot_token_id": self.eot_token_id,
            "max_gen_toks": self.max_gen_toks,
            "limit": self.limit,
            "log_samples": self.log_samples,
            "vllm_pool_manifest": str(self.vllm_pool_manifest),
            "publish": self.publish,
            "execution_units": [
                {
                    "weight": unit.weight.name if unit.weight else None,
                    "weight_sha256": unit.weight_sha256,
                    "wkv_mode": unit.wkv_mode,
                    "pool_manifest": str(unit.manifest_path),
                }
                for unit in self.execution_units
            ],
            "scoreboard_url": self.scoreboard_url,
            "scoreboard_token": "[REDACTED]" if self.scoreboard_token else None,
        }
