from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Any


DEFAULT_ENV_FILE = ".env.local"
ENV_FALLBACKS = (".env.remote", ".env")


def find_env_path(
    root: Path,
    env_file: str,
    *,
    use_fallbacks: bool = True,
) -> Path | None:
    candidates = [Path(env_file)]
    if use_fallbacks and env_file == DEFAULT_ENV_FILE:
        candidates.extend(Path(name) for name in ENV_FALLBACKS)
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else root / candidate
        # Preserve lexical symlink entries, including broken symlinks, so
        # security-sensitive callers can inspect and reject them instead of
        # silently treating an explicitly configured file as absent.
        if path.exists() or path.is_symlink():
            return path
    return None


def _parse_dotenv(content: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def load_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return _parse_dotenv(path.read_text())


def _load_private_dotenv(path: Path) -> dict[str, str]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise OSError("secure private environment reads require O_NOFOLLOW")
    descriptor = os.open(path, flags | nofollow)
    try:
        status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(status.st_mode)
            or stat.S_IMODE(status.st_mode) != 0o600
            or status.st_uid != os.geteuid()
        ):
            raise PermissionError(
                "eval private environment file must be owned by the current "
                "user, have mode 0600, and be a regular non-symlink file: "
                f"{path}"
            )
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            descriptor = -1
            return _parse_dotenv(stream.read())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def load_env(
    root: Path,
    env_file: str,
    *,
    use_fallbacks: bool = True,
    require_private: bool = False,
) -> tuple[dict[str, str], Path | None]:
    env = dict(os.environ)
    path = find_env_path(root, env_file, use_fallbacks=use_fallbacks)
    if path is None:
        return env, None
    values = _load_private_dotenv(path) if require_private else load_dotenv(path)
    for key, value in values.items():
        env.setdefault(key, value)
    return env, path


def env_value(env: dict[str, str], *names: str) -> str | None:
    for name in names:
        value = env.get(name)
        if value:
            return value
    return None


def pick(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default
