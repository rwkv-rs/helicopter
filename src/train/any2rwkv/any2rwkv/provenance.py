from __future__ import annotations

from urllib.parse import urlsplit

_ASCII_WHITESPACE = " \t\n\r\v\f"


def canonical_github_repository(value: object) -> str:
    """Canonicalize one strict ASCII GitHub owner/repository URL."""
    if not isinstance(value, str) or not value or not value.isascii():
        raise ValueError("GitHub repository URL must be non-empty ASCII text")
    if value != value.strip(_ASCII_WHITESPACE) or any(
        ord(character) < 0x20 or ord(character) == 0x7F for character in value
    ):
        raise ValueError(
            "GitHub repository URL contains ASCII whitespace or control text"
        )
    raw = value
    if raw.startswith("git+"):
        raw = raw.removeprefix("git+")
    if "git+" in raw or "%" in raw or "\\" in raw:
        raise ValueError("GitHub repository URL contains a forbidden encoding")
    parsed = urlsplit(raw)
    if parsed.scheme.lower() != "https" or parsed.hostname is None:
        raise ValueError("GitHub repository URL must use HTTPS")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("GitHub repository URL contains an invalid port") from error
    if (
        parsed.hostname.lower() != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("GitHub repository URL is not an unqualified github.com URL")
    path = parsed.path
    if "//" in path:
        raise ValueError("GitHub repository URL contains an empty path segment")
    path = path.removesuffix("/")
    parts = path.split("/")
    if len(parts) != 3 or parts[0] or not parts[1] or not parts[2]:
        raise ValueError("GitHub repository URL must identify exactly owner/repository")
    repository = parts[2]
    if repository.lower().endswith(".git"):
        repository = repository[:-4]
    if not repository or repository.lower().endswith(".git"):
        raise ValueError("GitHub repository URL contains an invalid .git suffix")
    owner = parts[1]
    allowed = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    )
    if any(character not in allowed for character in owner + repository):
        raise ValueError("GitHub owner/repository contains a forbidden character")
    return f"https://github.com/{owner.lower()}/{repository.lower()}"


def github_repository_matches(actual: object, expected: str) -> bool:
    """Return false for malformed or foreign repositories."""
    try:
        return canonical_github_repository(actual) == canonical_github_repository(
            expected
        )
    except ValueError:
        return False


__all__ = ["canonical_github_repository", "github_repository_matches"]
