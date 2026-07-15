from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BundledResource:
    relative_path: str
    sha256: str

    @property
    def manifest_digest(self) -> str:
        payload = f"bundled-resource-v1\0{self.relative_path}\0{self.sha256}".encode()
        return hashlib.sha256(payload).hexdigest()

    def resolve(self) -> Path:
        resource = files("lighteval_runner").joinpath(self.relative_path)
        path = Path(str(resource))
        if not path.is_file():
            raise ValueError(
                f"bundled evaluation resource is missing: {self.relative_path}"
            )
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != self.sha256:
            raise ValueError(
                f"bundled evaluation resource digest mismatch: {self.relative_path}"
            )
        return path
