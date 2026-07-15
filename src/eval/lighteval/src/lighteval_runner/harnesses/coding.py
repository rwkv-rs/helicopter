from __future__ import annotations

import signal
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .linux_sandbox import (
    LinuxSandboxUnavailable,
    ProcessLimits,
    run_isolated_python,
)


SANDBOX_REVISION = "landlock-seccomp-python-v3"


class SandboxUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    wall_seconds: float = 3.0
    cpu_seconds: int = 2
    memory_bytes: int = 256 * 1024 * 1024
    process_limit: int = 32
    output_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        if (
            min(
                self.wall_seconds,
                self.cpu_seconds,
                self.memory_bytes,
                self.process_limit,
                self.output_bytes,
            )
            <= 0
        ):
            raise ValueError("sandbox limits must be positive")


@dataclass(frozen=True, slots=True)
class SandboxResult:
    revision: str
    return_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    output_limit_exceeded: bool


def run_python_submission(
    source: str,
    *,
    stdin: str,
    policy: SandboxPolicy = SandboxPolicy(),
) -> SandboxResult:
    with tempfile.TemporaryDirectory(prefix="helicopter-coding-") as directory:
        submission = Path(directory) / "submission.py"
        submission.write_text(source, encoding="utf-8")
        with (
            tempfile.TemporaryFile(mode="w+b") as stdout_file,
            tempfile.TemporaryFile(mode="w+b") as stderr_file,
        ):
            try:
                completed = run_isolated_python(
                    submission,
                    stdin=stdin.encode(),
                    stdout=stdout_file,
                    stderr=stderr_file,
                    wall_seconds=policy.wall_seconds,
                    limits=ProcessLimits(
                        cpu_seconds=policy.cpu_seconds,
                        memory_bytes=policy.memory_bytes,
                        process_limit=policy.process_limit,
                        output_bytes=policy.output_bytes,
                    ),
                )
            except LinuxSandboxUnavailable as error:
                raise SandboxUnavailable(str(error)) from error
            stdout = _read_output(stdout_file, policy.output_bytes)
            stderr = _read_output(stderr_file, policy.output_bytes)
            output_limit_exceeded = (
                stdout_file.tell() >= policy.output_bytes
                or stderr_file.tell() >= policy.output_bytes
                or completed.return_code == -signal.SIGXFSZ
            )
            return SandboxResult(
                revision=SANDBOX_REVISION,
                return_code=completed.return_code,
                stdout=stdout,
                stderr=stderr,
                timed_out=completed.timed_out,
                output_limit_exceeded=output_limit_exceeded,
            )


def _read_output(stream, limit: int) -> str:
    size = stream.tell()
    stream.seek(0)
    payload = stream.read(min(size, limit))
    return payload.decode("utf-8", errors="replace")
