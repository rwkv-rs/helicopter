from __future__ import annotations

import ctypes
import errno
import os
import platform
import resource
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


_AUDIT_ARCH = {"x86_64": 0xC000003E, "aarch64": 0xC00000B7}
_SYSCALLS = {
    "x86_64": {
        "network": (
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            288,
            299,
            307,
        ),
        "host": (
            62,
            101,
            109,
            112,
            155,
            161,
            165,
            166,
            200,
            234,
            246,
            248,
            249,
            250,
            272,
            298,
            304,
            308,
            310,
            311,
            321,
            323,
            424,
            434,
            438,
        ),
    },
    "aarch64": {
        "network": (
            198,
            199,
            200,
            201,
            202,
            203,
            204,
            205,
            206,
            207,
            208,
            209,
            210,
            211,
            212,
            242,
            243,
            269,
        ),
        "host": (
            39,
            40,
            41,
            51,
            97,
            104,
            117,
            129,
            130,
            131,
            154,
            157,
            217,
            218,
            219,
            241,
            265,
            268,
            270,
            271,
            280,
            282,
            424,
            434,
            438,
        ),
    },
}

_LANDLOCK_CREATE_RULESET = 444
_LANDLOCK_ADD_RULE = 445
_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_SECCOMP = 22
_SECCOMP_MODE_FILTER = 2

_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_REFER = 1 << 13
_FS_TRUNCATE = 1 << 14
_FS_IOCTL_DEV = 1 << 15
_FS_BASE = sum(1 << bit for bit in range(13))
_FS_READ_ONLY = _FS_EXECUTE | _FS_READ_FILE | _FS_READ_DIR


class LinuxSandboxUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProcessLimits:
    cpu_seconds: int
    memory_bytes: int
    process_limit: int
    output_bytes: int


@dataclass(frozen=True, slots=True)
class IsolatedProcessResult:
    return_code: int | None
    timed_out: bool


class _LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _LandlockPathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ushort), ("filter", ctypes.POINTER(_SockFilter))]


def landlock_abi() -> int:
    if platform.system() != "Linux":
        raise LinuxSandboxUnavailable("coding sandbox requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(
        _LANDLOCK_CREATE_RULESET,
        ctypes.c_void_p(),
        ctypes.c_size_t(0),
        ctypes.c_uint(_LANDLOCK_CREATE_RULESET_VERSION),
    )
    if result < 0:
        error = ctypes.get_errno()
        raise LinuxSandboxUnavailable(f"Landlock is unavailable: {os.strerror(error)}")
    return int(result)


def run_isolated_python(
    submission: Path,
    *,
    stdin: bytes,
    stdout: BinaryIO,
    stderr: BinaryIO,
    wall_seconds: float,
    limits: ProcessLimits,
) -> IsolatedProcessResult:
    machine = platform.machine().lower()
    if machine not in _AUDIT_ARCH:
        raise LinuxSandboxUnavailable(
            f"coding sandbox does not support architecture: {machine}"
        )
    abi = landlock_abi()
    python = Path("/usr/bin/python3")
    if not python.is_file():
        raise LinuxSandboxUnavailable("coding sandbox requires /usr/bin/python3")

    try:
        process = subprocess.Popen(
            (str(python), "-I", "-S", "-B", str(submission)),
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            cwd=submission.parent,
            env={
                "HOME": str(submission.parent),
                "LANG": "C.UTF-8",
                "PATH": "/usr/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            close_fds=True,
            start_new_session=True,
            preexec_fn=_sandbox_preexec(submission.parent, abi, machine, limits),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise LinuxSandboxUnavailable(
            f"kernel sandbox initialization failed: {error}"
        ) from error

    try:
        process.communicate(input=stdin, timeout=wall_seconds)
        return IsolatedProcessResult(process.returncode, False)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return IsolatedProcessResult(None, True)


def _sandbox_preexec(
    work_directory: Path,
    abi: int,
    machine: str,
    limits: ProcessLimits,
):
    def apply() -> None:
        _restrict_filesystem(work_directory, abi)
        _set_limits(limits)
        _restrict_syscalls(machine)

    return apply


def _restrict_filesystem(work_directory: Path, abi: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    handled = _FS_BASE
    if abi >= 2:
        handled |= _FS_REFER
    if abi >= 3:
        handled |= _FS_TRUNCATE
    if abi >= 5:
        handled |= _FS_IOCTL_DEV
    attr = _LandlockRulesetAttr(handled)
    ruleset_fd = libc.syscall(
        _LANDLOCK_CREATE_RULESET,
        ctypes.byref(attr),
        ctypes.sizeof(attr),
        ctypes.c_uint(0),
    )
    if ruleset_fd < 0:
        _raise_errno("create Landlock ruleset")
    try:
        for path in (Path("/usr"), Path("/lib"), Path("/lib64")):
            if path.exists():
                _allow_path(libc, ruleset_fd, path, _FS_READ_ONLY)
        ld_cache = Path("/etc/ld.so.cache")
        if ld_cache.is_file():
            _allow_path(libc, ruleset_fd, ld_cache, _FS_READ_FILE)
        _allow_path(libc, ruleset_fd, work_directory, handled & ~_FS_EXECUTE)
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            _raise_errno("enable no_new_privs")
        if libc.syscall(_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0) != 0:
            _raise_errno("apply Landlock ruleset")
    finally:
        os.close(ruleset_fd)


def _allow_path(libc, ruleset_fd: int, path: Path, access: int) -> None:
    path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        rule = _LandlockPathBeneathAttr(access, path_fd, 0)
        if (
            libc.syscall(
                _LANDLOCK_ADD_RULE,
                ruleset_fd,
                _LANDLOCK_RULE_PATH_BENEATH,
                ctypes.byref(rule),
                0,
            )
            != 0
        ):
            _raise_errno(f"allow sandbox path {path}")
    finally:
        os.close(path_fd)


def _set_limits(limits: ProcessLimits) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (limits.memory_bytes, limits.memory_bytes))
    resource.setrlimit(
        resource.RLIMIT_NPROC, (limits.process_limit, limits.process_limit)
    )
    resource.setrlimit(
        resource.RLIMIT_FSIZE, (limits.output_bytes, limits.output_bytes)
    )
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _restrict_syscalls(machine: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    blocked = (*_SYSCALLS[machine]["network"], *_SYSCALLS[machine]["host"])
    instructions = [
        (0x20, 0, 0, 4),
        (0x15, 1, 0, _AUDIT_ARCH[machine]),
        (0x06, 0, 0, 0x80000000),
        (0x20, 0, 0, 0),
    ]
    for syscall_number in blocked:
        instructions.extend(
            (
                (0x15, 0, 1, syscall_number),
                (0x06, 0, 0, 0x00050000 | errno.EPERM),
            )
        )
    instructions.append((0x06, 0, 0, 0x7FFF0000))
    filters = (_SockFilter * len(instructions))(
        *(_SockFilter(*instruction) for instruction in instructions)
    )
    program = _SockFprog(len(filters), filters)
    if libc.prctl(_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER, ctypes.byref(program)) != 0:
        _raise_errno("apply seccomp filter")


def _raise_errno(action: str) -> None:
    error = ctypes.get_errno()
    raise OSError(error, f"failed to {action}: {os.strerror(error)}")
