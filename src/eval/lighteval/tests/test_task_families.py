from __future__ import annotations

import pytest

from lighteval_runner.harnesses.coding import (
    SandboxPolicy,
    SandboxResult,
    SandboxUnavailable,
    run_python_submission,
)
from lighteval_runner.tasks.coding import (
    CodingCase,
    CodingScoringInput,
    score_python_submission,
)
from lighteval_runner.tasks.function_calling import (
    InvalidFunctionCall,
    exact_function_call_match,
    parse_function_call,
)


def test_function_call_arguments_are_canonical_and_invalid_json_is_not_empty_object():
    actual = parse_function_call({"name": "search", "arguments": '{"b":2,"a":1}'})
    expected = parse_function_call({"name": "search", "arguments": {"a": 1, "b": 2}})
    assert exact_function_call_match(actual, expected)
    with pytest.raises(InvalidFunctionCall):
        parse_function_call({"name": "search", "arguments": "not-json"})


def test_coding_submission_scoring_consumes_only_sandbox_result(monkeypatch):
    monkeypatch.setattr(
        "lighteval_runner.tasks.coding.run_python_submission",
        lambda *_args, **_kwargs: SandboxResult(
            revision="landlock-seccomp-python-v3",
            return_code=0,
            stdout="42\n",
            stderr="",
            timed_out=False,
            output_limit_exceeded=False,
        ),
    )
    source = "value = int(input())\nprint(value * 2)"
    score, results = score_python_submission(
        CodingScoringInput(source, (CodingCase("21\n", "42\n"),)),
        policy=SandboxPolicy(wall_seconds=2.0),
    )
    assert score == 1.0
    assert results[0].revision == "landlock-seccomp-python-v3"


def test_coding_sandbox_blocks_network_access():
    source = "import socket\nsocket.create_connection(('127.0.0.1', 1), timeout=0.1)"
    try:
        result = run_python_submission(
            source, stdin="", policy=SandboxPolicy(wall_seconds=2.0)
        )
    except SandboxUnavailable as error:
        pytest.skip(str(error))
    else:
        assert result.return_code != 0
        assert result.timed_out is False


def test_coding_sandbox_allows_valid_submission():
    result = _run_in_available_sandbox("print(int(input()) * 2)", stdin="21\n")
    assert result.return_code == 0
    assert result.stdout == "42\n"


@pytest.mark.parametrize(
    "source",
    (
        "open('/etc/passwd').read()",
        "open('/tmp/helicopter-sandbox-escape', 'w').write('escape')",
    ),
)
def test_coding_sandbox_blocks_filesystem_escape(source):
    result = _run_in_available_sandbox(source)
    assert result.return_code != 0


def test_coding_sandbox_bounds_captured_output():
    result = _run_in_available_sandbox(
        "print('x' * 4096)", policy=SandboxPolicy(output_bytes=1024)
    )
    assert result.output_limit_exceeded is True
    assert len(result.stdout.encode()) <= 1024


def test_coding_sandbox_enforces_wall_timeout():
    result = _run_in_available_sandbox(
        "while True: pass",
        policy=SandboxPolicy(wall_seconds=0.1, cpu_seconds=2),
    )
    assert result.timed_out is True


def _run_in_available_sandbox(source, *, stdin="", policy=SandboxPolicy()):
    try:
        return run_python_submission(source, stdin=stdin, policy=policy)
    except SandboxUnavailable as error:
        pytest.skip(str(error))
