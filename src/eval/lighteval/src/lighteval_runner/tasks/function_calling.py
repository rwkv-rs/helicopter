from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from ..context import ContextDocument, ContextSection
from ..task_runtime import ModelOutputRejected, PreparedSample


class InvalidFunctionCall(ValueError):
    """Model output is not a valid signed function-call value."""


@dataclass(frozen=True, slots=True)
class FunctionCall:
    name: str
    arguments_json: str

    @property
    def arguments(self) -> dict[str, Any]:
        value = json.loads(self.arguments_json)
        assert isinstance(value, dict)
        return value


def parse_function_call(payload: Any) -> FunctionCall:
    if not isinstance(payload, Mapping):
        raise InvalidFunctionCall("function call must be an object")
    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(name, str) or not name.strip():
        raise InvalidFunctionCall("function call name must be a non-empty string")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise InvalidFunctionCall(
                "function call arguments are invalid JSON"
            ) from error
    if not isinstance(arguments, Mapping):
        raise InvalidFunctionCall("function call arguments must be an object")
    canonical = json.dumps(
        dict(arguments), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return FunctionCall(name=name, arguments_json=canonical)


def exact_function_call_match(actual: FunctionCall, expected: FunctionCall) -> bool:
    return (
        actual.name == expected.name
        and actual.arguments_json == expected.arguments_json
    )


class FunctionCallingRuntime:
    def prepare(self, row: Mapping[str, Any]) -> PreparedSample:
        prompt = row.get("prompt")
        tool = row.get("tool")
        expected = row.get("expected")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("function-calling prompt must be a non-empty string")
        if not isinstance(tool, Mapping):
            raise ValueError("function-calling tool contract must be an object")
        try:
            expected_call = parse_function_call(expected)
        except InvalidFunctionCall as error:
            raise ValueError(f"invalid function-calling reference: {error}") from error
        tool_json = json.dumps(
            _plain_value(tool),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        context = ContextDocument(
            (
                ContextSection(
                    "tool-contract",
                    "system",
                    f"Available tool contract: {tool_json}",
                    100,
                ),
                ContextSection("query", "user", prompt, 100),
            )
        )
        return PreparedSample(context=context, scoring_state=expected_call)

    def score(
        self,
        sample: PreparedSample,
        *,
        prompt: str,
        completion: str,
        output_token_ids: tuple[int, ...],
    ) -> Mapping[str, float]:
        del prompt, output_token_ids
        try:
            payload = json.loads(completion)
            actual = parse_function_call(payload)
        except (json.JSONDecodeError, InvalidFunctionCall) as error:
            raise ModelOutputRejected(str(error)) from error
        expected = sample.scoring_state
        if not isinstance(expected, FunctionCall):
            raise RuntimeError(
                "function-calling runtime received an invalid scoring state"
            )
        return {
            "exact_function_call": float(exact_function_call_match(actual, expected))
        }


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value
