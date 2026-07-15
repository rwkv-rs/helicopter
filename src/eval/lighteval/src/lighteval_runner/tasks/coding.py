from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..harnesses.coding import SandboxPolicy, SandboxResult, run_python_submission
from ..harnesses.coding import SandboxUnavailable
from ..context import ContextDocument, ContextSection
from ..task_runtime import HarnessFailure, PreparedSample


@dataclass(frozen=True, slots=True)
class CodingCase:
    stdin: str
    expected_stdout: str


@dataclass(frozen=True, slots=True)
class CodingScoringInput:
    raw_completion: str
    cases: tuple[CodingCase, ...]


def score_python_submission(
    scoring_input: CodingScoringInput,
    *,
    policy: SandboxPolicy = SandboxPolicy(),
) -> tuple[float, tuple[SandboxResult, ...]]:
    results = tuple(
        run_python_submission(
            scoring_input.raw_completion, stdin=case.stdin, policy=policy
        )
        for case in scoring_input.cases
    )
    passed = sum(
        result.return_code == 0
        and not result.timed_out
        and not result.output_limit_exceeded
        and result.stdout.strip() == case.expected_stdout.strip()
        for result, case in zip(results, scoring_input.cases, strict=True)
    )
    return (passed / len(scoring_input.cases) if scoring_input.cases else 0.0), results


class CodingRuntime:
    def prepare(self, row: Mapping[str, Any]) -> PreparedSample:
        prompt = row.get("prompt")
        raw_cases = row.get("cases")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("coding prompt must be a non-empty string")
        if not isinstance(raw_cases, (list, tuple)) or not raw_cases:
            raise ValueError("coding task must define at least one case")
        cases: list[CodingCase] = []
        for item in raw_cases:
            if not isinstance(item, Mapping):
                raise ValueError("coding case must be an object")
            stdin = item.get("stdin")
            expected_stdout = item.get("expected_stdout")
            if not isinstance(stdin, str) or not isinstance(expected_stdout, str):
                raise ValueError(
                    "coding case stdin and expected_stdout must be strings"
                )
            cases.append(CodingCase(stdin, expected_stdout))
        return PreparedSample(
            context=ContextDocument((ContextSection("query", "user", prompt, 100),)),
            scoring_state=tuple(cases),
        )

    def score(
        self,
        sample: PreparedSample,
        *,
        prompt: str,
        completion: str,
        output_token_ids: tuple[int, ...],
    ) -> Mapping[str, float]:
        del prompt, output_token_ids
        cases = sample.scoring_state
        if not isinstance(cases, tuple) or not all(
            isinstance(case, CodingCase) for case in cases
        ):
            raise RuntimeError("coding runtime received an invalid scoring state")
        try:
            score, _ = score_python_submission(CodingScoringInput(completion, cases))
        except SandboxUnavailable as error:
            raise HarnessFailure(str(error)) from error
        return {"sandbox_pass_rate": score}
