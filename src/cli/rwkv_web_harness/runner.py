from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .context import PromptContext
from .models import GenerationBackend, GenerationRequest, ModelBackendError
from .protocol import FinalAnswer, parse_turn
from .tools import ToolResult, WebToolkit
from .trace import TraceWriter


DEFAULT_SYSTEM_PROMPT = """You are a careful web research agent powered by a local RWKV model.
Use the web tools to gather evidence before answering. Do not invent URLs,
facts, or citations. Search broadly, open the most useful sources, and use
find_in_page when a page is long. Keep each action small and valid.
"""


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 8
    max_context_chars: int = 24000
    max_new_tokens: int = 256
    temperature: float = 0.0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@dataclass(frozen=True)
class RunResult:
    task_id: str
    status: str
    answer: str | None
    citations: tuple[str, ...]
    steps: int
    prompt_chars: int
    context_truncated: bool
    error: str | None = None
    trace_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentRunner:
    def __init__(
        self,
        *,
        backend: GenerationBackend,
        toolkit: WebToolkit,
        config: AgentConfig | None = None,
        trace: TraceWriter | None = None,
    ) -> None:
        self.backend = backend
        self.toolkit = toolkit
        self.config = config or AgentConfig()
        self.trace = trace

    def run(self, *, task_id: str, question: str) -> RunResult:
        if not question.strip():
            raise ValueError("question must not be empty")
        if self.config.max_steps < 1:
            raise ValueError("max_steps must be positive")
        context = PromptContext(
            system_prompt=f"{self.config.system_prompt.strip()}\n\n{self.toolkit.tool_descriptions}",
            task=question.strip(),
        )
        self._write("run_started", task_id=task_id, question=question)
        prompt_chars = 0
        context_truncated = False
        last_error: str | None = None

        for step in range(1, self.config.max_steps + 1):
            prompt, truncated = context.render(self.config.max_context_chars)
            prompt_chars = len(prompt)
            context_truncated = context_truncated or truncated
            self._write(
                "prompt",
                task_id=task_id,
                step=step,
                prompt_chars=len(prompt),
                context_truncated=truncated,
            )
            try:
                raw_output = self.backend.generate(
                    GenerationRequest(
                        prompt=prompt,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                    )
                )
            except ModelBackendError as exc:
                last_error = str(exc)
                self._write("model_error", task_id=task_id, step=step, error=last_error)
                return self._finish(
                    task_id=task_id,
                    status="failed",
                    answer=None,
                    citations=(),
                    steps=step,
                    prompt_chars=prompt_chars,
                    context_truncated=context_truncated,
                    error=last_error,
                )

            self._write("model_output", task_id=task_id, step=step, output=raw_output)
            context.add(f"Assistant action:\n{raw_output}")
            try:
                parsed = parse_turn(raw_output)
                if parsed.error:
                    raise ValueError(parsed.error)
            except ValueError as exc:
                last_error = str(exc)
                context.add(f"Harness validation error:\n{last_error}")
                self._write("protocol_error", task_id=task_id, step=step, error=last_error)
                continue

            if parsed.final is not None:
                return self._finish(
                    task_id=task_id,
                    status="completed",
                    answer=parsed.final.answer,
                    citations=parsed.final.citations,
                    steps=step,
                    prompt_chars=prompt_chars,
                    context_truncated=context_truncated,
                )

            if parsed.action is None:
                last_error = "parsed turn contained neither an action nor a final answer"
                context.add(f"Harness validation error:\n{last_error}")
                continue
            self._write(
                "tool_call",
                task_id=task_id,
                step=step,
                tool=parsed.action.name,
                arguments=parsed.action.arguments,
            )
            result = self.toolkit.execute(parsed.action)
            observation = result.observation()
            context.add(f"Tool result ({parsed.action.name}):\n{observation}")
            self._write(
                "tool_result",
                task_id=task_id,
                step=step,
                tool=result.tool,
                ok=result.ok,
                message=result.message,
                data=result.data,
            )

        last_error = last_error or f"agent did not finish within {self.config.max_steps} steps"
        return self._finish(
            task_id=task_id,
            status="failed",
            answer=None,
            citations=(),
            steps=self.config.max_steps,
            prompt_chars=prompt_chars,
            context_truncated=context_truncated,
            error=last_error,
        )

    def _finish(
        self,
        *,
        task_id: str,
        status: str,
        answer: str | None,
        citations: tuple[str, ...],
        steps: int,
        prompt_chars: int,
        context_truncated: bool,
        error: str | None = None,
    ) -> RunResult:
        result = RunResult(
            task_id=task_id,
            status=status,
            answer=answer,
            citations=citations,
            steps=steps,
            prompt_chars=prompt_chars,
            context_truncated=context_truncated,
            error=error,
            trace_path=str(self.trace.path) if self.trace else None,
        )
        self._write("run_finished", **result.as_dict())
        return result

    def _write(self, event: str, **payload: Any) -> None:
        if self.trace is not None:
            self.trace.write(event, **payload)
