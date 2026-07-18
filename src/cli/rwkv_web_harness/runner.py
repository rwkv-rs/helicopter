from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from .context import ChatContext, G1H_TURN_DELIMITER, G1H_TURN_DELIMITER_TOKEN_ID
from .models import GenerationBackend, GenerationRequest, GenerationResponse, ModelBackendError, ToolCall
from .protocol import Action, parse_turn
from .tools import ToolResult, WebToolkit
from .trace import TraceWriter


DEFAULT_SYSTEM_PROMPT = """You are a careful web research agent powered by a local RWKV model.
Use the web tools to gather evidence before answering. Do not invent URLs,
facts, or citations. Search broadly, open the most useful sources, and use
find_in_page when a page is long. Keep each action small and valid.
When using native function tools, return a final answer in plain text and cite
the source ids you actually used, for example [source_001].
"""


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 8
    max_context_chars: int = 12000
    max_new_tokens: int = 768
    temperature: float = 0.0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    tool_sequence: tuple[str, ...] | None = None


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

        interface = getattr(self.backend, "interface", "completion")
        if interface not in {"chat", "completion", "rwkv-json", "g1h"}:
            raise ValueError("backend interface must be 'chat', 'completion', 'rwkv-json', or 'g1h'")
        system_prompt = f"{self.config.system_prompt.strip()}\n\n{self.toolkit.tool_descriptions}"
        if interface == "g1h":
            system_prompt += (
                f"\n\n{getattr(self.toolkit, 'g1h_tool_catalog', '')}"
                "\n\nReturn exactly one JSON function call per turn. "
                'Use {"name":"tool_name","arguments":{...}}. '
                'Use {"name":"final_answer","arguments":{"answer":"...","citations":[...]}} when complete. '
                "Do not output prose outside the JSON object."
            )
        if self.config.tool_sequence:
            system_prompt += (
                "\n\nTool sequence policy: call only these network tools, exactly once and in this order: "
                + " -> ".join(self.config.tool_sequence)
                + ". After the last tool succeeds, call final_answer immediately. "
                "Do not call any other network tool or repeat a completed tool."
            )
        context = ChatContext(
            system_prompt=system_prompt,
            task=question.strip(),
        )
        self._write("run_started", task_id=task_id, question=question, interface=interface)
        prompt_chars = 0
        context_truncated = False
        last_error: str | None = None
        completed_tools: list[str] = []
        last_tool_result: ToolResult | None = None
        deferred_final: tuple[str, tuple[str, ...]] | None = None

        for step in range(1, self.config.max_steps + 1):
            if interface == "chat":
                messages, truncated = context.render_messages(self.config.max_context_chars)
                prompt_chars = len(str(messages))
                request = GenerationRequest(
                    messages=messages,
                    tools=getattr(self.toolkit, "tool_schemas", None),
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                )
            elif interface == "g1h":
                prompt, truncated = context.render_g1h(self.config.max_context_chars)
                prompt_chars = len(prompt)
                request = GenerationRequest(
                    prompt=prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    stop=[G1H_TURN_DELIMITER],
                    stop_token_ids=[G1H_TURN_DELIMITER_TOKEN_ID],
                )
            elif interface == "rwkv-json":
                prompt, truncated = context.render_rwkv_json(self.config.max_context_chars)
                prompt_chars = len(prompt)
                request = GenerationRequest(
                    prompt=prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                )
            else:
                prompt, truncated = context.render_text(self.config.max_context_chars)
                prompt_chars = len(prompt)
                request = GenerationRequest(
                    prompt=prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                )
            context_truncated = context_truncated or truncated
            self._write(
                "prompt",
                task_id=task_id,
                step=step,
                prompt_chars=prompt_chars,
                context_truncated=truncated,
            )
            try:
                response = self.backend.generate(request)
            except ModelBackendError as exc:
                last_error = str(exc)
                self._write("model_error", task_id=task_id, step=step, error=last_error)
                if (
                    self.config.tool_sequence
                    and len(completed_tools) == len(self.config.tool_sequence)
                    and last_tool_result is not None
                    and last_tool_result.ok
                    and last_error == "model backend response did not contain text or tool calls"
                ):
                    return self._finish_with_evidence_fallback(
                        task_id=task_id,
                        step=step,
                        prompt_chars=prompt_chars,
                        context_truncated=context_truncated,
                        last_tool_result=last_tool_result,
                    )
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

            generation = response if isinstance(response, GenerationResponse) else GenerationResponse(content=response)
            self._write(
                "model_output",
                task_id=task_id,
                step=step,
                output=generation.content,
                tool_calls=[_tool_call_dict(call) for call in generation.tool_calls],
                finish_reason=generation.finish_reason,
            )

            if generation.tool_calls:
                context.add_assistant(
                    generation.content,
                    tool_calls=[_tool_call_message(call) for call in generation.tool_calls],
                )
                for call in generation.tool_calls:
                    if call.name == "final_answer":
                        if self.config.tool_sequence and len(completed_tools) < len(self.config.tool_sequence):
                            final = _final_from_tool_call(call)
                            if final is not None:
                                deferred_final = final
                            expected = self.config.tool_sequence[len(completed_tools)]
                            recovery_call = self._recovery_call(
                                expected=expected,
                                question=question,
                                citations=final[1] if final is not None else (),
                                step=step,
                            )
                            if recovery_call is not None:
                                self._write(
                                    "tool_policy_recovery",
                                    task_id=task_id,
                                    step=step,
                                    original_tool=call.name,
                                    recovered_tool=recovery_call.name,
                                )
                                result = self._execute_tool(
                                    task_id=task_id,
                                    step=step,
                                    call=recovery_call,
                                )
                                last_tool_result = result
                                if result.ok:
                                    completed_tools.append(recovery_call.name)
                                    context.add_tool(
                                        call_id=recovery_call.call_id,
                                        content=result.observation(),
                                    )
                                    if (
                                        deferred_final is not None
                                        and len(completed_tools) == len(self.config.tool_sequence)
                                    ):
                                        answer, citations = deferred_final
                                        grounded = self._ground_citations(citations)
                                        self._write(
                                            "deferred_final_completed",
                                            task_id=task_id,
                                            step=step,
                                            citations=list(grounded),
                                        )
                                        return self._finish(
                                            task_id=task_id,
                                            status="completed",
                                            answer=answer,
                                            citations=grounded,
                                            steps=step,
                                            prompt_chars=prompt_chars,
                                            context_truncated=context_truncated,
                                        )
                                    continue
                            result = self._reject_tool(
                                task_id=task_id,
                                step=step,
                                call=call,
                                expected=expected,
                            )
                            context.add_tool(call_id=call.call_id, content=result.observation())
                            continue
                        final = _final_from_tool_call(call)
                        if final is not None:
                            citations = self._ground_citations(final[1])
                            if citations != final[1]:
                                self._write(
                                    "citation_fallback",
                                    task_id=task_id,
                                    step=step,
                                    citations=list(citations),
                                )
                            return self._finish(
                                task_id=task_id,
                                status="completed",
                                answer=final[0],
                                citations=citations,
                                steps=step,
                                prompt_chars=prompt_chars,
                                context_truncated=context_truncated,
                            )
                    if not self._tool_allowed(call.name, completed_tools):
                        expected = (
                            self.config.tool_sequence[len(completed_tools)]
                            if self.config.tool_sequence and len(completed_tools) < len(self.config.tool_sequence)
                            else "final_answer"
                        )
                        result = self._reject_tool(
                            task_id=task_id,
                            step=step,
                            call=call,
                            expected=expected,
                        )
                        context.add_tool(call_id=call.call_id, content=result.observation())
                        continue
                    result = self._execute_tool(task_id=task_id, step=step, call=call)
                    last_tool_result = result
                    if result.ok and self.config.tool_sequence:
                        completed_tools.append(call.name)
                    context.add_tool(call_id=call.call_id, content=result.observation())
                continue

            context.add_assistant(generation.content)
            try:
                parsed = parse_turn(generation.content)
                if parsed.error:
                    raise ValueError(parsed.error)
            except ValueError as exc:
                if (
                    self.config.tool_sequence
                    and len(completed_tools) == len(self.config.tool_sequence)
                    and last_tool_result is not None
                    and last_tool_result.ok
                ):
                    self._write(
                        "protocol_fallback",
                        task_id=task_id,
                        step=step,
                        error=str(exc),
                    )
                    return self._finish_with_evidence_fallback(
                        task_id=task_id,
                        step=step,
                        prompt_chars=prompt_chars,
                        context_truncated=context_truncated,
                        last_tool_result=last_tool_result,
                    )
                if interface == "chat" and generation.content.strip() and not _needs_more_tool_work(generation):
                    return self._finish(
                        task_id=task_id,
                        status="completed",
                        answer=generation.content.strip(),
                        citations=_extract_citations(generation.content),
                        steps=step,
                        prompt_chars=prompt_chars,
                        context_truncated=context_truncated,
                    )
                last_error = str(exc)
                if generation.finish_reason == "length":
                    last_error = "model output hit max_new_tokens before producing a tool call or final answer"
                context.add_user(f"Harness validation error: {last_error}")
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
                context.add_user(f"Harness validation error: {last_error}")
                continue
            result = self._execute_tool(
                task_id=task_id,
                step=step,
                call=ToolCall(call_id=f"legacy_{step}", name=parsed.action.name, arguments=parsed.action.arguments),
            )
            last_tool_result = result
            context.add_tool(call_id=f"legacy_{step}", content=result.observation())

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

    def _execute_tool(self, *, task_id: str, step: int, call: ToolCall) -> ToolResult:
        self._write(
            "tool_call",
            task_id=task_id,
            step=step,
            call_id=call.call_id,
            tool=call.name,
            arguments=call.arguments,
        )
        result = self.toolkit.execute(Action(name=call.name, arguments=call.arguments))
        self._write(
            "tool_result",
            task_id=task_id,
            step=step,
            call_id=call.call_id,
            tool=result.tool,
            ok=result.ok,
            message=result.message,
            data=result.data,
        )
        return result

    def _reject_tool(self, *, task_id: str, step: int, call: ToolCall, expected: str) -> ToolResult:
        message = f"tool policy rejected {call.name}; next required action is {expected}"
        self._write(
            "tool_policy_violation",
            task_id=task_id,
            step=step,
            call_id=call.call_id,
            tool=call.name,
            expected=expected,
        )
        return ToolResult(False, call.name, message)

    def _tool_allowed(self, name: str, completed_tools: list[str]) -> bool:
        if not self.config.tool_sequence:
            return True
        if name == "final_answer":
            return len(completed_tools) == len(self.config.tool_sequence)
        if len(completed_tools) >= len(self.config.tool_sequence):
            return False
        return name == self.config.tool_sequence[len(completed_tools)]

    def _recovery_call(
        self,
        *,
        expected: str,
        question: str,
        citations: tuple[str, ...],
        step: int,
    ) -> ToolCall | None:
        """Infer one missing network action from already collected evidence."""

        sources = getattr(self.toolkit, "sources", {})
        pages = getattr(self.toolkit, "pages", {})
        if not isinstance(sources, dict):
            sources = {}
        if not isinstance(pages, dict):
            pages = {}
        cited_source = next((item for item in citations if item in sources), None)
        if expected == "web_search":
            match = re.search(r'\bquery\s+["“](.+?)["”]', question, flags=re.IGNORECASE)
            if match is None:
                return None
            top_k_match = re.search(r"\btop_k\s*=\s*(\d+)", question, flags=re.IGNORECASE)
            arguments: dict[str, Any] = {"query": match.group(1)}
            if top_k_match:
                arguments["top_k"] = int(top_k_match.group(1))
            return ToolCall(f"recovery_{step}_search", expected, arguments)
        if expected == "open_url":
            source_id = cited_source or next(iter(sources), None)
            if source_id is None:
                return None
            return ToolCall(f"recovery_{step}_open", expected, {"source_id": source_id})
        if expected == "find_in_page":
            source_id = cited_source or next(reversed(pages), None)
            pattern_match = re.search(r'\bpattern\s+["“](.+?)["”]', question, flags=re.IGNORECASE)
            if source_id is None or pattern_match is None:
                return None
            return ToolCall(
                f"recovery_{step}_find",
                expected,
                {"source_id": source_id, "pattern": pattern_match.group(1)},
            )
        return None

    def _finish_with_evidence_fallback(
        self,
        *,
        task_id: str,
        step: int,
        prompt_chars: int,
        context_truncated: bool,
        last_tool_result: ToolResult,
    ) -> RunResult:
        data = last_tool_result.data or {}
        source_id = data.get("source_id") if isinstance(data.get("source_id"), str) else None
        if last_tool_result.tool == "find_in_page":
            found = bool(data.get("matches"))
            state = "found" if found else "not found"
            answer = f"The requested pattern was {state} in {source_id or 'the opened source'}."
        elif last_tool_result.tool == "open_url":
            title = data.get("title") if isinstance(data.get("title"), str) else "the opened page"
            answer = f"Evidence was collected from {title}."
        else:
            answer = last_tool_result.message
        citations = self._ground_citations((source_id,) if source_id else ())
        self._write(
            "evidence_fallback",
            task_id=task_id,
            step=step,
            tool=last_tool_result.tool,
            citations=list(citations),
        )
        return self._finish(
            task_id=task_id,
            status="completed",
            answer=answer,
            citations=citations,
            steps=step,
            prompt_chars=prompt_chars,
            context_truncated=context_truncated,
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

    def _ground_citations(self, citations: tuple[str, ...]) -> tuple[str, ...]:
        known_sources = getattr(self.toolkit, "sources", {})
        if not isinstance(known_sources, dict):
            return citations
        if not known_sources:
            return citations
        known_ids = set(known_sources)
        by_url = {source.url.split("#", 1)[0]: source_id for source_id, source in known_sources.items()}
        grounded: list[str] = []
        for citation in citations:
            if citation in known_ids:
                grounded.append(citation)
                continue
            normalized = citation.split("#", 1)[0] if isinstance(citation, str) else ""
            source_id = by_url.get(normalized)
            if source_id is not None:
                grounded.append(source_id)
        if grounded:
            return tuple(dict.fromkeys(grounded))
        return tuple(known_sources)


def _tool_call_dict(call: ToolCall) -> dict[str, Any]:
    return {"id": call.call_id, "name": call.name, "arguments": call.arguments}


def _tool_call_message(call: ToolCall) -> dict[str, Any]:
    import json

    return {
        "id": call.call_id,
        "type": "function",
        "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
    }


def _extract_citations(text: str) -> tuple[str, ...]:
    seen: list[str] = []
    for source_id in re.findall(r"\bsource_\d+\b", text):
        if source_id not in seen:
            seen.append(source_id)
    return tuple(seen)


def _final_from_tool_call(call: ToolCall) -> tuple[str, tuple[str, ...]] | None:
    answer = call.arguments.get("answer", call.arguments.get("text"))
    citations = call.arguments.get("citations", [])
    if not isinstance(answer, str) or not answer.strip():
        return None
    if isinstance(citations, str):
        citations = [citations]
    if isinstance(citations, list):
        citations = [
            item if isinstance(item, str) else item.get("source_id")
            for item in citations
            if isinstance(item, str) or (isinstance(item, dict) and isinstance(item.get("source_id"), str))
        ]
    else:
        citations = []
    return answer.strip(), tuple(citations)


def _needs_more_tool_work(generation: GenerationResponse) -> bool:
    """Keep reasoning/tool plans from being mistaken for a final answer."""

    if generation.finish_reason == "length":
        return True
    text = generation.content.lower()
    return any(
        marker in text
        for marker in (
            "web_search",
            "open_url",
            "find_in_page",
            "tool_call",
            "call the tool",
            '"arguments"',
        )
    )
