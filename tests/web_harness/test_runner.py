import tempfile
import unittest
from pathlib import Path

from rwkv_web_harness.models import GenerationRequest, GenerationResponse, ModelBackendError, ToolCall
from rwkv_web_harness.runner import AgentConfig, AgentRunner
from rwkv_web_harness.tools import Page, Source, ToolResult
from rwkv_web_harness.trace import TraceWriter


class ScriptedBackend:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> str:
        self.requests.append(request)
        return next(self.outputs)


class FakeToolkit:
    tool_descriptions = "Available tools: web_search"
    tool_schemas = [{"type": "function", "function": {"name": "web_search"}}]

    def execute(self, action):
        return ToolResult(True, action.name, "ok", {"source_id": "source_001"})


class RunnerTests(unittest.TestCase):
    def test_native_chat_tool_call_then_plain_final(self) -> None:
        class NativeBackend:
            interface = "chat"

            def __init__(self) -> None:
                self.requests: list[GenerationRequest] = []
                self.outputs = iter(
                    [
                        GenerationResponse(
                            content="",
                            tool_calls=(
                                ToolCall(
                                    call_id="call_1",
                                    name="web_search",
                                    arguments={"query": "RWKV"},
                                ),
                            ),
                        ),
                        GenerationResponse(content="RWKV is cited in [source_001]."),
                    ]
                )

            def generate(self, request: GenerationRequest) -> GenerationResponse:
                self.requests.append(request)
                return next(self.outputs)

        backend = NativeBackend()
        result = AgentRunner(
            backend=backend,
            toolkit=FakeToolkit(),
            config=AgentConfig(max_steps=3),
        ).run(task_id="native", question="What is RWKV?")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.citations, ("source_001",))
        self.assertEqual(len(backend.requests), 2)
        self.assertEqual(backend.requests[0].messages[0]["role"], "system")
        self.assertEqual(backend.requests[0].tools, FakeToolkit.tool_schemas)
        self.assertEqual(backend.requests[1].messages[-1]["role"], "tool")

    def test_g1h_interface_uses_diamond_context_and_stop_token(self) -> None:
        class G1HBackend:
            interface = "g1h"

            def __init__(self) -> None:
                self.requests: list[GenerationRequest] = []
                self.outputs = iter(
                    [
                        GenerationResponse(
                            content="",
                            tool_calls=(ToolCall("call_1", "web_search", {"query": "RWKV"}),),
                        ),
                        GenerationResponse(
                            content="",
                            tool_calls=(
                                ToolCall(
                                    call_id="call_2",
                                    name="final_answer",
                                    arguments={"answer": "RWKV", "citations": ["source_001"]},
                                ),
                            ),
                        ),
                    ]
                )

            def generate(self, request: GenerationRequest) -> GenerationResponse:
                self.requests.append(request)
                return next(self.outputs)

        backend = G1HBackend()
        result = AgentRunner(backend=backend, toolkit=FakeToolkit(), config=AgentConfig(max_steps=2)).run(
            task_id="g1h", question="What is RWKV?"
        )
        self.assertEqual(result.status, "completed")
        self.assertIn("User✿", backend.requests[0].prompt)
        self.assertIn("Bot✿<think></think>", backend.requests[0].prompt)
        self.assertEqual(backend.requests[0].stop, ["✿"])
        self.assertEqual(backend.requests[0].stop_token_ids, [10060])

    def test_tool_loop_then_final_answer_writes_trace(self) -> None:
        backend = ScriptedBackend(
            [
                '<tool_call>{"name":"web_search","arguments":{"query":"RWKV"}}</tool_call>',
                '<final_answer>{"answer":"RWKV","citations":["source_001"]}</final_answer>',
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.jsonl"
            with TraceWriter(trace_path) as trace:
                result = AgentRunner(
                    backend=backend,
                    toolkit=FakeToolkit(),
                    config=AgentConfig(max_steps=3),
                    trace=trace,
                ).run(task_id="test", question="What is RWKV?")
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.answer, "RWKV")
            self.assertEqual(result.citations, ("source_001",))
            self.assertEqual(len(backend.requests), 2)
            self.assertIn("run_finished", trace_path.read_text(encoding="utf-8"))

    def test_runner_fails_after_step_budget(self) -> None:
        backend = ScriptedBackend(['<tool_call>{"name":"web_search","arguments":{"query":"RWKV"}}</tool_call>'] * 2)
        result = AgentRunner(
            backend=backend,
            toolkit=FakeToolkit(),
            config=AgentConfig(max_steps=2),
        ).run(task_id="test", question="What is RWKV?")
        self.assertEqual(result.status, "failed")
        self.assertIn("within 2 steps", result.error)

    def test_tool_sequence_blocks_unexpected_network_tools(self) -> None:
        class PolicyBackend:
            interface = "g1h"

            def __init__(self) -> None:
                self.outputs = iter(
                    [
                        GenerationResponse(
                            content="",
                            tool_calls=(ToolCall("search", "web_search", {"query": "RWKV"}),),
                        ),
                        GenerationResponse(
                            content="",
                            tool_calls=(ToolCall("unexpected", "find_in_page", {"source_id": "source_001", "pattern": "RWKV"}),),
                        ),
                        GenerationResponse(
                            content="",
                            tool_calls=(
                                ToolCall(
                                    "final",
                                    "final_answer",
                                    {"answer": "done", "citations": ["source_001"]},
                                ),
                            ),
                        ),
                    ]
                )

            def generate(self, request: GenerationRequest) -> GenerationResponse:
                return next(self.outputs)

        result = AgentRunner(
            backend=PolicyBackend(),
            toolkit=FakeToolkit(),
            config=AgentConfig(max_steps=3, tool_sequence=("web_search",)),
        ).run(task_id="policy", question="What is RWKV?")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, "done")

    def test_tool_sequence_recovers_a_premature_final_answer(self) -> None:
        class RecoveryToolkit:
            tool_descriptions = "Available tools: web_search, open_url, find_in_page"
            tool_schemas = []
            sources = {"source_001": Source("source_001", "Example", "https://example.com")}
            pages = {"source_001": Page("source_001", "Example", "https://example.com", "Python evidence")}

            def execute(self, action):
                return ToolResult(
                    True,
                    action.name,
                    "ok",
                    {"source_id": "source_001", "matches": ["Python evidence"]},
                )

        class RecoveryBackend:
            interface = "g1h"

            def __init__(self) -> None:
                self.outputs = iter(
                    [
                        GenerationResponse(
                            content="",
                            tool_calls=(ToolCall("search", "web_search", {"query": "Python"}),),
                        ),
                        GenerationResponse(
                            content="",
                            tool_calls=(ToolCall("open", "open_url", {"source_id": "source_001"}),),
                        ),
                        GenerationResponse(
                            content="",
                            tool_calls=(
                                ToolCall(
                                    "final",
                                    "final_answer",
                                    {"answer": "Python evidence", "citations": ["source_001"]},
                                ),
                            ),
                        ),
                    ]
                )

            def generate(self, request: GenerationRequest) -> GenerationResponse:
                return next(self.outputs)

        result = AgentRunner(
            backend=RecoveryBackend(),
            toolkit=RecoveryToolkit(),
            config=AgentConfig(max_steps=3, tool_sequence=("web_search", "open_url", "find_in_page")),
        ).run(
            task_id="recovery",
            question='Use web_search, then open_url, then find_in_page using the pattern "Python".',
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.citations, ("source_001",))

    def test_tool_sequence_uses_evidence_fallback_after_empty_model_response(self) -> None:
        class FallbackBackend:
            interface = "g1h"

            def __init__(self) -> None:
                self.outputs = iter(
                    [
                        GenerationResponse(content="", tool_calls=(ToolCall("search", "web_search", {}),)),
                        GenerationResponse(content="", tool_calls=(ToolCall("open", "open_url", {}),)),
                        GenerationResponse(content="", tool_calls=(ToolCall("find", "find_in_page", {}),)),
                    ]
                )

            def generate(self, request: GenerationRequest) -> GenerationResponse:
                try:
                    return next(self.outputs)
                except StopIteration as exc:
                    raise ModelBackendError("model backend response did not contain text or tool calls") from exc

        result = AgentRunner(
            backend=FallbackBackend(),
            toolkit=FakeToolkit(),
            config=AgentConfig(max_steps=4, tool_sequence=("web_search", "open_url", "find_in_page")),
        ).run(task_id="fallback", question="What is RWKV?")
        self.assertEqual(result.status, "completed")


if __name__ == "__main__":
    unittest.main()
