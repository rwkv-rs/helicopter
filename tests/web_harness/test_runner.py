import tempfile
import unittest
from pathlib import Path

from rwkv_web_harness.models import GenerationRequest
from rwkv_web_harness.runner import AgentConfig, AgentRunner
from rwkv_web_harness.tools import ToolResult
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

    def execute(self, action):
        return ToolResult(True, action.name, "ok", {"source_id": "source_001"})


class RunnerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
