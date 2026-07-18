import json
import tempfile
import unittest
from pathlib import Path

from rwkv_web_harness.batch import load_task_specs, summarize_cases, validate_case


class BatchTests(unittest.TestCase):
    def test_checked_in_suite_contains_100_unique_tasks(self) -> None:
        root = Path(__file__).resolve().parents[2]
        specs = load_task_specs(root / "configs" / "web_harness_tasks_100.jsonl")
        self.assertEqual(len(specs), 100)
        self.assertEqual(len({spec.task_id for spec in specs}), 100)
        self.assertEqual(specs[0].expected_tools, ("web_search",))
        self.assertEqual(specs[-1].expected_tools, ("web_search", "open_url", "find_in_page"))

    def test_validation_requires_successful_network_payload_and_citation(self) -> None:
        spec = load_task_specs_from_rows(
            [
                {
                    "task_id": "case",
                    "question": "search and open",
                    "expected_tools": ["web_search", "open_url"],
                }
            ]
        )[0]
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.jsonl"
            events = [
                {"event": "tool_call", "tool": "web_search"},
                {
                    "event": "tool_result",
                    "tool": "web_search",
                    "ok": True,
                    "data": {"query": "RWKV", "results": [{"source_id": "source_001"}]},
                },
                {"event": "tool_call", "tool": "open_url"},
                {
                    "event": "tool_result",
                    "tool": "open_url",
                    "ok": True,
                    "data": {"source_id": "source_001", "url": "https://example.com", "content": "evidence"},
                },
            ]
            trace_path.write_text(
                "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
                encoding="utf-8",
            )
            validation = validate_case(
                {
                    "status": "completed",
                    "citations": ["source_001"],
                },
                spec,
                trace_path,
            )
        self.assertTrue(validation["passed"])
        self.assertEqual(validation["network_evidence_count"], 2)

    def test_validation_rejects_completed_without_trace_evidence(self) -> None:
        spec = load_task_specs_from_rows(
            [{"task_id": "case", "question": "search", "expected_tools": ["web_search"]}]
        )[0]
        with tempfile.TemporaryDirectory() as directory:
            validation = validate_case(
                {"status": "completed", "citations": ["source_001"]},
                spec,
                Path(directory) / "missing.jsonl",
            )
        self.assertFalse(validation["passed"])
        self.assertFalse(validation["checks"]["network_evidence"])

    def test_summarizes_pass_rate(self) -> None:
        report = summarize_cases(
            [
                {"result": {"status": "completed"}, "validation": {"passed": True, "checks": {"network_evidence": True}}},
                {"result": {"status": "failed"}, "validation": {"passed": False, "checks": {"network_evidence": False}}},
            ],
            suite="test",
        )
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["passed"], 1)
        self.assertEqual(report["network_verified"], 1)
        self.assertEqual(report["pass_rate"], 0.5)


def load_task_specs_from_rows(rows: list[dict[str, object]]):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.flush()
        return load_task_specs(handle.name)


if __name__ == "__main__":
    unittest.main()
