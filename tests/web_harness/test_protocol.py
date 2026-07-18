import unittest

from rwkv_web_harness.protocol import parse_turn


class ProtocolTests(unittest.TestCase):
    def test_parses_tool_call(self) -> None:
        parsed = parse_turn('<tool_call>{"name":"web_search","arguments":{"query":"RWKV"}}</tool_call>')
        self.assertIsNone(parsed.error)
        self.assertEqual(parsed.action.name, "web_search")
        self.assertEqual(parsed.action.arguments["query"], "RWKV")

    def test_parses_json_final_answer(self) -> None:
        parsed = parse_turn('<final_answer>{"answer":"RWKV","citations":["source_001"]}</final_answer>')
        self.assertIsNone(parsed.error)
        self.assertEqual(parsed.final.answer, "RWKV")
        self.assertEqual(parsed.final.citations, ("source_001",))

    def test_rejects_unstructured_output(self) -> None:
        parsed = parse_turn("I think the answer is RWKV")
        self.assertIsNotNone(parsed.error)


if __name__ == "__main__":
    unittest.main()
