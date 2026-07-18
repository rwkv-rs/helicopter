import unittest

from rwkv_web_harness.models import ToolCall, _extract_response


class ModelResponseTests(unittest.TestCase):
    def test_extracts_native_tool_calls(self) -> None:
        response = _extract_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "function": {"name": "web_search", "arguments": '{"query":"RWKV"}'},
                                }
                            ],
                        }
                    }
                ]
            }
        )
        self.assertEqual(response.content, "")
        self.assertEqual(response.tool_calls, (ToolCall("call_abc", "web_search", {"query": "RWKV"}),))

    def test_extracts_completion_text(self) -> None:
        response = _extract_response({"choices": [{"text": " answer ", "finish_reason": "stop"}]})
        self.assertEqual(response.content, "answer")
        self.assertEqual(response.tool_calls, ())
        self.assertEqual(response.finish_reason, "stop")

    def test_extracts_g1h_agentic_json_command(self) -> None:
        response = _extract_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": (
                                "</think>\n"
                                '{"analysis":"search first", "commands":['
                                '{"keystrokes":"web_search \'RWKV vLLM\'\\n"}]}'
                            )
                        },
                    }
                ]
            }
        )
        self.assertEqual(response.content, "")
        self.assertEqual(response.tool_calls[0].name, "web_search")
        self.assertEqual(response.tool_calls[0].arguments, {"query": "RWKV vLLM"})

    def test_extracts_prefilled_g1h_json_call(self) -> None:
        response = _extract_response(
            {"choices": [{"text": '"name":"final_answer","arguments":{"answer":"done"}}'}]}
        )
        self.assertEqual(response.tool_calls[0].name, "final_answer")

    def test_extracts_g1h_json_array(self) -> None:
        response = _extract_response(
            {"choices": [{"text": '[{"name":"web_search","arguments":{"query":"RWKV"}}]✿'}]}
        )
        self.assertEqual(response.tool_calls[0].name, "web_search")

    def test_extracts_g1h_final_answer_object(self) -> None:
        response = _extract_response(
            {
                "choices": [
                    {
                        "text": '{"final_answer":"done","citations":[{"source_id":"source_001"}]}✿',
                    }
                ]
            }
        )
        self.assertEqual(response.tool_calls[0].name, "final_answer")
        self.assertEqual(response.tool_calls[0].arguments["citations"][0]["source_id"], "source_001")

    def test_extracts_first_call_before_closing_think(self) -> None:
        response = _extract_response(
            {
                "choices": [
                    {
                        "text": (
                            '{"name":"web_search","arguments":{"query":"RWKV"}}\n'
                            '{"name":"final_answer","arguments":{"answer":"later"}}'
                            '</think>'
                        )
                    }
                ]
            }
        )
        self.assertEqual(response.tool_calls[0].name, "web_search")

    def test_repairs_stray_bracket_in_g1h_final_call(self) -> None:
        response = _extract_response(
            {
                "choices": [
                    {
                        "text": (
                            '{"name":"final_answer","arguments":{"answer":"done",'
                            '"citations":[{"source_id":"source_001"}]}}]}'
                        )
                    }
                ]
            }
        )
        self.assertEqual(response.tool_calls[0].name, "final_answer")

    def test_finds_first_call_in_repeated_g1h_history(self) -> None:
        response = _extract_response(
            {
                "choices": [
                    {
                        "text": (
                            '{"name":"final_answer","arguments":{"answer":"done",'
                            '"citations":[{"source_id":"source_001"}]}}'
                            "\n\nUser: repeated history\nAssistant: <think>old</think>\n"
                            '{"name":"final_answer","arguments":{"answer":"truncated"'
                        )
                    }
                ]
            }
        )
        self.assertEqual(response.tool_calls[0].name, "final_answer")
        self.assertEqual(response.tool_calls[0].arguments["answer"], "done")

    def test_does_not_turn_tool_observation_into_a_new_call(self) -> None:
        response = _extract_response(
            {
                "choices": [
                    {
                        "text": '{"ok":true,"tool":"find_in_page","message":"pattern found"}',
                    }
                ]
            }
        )
        self.assertEqual(response.tool_calls, ())
        self.assertIn('"pattern found"', response.content)


if __name__ == "__main__":
    unittest.main()
