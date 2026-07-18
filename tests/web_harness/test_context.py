import unittest

from rwkv_web_harness.context import ChatContext


class ContextTests(unittest.TestCase):
    def test_g1h_uses_compact_think_and_diamond_delimiters(self) -> None:
        context = ChatContext(system_prompt="Tools: web_search", task="Find RWKV")
        context.add_assistant(
            None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": "{\"query\":\"RWKV\"}",
                    },
                }
            ],
        )
        context.add_tool(call_id="call_1", content="{\"ok\":true}")

        prompt, truncated = context.render_g1h(4000)

        self.assertFalse(truncated)
        self.assertIn("User✿Find RWKV✿", prompt)
        self.assertIn('Bot✿<think></think>{\"name\":\"web_search\",\"arguments\":{\"query\":\"RWKV\"}}✿', prompt)
        self.assertIn("User✿{\"ok\":true}✿", prompt)
        self.assertTrue(prompt.endswith("Bot✿<think></think>"))

    def test_rwkv_json_uses_compact_think_prefill(self) -> None:
        context = ChatContext(system_prompt="Tools: web_search", task="Find RWKV")

        prompt, _ = context.render_rwkv_json(4000)

        self.assertIn("Assistant: <think></think>", prompt)
        self.assertNotIn("Assistant: <think>\\n</think>", prompt)


if __name__ == "__main__":
    unittest.main()
