import json
import unittest

from rwkv_web_harness.tools import _parse_html_results, _parse_searxng_results


class ToolParsingTests(unittest.TestCase):
    def test_parses_searxng_json_results(self) -> None:
        raw = json.dumps(
            {"results": [{"title": "RWKV", "url": "https://example.com/rwkv", "content": "A result"}]}
        ).encode("utf-8")
        results = _parse_searxng_results(raw, 5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "RWKV")
        self.assertEqual(results[0].url, "https://example.com/rwkv")

    def test_parses_duckduckgo_lite_html_results(self) -> None:
        raw = b"""
        <a class='result-link' href='//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F'>Example result</a>
        <td class='result-snippet'>Useful <b>evidence</b> here.</td>
        """
        results = _parse_html_results(raw, 5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://example.com/")
        self.assertIn("evidence", results[0].snippet)


if __name__ == "__main__":
    unittest.main()
