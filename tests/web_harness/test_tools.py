import json
import gzip
import unittest
from unittest.mock import patch

from rwkv_web_harness.protocol import Action
from rwkv_web_harness.tools import WebToolkit, WebToolError, _http_get, _parse_html_results, _parse_searxng_results


class ToolParsingTests(unittest.TestCase):
    def test_exposes_native_tool_schemas(self) -> None:
        toolkit = WebToolkit()
        names = [item["function"]["name"] for item in toolkit.tool_schemas]
        self.assertEqual(names, ["web_search", "open_url", "find_in_page"])

    def test_exposes_g1h_flat_tool_catalog(self) -> None:
        toolkit = WebToolkit()
        self.assertTrue(toolkit.g1h_tool_catalog.startswith("Tools:\n"))
        self.assertIn('"name":"final_answer"', toolkit.g1h_tool_catalog)

    def test_html_search_falls_back_after_provider_challenge(self) -> None:
        google_html = b'<h3><a href="https://example.com/rwkv">RWKV result</a></h3>'
        with patch(
            "rwkv_web_harness.tools._http_get",
            side_effect=[WebToolError("challenge"), google_html],
        ) as http_get:
            result = WebToolkit(search_url="https://primary.example/search", search_backend="html").execute(
                Action("web_search", {"query": "RWKV", "top_k": 1})
            )
        self.assertTrue(result.ok)
        self.assertEqual(result.data["provider"], "https://html.duckduckgo.com/html/")
        self.assertEqual(result.data["results"][0]["url"], "https://example.com/rwkv")
        self.assertEqual(http_get.call_count, 2)

    def test_http_get_decompresses_gzip_pages(self) -> None:
        class Response:
            headers = {"Content-Encoding": "gzip"}

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def read(self, _):
                return gzip.compress(b"<html>evidence</html>")

        with patch("rwkv_web_harness.tools.urlopen", return_value=Response()):
            self.assertEqual(
                _http_get("https://example.com", timeout=5, user_agent="test"),
                b"<html>evidence</html>",
            )

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
