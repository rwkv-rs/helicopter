import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from rwkv_web_harness.config import HarnessSettings
from rwkv_web_harness.preflight import probe_model, probe_search


class _Response:
    def __init__(self, payload: bytes, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, _size: int | None = None) -> bytes:
        return self.payload


class PreflightTests(unittest.TestCase):
    def test_model_probe_sends_bearer_token_and_checks_model(self) -> None:
        response = _Response(json.dumps({"data": [{"id": "g1h-1.5b"}]}).encode())
        with patch("rwkv_web_harness.preflight.urlopen", return_value=response) as urlopen:
            result = probe_model(
                "http://127.0.0.1:19315/v1",
                model="g1h-1.5b",
                api_key="secret",
                timeout=2,
            )
        self.assertTrue(result.ok)
        self.assertEqual(result.model, "g1h-1.5b")
        self.assertEqual(urlopen.call_args.args[0].headers["Authorization"], "Bearer secret")

    def test_model_probe_reports_wrong_model(self) -> None:
        response = _Response(json.dumps({"data": [{"id": "other"}]}).encode())
        with patch("rwkv_web_harness.preflight.urlopen", return_value=response):
            result = probe_model(
                "http://127.0.0.1:19315/v1",
                model="g1h-1.5b",
                api_key=None,
                timeout=2,
            )
        self.assertFalse(result.ok)
        self.assertIn("not advertised", result.message)

    def test_model_probe_distinguishes_auth_failure(self) -> None:
        error = HTTPError(
            "http://127.0.0.1:19315/v1/models",
            401,
            "Unauthorized",
            {},
            None,
        )
        with patch("rwkv_web_harness.preflight.urlopen", side_effect=error):
            result = probe_model(
                "http://127.0.0.1:19315/v1",
                model="g1h-1.5b",
                api_key=None,
                timeout=2,
            )
        self.assertFalse(result.ok)
        self.assertEqual(result.status, 401)
        self.assertIn("HTTP 401", result.message)

    def test_search_probe_uses_backend_query(self) -> None:
        with patch("rwkv_web_harness.preflight.urlopen", return_value=_Response(b"ok")) as urlopen:
            result = probe_search(
                "https://www.bing.com/search",
                backend="html",
                timeout=2,
            )
        self.assertTrue(result.ok)
        self.assertIn("q=RWKV", urlopen.call_args.args[0].full_url)


if __name__ == "__main__":
    unittest.main()
