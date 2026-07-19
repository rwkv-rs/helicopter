import unittest

from rwkv_web_harness.config import HarnessSettings, resolve_search_url


class ConfigTests(unittest.TestCase):
    def test_defaults_are_g1h_deployable_without_secrets(self) -> None:
        settings = HarnessSettings()
        self.assertEqual(settings.model_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(settings.resolved_search_url, "https://lite.duckduckgo.com/lite/")

    def test_search_backend_defaults_are_consistent(self) -> None:
        self.assertEqual(resolve_search_url("html"), "https://lite.duckduckgo.com/lite/")
        self.assertEqual(resolve_search_url("searxng"), "http://127.0.0.1:8080/search")
        self.assertEqual(resolve_search_url("html", "https://www.bing.com/search"), "https://www.bing.com/search")

    def test_rejects_invalid_runtime_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "absolute http"):
            HarnessSettings(model_url="127.0.0.1:8000/v1")
        with self.assertRaisesRegex(ValueError, "max_steps"):
            HarnessSettings(max_steps=0)
        with self.assertRaisesRegex(ValueError, "timeout"):
            HarnessSettings(timeout=0)


if __name__ == "__main__":
    unittest.main()
