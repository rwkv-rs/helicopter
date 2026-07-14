from __future__ import annotations

import unittest

from any2rwkv.oracle import run_gdn_oracle


class OracleTests(unittest.TestCase):
    def test_frozen_32_case_fp64_oracle_passes(self) -> None:
        result = run_gdn_oracle()
        self.assertEqual(result["fixture_count"], 32)
        self.assertTrue(result["passed"], result["failures"])


if __name__ == "__main__":
    unittest.main()

