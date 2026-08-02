from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_text_generation.py"
SPEC = importlib.util.spec_from_file_location("validate_text_generation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_token_statistics_handles_empty_generation() -> None:
    assert MODULE.token_statistics([]) == {
        "token_count": 0,
        "unique_token_ratio": 0.0,
        "max_token_run": 0,
    }


def test_token_statistics_reports_repetition_without_quality_threshold() -> None:
    assert MODULE.token_statistics([1, 1, 1, 2, 3, 3]) == {
        "token_count": 6,
        "unique_token_ratio": 0.5,
        "max_token_run": 3,
    }
