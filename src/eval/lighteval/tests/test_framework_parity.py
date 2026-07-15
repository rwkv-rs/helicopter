from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lighteval_runner.registry import LIGHTEVAL_REVISION, get_task_definition


FIXTURE = Path(__file__).parent / "test_data/lighteval_parity_v1.json"
FIXTURE_SHA256 = "f84ef1847f944723fed4de7271cde147461c4bba45dfc18585ebc10f77120d32"


def test_pinned_lighteval_prompt_reference_metric_parity() -> None:
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["lighteval_revision"] == LIGHTEVAL_REVISION
    for case in payload["cases"]:
        runtime = get_task_definition(case["task"]).load_runtime()
        prepared = runtime.prepare(case["row"])
        assert [section.content for section in prepared.context.sections] == [
            case["expected_prompt"]
        ]
        metrics = runtime.score(
            prepared,
            prompt=case["expected_prompt"],
            completion=case["completion"],
            output_token_ids=(1, 0),
        )
        assert metrics == case["expected_metrics"]
