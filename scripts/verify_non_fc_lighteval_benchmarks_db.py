from __future__ import annotations

import asyncio
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI_SRC = ROOT / "src/cli"
SCOREBOARD_SRC = ROOT / "src/scoreboard-server"

for path in (CLI_SRC, SCOREBOARD_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from helicopter_cli.benchmark_catalog_defaults import (  # noqa: E402
    CATALOG_RUN_STATUS,
    CATALOG_SCOPE,
    CATALOG_SOURCE,
    CATALOG_TARGET_KIND,
    EXPECTED_FIELDS,
    REQUIRED_TASKS,
    TARGET_PER_DOMAIN,
)
from helicopter_cli.non_fc_lighteval_catalog import (  # noqa: E402
    DEFAULT_CUSTOM_TASKS,
    EXCLUDED_DIRECT_PATTERNS,
    has_perplexity_metric,
    matches_any,
)
from helicopter_cli.lighteval_tasks import load_registry  # noqa: E402
from scoreboard_server.db.connection import close_db, init_db  # noqa: E402
from scoreboard_server.db.repository import ScoreboardStore  # noqa: E402
from scoreboard_server.db.settings import DatabaseSettings  # noqa: E402


def fail(message: str) -> None:
    raise SystemExit(f"non-FC LightEval DB benchmark verification failed: {message}")


async def load_catalog_rows() -> list[dict[str, object]]:
    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=False)
    try:
        store = ScoreboardStore(settings=settings)
        return await store.list_benchmark_catalog(scope=CATALOG_SCOPE, fields=EXPECTED_FIELDS)
    finally:
        await close_db()


def verify_rows(rows: list[dict[str, object]]) -> None:
    counts = Counter(str(row.get("field") or "") for row in rows)
    expected_counts = Counter({field: TARGET_PER_DOMAIN for field in EXPECTED_FIELDS})
    if counts != expected_counts:
        fail(f"field counts are {dict(counts)!r}, expected {dict(expected_counts)!r}")
    names = [str(row.get("name") or "") for row in rows]
    if len(names) != len(set(names)):
        fail("benchmark names must be unique")
    for required in REQUIRED_TASKS:
        if required not in names:
            fail(f"required task missing: {required}")
    for row in rows:
        name = str(row.get("name") or "")
        if row.get("source") != CATALOG_SOURCE:
            fail(f"{name} source must be {CATALOG_SOURCE}")
        if row.get("target_kind") != CATALOG_TARGET_KIND:
            fail(f"{name} target_kind must be {CATALOG_TARGET_KIND}")
        if row.get("run_status") != CATALOG_RUN_STATUS:
            fail(f"{name} run_status must be {CATALOG_RUN_STATUS}")
        if matches_any(name, EXCLUDED_DIRECT_PATTERNS):
            fail(f"{name} matches an excluded FC/agent/tool-use pattern")

    registry = load_registry(custom_tasks=str(DEFAULT_CUSTOM_TASKS), load_multilingual=True)
    task_set = set(registry._task_registry)
    missing = [name for name in names if name not in task_set]
    if missing:
        fail(f"{len(missing)} DB rows are not exact LightEval tasks: {missing[:10]!r}")
    unsupported = []
    for name in names:
        if has_perplexity_metric(registry._task_registry[name]):
            unsupported.append(name)
    if unsupported:
        fail(f"{len(unsupported)} DB rows require PERPLEXITY, unsupported by endpoint-litellm: {unsupported[:10]!r}")


def main() -> int:
    rows = asyncio.run(load_catalog_rows())
    verify_rows(rows)
    counts = Counter(str(row["field"]) for row in rows)
    print("verified\tbenchmark_catalog")
    print(f"scope\t{CATALOG_SCOPE}")
    for field in EXPECTED_FIELDS:
        print(f"field\t{field}\t{counts[field]}")
    print(f"direct_lighteval_tasks\t{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
