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

from helicopter_cli.benchmark_catalog_defaults import CATALOG_SCOPE, EXPECTED_FIELDS  # noqa: E402
from helicopter_cli.non_fc_lighteval_catalog import build_manifest  # noqa: E402
from scoreboard_server.db.connection import close_db, init_db  # noqa: E402
from scoreboard_server.db.repository import ScoreboardStore  # noqa: E402
from scoreboard_server.db.settings import DatabaseSettings  # noqa: E402


async def seed() -> int:
    manifest = build_manifest(root=ROOT)
    rows = [dict(row, scope=CATALOG_SCOPE) for row in manifest["benchmarks"]]
    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=True)
    try:
        store = ScoreboardStore(settings=settings)
        written = await store.upsert_benchmark_catalog(rows=rows)
        removed = await store.prune_benchmark_catalog(scope=CATALOG_SCOPE, rows=rows)
        counts = Counter(str(row["field"]) for row in rows)
    finally:
        await close_db()
    print(f"seeded\tbenchmark_catalog\t{written}")
    print(f"pruned\tbenchmark_catalog\t{removed}")
    print(f"scope\t{CATALOG_SCOPE}")
    for field in EXPECTED_FIELDS:
        print(f"field\t{field}\t{counts[field]}")
    return 0


def main() -> int:
    return asyncio.run(seed())


if __name__ == "__main__":
    raise SystemExit(main())
