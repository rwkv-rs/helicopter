from __future__ import annotations

import argparse
import json
from pathlib import Path

from scoreboard_server.application import create_app
from scoreboard_server.settings import ScoreboardSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the scoreboard HTTP contract")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    app = create_app(
        ScoreboardSettings(
            cors_origins=("http://127.0.0.1:3000",),
            tokens=(),
        ),
        database_url="sqlite:///./openapi-unused.db",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(app.openapi(), sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
