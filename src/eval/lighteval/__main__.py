from __future__ import annotations

import argparse
import os
from pathlib import Path

from .evaluate import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="helicopter-lighteval")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return run(
        config_path=args.config,
        env=os.environ,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
