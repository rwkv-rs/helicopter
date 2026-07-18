from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI_SRC = ROOT / "src/cli"

if str(CLI_SRC) not in sys.path:
    sys.path.insert(0, str(CLI_SRC))

from helicopter_cli.non_fc_lighteval_catalog import build_manifest  # noqa: E402


def main() -> int:
    manifest = build_manifest(root=ROOT)
    print("built direct HF/LightEval non-FC benchmark catalog in memory; no JSON file written")
    for domain in manifest["domains"]:
        print(f"{domain['field']}\t{domain['count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
