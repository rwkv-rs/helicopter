"""Run lm-eval after registering the installed Any-to-RWKV model family."""

from __future__ import annotations

import runpy

from . import register_any_to_rwkv_auto_classes


def main() -> None:
    register_any_to_rwkv_auto_classes()
    runpy.run_module("lm_eval.__main__", run_name="__main__")


if __name__ == "__main__":
    main()
