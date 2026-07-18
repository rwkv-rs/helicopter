from __future__ import annotations

import os


if os.environ.get("HELICOPTER_PATCH_LIGHTEVAL_LITELLM_LOGPROBS") == "1":
    import helicopter_cli.lighteval_litellm_logprobs  # noqa: F401
