from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Mapping

from .config import (
    EvaluationConfigurationError,
    load_evaluation_config,
    load_evaluation_environment,
    public_environment,
    resolve_weights,
)
from .plan import build_plan, public_plan
from .preflight import run_preflight
from .registry import load_default_registry
from .http_client import ScoreboardError


@contextmanager
def _process_environment(values: Mapping[str, str]):
    missing = object()
    previous: dict[str, str | object] = {
        key: os.environ.get(key, missing) for key in values
    }
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is missing:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run(*, config_path: Path, env: Mapping[str, str], dry_run: bool) -> int:
    with _process_environment(env):
        try:
            config = load_evaluation_config(config_path)
            environment = load_evaluation_environment(env)
            weights = resolve_weights(config, environment)
            readiness = run_preflight(environment)
            registry = load_default_registry()
            plan = build_plan(config, weights, registry)
        except (EvaluationConfigurationError, ScoreboardError, OSError) as error:
            raise SystemExit(str(error)) from error

        if dry_run:
            output = {
                "status": "ready",
                "environment": public_environment(environment),
                "readiness": readiness,
                "plan": public_plan(plan),
            }
            print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        from .campaign import run_campaign
        from .manifest import ManifestError

        try:
            return run_campaign(plan=plan, environment=environment)
        except (ManifestError, ScoreboardError, OSError) as error:
            raise SystemExit(f"evaluation failed: {error}") from error
