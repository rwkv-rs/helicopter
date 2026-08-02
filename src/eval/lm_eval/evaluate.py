from __future__ import annotations

import importlib.metadata
import json
import os
import stat
import uuid
from pathlib import Path
from typing import Mapping

from helicopter_lighteval.http_pool import PoolError, VLLMHttpPool
from helicopter_lighteval.publish import PublicationError

from .config import ConfigError, LMEvalConfig
from .model import RWKVVLLMHttpLM


def run(*, config_path: Path, env: Mapping[str, str], dry_run: bool) -> int:
    pools: list[VLLMHttpPool] = []
    try:
        config = LMEvalConfig.read(config_path, env)
        task_manager, resolved_tasks = _resolve_tasks(config.tasks)
        version = importlib.metadata.version("lm-eval")
        if version != "0.4.12":
            raise ConfigError(f"lm-eval version must be 0.4.12, found {version}")
        client = None
        campaign = None
        expected_tasks = None
        scoreboard = None
        if config.publish:
            from helicopter_lighteval.publish import ScoreboardClient
            from .publish import campaign_payload, expected_tasks as build_expected
            from .publish import task_metadata

            if config.scoreboard_url is None or config.scoreboard_token is None:
                raise ConfigError("published evaluation requires Scoreboard access")
            metadata = task_metadata(task_manager, config.tasks, resolved_tasks)
            expected_tasks = build_expected(config, metadata)
            campaign = campaign_payload(config, metadata, expected_tasks)
            client = ScoreboardClient(config.scoreboard_url, config.scoreboard_token)
            scoreboard = client.preflight("lm-eval", version)
        readiness: list[dict[str, object]] = []
        for unit in config.execution_units:
            pool = VLLMHttpPool(unit.manifest)
            pools.append(pool)
            model_id = pool.preflight()
            readiness.append(
                {
                    "model_id": model_id,
                    "weight_sha256": unit.weight_sha256,
                    "wkv_mode": unit.wkv_mode,
                    "global_step": unit.manifest.global_step,
                    "max_model_len": unit.manifest.max_model_len,
                    "effective_max_length": unit.manifest.max_model_len - 2,
                }
            )
        if dry_run:
            print(
                json.dumps(
                    {
                        "status": "ready",
                        "evaluator": {"name": "lm-eval", "version": version},
                        "config": config.public(),
                        **(readiness[0] if len(readiness) == 1 else {}),
                        "execution_units": readiness,
                        "resolved_tasks": list(resolved_tasks),
                        "task_selection": {
                            "selectors": list(config.tasks),
                            "resolved": _describe_tasks(
                                task_manager, resolved_tasks
                            ),
                        },
                        "expected_task_count": (
                            len(expected_tasks) if expected_tasks is not None else 0
                        ),
                        "scoreboard": scoreboard,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        import lm_eval

        campaign_id = None
        if config.publish:
            if client is None or campaign is None or expected_tasks is None:
                raise ConfigError("published evaluation was not initialized")
            receipt = client.create_campaign(campaign, str(campaign["run_key"]))
            try:
                campaign_id = str(uuid.UUID(str(receipt["campaign_id"])))
            except ValueError as error:
                raise PublicationError(
                    "Scoreboard returned an invalid campaign id"
                ) from error

        completed = 0
        for index, (unit, pool, ready) in enumerate(
            zip(config.execution_units, pools, readiness, strict=True)
        ):
            model = RWKVVLLMHttpLM(
                pool=pool,
                eot_token_id=config.eot_token_id,
                batch_size=config.batch_size,
                max_gen_toks=config.max_gen_toks,
            )
            results = lm_eval.simple_evaluate(
                model=model,
                tasks=list(resolved_tasks),
                batch_size=config.batch_size,
                limit=config.limit,
                log_samples=config.log_samples or config.publish,
                task_manager=task_manager,
            )
            if results is None:
                raise RuntimeError("lm-eval returned no results")
            output_dir = _unit_output_dir(config.output_dir, unit, index)
            _write_results(
                config=config,
                model_id=str(ready["model_id"]),
                version=version,
                results=results,
                output_dir=output_dir,
                manifest=unit.manifest,
                weight_sha256=unit.weight_sha256,
            )
            if campaign_id is not None:
                from .publish import publish_unit

                unit_expected = [
                    task
                    for task in expected_tasks
                    if task["weight_sha256"] == unit.weight_sha256
                    and task["wkv_mode"] == unit.wkv_mode
                ]
                completed += publish_unit(
                    output_dir=output_dir,
                    campaign_id=campaign_id,
                    expected=unit_expected,
                    unit=unit,
                    config=config,
                    client=client,
                )
        if campaign_id is not None:
            if completed != len(expected_tasks):
                raise PublicationError(
                    "not every expected evaluation task was published"
                )
            client.finalize(campaign_id, len(expected_tasks))
        print(f"lm-eval results written to {config.output_dir}")
        return 0
    except (ConfigError, PoolError, PublicationError) as error:
        raise SystemExit(str(error)) from error
    except Exception as error:
        name = f"{type(error).__module__}.{type(error).__qualname__}"
        detail = str(error).strip()
        suffix = f": {detail}" if detail else ""
        raise SystemExit(f"lm-eval evaluation failed: {name}{suffix}") from error
    finally:
        for pool in pools:
            pool.close()


def _unit_output_dir(configured: Path, unit, index: int) -> Path:
    if index == 0 and unit.weight_sha256 is None:
        return configured
    digest = unit.weight_sha256 or f"unidentified-{index}"
    return configured / digest / unit.wkv_mode


def _resolve_tasks(tasks: tuple[str, ...]):
    from lm_eval.tasks import TaskManager

    manager = TaskManager()
    resolved: list[str] = []
    for selector in tasks:
        matches = manager.match_tasks([selector])
        if not matches:
            raise ConfigError(f"unknown lm-eval task, group, tag, or pattern: {selector}")
        for task_name in matches:
            if task_name not in resolved:
                resolved.append(task_name)
    return manager, tuple(resolved)


def _describe_tasks(manager, tasks: tuple[str, ...]) -> list[dict[str, object]]:
    descriptions: list[dict[str, object]] = []
    for task_name in tasks:
        entry = manager.task_index[task_name]
        config = entry.cfg or {}
        descriptions.append(
            {
                "name": task_name,
                "kind": entry.kind.name.lower(),
                "output_type": config.get("output_type"),
            }
        )
    return descriptions


def _write_results(
    *,
    config: LMEvalConfig,
    model_id: str,
    version: str,
    results: Mapping[str, object],
    output_dir: Path | None = None,
    manifest=None,
    weight_sha256: str | None = None,
) -> None:
    from lm_eval.utils import handle_non_serializable

    destination = output_dir or config.output_dir
    execution_manifest = manifest or config.manifest
    _prepare_output_dir(destination)
    raw = json.loads(
        json.dumps(results, default=handle_non_serializable, ensure_ascii=False)
    )
    summary = {
        "schema_version": 1,
        "evaluator": {"name": "lm-eval", "version": version},
        "backend": "vllm_http",
        "model_id": model_id,
        "weight_sha256": weight_sha256,
        "global_step": execution_manifest.global_step,
        "wkv_mode": execution_manifest.wkv_mode,
        "max_model_len": execution_manifest.max_model_len,
        "effective_max_length": execution_manifest.max_model_len - 2,
        "tasks": list(config.tasks),
        "metrics": raw.get("results", {}),
        "versions": raw.get("versions", {}),
    }
    _atomic_json(destination / "results.json", raw)
    _atomic_json(destination / "summary.json", summary)
    _write_sample_artifacts(destination, raw, version)


def _write_sample_artifacts(
    output_dir: Path,
    results: Mapping[str, object],
    version: str,
) -> None:
    raw_samples = results.get("samples")
    artifacts: list[dict[str, object]] = []
    if isinstance(raw_samples, Mapping):
        samples_dir = output_dir / "samples"
        _prepare_output_dir(samples_dir)
        for index, (task_name, rows) in enumerate(sorted(raw_samples.items())):
            if not isinstance(task_name, str) or not isinstance(rows, list):
                raise RuntimeError("lm-eval samples must map task names to arrays")
            path = samples_dir / f"{index:04d}.json"
            _atomic_json(path, rows)
            artifacts.append(
                {
                    "task_name": task_name,
                    "path": path.relative_to(output_dir).as_posix(),
                    "samples": len(rows),
                }
            )
    _atomic_json(
        output_dir / "artifacts.json",
        {
            "schema_version": 1,
            "evaluator": {"name": "lm-eval", "version": version},
            "results_path": "results.json",
            "summary_path": "summary.json",
            "sample_artifacts": artifacts,
        },
    )


def _prepare_output_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    status = path.lstat()
    if (
        not stat.S_ISDIR(status.st_mode)
        or path.is_symlink()
        or status.st_uid != os.geteuid()
    ):
        raise ConfigError("output_dir must be an owned regular directory")
    path.chmod(0o700)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(path)
