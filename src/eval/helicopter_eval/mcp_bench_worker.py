from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import shlex
import sys
from pathlib import Path
from typing import Any


class WorkerState:
    def __init__(self, runtime_root: Path) -> None:
        self.runtime_root = runtime_root
        self.manager = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCP-Bench runtime bridge worker")
    parser.add_argument("--runtime-root", required=True)
    return parser.parse_args()


def configure_runtime_paths(runtime_root: Path) -> None:
    venv_bin = runtime_root / ".venv" / "bin"
    path_value = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{venv_bin}:{path_value}" if path_value else str(venv_bin)
    sys.path.insert(0, str(runtime_root))


def load_server_configs(runtime_root: Path, server_names: list[str]) -> list[dict[str, Any]]:
    commands_path = runtime_root / "mcp_servers" / "commands.json"
    api_key_path = runtime_root / "mcp_servers" / "api_key"
    commands = json.loads(commands_path.read_text(encoding="utf-8"))
    api_keys = load_api_keys(api_key_path)
    configs: list[dict[str, Any]] = []
    for server_name in server_names:
        raw = commands.get(server_name)
        if not isinstance(raw, dict):
            raise ValueError(f"server config not found: {server_name}")
        env = dict(os.environ)
        for key in raw.get("env", []) or []:
            if key in api_keys:
                env[key] = api_keys[key]
        cwd = resolve_cwd(runtime_root, str(raw.get("cwd") or ""))
        configs.append(
            {
                "name": server_name,
                "command": shlex.split(str(raw.get("cmd") or "")),
                "env": env,
                "cwd": str(cwd),
                "transport": str(raw.get("transport") or "stdio"),
                "port": raw.get("port"),
                "endpoint": str(raw.get("endpoint") or "/mcp"),
            }
        )
    return configs


def load_api_keys(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            payload[key.strip()] = value.strip()
    return payload


def resolve_cwd(runtime_root: Path, raw_cwd: str) -> Path:
    if not raw_cwd:
        return runtime_root
    if raw_cwd.startswith("../"):
        return (runtime_root / "mcp_servers" / raw_cwd[3:]).resolve()
    return (runtime_root / raw_cwd).resolve()


def extract_text_from_result(result: Any) -> str:
    content = getattr(result, "content", None)
    if content:
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
        if parts:
            return "".join(parts)
    return str(result)


async def handle_open_task(state: WorkerState, payload: dict[str, Any]) -> dict[str, Any]:
    if state.manager is not None:
        with contextlib.redirect_stdout(sys.stderr):
            await state.manager.close_all_connections()
        state.manager = None
    from mcp_modules.server_manager_persistent import PersistentMultiServerManager

    configs = load_server_configs(state.runtime_root, list(payload.get("servers") or []))
    manager = PersistentMultiServerManager(configs)
    with contextlib.redirect_stdout(sys.stderr):
        available_tools = await manager.connect_all_servers()
    state.manager = manager
    return {"available_tools": available_tools}


async def handle_call_tool(state: WorkerState, payload: dict[str, Any]) -> dict[str, Any]:
    if state.manager is None:
        raise RuntimeError("no active MCP-Bench task session")
    tool_name = str(payload.get("tool_name") or "")
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}
    with contextlib.redirect_stdout(sys.stderr):
        result_obj = await state.manager.call_tool(tool_name, arguments)
    is_error = bool(getattr(result_obj, "isError", False))
    text = extract_text_from_result(result_obj)
    return {
        "success": not is_error,
        "result": text if not is_error else "",
        "error": text if is_error else "",
    }


async def handle_evaluate(_state: WorkerState, payload: dict[str, Any]) -> dict[str, Any]:
    request = payload.get("request") or {}
    if not isinstance(request, dict):
        raise ValueError("invalid evaluation request payload")
    from benchmark.evaluator import TaskEvaluator
    from llm.provider import LLMProvider
    from openai import AsyncOpenAI

    judge = request.get("judge_config") or {}
    if not isinstance(judge, dict):
        raise ValueError("judge_config must be an object")
    client = AsyncOpenAI(
        api_key=str(judge.get("api_key") or ""),
        base_url=str(judge.get("base_url") or ""),
    )
    provider = LLMProvider(
        client=client,
        deployment_name=str(judge.get("model") or ""),
        provider_type="openai_compatible",
    )
    evaluator = TaskEvaluator(provider, enable_judge_stability=False)
    with contextlib.redirect_stdout(sys.stderr):
        evaluation = await evaluator.evaluate(
            task=str(request.get("task") or ""),
            execution_results=list(request.get("execution_results") or []),
            final_solution=str(request.get("final_solution") or ""),
            total_rounds=int(request.get("total_rounds") or 0),
            available_tools=dict(request.get("available_tools") or {}),
            planning_json_compliance=float(request.get("planning_json_compliance") or 0.0),
            accumulated_information=str(request.get("accumulated_information") or ""),
            concrete_task_description=str(request.get("concrete_task_description") or ""),
            dependency_analysis=str(request.get("dependency_analysis") or ""),
        )
    if not isinstance(evaluation, dict):
        raise RuntimeError("official evaluator returned a non-dict payload")
    return evaluation


async def handle_close_task(state: WorkerState, _payload: dict[str, Any]) -> dict[str, Any]:
    if state.manager is not None:
        with contextlib.redirect_stdout(sys.stderr):
            await state.manager.close_all_connections()
        state.manager = None
    return {}


async def dispatch(state: WorkerState, action: str, payload: dict[str, Any]) -> dict[str, Any]:
    if action == "open_task":
        return await handle_open_task(state, payload)
    if action == "call_tool":
        return await handle_call_tool(state, payload)
    if action == "evaluate":
        return await handle_evaluate(state, payload)
    if action == "close_task":
        return await handle_close_task(state, payload)
    if action == "shutdown":
        await handle_close_task(state, {})
        return {"shutdown": True}
    raise ValueError(f"unknown action: {action}")


def main() -> int:
    args = parse_args()
    runtime_root = Path(args.runtime_root).expanduser().resolve()
    configure_runtime_paths(runtime_root)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    state = WorkerState(runtime_root)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        for line in sys.stdin:
            raw = line.strip()
            if not raw:
                continue
            try:
                request = json.loads(raw)
                action = str(request.get("action") or "")
                payload = request.get("payload") or {}
                if not isinstance(payload, dict):
                    payload = {}
                data = loop.run_until_complete(dispatch(state, action, payload))
                sys.stdout.write(json.dumps({"ok": True, "data": data}, ensure_ascii=False) + "\n")
                sys.stdout.flush()
                if action == "shutdown":
                    break
            except Exception as exc:
                sys.stdout.write(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False) + "\n")
                sys.stdout.flush()
    finally:
        try:
            loop.run_until_complete(handle_close_task(state, {}))
        except Exception:
            pass
        loop.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
