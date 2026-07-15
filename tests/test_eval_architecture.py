from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DB = ROOT / "src/scoreboard-server/scoreboard_server/db"
SELF = Path(__file__).resolve()


def _production_sources():
    roots = (
        ROOT / "src/cli",
        ROOT / "src/eval",
        ROOT / "src/scoreboard-client",
        ROOT / "src/scoreboard-server/scoreboard_server",
        ROOT / "scripts",
    )
    for suffix in ("*.py", "*.ts", "*.tsx", "*.js", "*.mjs"):
        for root in roots:
            for path in root.glob(f"**/{suffix}"):
                if path == SELF or any(
                    part
                    in {"tests", "test_data", ".venv", "node_modules", ".next", ".git"}
                    for part in path.parts
                ):
                    continue
                yield path


def test_only_scoreboard_persistence_can_load_database_drivers_or_query_api() -> None:
    forbidden = re.compile(
        r"(?:\bimport\s+(?:asyncpg|aiosqlite|tortoise|sqlalchemy)|"
        r"\bfrom\s+(?:asyncpg|aiosqlite|tortoise|sqlalchemy)|"
        r"scoreboard_server\.db|SCOREBOARD_DATABASE_URL|postgres(?:ql)?://|"
        r"from\s+['\"](?:pg|postgres|@prisma/client)['\"])"
    )
    violations: list[str] = []
    for path in _production_sources():
        try:
            path.relative_to(BACKEND_DB)
            continue
        except ValueError:
            pass
        if forbidden.search(path.read_text(encoding="utf-8", errors="replace")):
            violations.append(str(path.relative_to(ROOT)))
    assert violations == []


def test_eval_has_one_component_owner_and_no_removed_facades() -> None:
    forbidden_paths = (
        ROOT / "src/eval/helicopter_eval",
        ROOT / "src/eval/common",
        ROOT / "src/eval/shared",
        ROOT / "src/cli/helicopter_cli/lighteval_rwkv_skills_tasks.py",
        ROOT / "src/cli/helicopter_cli/function_calling.py",
        ROOT / "src/cli/helicopter_cli/agent_harness.py",
    )
    assert all(not path.exists() for path in forbidden_paths)
    assert (ROOT / "src/eval/lighteval/pyproject.toml").is_file()
    assert not (ROOT / "src/eval/__init__.py").exists()


def test_eval_import_graph_respects_component_boundaries() -> None:
    package = ROOT / "src/eval/lighteval/src/lighteval_runner"
    violations: list[str] = []
    for path in package.rglob("*.py"):
        relative = path.relative_to(package)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = {
            (node.level, node.module or "")
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        if relative.parts[0] == "tasks" and any(
            module.startswith(("application", "provider", "results"))
            for _level, module in imports
        ):
            violations.append(str(relative))
        if relative.name == "contracts.py" and any(level for level, _module in imports):
            violations.append(str(relative))
        if any("scoreboard_server" in module for _level, module in imports):
            violations.append(str(relative))
    assert violations == []
