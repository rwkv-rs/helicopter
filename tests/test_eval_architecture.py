from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()


def _non_scoreboard_backend_sources():
    roots = (
        ROOT / "src/cli",
        ROOT / "src/eval",
        ROOT / "src/scoreboard-client",
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


def test_non_scoreboard_callers_cannot_load_database_drivers_or_query_api() -> None:
    forbidden = re.compile(
        r"(?:\bimport\s+(?:asyncpg|aiosqlite|tortoise|sqlalchemy)|"
        r"\bfrom\s+(?:asyncpg|aiosqlite|tortoise|sqlalchemy)|"
        r"scoreboard_server\.db|SCOREBOARD_DATABASE_URL|postgres(?:ql)?://|"
        r"from\s+['\"](?:pg|postgres|@prisma/client)['\"])"
    )
    violations: list[str] = []
    for path in _non_scoreboard_backend_sources():
        if forbidden.search(path.read_text(encoding="utf-8", errors="replace")):
            violations.append(str(path.relative_to(ROOT)))
    assert violations == []


def test_eval_component_has_parallel_sources_and_no_removed_facades() -> None:
    component = ROOT / "src/eval/lighteval"
    forbidden_paths = (
        ROOT / "src/eval/helicopter_eval",
        ROOT / "src/eval/common",
        ROOT / "src/eval/shared",
        component / "src/lighteval_runner",
        component / "src/helicopter_lighteval/results",
        component / "src/helicopter_lighteval/provider",
        component / "src/helicopter_lighteval/data_sources",
        component / "src/helicopter_lighteval/context.py",
    )
    assert all(not path.exists() for path in forbidden_paths)
    assert (component / "pyproject.toml").is_file()
    assert not (ROOT / "src/eval/__init__.py").exists()
    assert sorted(
        path.name for path in (component / "src/helicopter_lighteval").glob("*.py")
    ) == [
        "__init__.py",
        "evaluation.py",
        "scoreboard.py",
        "vllm_rwkv.py",
    ]
    assert sorted(
        path.name
        for path in (component / "src/helicopter_lighteval/datasets").glob("*.py")
    ) == [
        "__init__.py",
        "coding.py",
        "instruction_following.py",
        "knowledge.py",
        "math.py",
    ]
    assert (component / "results").is_dir() or not (component / "results").exists()


def test_scoreboard_and_vllm_product_boundaries_are_not_imported_by_evaluator() -> None:
    package = ROOT / "src/eval/lighteval/src/helicopter_lighteval"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in package.rglob("*.py")
    )
    assert "scoreboard_server" not in source
    assert "sqlalchemy" not in source
    assert "asyncpg" not in source
    assert "vllm_rwkv" in source


def test_publication_route_does_not_import_database_layer() -> None:
    route = (
        ROOT
        / "src/scoreboard-server/scoreboard_server/routes/api/evaluation_publications.py"
    )
    assert "scoreboard_server.db" not in route.read_text(encoding="utf-8")


def test_local_installer_defaults_and_forwards_rwkv_build_profile() -> None:
    installer = (ROOT / "scripts/install_local.sh").read_text(encoding="utf-8")
    assert 'VLLM_BUILD_PROFILE="${VLLM_BUILD_PROFILE:-rwkv}"' in installer
    assert 'VLLM_BUILD_PROFILE="${VLLM_BUILD_PROFILE:-full}"' not in installer
    assert installer.count('VLLM_BUILD_PROFILE="$VLLM_BUILD_PROFILE"') == 2
