from __future__ import annotations

from pathlib import Path

from scoreboard_server.adapters import screenshot


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "scoreboard_server"
SRC_ROOT = PACKAGE_ROOT.parents[1]
ROUTE_LEAVES = {
    "api/admin/backpressure.py",
    "api/admin/eval/cancel.py",
    "api/admin/eval/draft.py",
    "api/admin/eval/options.py",
    "api/admin/eval/pause.py",
    "api/admin/eval/resume.py",
    "api/admin/eval/start.py",
    "api/admin/eval/status.py",
    "api/admin/health.py",
    "api/capture_page.py",
    "api/eval_context.py",
    "api/eval_records.py",
    "api/health.py",
    "api/leaderboard.py",
    "api/meta.py",
    "api/refresh.py",
    "api/score_history/detail.py",
    "api/score_history/index.py",
    "api/score_history/options.py",
}


def _leaf_modules(directory: str) -> set[str]:
    root = PACKAGE_ROOT / directory
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if path.name != "__init__.py"
    }


def test_scoreboard_server_uses_layered_backend_layout() -> None:
    expected_modules = {
        "adapters/screenshot.py",
        "cores/charts.py",
        "cores/leaderboard.py",
        "cores/normalize.py",
        "db/connection.py",
        "db/lease.py",
        "db/models/__init__.py",
        "db/models/benchmark.py",
        "db/models/checker.py",
        "db/models/completion.py",
        "db/models/eval_record.py",
        "db/models/scheduler_lease.py",
        "db/models/score.py",
        "db/models/score_model.py",
        "db/models/task.py",
        "db/repository.py",
        "db/resume.py",
        "db/schema.py",
        "db/settings.py",
        "application.py",
    }
    expected_modules.update(f"routes/{path}" for path in ROUTE_LEAVES)
    expected_modules.update(f"dtos/{path}" for path in ROUTE_LEAVES)
    expected_modules.update(f"services/{path}" for path in ROUTE_LEAVES)
    missing = sorted(path for path in expected_modules if not (PACKAGE_ROOT / path).is_file())
    assert missing == []

    disallowed_root_modules = {
        "app.py",
        "config.py",
        "db.py",
        "lease.py",
        "models.py",
        "normalize.py",
        "schema.py",
        "screenshot.py",
        "service.py",
    }
    present = sorted(path for path in disallowed_root_modules if (PACKAGE_ROOT / path).exists())
    assert present == []

    assert not (PACKAGE_ROOT / "db/models.py").exists()
    assert not (PACKAGE_ROOT / "dtos/lease.py").exists()
    assert not (PACKAGE_ROOT / "dtos/resume.py").exists()
    assert not (PACKAGE_ROOT / "routes/admin.py").exists()
    assert not (PACKAGE_ROOT / "routes/app.py").exists()
    assert not (PACKAGE_ROOT / "routes/scoreboard.py").exists()
    assert not (PACKAGE_ROOT / "services/scoreboard.py").exists()

    assert _leaf_modules("routes") == ROUTE_LEAVES
    assert _leaf_modules("dtos") == ROUTE_LEAVES
    assert _leaf_modules("services") == ROUTE_LEAVES


def test_screenshot_adapter_uses_client_sibling_directory() -> None:
    assert screenshot._CLIENT_DIR == SRC_ROOT / "scoreboard-client"
    assert screenshot._SCRIPT.is_file()
