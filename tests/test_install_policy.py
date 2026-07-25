from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_lighteval_is_a_locked_root_group_without_a_child_package() -> None:
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    group = manifest["dependency-groups"]["lighteval"]

    assert group[0] == {"include-group": "vllm-rwkv"}
    assert "lighteval[extended-tasks,math]==0.13.0" in group
    assert not any(
        "git+" in str(item) or str(item).startswith("vllm") for item in group[1:]
    )
    assert not (ROOT / "src/eval/lighteval/pyproject.toml").exists()
    assert not (ROOT / "src/eval/lighteval/uv.lock").exists()


def test_installers_expose_the_complete_evaluation_components() -> None:
    local = (ROOT / "scripts/install_local.sh").read_text(encoding="utf-8")
    remote = (ROOT / "scripts/install_remote.sh").read_text(encoding="utf-8")

    for component in (
        "lighteval",
        "scoreboard-server",
        "scoreboard-client",
    ):
        assert component in local
        assert component in remote
    assert "component_enabled lighteval" in local
    assert "vllm_package_enabled && install_vllm_package" in local
    assert "python_component_enabled && sync_uv_env" in local
    assert "--frozen-lockfile" in local


def test_installer_separates_lock_refresh_from_dependency_upgrade() -> None:
    local = (ROOT / "scripts/install_local.sh").read_text(encoding="utf-8")
    remote = (ROOT / "scripts/install_remote.sh").read_text(encoding="utf-8")
    example = (ROOT / ".env.example").read_text(encoding="utf-8")

    for source in (local, remote):
        assert "0 | lock | 1" in source
        assert "lock to refresh lockfiles without a broad upgrade" in source
        assert "validate_uv_upgrade" in source
    assert "append_uv_sync_policy sync_args" in local
    assert "0) sync_args_ref+=(--locked)" in local
    assert "lock) ;;" in local
    assert "1) sync_args_ref+=(--upgrade)" in local
    assert '[[ "$UV_UPGRADE" == "0" ]] && install_args+=(--frozen-lockfile)' in local
    assert "lock = refresh locks without broad upgrades" in example


def test_installer_exports_an_absolute_native_build_tmpdir() -> None:
    local = (ROOT / "scripts/install_local.sh").read_text(encoding="utf-8")

    assert 'if [[ "$BUILD_TMPDIR" != /* ]]; then' in local
    assert 'BUILD_TMPDIR="$ROOT/$BUILD_TMPDIR"' in local
    assert 'BUILD_TMPDIR="$(cd "$BUILD_TMPDIR" && pwd -P)"' in local
    assert '[[ -w "$BUILD_TMPDIR" ]]' in local
    assert 'export TMPDIR="$BUILD_TMPDIR"' in local


def test_full_install_profile_remains_disabled() -> None:
    for script in ("install_local.sh", "install_remote.sh"):
        source = (ROOT / "scripts" / script).read_text(encoding="utf-8")
        assert 'full) die "INSTALL_PROFILE=full is disabled' in source
        assert "INSTALL_COMPONENTS=full is disabled" in source
