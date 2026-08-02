from __future__ import annotations

import email
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

from any2rwkv.kernel import (
    FLA_RWKV7_REQUIREMENT,
    FLA_RWKV7_REVISION,
    FLA_RWKV7_SOURCE_URL,
)
from any2rwkv.preflight import (
    TRANSFORMERS_REQUIREMENT,
    TRANSFORMERS_REVISION,
    TRANSFORMERS_SOURCE_URL,
)
from packaging.requirements import Requirement

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PRODUCT_ROOT = PACKAGE_ROOT.parents[2]


def _project_requirements(pyproject: Path) -> dict[str, Requirement]:
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    requirements = (Requirement(value) for value in project["dependencies"])
    return {requirement.name: requirement for requirement in requirements}


def _assert_exact_vcs_requirements(requirements: dict[str, Requirement]) -> None:
    transformers = requirements["transformers"]
    assert transformers.url == f"git+{TRANSFORMERS_SOURCE_URL}@{TRANSFORMERS_REVISION}"
    assert not transformers.specifier

    fla = requirements["flash-linear-attention"]
    assert fla.url == f"git+{FLA_RWKV7_SOURCE_URL}@{FLA_RWKV7_REVISION}"
    assert fla.extras == {"flash-rwkv"}
    assert not fla.specifier
    assert "rwkv7-hf-adapter" not in requirements


def test_manifest_pins_standalone_runtime_revisions() -> None:
    pyproject = PACKAGE_ROOT / "pyproject.toml"
    document = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    _assert_exact_vcs_requirements(_project_requirements(pyproject))
    assert document.get("tool", {}).get("uv", {}).get("sources") is None
    assert Requirement(TRANSFORMERS_REQUIREMENT).url == (
        f"git+{TRANSFORMERS_SOURCE_URL}@{TRANSFORMERS_REVISION}"
    )
    assert Requirement(FLA_RWKV7_REQUIREMENT).url == (
        f"git+{FLA_RWKV7_SOURCE_URL}@{FLA_RWKV7_REVISION}"
    )
    uv = document["tool"]["uv"]
    assert uv["extra-build-dependencies"]["causal-conv1d"] == [
        {"requirement": "torch", "match-runtime": True}
    ]
    assert uv["extra-build-variables"]["causal-conv1d"] == {
        "CAUSAL_CONV1D_FORCE_BUILD": "TRUE"
    }


def test_product_root_does_not_duplicate_standalone_runtime_dependencies() -> None:
    document = tomllib.loads(
        (PRODUCT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert "any2rwkv" not in document.get("dependency-groups", {})


def test_standalone_wheel_metadata_retains_exact_runtime_revisions(
    tmp_path: Path,
) -> None:
    uv = shutil.which("uv")
    assert uv is not None, "uv is required to exercise the standalone PEP 517 build"

    standalone = tmp_path / "standalone"
    standalone.mkdir()
    shutil.copy2(PACKAGE_ROOT / "pyproject.toml", standalone / "pyproject.toml")
    shutil.copytree(PACKAGE_ROOT / "any2rwkv", standalone / "any2rwkv")
    assert not (tmp_path / "rwkv-hf").exists()

    wheel_dir = tmp_path / "wheel"
    subprocess.run(
        [
            uv,
            "build",
            "--wheel",
            "--offline",
            "--no-build-isolation",
            "--python",
            sys.executable,
            "--out-dir",
            str(wheel_dir),
            str(standalone),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    wheels = list(wheel_dir.glob("any2rwkv-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata_bytes = archive.read(metadata_name)

    metadata = email.message_from_bytes(metadata_bytes)
    parsed_requirements = [
        Requirement(value) for value in metadata.get_all("Requires-Dist", [])
    ]
    requirements = {
        requirement.name: requirement for requirement in parsed_requirements
    }
    _assert_exact_vcs_requirements(requirements)
    assert b"rwkv7-hf-adapter" not in metadata_bytes
    assert b"../rwkv-hf" not in metadata_bytes
