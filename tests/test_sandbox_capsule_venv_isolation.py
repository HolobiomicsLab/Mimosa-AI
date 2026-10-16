#!/usr/bin/env python3
"""Each graded capsule gets a fresh venv; one capsule's installs never reach another.

2026-10-16 night campaign: all SAB tasks installed into one shared grading
venv. deepchem 2.7.1 (Task 1, ClinTox) pinned scipy to 1.8.1 for every later
task, and that scipy crashed scikit-learn's PCA in Task 17 (drugex). The same
capsule regraded in a fresh venv passed VER and SR. Grading now builds an
immutable base layer once and a private venv per capsule on top of it.
Real venvs and pip are used, offline: local wheels stand in for scipy
(``mimosaprobe`` 1.0 and 2.0) and deepchem (``mimosadep``, which needs the
older ``mimosaprobe==1.0``, as deepchem 2.7.1 needs ``scipy<1.9``).
"""

import base64
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.benchmark_evaluation import execution_sandbox  # noqa: E402
from sources.benchmark_evaluation.execution_sandbox import (
    ExecutionSandbox,  # noqa: E402
)


def _wheel(directory: Path, name: str, version: str, requires: str = "") -> None:
    """Write a minimal pure-Python wheel ``<name>==<version>``."""
    stem = f"{name}-{version}"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    if requires:
        metadata += f"Requires-Dist: {requires}\n"
    files = {
        f"{name}.py": f"__version__ = '{version}'\n",
        f"{stem}.dist-info/METADATA": metadata,
        f"{stem}.dist-info/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    rows = []
    for path, text in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b"=").decode()
        rows.append(f"{path},sha256={digest},{len(text.encode())}")
    rows.append(f"{stem}.dist-info/RECORD,,")
    files[f"{stem}.dist-info/RECORD"] = "\n".join(rows) + "\n"
    with zipfile.ZipFile(directory / f"{stem}-py3-none-any.whl", "w") as archive:
        for path, text in files.items():
            archive.writestr(path, text)


def _version(python: Path) -> str:
    """Return the mimosaprobe version a venv's interpreter imports."""
    out = subprocess.run([str(python), "-c", "import mimosaprobe; print(mimosaprobe.__version__)"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


@pytest.fixture
def offline_wheels(tmp_path, monkeypatch):
    """Serve the probe wheels to pip without network access."""
    try:
        execution_sandbox.resolve_sandbox_python()
    except RuntimeError:
        pytest.skip(f"Python {execution_sandbox.SANDBOX_PYTHON_VERSION} not available")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _wheel(wheels, "mimosaprobe", "1.0")
    _wheel(wheels, "mimosaprobe", "2.0")
    _wheel(wheels, "mimosadep", "1.0", requires="mimosaprobe==1.0")
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("PIP_FIND_LINKS", str(wheels))
    monkeypatch.setattr(ExecutionSandbox, "_shared_venvs", {})
    yield
    ExecutionSandbox.cleanup_shared_venv()


def test_capsule_install_stays_in_its_own_venv(tmp_path, monkeypatch, offline_wheels):
    # Base spec allows both versions and resolves to the newest (2.0), like
    # "scipy<1.14.0" -> 1.13.1; the deepchem-like capsule needs 1.0.
    needs = {"deepchem_like": ["mimosadep"], "plain": []}
    monkeypatch.setattr(ExecutionSandbox, "_discover_capsule_dependencies",
                        lambda self: (needs[self.capsule_path.name], []))
    for name in needs:
        (tmp_path / name).mkdir()
    base = ["mimosaprobe<3"]

    first = ExecutionSandbox(tmp_path / "deepchem_like", base_packages=base)
    second = ExecutionSandbox(tmp_path / "plain", base_packages=base)

    assert first.base_venv_path == second.base_venv_path  # base stack built once
    assert first.venv_path != second.venv_path
    assert _version(first.python_exe) == "1.0"  # capsule's downgrade applies to it
    assert _version(second.python_exe) == "2.0"  # later capsule unaffected
    assert _version(first.base_venv_path / "bin" / "python") == "2.0"  # base immutable
    first.cleanup()
    assert not first.venv_path.exists()
    assert _version(second.python_exe) == "2.0"
