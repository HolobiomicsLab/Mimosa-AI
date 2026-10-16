#!/usr/bin/env python3
"""
Tests for source-build constraint handling in ExecutionSandbox.

Bug: phonopy 2.x ships sdist-only releases and declares an unpinned
``build-system.requires = ["scikit-build-core", "nanobind", "numpy"]``.
pip's PEP 517 build isolation ignores ``-c`` constraint files, so the build
environment drifted to scikit-build-core >= 0.10, which removed the
``cmake.verbose`` setting phonopy 2.29's pyproject still sets. Every
``pip install -c constraints.txt phonopy`` died at "Getting requirements to
build wheel" with "ERROR: Use build.verbose instead of cmake.verbose for
scikit-build-core >= 0.10", and the phonon tasks (plot_phonon_dos /
plot_phonon_band_structure) were excluded as infra in every campaign from
2026-09-02 to 2026-09-18.

Fix: packages listed in SOURCE_BUILD_CONSTRAINTS are split out of the merged
install transaction and installed in their own pip call with the
``PIP_CONSTRAINT`` environment variable pointing at per-package backend pins
(the one mechanism pip honours inside build environments). Verified against
pip 25.1.1: phonopy 2.29.1 builds with scikit-build-core 0.9.10 and imports
against numpy 1.26.4.

These tests are offline: pip invocations are captured via a fake
``_run_process``.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.benchmark_evaluation.execution_sandbox import (
    SOURCE_BUILD_CONSTRAINTS,
    EvalInfraError,
    ExecutionSandbox,
)


def _bare_sandbox(
    tmp_path: Path, constraints_file: str | None = None
) -> ExecutionSandbox:
    """ExecutionSandbox without the heavy venv/env setup done in __init__."""
    sb = object.__new__(ExecutionSandbox)
    sb.logger = logging.getLogger("tests.sandbox_source_build")
    base_dir = tmp_path / "mimosa_base_venv_test"
    base_dir.mkdir()
    sb.venv_path = base_dir / "venv"
    sb.venv_path.mkdir()
    sb.python_exe = sb.venv_path / "bin" / "python"
    sb.pip_exe = sb.venv_path / "bin" / "pip"
    sb._constraints_file = constraints_file
    return sb


class _FakeRunProcess:
    """Records pip calls; returns canned CompletedProcess-like results."""

    def __init__(self, returncode=0, stderr=""):
        self.calls: list[dict] = []
        self._returncode = returncode
        self._stderr = stderr

    def __call__(self, cmd, timeout, cwd=None, env=None):
        self.calls.append(
            {"cmd": list(cmd), "timeout": timeout, "cwd": cwd, "env": env}
        )
        return subprocess.CompletedProcess(cmd, self._returncode, "", self._stderr)


# --- _install_source_constrained --------------------------------------------


def test_pip_constraint_env_and_constraints_file(tmp_path):
    cons = tmp_path / "constraints.txt"
    cons.write_text("phonopy<2.30\n")
    sb = _bare_sandbox(tmp_path, constraints_file=str(cons))
    fake = _FakeRunProcess()
    sb._run_process = fake

    sb._install_source_constrained(["phonopy"])

    assert len(fake.calls) == 1
    call = fake.calls[0]
    # Runtime constraints still apply; the requirement itself is passed alone.
    assert call["cmd"] == [
        str(sb.pip_exe),
        "install",
        "--quiet",
        "-c",
        str(cons),
        "phonopy",
    ]
    # The build constraints file is written next to the venv with the pins.
    build_cons = sb.venv_path.parent / "build_constraints_phonopy.txt"
    assert build_cons.exists()
    assert (
        build_cons.read_text() == "\n".join(SOURCE_BUILD_CONSTRAINTS["phonopy"]) + "\n"
    )
    # PIP_CONSTRAINT is the mechanism that reaches the isolated build env.
    assert call["env"]["PIP_CONSTRAINT"] == str(build_cons)
    assert call["env"]["PATH"] == os.environ["PATH"]  # env copied, not replaced


def test_failure_raises_infra_error_not_agent_failure(tmp_path):
    sb = _bare_sandbox(tmp_path, constraints_file=str(tmp_path / "c.txt"))
    sb._run_process = _FakeRunProcess(
        returncode=1,
        stderr="ERROR: Use build.verbose instead of cmake.verbose",
    )

    with pytest.raises(EvalInfraError) as excinfo:
        sb._install_source_constrained(["phonopy"])
    # The message keeps the diagnosis shape used by _install_packages.
    assert "Required package install failed" in str(excinfo.value)
    assert "cmake.verbose" in str(excinfo.value)


def test_without_constraints_file_pip_constraint_still_set(tmp_path):
    sb = _bare_sandbox(tmp_path, constraints_file=None)
    fake = _FakeRunProcess()
    sb._run_process = fake

    sb._install_source_constrained(["phonopy"])

    cmd = fake.calls[0]["cmd"]
    assert "-c" not in cmd
    assert fake.calls[0]["env"]["PIP_CONSTRAINT"].endswith(
        "build_constraints_phonopy.txt"
    )


# --- _setup_environment split ------------------------------------------------


def test_setup_environment_splits_source_pinned_out_of_merged(tmp_path):
    sb = _bare_sandbox(tmp_path, constraints_file=str(tmp_path / "c.txt"))
    sb.base_packages = ["numpy<2.0"]

    installed: list[list[str]] = []
    source_installed: list[list[str]] = []

    def fake_install(packages):
        installed.append(list(packages))

    def fake_source(packages):
        source_installed.append(list(packages))

    sb._install_packages = fake_install
    sb._install_source_constrained = fake_source
    sb._run_special_case_installs = lambda present: None
    sb._discover_capsule_dependencies = lambda: (["phonopy", "seekpath"], ["phonopy"])

    sb._setup_environment()

    # phonopy leaves the merged transaction; other per-task deps stay in it.
    assert installed == [["numpy<2.0", "seekpath"]]
    assert source_installed == [["phonopy"]]


def test_setup_environment_without_source_pinned_keeps_one_call(tmp_path):
    sb = _bare_sandbox(tmp_path, constraints_file=str(tmp_path / "c.txt"))
    sb.base_packages = ["numpy<2.0"]

    calls: list[list[str]] = []
    sb._install_packages = lambda packages: calls.append(list(packages))

    def fake_source(packages):
        if packages:  # the real method is a no-op for an empty list
            calls.append(["SOURCE"] + list(packages))

    sb._install_source_constrained = fake_source
    sb._run_special_case_installs = lambda present: None
    sb._discover_capsule_dependencies = lambda: (["h5py"], [])

    sb._setup_environment()

    assert calls == [["numpy<2.0", "h5py"]]
