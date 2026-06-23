"""Environment precheck & auto-provisioning for Mimosa.

Mimosa targets **Python 3.12**.  This module verifies that the host has a
working ``python3.12`` interpreter together with a usable ``pip`` for it, and
— when missing — attempts to install them automatically.

Public API
----------
- :func:`is_python312_installed`  -> bool
- :func:`is_pip_available`        -> bool
- :func:`install_python312`       -> bool
- :func:`install_pip_for_python312` -> bool
- :func:`ensure_environment`      -> bool  (the orchestrator used by ``main``)

The orchestrator is intentionally idempotent and side-effect-safe: it only
attempts installs when a check fails, prints clear status lines, and returns a
boolean so the caller can decide whether to abort.

Auto-install strategy (best-effort, platform aware):
  * Debian/Ubuntu      -> ``apt-get`` (+ deadsnakes PPA when needed)
  * RHEL/Fedora/CentOS -> ``dnf`` / ``yum``
  * Arch               -> ``pacman``
  * macOS              -> ``brew``
pip is bootstrapped via ``python3.12 -m ensurepip`` first, falling back to the
official ``get-pip.py`` bootstrap script downloaded from ``bootstrap.pypa.io``.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request

try:
    # Reuse Mimosa's pretty-print helpers when available.
    from sources.cli.pretty_print import print_ok, print_warn, print_err, print_info
except Exception:  # pragma: no cover - fallback if imported in isolation
    def print_ok(msg: str) -> None:
        print(f"  [OK]   {msg}")

    def print_warn(msg: str) -> None:
        print(f"  [WARN] {msg}")

    def print_err(msg: str) -> None:
        print(f"  [ERR]  {msg}")

    def print_info(msg: str) -> None:
        print(f"  [..]   {msg}")


PYTHON312 = "python3.12"
_REQUIRED_MAJOR = 3
_REQUIRED_MINOR = 12
_GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _run(cmd: list[str], *, check: bool = False, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a command, streaming output by default.

    Returns the CompletedProcess. When ``check`` is True a non-zero exit raises
    ``subprocess.CalledProcessError``.
    """
    print_info(f"$ {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
    )


def _which_python312() -> str | None:
    """Return the path to a ``python3.12`` executable, if any."""
    path = shutil.which(PYTHON312)
    if path:
        return path
    # The current interpreter might already be 3.12 even if not aliased.
    if (sys.version_info.major, sys.version_info.minor) == (_REQUIRED_MAJOR, _REQUIRED_MINOR):
        return sys.executable
    return None


def _is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def _sudo_prefix() -> list[str]:
    """Prepend ``sudo`` for privileged installs when not already root."""
    if _is_root():
        return []
    if shutil.which("sudo"):
        return ["sudo"]
    return []


def _detect_linux_package_manager() -> str | None:
    for mgr in ("apt-get", "dnf", "yum", "pacman", "zypper"):
        if shutil.which(mgr):
            return mgr
    return None


# ── Checks ──────────────────────────────────────────────────────────────────

def is_python312_installed() -> bool:
    """True if a ``python3.12`` interpreter is reachable on this host."""
    exe = _which_python312()
    if not exe:
        return False
    try:
        proc = _run([exe, "--version"], capture=True)
        version = (proc.stdout or proc.stderr or "").strip()
        return proc.returncode == 0 and "3.12" in version
    except Exception:
        return False


def is_pip_available(python_exe: str | None = None) -> bool:
    """True if ``pip`` is usable through ``python3.12 -m pip``."""
    exe = python_exe or _which_python312()
    if not exe:
        return False
    try:
        proc = _run([exe, "-m", "pip", "--version"], capture=True)
        return proc.returncode == 0
    except Exception:
        return False


# ── Installers ────────────────────────────────────────────────────────────────

def install_python312() -> bool:
    """Attempt to auto-install Python 3.12 for the current platform.

    Returns True on success (interpreter becomes reachable), else False.
    """
    system = platform.system()
    print_info(f"Attempting to install {PYTHON312} on {system}…")

    try:
        if system == "Linux":
            ok = _install_python312_linux()
        elif system == "Darwin":
            ok = _install_python312_macos()
        else:
            print_err(f"Automatic Python 3.12 install is unsupported on '{system}'.")
            print_info("Please install Python 3.12 manually: https://www.python.org/downloads/")
            return False
    except subprocess.CalledProcessError as exc:
        print_err(f"Python 3.12 installation command failed: {exc}")
        return False
    except Exception as exc:  # pragma: no cover - defensive
        print_err(f"Unexpected error during Python 3.12 install: {exc}")
        return False

    if ok and is_python312_installed():
        print_ok("Python 3.12 installed successfully.")
        return True
    print_err("Python 3.12 still not detected after installation attempt.")
    return False


def _install_python312_linux() -> bool:
    mgr = _detect_linux_package_manager()
    if not mgr:
        print_err("No supported Linux package manager found (apt/dnf/yum/pacman/zypper).")
        return False
    sudo = _sudo_prefix()

    if mgr == "apt-get":
        _run(sudo + ["apt-get", "update"])
        # Try the distro package first; fall back to the deadsnakes PPA for
        # older Ubuntu releases that don't ship 3.12.
        result = _run(sudo + ["apt-get", "install", "-y", "python3.12", "python3.12-venv"])
        if result.returncode != 0:
            print_warn("python3.12 not in default repos — adding deadsnakes PPA.")
            _run(sudo + ["apt-get", "install", "-y", "software-properties-common"])
            _run(sudo + ["add-apt-repository", "-y", "ppa:deadsnakes/ppa"])
            _run(sudo + ["apt-get", "update"])
            _run(sudo + ["apt-get", "install", "-y", "python3.12", "python3.12-venv"])
        return True

    if mgr in ("dnf", "yum"):
        _run(sudo + [mgr, "install", "-y", "python3.12"])
        return True

    if mgr == "pacman":
        # Arch rolling release ships the current python; python312 is in AUR.
        _run(sudo + ["pacman", "-Sy", "--noconfirm", "python"])
        return True

    if mgr == "zypper":
        _run(sudo + ["zypper", "--non-interactive", "install", "python312"])
        return True

    return False


def _install_python312_macos() -> bool:
    if not shutil.which("brew"):
        print_err("Homebrew not found. Install it from https://brew.sh then re-run.")
        return False
    _run(["brew", "install", "python@3.12"])
    return True


def install_pip_for_python312() -> bool:
    """Ensure pip is available for python3.12.

    Strategy:
      1. ``python3.12 -m ensurepip --upgrade`` (no network, bundled wheels).
      2. Fall back to the official ``get-pip.py`` bootstrap if ensurepip fails.
    Returns True once ``python3.12 -m pip`` works.
    """
    exe = _which_python312()
    if not exe:
        print_err("Cannot install pip: python3.12 is not available.")
        return False

    print_info("Bootstrapping pip via ensurepip…")
    try:
        proc = _run([exe, "-m", "ensurepip", "--upgrade"])
        if proc.returncode == 0 and is_pip_available(exe):
            _upgrade_pip(exe)
            print_ok("pip is available for python3.12.")
            return True
    except Exception as exc:
        print_warn(f"ensurepip failed: {exc}")

    print_warn("ensurepip unavailable — falling back to get-pip.py bootstrap.")
    if _bootstrap_get_pip(exe) and is_pip_available(exe):
        _upgrade_pip(exe)
        print_ok("pip is available for python3.12.")
        return True

    print_err("Failed to make pip available for python3.12.")
    return False


def _bootstrap_get_pip(python_exe: str) -> bool:
    """Download and run the official get-pip.py bootstrap script."""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            get_pip = os.path.join(tmp, "get-pip.py")
            print_info(f"Downloading {_GET_PIP_URL}")
            urllib.request.urlretrieve(_GET_PIP_URL, get_pip)
            proc = _run([python_exe, get_pip])
            return proc.returncode == 0
    except Exception as exc:
        print_err(f"get-pip.py bootstrap failed: {exc}")
        return False


def _upgrade_pip(python_exe: str) -> None:
    """Best-effort pip self-upgrade; never fatal."""
    try:
        _run([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
    except Exception:
        pass


# ── Orchestrator ───────────────────────────────────────────────────────────────

def ensure_environment(auto_install: bool = True) -> bool:
    """Verify (and optionally auto-provision) the Mimosa runtime environment.

    Steps:
      1. Ensure ``python3.12`` is installed (auto-install if missing).
      2. Ensure ``pip`` is available for ``python3.12`` (auto-install if missing).

    Args:
        auto_install: when False, only checks are run (no installs attempted).

    Returns:
        True if the environment satisfies the requirements at the end.
    """
    print_info("Verifying Mimosa runtime environment (Python 3.12 + pip)…")

    # 1) Python 3.12
    if is_python312_installed():
        print_ok("Python 3.12 is installed.")
    else:
        print_warn("Python 3.12 not found.")
        if not auto_install:
            print_err("Auto-install disabled — please install Python 3.12 manually.")
            return False
        if not install_python312():
            return False

    exe = _which_python312()

    # 2) pip for python3.12
    if is_pip_available(exe):
        print_ok("pip is available for python3.12.")
    else:
        print_warn("pip is not available for python3.12.")
        if not auto_install:
            print_err("Auto-install disabled — please install pip for python3.12 manually.")
            return False
        if not install_pip_for_python312():
            return False

    print_ok("Environment ready for Mimosa.")
    return True


if __name__ == "__main__":
    ok = ensure_environment(auto_install="--check-only" not in sys.argv)
    sys.exit(0 if ok else 1)
