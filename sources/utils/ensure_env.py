"""Environment precheck & auto-provisioning for the Mimosa workflow runner.

Mimosa itself can run under any recent Python, but it executes generated
workflows through :class:`sources.core.workflow_runner.WorkflowRunner`, which
targets **Python 3.12**. That runner already knows how to:

  * resolve a ``python3.12`` interpreter on ``PATH``,
  * create a managed virtualenv (``python3.12 -m venv``),
  * bootstrap pip *inside that venv* (``ensurepip``),
  * pip-install workflow dependencies into the venv.

What the runner **cannot** do is provision the interpreter itself: if
``python3.12`` is absent it raises ``RuntimeError`` and dies. On Debian/Ubuntu
it also silently needs the ``python3.12-venv`` package, without which
``python3.12 -m venv`` fails even when the interpreter is present.

This module fills exactly that gap (and nothing the runner already covers):

  1. ensure a working ``python3.12`` interpreter exists,
  2. ensure ``python3.12 -m venv`` works (the runner's actual dependency),

auto-installing the interpreter + venv support when missing. We deliberately do
*not* bootstrap system pip via ``get-pip.py`` — the runner manages pip inside
its own venv, so doing so here would be redundant scope.

Public API
----------
- :func:`is_python312_installed`   -> bool
- :func:`is_venv_available`        -> bool   (``python3.12 -m venv`` works)
- :func:`install_python312`        -> bool
- :func:`install_venv_support`     -> bool
- :func:`ensure_environment`       -> bool   (orchestrator used by ``main``)

Auto-install strategy (best-effort, platform aware):
  * Debian/Ubuntu      -> ``apt-get`` (+ deadsnakes PPA when needed)
  * RHEL/Fedora/CentOS -> ``dnf`` / ``yum``
  * Arch               -> ``pacman``
  * openSUSE           -> ``zypper``
  * macOS              -> ``brew``
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile

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


def is_venv_available(python_exe: str | None = None) -> bool:
    """True if ``python3.12 -m venv`` can actually create a venv.

    This is the capability the WorkflowRunner relies on (it builds a managed
    venv and bootstraps pip inside it). On Debian/Ubuntu the ``venv`` module is
    split into the separate ``python3.12-venv`` package, so importing it is not
    enough — we create a throwaway venv to be sure pip can be seeded.
    """
    exe = python_exe or _which_python312()
    if not exe:
        return False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            probe = os.path.join(tmp, "venv_probe")
            proc = _run([exe, "-m", "venv", probe], capture=True)
            return proc.returncode == 0
    except Exception:
        return False


# ── Installers ────────────────────────────────────────────────────────────────

def install_python312() -> bool:
    """Attempt to auto-install Python 3.12 (with venv support) for this host.

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


def install_venv_support() -> bool:
    """Ensure ``python3.12 -m venv`` works (installs distro venv package).

    Only meaningful on Debian/Ubuntu where venv is a separate package; on other
    platforms venv ships with the interpreter and this is a no-op verification.
    Returns True once ``python3.12 -m venv`` succeeds.
    """
    exe = _which_python312()
    if not exe:
        print_err("Cannot ensure venv support: python3.12 is not available.")
        return False

    system = platform.system()
    if system == "Linux" and _detect_linux_package_manager() == "apt-get":
        sudo = _sudo_prefix()
        print_info("Installing python3.12-venv (Debian/Ubuntu splits venv out)…")
        try:
            _run(sudo + ["apt-get", "install", "-y", "python3.12-venv"])
        except Exception as exc:
            print_warn(f"Could not install python3.12-venv: {exc}")

    if is_venv_available(exe):
        print_ok("python3.12 venv support is available.")
        return True

    print_err(
        "python3.12 -m venv is not functional. Install the venv package for "
        "your distro (e.g. 'python3.12-venv' on Debian/Ubuntu)."
    )
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
        # older Ubuntu releases that don't ship 3.12. We install -venv too so
        # the WorkflowRunner's `python3.12 -m venv` works out of the box.
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


# ── Orchestrator ───────────────────────────────────────────────────────────────

def ensure_environment(auto_install: bool = True) -> bool:
    """Verify (and optionally auto-provision) the workflow-runner environment.

    Complements :class:`WorkflowRunner` (which manages its own venv + pip) by
    guaranteeing the two things the runner cannot self-provision:

      1. a working ``python3.12`` interpreter,
      2. functional ``python3.12 -m venv`` (the runner builds a venv and
         bootstraps pip inside it via ``ensurepip``).

    Args:
        auto_install: when False, only checks are run (no installs attempted).

    Returns:
        True if the environment satisfies the runner's requirements.
    """
    print_info("Verifying Python 3.12 runner environment (interpreter + venv)…")

    # 1) Python 3.12 interpreter
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

    # 2) venv support (pip is bootstrapped by the runner inside the venv)
    if is_venv_available(exe):
        print_ok("python3.12 venv support is available.")
    else:
        print_warn("python3.12 -m venv is not functional.")
        if not auto_install:
            print_err("Auto-install disabled — please install python3.12 venv support manually.")
            return False
        if not install_venv_support():
            return False

    print_ok("Python 3.12 runner environment is ready.")
    return True


if __name__ == "__main__":
    ok = ensure_environment(auto_install="--check-only" not in sys.argv)
    sys.exit(0 if ok else 1)
