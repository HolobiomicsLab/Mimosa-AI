"""Filesystem locations for repo-checkout and installed (pip / uv tool) runs.

Read-only assets shipped with the package (prompts, module templates) always
resolve relative to the package itself, whether that is a git checkout or
site-packages. Mutable state (memory, workflows, capsules, caches, persisted
config) stays inside the checkout for repo runs — preserving the historical
layout — and moves to XDG user directories (``~/.config/mimosa``,
``~/.local/share/mimosa``, ``~/.cache/mimosa``) for installed runs.
"""

import os
from pathlib import Path

APP_NAME = "mimosa"

# Directory containing ``sources/``, ``config.py`` and ``main.py`` —
# the repo root in a checkout, site-packages when installed as a wheel.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def is_repo_checkout() -> bool:
    """Return True when running from a source checkout, not an installed wheel.

    Wheels never ship ``pyproject.toml`` next to the package, so its
    presence at ``PACKAGE_ROOT`` identifies a checkout.
    """
    return (PACKAGE_ROOT / "pyproject.toml").is_file()


def resource_path(relative: str) -> str:
    """Return the absolute path of a read-only file shipped with the package.

    Args:
        relative: Path relative to the package root, e.g. ``sources/prompts/x.md``.
    """
    return str(PACKAGE_ROOT / relative)


def config_dir() -> Path:
    """Return the user configuration directory (``~/.config/mimosa`` by default)."""
    return _xdg_dir("XDG_CONFIG_HOME", "~/.config")


def data_dir() -> Path:
    """Return the user data directory (``~/.local/share/mimosa`` by default)."""
    return _xdg_dir("XDG_DATA_HOME", "~/.local/share")


def cache_dir() -> Path:
    """Return the user cache directory (``~/.cache/mimosa`` by default)."""
    return _xdg_dir("XDG_CACHE_HOME", "~/.cache")


def _xdg_dir(env_var: str, fallback: str) -> Path:
    base = Path(os.environ.get(env_var) or fallback).expanduser()
    return base / APP_NAME


def user_config_file() -> Path:
    """Return the persisted configuration file written by the onboarding CLI."""
    return config_dir() / "config.json"


def user_env_file() -> Path:
    """Return the optional dotenv file holding API keys (``~/.config/mimosa/.env``)."""
    return config_dir() / ".env"


def state_dir(repo_relative: str, installed_name: str) -> str:
    """Return the absolute directory for mutable runtime state.

    Args:
        repo_relative: Location inside a checkout, e.g. ``sources/memory``.
        installed_name: Folder name under the XDG data dir, e.g. ``memory``.
    """
    if is_repo_checkout():
        return str(PACKAGE_ROOT / repo_relative)
    return str(data_dir() / installed_name)


def default_memory_dir() -> str:
    """Return the default agent-memory directory."""
    return state_dir("sources/memory", "memory")


def default_workflow_dir() -> str:
    """Return the default generated-workflows directory."""
    return state_dir("sources/workflows", "workflows")


def default_runs_capsule_dir() -> str:
    """Return the default run-capsule output directory."""
    return state_dir("runs_capsule", "runs_capsule")


def default_tmp_dir() -> str:
    """Return the scratch directory used by the WorkflowRunner."""
    if is_repo_checkout():
        return str(PACKAGE_ROOT / "tmp")
    return str(cache_dir() / "tmp")


def pricing_cache_file() -> str:
    """Return the cache file for OpenRouter pricing data."""
    if is_repo_checkout():
        return str(PACKAGE_ROOT / "sources" / "cache" / "openrouter_pricing.json")
    return str(cache_dir() / "openrouter_pricing.json")


if __name__ == "__main__":
    print(f"PACKAGE_ROOT      = {PACKAGE_ROOT}")
    print(f"is_repo_checkout  = {is_repo_checkout()}")
    print(f"config_dir        = {config_dir()}")
    print(f"data_dir          = {data_dir()}")
    print(f"cache_dir         = {cache_dir()}")
    print(f"memory_dir        = {default_memory_dir()}")
    print(f"workflow_dir      = {default_workflow_dir()}")
    print(f"runs_capsule_dir  = {default_runs_capsule_dir()}")
    print(f"tmp_dir           = {default_tmp_dir()}")
    print(f"pricing_cache     = {pricing_cache_file()}")
