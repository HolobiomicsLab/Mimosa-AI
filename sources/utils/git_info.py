"""Git provenance of the orchestrator checkout — {commit, branch, dirty}.

Shared by the benchmark run notes (``csv_mode``) and the ASTRA environment
capture (``sources/transparency/env_capture``) so both record the same facts
the same way. Stdlib only, no network.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_GIT_TIMEOUT_S = 5


def get_git_info(repo_dir: Path | None = None) -> dict:
    """Return {commit, branch, dirty} for the repo containing ``repo_dir``.

    ``repo_dir`` defaults to this module's directory (git resolves the
    enclosing repository itself). Values are ``None`` when git metadata is
    unavailable (no git binary, not a checkout, timeout), so a run outside a
    repo still records an honest shape instead of crashing.
    """
    cwd = repo_dir or Path(__file__).resolve().parent

    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
                timeout=_GIT_TIMEOUT_S,
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


if __name__ == "__main__":
    info = get_git_info()
    assert set(info) == {"commit", "branch", "dirty"}, info
    missing = get_git_info(Path("/"))
    assert missing["commit"] is None and missing["dirty"] is None, missing
    print(f"[OK] git_info smoke check passed ({info})")
