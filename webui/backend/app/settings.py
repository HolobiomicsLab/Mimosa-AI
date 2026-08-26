"""Resolve where Mimosa keeps its on-disk artifacts.

The Observatory never runs Mimosa; it reads the directories Mimosa writes:
``<root>/sources/workflows`` (per-run evolution artifacts) and
``<root>/sources/memory`` (per-agent traces), plus the shared toolomics
workspace and, optionally, an ASB benchmark corpus (``MIMOSA_CORPUS_DIR``).
Everything is env-overridable so the API can point at any checkout, install,
or a copied dataset without code changes; the defaults derive from this
file's location in the repo, never from a developer machine's paths.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# webui/backend/app/settings.py → three parents up is the repo checkout root.
_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings:
    """Filesystem locations the API reads from. Env vars override every field."""

    def __init__(self) -> None:
        # Root of a Mimosa checkout/install whose runtime data we observe.
        root = Path(os.environ.get("MIMOSA_ROOT", _REPO_ROOT)).expanduser()
        self.root = root
        self.workflow_dir = Path(
            os.environ.get("MIMOSA_WORKFLOW_DIR", root / "sources" / "workflows")
        ).expanduser()
        self.memory_dir = Path(
            os.environ.get("MIMOSA_MEMORY_DIR", root / "sources" / "memory")
        ).expanduser()
        self.workspace_dir = Path(
            os.environ.get("MIMOSA_WORKSPACE_DIR", root / "workspace")
        ).expanduser()
        # Per-run workspace snapshots survive the live-workspace churn.
        self.snapshot_glob = os.environ.get("MIMOSA_SNAPSHOT_GLOB", "/tmp/mimosa_run_*")
        # ASTRA capsules the transparency exporter writes for a family's best
        # run, and the root scanned for ASB-as-evaluator evaluation capsules
        # (eval_astra.yaml) produced by asb_eval against this checkout's runs.
        self.capsule_dir = Path(
            os.environ.get("MIMOSA_CAPSULE_DIR", root / "runs_capsule")
        ).expanduser()
        self.eval_dir = Path(
            os.environ.get("MIMOSA_EVAL_DIR", root / "evaluations")
        ).expanduser()
        # Root of an ASB benchmark corpus (<corpus>/<challenge>/cards/…) for
        # the task-definition join. Unset by default — the corpus is a
        # separate checkout, so there is no honest in-repo default.
        corpus = os.environ.get("MIMOSA_CORPUS_DIR", "")
        self.corpus_dir: Path | None = Path(corpus).expanduser() if corpus else None
        # CORS origins for the dev frontend.
        self.cors_origins = os.environ.get(
            "MIMOSA_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")

        # ── Phase 2: setup & launch ──
        # The Mimosa config file the setup page reads/writes (checkout uses
        # config_default.json; installs use ~/.config/mimosa/config.json).
        default_config = root / "config_default.json"
        if not default_config.exists():
            default_config = Path.home() / ".config" / "mimosa" / "config.json"
        self.config_path = Path(
            os.environ.get("MIMOSA_CONFIG", default_config)
        ).expanduser()
        # Python that can import Mimosa (its venv) — used by the launch bridge.
        self.mimosa_python = Path(
            os.environ.get("MIMOSA_PYTHON", root / ".venv" / "bin" / "python")
        ).expanduser()
        # dotenv files scanned for API-key presence (never for values).
        self.env_files = [
            root / ".env",
            Path.home() / ".config" / "mimosa" / ".env",
        ]

    @property
    def known_key_names(self) -> list[str]:
        return [
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
            "DEEPSEEK_API_KEY", "MISTRAL_API_KEY", "HF_TOKEN",
        ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
