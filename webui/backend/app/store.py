"""Read-only access to a run's on-disk artifacts under ``sources/workflows``.

Every function is defensive: partial, crashed, or in-flight runs are the norm
(a 0-byte ``state_result.json`` means "still running or crashed"), so nothing
here raises on missing/empty files — callers get ``None`` and decide.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .settings import get_settings

# A run dir is recognised by any of these artifacts existing inside it.
_RUN_MARKERS = ("run_metrics.json", "state_result.json")
_RUN_GLOB_MARKERS = ("goal_*.txt", "lineage_*.json", "workflow_genotype_*.py")
_UUID_TS = re.compile(r"(\d{8})_(\d{6})")


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _read_json(path: Path) -> Any | None:
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _find(run_dir: Path, pattern: str) -> Path | None:
    """First file in *run_dir* matching a glob, or None."""
    for p in sorted(run_dir.glob(pattern)):
        return p
    return None


def parse_created_at(run_id: str) -> str | None:
    """Recover an ISO timestamp from a run id like ``20260703_151955_649ef575``."""
    m = _UUID_TS.search(run_id)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(0), "%Y%m%d_%H%M%S").isoformat()
    except ValueError:
        return None


def workflow_dir() -> Path:
    return get_settings().workflow_dir


def run_path(run_id: str) -> Path:
    return workflow_dir() / run_id


def is_run_dir(path: Path) -> bool:
    if not path.is_dir() or path.name.startswith((".", "_")):
        return False
    if any((path / m).exists() for m in _RUN_MARKERS):
        return True
    return any(_find(path, g) for g in _RUN_GLOB_MARKERS)


def list_run_ids() -> list[str]:
    root = workflow_dir()
    if not root.is_dir():
        return []
    ids = [p.name for p in root.iterdir() if is_run_dir(p)]
    # Newest first (uuid prefix is a sortable timestamp).
    return sorted(ids, reverse=True)


def _coerce_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_run_metrics(run_id: str) -> dict[str, Any] | None:
    data = _read_json(run_path(run_id) / "run_metrics.json")
    if not isinstance(data, dict):
        return None
    # Drop the 384-float embedding from list responses; keep it addressable.
    data = dict(data)
    data.pop("qd_descriptor", None)
    return data


def read_lineage(run_id: str) -> dict[str, Any] | None:
    p = _find(run_path(run_id), "lineage_*.json")
    data = _read_json(p) if p else None
    return data if isinstance(data, dict) else None


def read_evaluation_scores(run_id: str) -> dict[str, Any] | None:
    """Pull the coerced ``evaluation.<type>`` block out of state_result.json.

    Values in that block are stored as strings ("1.0", "17", "False"); we
    coerce the numeric/boolean ones so the API returns real JSON types.
    """
    state = _read_json(run_path(run_id) / "state_result.json")
    if not isinstance(state, dict):
        return None
    evaluation = state.get("evaluation")
    if not isinstance(evaluation, dict) or not evaluation:
        return None
    eval_type, raw = next(iter(evaluation.items()))
    if not isinstance(raw, dict):
        return None
    scores: dict[str, Any] = {"eval_type": eval_type}
    for key, val in raw.items():
        if key in {"failure_fingerprint"}:
            continue  # numeric-vector detail, not needed for summaries
        if isinstance(val, str) and val in {"True", "False"}:
            scores[key] = val == "True"
        elif isinstance(val, str) and re.fullmatch(r"-?\d+(\.\d+)?", val):
            scores[key] = float(val) if "." in val else int(val)
        else:
            scores[key] = val
    return scores


# Per-claim block in a verifier evaluation.txt, e.g.
#   [claim_id] (importance=10; literal deliverable) The workflow's output ...
#     relevant_files: ['hello.py']
#     kind=executable status=pass score=1.0
#     details: Output contains 'hello'
_CLAIM_HEADER = re.compile(r"^\[([^\]]+)\]\s*\(importance=(\d+);\s*(.*?)\)\s*(.*)$")
_KIND_LINE = re.compile(r"kind=(\S+)\s+status=(\S+)\s+score=(\S+)")
_QUOTED = re.compile(r"'([^']*)'|\"([^\"]*)\"")


def _parse_relevant_files(value: str) -> list[str]:
    return [a or b for a, b in _QUOTED.findall(value)]


def read_evaluation_claims(run_id: str) -> list[dict[str, Any]] | None:
    """Parse the per-claim pass/fail detail out of a verifier evaluation.txt.

    Returns one dict per claim (id, importance, rationale, description, status,
    score, kind, relevant_files, details). Returns None for non-verifier report
    formats (no claim blocks) or a missing/empty file.
    """
    text = _read_text(run_path(run_id) / "evaluation.txt")
    if not text:
        return None
    claims: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in text.splitlines():
        header = _CLAIM_HEADER.match(raw) if raw.startswith("[") else None
        if header:
            if current:
                claims.append(current)
            current = {
                "id": header.group(1),
                "importance": int(header.group(2)),
                "rationale": header.group(3).strip(),
                "description": header.group(4).strip(),
                "relevant_files": [],
                "kind": None,
                "status": None,
                "score": None,
                "details": None,
            }
            continue
        if current is None:
            continue
        line = raw.strip()
        if line.startswith("relevant_files:"):
            current["relevant_files"] = _parse_relevant_files(line.split(":", 1)[1])
        elif line.startswith("kind="):
            kind = _KIND_LINE.search(line)
            if kind:
                current["kind"] = kind.group(1)
                current["status"] = kind.group(2)
                current["score"] = _coerce_number(kind.group(3))
        elif line.startswith("details:"):
            current["details"] = line.split(":", 1)[1].strip()
    if current:
        claims.append(current)
    return claims or None


_LIVE_WINDOW_S = 180  # a run touched within this window is treated as in-flight


def _newest_mtime(run_dir: Path) -> float:
    newest = 0.0
    try:
        for child in run_dir.iterdir():
            try:
                newest = max(newest, child.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        return 0.0
    return newest


def run_status(run_id: str) -> str:
    """completed | error | crashed | running — from artifact presence/flags.

    A run with metrics is finished (``on_error`` distinguishes error from ok).
    Without metrics we fall back to state_result presence, and finally to file
    recency: a dir last touched recently is still running, otherwise it died
    mid-run and is reported as crashed rather than falsely "running".
    """
    run_dir = run_path(run_id)
    metrics = _read_json(run_dir / "run_metrics.json")
    if isinstance(metrics, dict):
        return "error" if metrics.get("on_error") else "completed"
    state = run_dir / "state_result.json"
    if state.exists():
        try:
            if state.stat().st_size > 2:
                return "completed"
        except OSError:
            return "crashed"
    if time.time() - _newest_mtime(run_dir) < _LIVE_WINDOW_S:
        return "running"
    return "crashed"


def overall_score(run_id: str, metrics: dict[str, Any] | None = None) -> float | None:
    metrics = metrics if metrics is not None else read_run_metrics(run_id)
    if metrics and metrics.get("overall_score") is not None:
        return _coerce_number(metrics.get("overall_score"))
    scores = read_evaluation_scores(run_id)
    if scores:
        return _coerce_number(scores.get("overall_score"))
    return None


def run_summary(run_id: str) -> dict[str, Any]:
    metrics = read_run_metrics(run_id)
    lineage = read_lineage(run_id)
    goal = _read_text(_find(run_path(run_id), "goal_*.txt") or Path("/nonexistent"))
    return {
        "id": run_id,
        "created_at": parse_created_at(run_id),
        "goal": (goal or "").strip() or (lineage or {}).get("goal_snippet"),
        "status": run_status(run_id),
        "score": overall_score(run_id, metrics),
        "cost_usd": _coerce_number((metrics or {}).get("cumulative_cost_usd")),
        "iteration": (metrics or lineage or {}).get("iteration"),
        "evolution_kind": (metrics or lineage or {}).get("evolution_kind"),
        "parents": (metrics or {}).get("parent_uuids")
        or (lineage or {}).get("parents")
        or [],
        "wall_time_s": _coerce_number((metrics or {}).get("iteration_wall_time_s")),
        "is_single_agent": run_id.startswith("single_agent_"),
    }


def list_runs() -> list[dict[str, Any]]:
    return [run_summary(rid) for rid in list_run_ids()]


# Artifacts a run dir may contain, mapped to a stable logical name + kind.
_ARTIFACT_SPECS: list[tuple[str, str, str]] = [
    ("workflow_graph", "workflow_*.png", "image"),
    ("evolution_tree", "evolution_tree.png", "image"),
    ("reward_progress", "reward_progress.png", "image"),
    ("assertion_progress", "assertion_progress.png", "image"),
    ("genotype", "workflow_genotype_*.py", "code"),
    ("system_prompt", "system_prompt_*.md", "markdown"),
    ("evolution_prompt", "evolution_prompt_*.md", "markdown"),
    ("goal", "goal_*.txt", "text"),
    ("original_task", "original_task_*.txt", "text"),
    ("evaluation", "evaluation.txt", "text"),
    ("textual_gradient", "textual_gradient.txt", "text"),
    ("state_result", "state_result.json", "json"),
    ("run_metrics", "run_metrics.json", "json"),
    ("lineage", "lineage_*.json", "json"),
]


def list_artifacts(run_id: str) -> list[dict[str, Any]]:
    run_dir = run_path(run_id)
    if not run_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for name, pattern, kind in _ARTIFACT_SPECS:
        path = _find(run_dir, pattern)
        if not path:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        out.append(
            {
                "name": name,
                "filename": path.name,
                "kind": kind,
                "size": size,
                "empty": size <= 2,
            }
        )
    return out


def artifact_path(run_id: str, name: str) -> tuple[Path, str] | None:
    """Resolve a logical artifact name to (path, kind), or None."""
    for spec_name, pattern, kind in _ARTIFACT_SPECS:
        if spec_name == name:
            path = _find(run_path(run_id), pattern)
            return (path, kind) if path and path.exists() else None
    return None


def run_detail(run_id: str) -> dict[str, Any] | None:
    run_dir = run_path(run_id)
    if not run_dir.is_dir():
        return None
    summary = run_summary(run_id)
    summary.update(
        {
            "original_task": (
                _read_text(_find(run_dir, "original_task_*.txt") or Path("/x")) or ""
            ).strip()
            or None,
            "textual_gradient": _read_text(run_dir / "textual_gradient.txt"),
            "evaluation_text": _read_text(run_dir / "evaluation.txt"),
            "evaluation_scores": read_evaluation_scores(run_id),
            "evaluation_claims": read_evaluation_claims(run_id),
            "genotype": _read_text(_find(run_dir, "workflow_genotype_*.py") or Path("/x")),
            "evolution_prompt": _read_text(
                _find(run_dir, "evolution_prompt_*.md") or Path("/x")
            ),
            "metrics": read_run_metrics(run_id),
            "artifacts": list_artifacts(run_id),
        }
    )
    return summary
