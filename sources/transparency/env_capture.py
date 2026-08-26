"""Capture the orchestrator environment into the ASTRA capsule.

Emitted as an ``environment:`` block inside ``astra.yaml`` (a registered
extension beside ``extraction:``). Stdlib only, zero network, and every
field degrades to an honest ``"absent (<reason>)"`` string instead of
crashing the export or silently vanishing.

Scope is EXPLICITLY PARTIAL: this captures the orchestrator process only.
The generated science workflows execute in a separate python3.12 sandbox
venv (``sources/utils/ensure_env.py``) plus live Toolomics services, none of
which is captured here — the emitted ``runner_env`` field says so verbatim.

Public-repo discipline: no absolute local paths and no key material may
appear in any emitted value. The run config is therefore NEVER embedded —
only a sha256 digest of it, computed after recursively dropping every key
whose name contains a secret marker (key/token/secret/password/api).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    sys.path.insert(
        0,
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )

from sources.utils.git_info import get_git_info

# Substring markers (case-insensitive) of config keys that must never reach
# the digest input, even hashed alongside benign keys.
_SECRET_MARKERS = ("key", "token", "secret", "password", "api")

# Model-role fields read verbatim from the run config (concrete model ids —
# upstream policy keeps tiers out of configs).
_MODEL_ROLE_FIELDS = (
    "smolagent_model_id",
    "planner_llm_model",
    "workflow_llm_model",
    "judge_model",
    "judge_extraction_model",
    "vision_judge_model",
    "capsule_namer_model",
)

_RUNNER_ENV_NOTE = "partial (orchestrator only; sandbox runner venv uncaptured)"

# The exporter's own decision-extraction calls (temperature 0.0) live in the
# same memory dir; they are the capture instrument, not the subject run, and
# folding them in would misstate the run's temperature range.
_EXTRACTOR_CALL_PREFIX = "astra_decision_step_"


def capture_environment(
    config: Any, memory_path: Path, run_metrics_path: Path | None
) -> dict[str, Any]:
    """Assemble the ``environment:`` block for one exported run.

    Args:
        config: The run's live config object (``config.Config`` in
            production; anything attribute-shaped in tests).
        memory_path: ``sources/memory/<uuid>`` — raw LLM-call JSONs, read
            for the temperature aggregation.
        run_metrics_path: ``sources/workflows/<uuid>/run_metrics.json`` or
            ``None`` when the workflow dir is not configured.
    """
    return {
        "git": get_git_info(Path(__file__).resolve().parent),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "model_roles": _model_roles(config),
        "temperature": _temperature_aggregate(memory_path),
        "config_digest": _config_digest(config),
        "grounding": _grounding_block(run_metrics_path),
        "runner_env": _RUNNER_ENV_NOTE,
    }


def _model_roles(config: Any) -> dict[str, Any] | str:
    """Model ids per role, verbatim from the config; honest-empty otherwise."""
    roles = {
        field: getattr(config, field, None)
        for field in _MODEL_ROLE_FIELDS
        if getattr(config, field, None)
    }
    return roles or "absent (config declares no model roles)"


def _temperature_aggregate(memory_path: Path) -> dict[str, Any] | str:
    """{min, max, n_calls} over every temperature the raw memory recorded."""
    if not memory_path.is_dir():
        return "absent (memory dir unavailable)"
    temperatures: list[float] = []
    for json_path in sorted(memory_path.glob("*.json")):
        if json_path.name.startswith(_EXTRACTOR_CALL_PREFIX):
            continue
        temperatures.extend(_temperatures_in_file(json_path))
    if not temperatures:
        return "absent (no temperature recorded in memory JSONs)"
    return {
        "min": min(temperatures),
        "max": max(temperatures),
        "n_calls": len(temperatures),
    }


def _temperatures_in_file(json_path: Path) -> list[float]:
    """Numeric ``temperature`` values in one memory JSON; [] when unreadable.

    Two shapes exist on disk: LLM-call cache files (a dict with a top-level
    ``temperature``) and agent traces like ``task_*.json`` (a list of step
    dicts). Unreadable or unexpected files contribute nothing — the caller's
    empty aggregate already says "no temperature recorded".
    """
    try:
        document = json.loads(json_path.read_text())
    except (OSError, ValueError):
        return []
    records = document if isinstance(document, list) else [document]
    return [
        record["temperature"]
        for record in records
        if isinstance(record, dict) and _is_number(record.get("temperature"))
    ]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _config_digest(config: Any) -> str:
    """sha256 over the secret-scrubbed, canonically-serialised config."""
    try:
        raw = config.jsonify() if hasattr(config, "jsonify") else dict(vars(config))
    except Exception:
        return "absent (config not serializable)"
    scrubbed = _drop_secret_keys(raw)
    try:
        canonical = json.dumps(scrubbed, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return "absent (config not serializable)"
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _drop_secret_keys(value: Any) -> Any:
    """Recursively drop every mapping key whose name looks secret-bearing."""
    if isinstance(value, dict):
        return {
            key: _drop_secret_keys(item)
            for key, item in value.items()
            if not _is_secret_name(key)
        }
    if isinstance(value, (list, tuple)):
        return [_drop_secret_keys(item) for item in value]
    return value


def _is_secret_name(name: Any) -> bool:
    lowered = str(name).lower()
    return any(marker in lowered for marker in _SECRET_MARKERS)


def _grounding_block(run_metrics_path: Path | None) -> dict[str, Any] | str:
    """The run's grounding block, verbatim, labelled as self-declared.

    ``declared_by: subject`` marks that these numbers come from the run's own
    metrics file — independent observation is the evaluator's layer, never
    conflated with this one.
    """
    if run_metrics_path is None:
        return "absent (workflow dir not configured)"
    try:
        metrics = json.loads(Path(run_metrics_path).read_text())
    except OSError:
        return "absent (run_metrics.json unavailable)"
    except ValueError:
        return "absent (run_metrics.json unreadable)"
    grounding = metrics.get("grounding")
    if not isinstance(grounding, dict):
        return "absent (run_metrics.json has no grounding block)"
    return {**grounding, "declared_by": "subject"}


if __name__ == "__main__":
    import tempfile
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as tmp:
        memory = Path(tmp) / "memory"
        memory.mkdir()
        (memory / "verifier_x.json").write_text(json.dumps({"temperature": 0.2}))
        (memory / "task_agent.json").write_text(
            json.dumps([{"temperature": 1.0}, {"no_temp": True}])
        )
        (memory / "astra_decision_step_0.json").write_text(
            json.dumps({"temperature": 0.0})  # instrument call — excluded
        )
        metrics_path = Path(tmp) / "run_metrics.json"
        metrics_path.write_text(json.dumps({"grounding": {"hit_rate": 0.5}}))
        config = SimpleNamespace(
            smolagent_model_id="openrouter/deepseek/deepseek-v4-flash",
            judge_model="openrouter/deepseek/deepseek-v4-flash",
            openrouter_api_key="sk-SHOULD-NEVER-LEAK",
        )
        env = capture_environment(config, memory, metrics_path)
        assert env["temperature"] == {"min": 0.2, "max": 1.0, "n_calls": 2}, env
        assert env["grounding"] == {"hit_rate": 0.5, "declared_by": "subject"}, env
        assert env["model_roles"]["judge_model"].endswith("deepseek-v4-flash")
        assert env["config_digest"].startswith("sha256:")
        assert "SHOULD-NEVER-LEAK" not in json.dumps(env)
        no_secret = SimpleNamespace(
            smolagent_model_id="openrouter/deepseek/deepseek-v4-flash",
            judge_model="openrouter/deepseek/deepseek-v4-flash",
        )
        assert env["config_digest"] == _config_digest(no_secret), (
            "digest must be independent of dropped secret keys"
        )
        gone = capture_environment(config, Path(tmp) / "nope", None)
        assert gone["temperature"] == "absent (memory dir unavailable)", gone
        assert gone["grounding"] == "absent (workflow dir not configured)", gone
        assert gone["runner_env"].startswith("partial"), gone
    print("[OK] env_capture smoke check passed")
