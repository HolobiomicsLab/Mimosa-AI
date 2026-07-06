"""Bridge executed *inside the Mimosa venv* by the Observatory backend.

The backend venv is intentionally minimal (no torch/smolagents), so anything
that needs Mimosa itself — LLM-backed goal refinement, mode classification, and
launching a run — is shelled out to this script with `MIMOSA_ROOT` on the path
and the project `.env` loaded. Communication is JSON: args on stdin (refine,
classify) or a single JSON argv (run), results on stdout as one JSON line
prefixed with ``@@RESULT@@`` so log noise before it is ignored.

Subcommands:
  refine    stdin {objective, history:[{question,answer}]} -> {is_clear, question, refined_prompt}
  classify  stdin {objective} -> {mode, confidence, reasoning, suggested_label}
  run       argv[2]=JSON {objective, mode, learn, judge, workflow_dir, memory_dir, workspace_dir}
"""

from __future__ import annotations

import json
import os
import sys

RESULT_MARKER = "@@RESULT@@"


def _emit(obj: dict) -> None:
    sys.stdout.write(RESULT_MARKER + json.dumps(obj) + "\n")
    sys.stdout.flush()


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()  # cwd is MIMOSA_ROOT
        load_dotenv(os.path.expanduser("~/.config/mimosa/.env"))
    except Exception:
        pass


def _persisted_config():
    """Load Mimosa config, preferring the persisted file for the user's models."""
    import config as cfgmod

    cfg = cfgmod.Config()
    for candidate in (
        os.path.join(os.getcwd(), "config_default.json"),
        os.path.expanduser("~/.config/mimosa/config.json"),
    ):
        if os.path.exists(candidate):
            try:
                cfg.load(candidate)
                break
            except Exception:
                continue
    return cfg


def _llm_json(
    system: str,
    user: str,
    temperature: float,
    expected_keys: dict[str, str],
    max_tokens: int = 1024,
) -> dict:
    """Robust JSON-returning LLM call, reusing the CLI onboarding's own helpers.

    ``_build_llm`` sets a low reasoning effort so the token budget isn't spent
    reasoning and truncating the JSON, and ``_call_llm_json`` recovers malformed
    output by stripping fences, repairing, retrying with self-correction, and
    finally regex-extracting ``expected_keys`` — the path that made the CLI
    reliable and that the bridge previously reimplemented without any recovery.
    """
    from sources.cli.onboard_cli import _build_llm, _call_llm_json

    llm = _build_llm(_persisted_config(), temperature=temperature, max_tokens=max_tokens)
    return _call_llm_json(llm, system, user, expected_keys=expected_keys)


def cmd_refine(payload: dict) -> dict:
    from sources.cli.onboard_cli import _CLARIFIER_SYSTEM

    objective = payload.get("objective", "")
    history = payload.get("history", [])
    convo = objective
    if history:
        convo += "\n\nClarifications so far:\n" + "\n".join(
            f"Q: {h.get('question','')}\nA: {h.get('answer','')}" for h in history
        )
    try:
        out = _llm_json(
            _CLARIFIER_SYSTEM,
            convo,
            temperature=0.1,
            expected_keys={"is_clear": "bool", "question": "str", "refined_prompt": "str"},
        )
    except Exception as exc:  # refinement is optional — never block the wizard
        return {
            "is_clear": True,
            "question": None,
            "refined_prompt": objective,
            "degraded": True,
            "note": f"{type(exc).__name__}: {exc}",
        }
    return {
        "is_clear": bool(out.get("is_clear")),
        "question": out.get("question"),
        "refined_prompt": out.get("refined_prompt") or objective,
    }


def cmd_classify(payload: dict) -> dict:
    from sources.cli.onboard_cli import _CLASSIFIER_SYSTEM

    objective = payload.get("objective", "")
    try:
        out = _llm_json(
            _CLASSIFIER_SYSTEM,
            objective,
            temperature=0.0,
            max_tokens=512,
            expected_keys={
                "mode": "str",
                "confidence": "number",
                "reasoning": "str",
                "suggested_label": "str",
            },
        )
    except Exception as exc:  # fall back to task mode rather than erroring
        return {
            "mode": "task",
            "confidence": None,
            "reasoning": "Automatic mode selection was unavailable; defaulting to task mode.",
            "suggested_label": None,
            "degraded": True,
            "note": f"{type(exc).__name__}: {exc}",
        }
    mode = out.get("mode")
    if mode not in ("task", "goal"):
        mode = "task"
    return {
        "mode": mode,
        "confidence": out.get("confidence"),
        "reasoning": out.get("reasoning"),
        "suggested_label": out.get("suggested_label"),
    }


def cmd_run(payload: dict) -> None:
    import asyncio

    cfg = _persisted_config()
    # Force the run's outputs into the dirs the Observatory watches.
    if payload.get("workflow_dir"):
        cfg.workflow_dir = payload["workflow_dir"]
    if payload.get("memory_dir"):
        cfg.memory_dir = payload["memory_dir"]
    if payload.get("workspace_dir"):
        cfg.workspace_dir = payload["workspace_dir"]
    cfg.create_paths()

    objective = payload["objective"]
    judge = bool(payload.get("judge", True))
    learn = bool(payload.get("learn", False))
    mode = payload.get("mode", "task")

    print(f"[bridge] launching {mode} mode · learn={learn} · judge={judge}", flush=True)
    print(f"[bridge] workflow_dir={cfg.workflow_dir}", flush=True)

    if mode == "goal":
        from sources.core.planner import Planner

        planner = Planner(cfg, enable_tts=False)
        planner.use_visualization = False
        asyncio.run(planner.start_planner(goal=objective, judge=judge))
    else:
        from sources.core.evolution_engine import EvolutionEngine

        engine = EvolutionEngine(cfg)
        asyncio.run(
            engine.start_workflow_evolution(
                goal=objective,
                template_uuid=None,
                judge=judge,
                enable_evolution=learn,
            )
        )
    print("[bridge] run complete", flush=True)


def main() -> int:
    _load_env()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        if cmd == "run":
            cmd_run(json.loads(sys.argv[2]))
            return 0
        payload = json.loads(sys.stdin.read() or "{}")
        if cmd == "refine":
            _emit({"ok": True, "result": cmd_refine(payload)})
        elif cmd == "classify":
            _emit({"ok": True, "result": cmd_classify(payload)})
        else:
            _emit({"ok": False, "error": f"unknown command: {cmd}"})
            return 2
        return 0
    except Exception as exc:  # surface a clean error the backend can relay
        if cmd == "run":
            print(f"[bridge] ERROR: {type(exc).__name__}: {exc}", flush=True)
            return 1
        _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
