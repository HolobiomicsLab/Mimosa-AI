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
import re
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


def _parse_json(text: str) -> dict:
    """Best-effort JSON extraction from a model response (may be fenced/prose)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)
    return json.loads(text)


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


def _llm_json(system: str, user: str, temperature: float) -> dict:
    """One JSON-returning LLM call via litellm, using the planner model."""
    import litellm

    cfg = _persisted_config()
    model = cfg.planner_llm_model
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 1024,
    }
    if not model.lower().startswith("anthropic"):
        kwargs["temperature"] = temperature  # Anthropic rejects explicit temperature
    resp = litellm.completion(**kwargs)
    return _parse_json(resp.choices[0].message.content)


def cmd_refine(payload: dict) -> dict:
    from sources.cli.onboard_cli import _CLARIFIER_SYSTEM

    objective = payload.get("objective", "")
    history = payload.get("history", [])
    convo = objective
    if history:
        convo += "\n\nClarifications so far:\n" + "\n".join(
            f"Q: {h.get('question','')}\nA: {h.get('answer','')}" for h in history
        )
    out = _llm_json(_CLARIFIER_SYSTEM, convo, temperature=0.1)
    return {
        "is_clear": bool(out.get("is_clear")),
        "question": out.get("question"),
        "refined_prompt": out.get("refined_prompt") or objective,
    }


def cmd_classify(payload: dict) -> dict:
    from sources.cli.onboard_cli import _CLASSIFIER_SYSTEM

    out = _llm_json(_CLASSIFIER_SYSTEM, payload.get("objective", ""), temperature=0.0)
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
