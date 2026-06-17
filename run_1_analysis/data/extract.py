"""Parse all workflow folders, build per-goal lineage table.

Outputs:
  /tmp/evo_report/workflows.csv       — one row per workflow with metrics
  /tmp/evo_report/goals.json          — goal_id -> {label, snippet, full_text, workflows}
  /tmp/evo_report/qd_archive.csv      — QD archive admission events
  /tmp/evo_report/variation_log.csv   — variation engine decisions
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path("/home/martin/Projects/CNRS/Mimosa-AI/sources/workflows/run_1")
OUT = Path("/tmp/evo_report")
OUT.mkdir(parents=True, exist_ok=True)


def short_label(snippet: str) -> str:
    """Return a short human-readable tag for a goal."""
    s = snippet.lower()
    if "clintox" in s:
        return "clintox"
    if "home range" in s or "elk" in s:
        return "elk_homerange"
    if "dkpes" in s:
        return "dkpes"
    if "bulk modulus" in s:
        return "bulk_modulus"
    if "shap" in s or "material diffusion" in s:
        return "shap_diffusion"
    # Fallback: first three words
    words = re.findall(r"\w+", snippet)
    return "_".join(words[:3]).lower() or "unknown"


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def parse_workflows() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    goals: dict[str, dict] = {}

    for folder in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        uuid = folder.name
        lin = load_json(folder / f"lineage_{uuid}.json") or {}
        metrics = load_json(folder / "run_metrics.json") or {}
        state = load_json(folder / "state_result.json") or {}

        snippet = (lin.get("goal_snippet") or "").strip()
        full_goal = (state.get("goal") or "").strip()
        if not snippet and full_goal:
            snippet = full_goal[:160]
        if not snippet:
            # try goal file
            for goal_file in folder.glob("goal_*.txt"):
                snippet = goal_file.read_text()[:160].strip()
                break

        # Normalize: strip whitespace, use first 120 chars to dedupe truncated snippets
        normalized = re.sub(r"\s+", " ", snippet).strip()[:120]
        goal_id = hashlib.md5(normalized.encode("utf-8")).hexdigest()[:8] if normalized else "unknown"
        label = short_label(snippet) if snippet else "unknown"
        if goal_id not in goals:
            goals[goal_id] = {
                "goal_id": goal_id,
                "label": label,
                "snippet": snippet,
                "full_text": full_goal,
                "workflows": [],
            }
        goals[goal_id]["workflows"].append(uuid)
        if full_goal and not goals[goal_id]["full_text"]:
            goals[goal_id]["full_text"] = full_goal

        # check for textual gradient (may indicate evolution feedback was produced)
        grad_path = folder / "textual_gradient.txt"
        has_gradient = grad_path.exists() and grad_path.stat().st_size > 0
        gradient_len = grad_path.stat().st_size if has_gradient else 0

        # workflow_genotype size (proxy for code complexity)
        geno_path = folder / f"workflow_genotype_{uuid}.py"
        geno_lines = 0
        if geno_path.exists():
            try:
                geno_lines = sum(1 for _ in geno_path.open())
            except Exception:
                pass

        var_state = metrics.get("variation_state") or {}
        rows.append({
            "uuid": uuid,
            "goal_id": goal_id,
            "goal_label": label,
            "iteration": metrics.get("iteration"),
            "evolution_kind": metrics.get("evolution_kind"),
            "parent_uuids": ";".join(metrics.get("parent_uuids") or []),
            "n_parents": len(metrics.get("parent_uuids") or []),
            "created_at": lin.get("created_at"),
            "wall_time_s": metrics.get("iteration_wall_time_s"),
            "cost_usd": metrics.get("iteration_cost_usd"),
            "cumulative_cost_usd": metrics.get("cumulative_cost_usd"),
            "overall_score": metrics.get("overall_score"),
            "overall_score_uncapped": metrics.get("overall_score_uncapped"),
            "qd_score": metrics.get("qd_score"),
            "novelty_score": metrics.get("novelty_score"),
            "qd_descriptor": json.dumps(metrics.get("qd_descriptor")),
            "stagnation": var_state.get("stagnation"),
            "success_rate": var_state.get("success_rate"),
            "effective_boldness": var_state.get("effective_boldness"),
            "parent_score": var_state.get("parent_score"),
            "scope_band": var_state.get("scope_band"),
            "agent_budget": var_state.get("agent_budget"),
            "genotype_lines": geno_lines,
            "has_gradient": has_gradient,
            "gradient_chars": gradient_len,
        })
    return rows, goals


def parse_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    rows, goals = parse_workflows()
    # tag qd_archive and variation_log with goal_id by uuid
    uuid_to_goal = {r["uuid"]: r["goal_id"] for r in rows}

    qd_rows = parse_jsonl(ROOT / "qd_archive.jsonl")
    for r in qd_rows:
        r["goal_id"] = uuid_to_goal.get(r.get("uuid"), "unknown")
        r["qd_descriptor"] = json.dumps(r.get("qd_descriptor"))
    var_rows = parse_jsonl(ROOT / "variation_log.jsonl")
    for r in var_rows:
        r["goal_id"] = uuid_to_goal.get(r.get("from_uuid"), "unknown")
        r["parent_uuids"] = ";".join(r.get("parent_uuids") or [])

    write_csv(rows, OUT / "workflows.csv")
    write_csv(qd_rows, OUT / "qd_archive.csv")
    write_csv(var_rows, OUT / "variation_log.csv")
    (OUT / "goals.json").write_text(json.dumps(goals, indent=2))

    # print summary
    print(f"workflows: {len(rows)}")
    print(f"goals: {len(goals)}")
    for gid, g in goals.items():
        print(f"  {gid}  {g['label']:>18}  n={len(g['workflows'])}")
    print(f"qd_archive lines: {len(qd_rows)}")
    print(f"variation_log lines: {len(var_rows)}")


if __name__ == "__main__":
    main()
