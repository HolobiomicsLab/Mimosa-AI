"""Parse evaluation.txt of every productive iteration to extract score components
(base_mean, information_bonus, hard_fail_capped, claim pass/fail/error counts).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path("/home/martin/Projects/CNRS/Mimosa-AI/sources/workflows/run_1")
OUT = Path("/tmp/evo_report")

RE_CLAIMS = re.compile(r"Claims:\s*(\d+)\s+pass=(\d+)\s+fail=(\d+)\s+error=(\d+)\s+unsure=(\d+)\s+scored=(\d+)")
RE_OVERALL = re.compile(r"Overall:\s*([\d.]+)\s*\(pre-cheat\s*([\d.]+),\s*uncapped\s*([\d.]+),\s*hard_fail_capped=(\w+)")
RE_BASE = re.compile(r"base_mean=([\d.]+)\s+information_bonus=([\d.]+)\s+n_high_importance_pass=(\d+)\s+high_importance_pass_mass=([\d.]+)\s+cheat_penalty=([\d.]+)")


def parse_eval(path: Path) -> dict:
    out = {"path": str(path)}
    try:
        head = path.read_text().splitlines()[:6]
    except Exception:
        return out
    text = "\n".join(head)
    m = RE_CLAIMS.search(text)
    if m:
        out["n_claims"] = int(m.group(1))
        out["n_pass"] = int(m.group(2))
        out["n_fail"] = int(m.group(3))
        out["n_error"] = int(m.group(4))
        out["n_unsure"] = int(m.group(5))
        out["n_scored"] = int(m.group(6))
    m = RE_OVERALL.search(text)
    if m:
        out["overall"] = float(m.group(1))
        out["pre_cheat"] = float(m.group(2))
        out["uncapped"] = float(m.group(3))
        out["hard_fail_capped"] = m.group(4).lower() == "true"
    m = RE_BASE.search(text)
    if m:
        out["base_mean"] = float(m.group(1))
        out["info_bonus"] = float(m.group(2))
        out["n_high_imp_pass"] = int(m.group(3))
        out["high_imp_pass_mass"] = float(m.group(4))
        out["cheat_penalty"] = float(m.group(5))
    return out


def main() -> None:
    rows = []
    for folder in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        ev = folder / "evaluation.txt"
        if not ev.exists() or ev.stat().st_size == 0:
            continue
        r = parse_eval(ev)
        r["uuid"] = folder.name
        rows.append(r)
    keys = sorted({k for r in rows for k in r.keys()})
    with (OUT / "eval_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"parsed {len(rows)} evaluations")
    # report which had hard_fail_capped True
    capped = [r for r in rows if r.get("hard_fail_capped")]
    print(f"hard_fail_capped=True in {len(capped)} workflows")
    err_heavy = sorted(rows, key=lambda r: -r.get("n_error", 0))[:5]
    print("top n_error workflows:")
    for r in err_heavy:
        print(f"  {r['uuid']}  errors={r.get('n_error')}/scored={r.get('n_scored')}  overall={r.get('overall')}")


if __name__ == "__main__":
    main()
