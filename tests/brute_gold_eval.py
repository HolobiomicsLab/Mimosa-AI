#!/usr/bin/env python3
"""
Brute sanity check of the ScienceAgentBench eval pipeline against gold solutions.

With no arguments it runs the FULL benchmark (every task in ScienceAgentBench.csv):
for each task it EXECUTES the gold program in the sandbox (VER) with the task's
real input data, then runs the task's eval script on the produced output (SR).
A correct pipeline scores VER=1 and SR=1 on the gold, so a fully-provisioned run
should approach 100% (figure tasks need OPENAI_API_KEY or they are excluded).

Uses the SAME ExecutionSandbox + methods (run_generated_code / run_eval_script)
as Mimosa's real eval, and by default the SAME base packages
(ExecutionSandbox.BASIC_PACKAGES: torch/tensorflow/rdkit/…). The first task pays
a one-time heavy install into the shared venv; every task after reuses it.

Input datasets are untracked and live in the MAIN working tree, not in a git
worktree — this harness resolves them (and the CSV) by walking up to the main
repo. If a gold program or its data is unavailable it falls back to SEED mode:
it feeds the gold OUTPUT through the eval script (SR only).

Not named `test_*` so pytest does not auto-collect it.

    python3.12 tests/brute_gold_eval.py                # full benchmark, full base
    python3.12 tests/brute_gold_eval.py --list         # list the tasks, run nothing
    python3.12 tests/brute_gold_eval.py --light CogSci_pattern_high_sim_eval
    python3.12 tests/brute_gold_eval.py --seed-only clintox_nn_eval
"""

import csv
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.benchmark_evaluation.execution_sandbox import ExecutionSandbox, EvalInfraError

SAB_ROOT = Path(__file__).resolve().parent.parent / "datasets" / "ScienceAgentBench"
EVAL_DIR = SAB_ROOT / "eval_programs"
GOLD_RESULTS_DIR = EVAL_DIR / "gold_results"
GOLD_PROG_DIR = SAB_ROOT / "gold_programs"

# Lighter base for fast subset runs (--light); pipreqs still adds per-gold deps.
LIGHT_BASE = ["numpy", "pandas", "scikit-learn", "pipreqs", "pip-tools"]

FLAGS = {"--seed-only", "--light", "--list"}


def _resolve_up(*rel_parts: str) -> Path | None:
    """Find datasets/<...> which may live in the main working tree above a worktree."""
    direct = SAB_ROOT.parent / Path(*rel_parts)
    if direct.exists():
        return direct
    for anc in Path(__file__).resolve().parents:
        cand = anc / "datasets" / Path(*rel_parts)
        if cand.exists():
            return cand
    return None


DATASETS_DIR = _resolve_up("ScienceAgentBench", "datasets")
TASK_CSV = _resolve_up("ScienceAgentBench.csv")


def _load_tasks() -> tuple[list[str], dict[str, str]]:
    """Return (all eval basenames, {eval_base: gold_base}) from the benchmark CSV."""
    if not TASK_CSV or not TASK_CSV.exists():
        return [], {}
    evals, mapping = [], {}
    with open(TASK_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eval_base = (row.get("eval_script_name") or "").replace(".py", "").strip()
            gold_base = (row.get("gold_program_name") or "").replace(".py", "").strip()
            if eval_base:
                evals.append(eval_base)
                mapping[eval_base] = gold_base
    return evals, mapping


ALL_TASKS, EVAL_TO_GOLD = _load_tasks()


def _parse_io_paths(eval_text: str) -> tuple[str | None, str | None]:
    """Extract the pred_results and gold_results filenames the eval script uses."""
    pred = re.search(r"pred_results/([A-Za-z0-9_.\-/]+)", eval_text)
    gold = re.search(r"gold_results/([A-Za-z0-9_.\-/]+)", eval_text)
    return (pred.group(1) if pred else None, gold.group(1) if gold else None)


def _copy_input_data(gold_text: str, capsule: Path) -> tuple[list[str], list[str]]:
    """Copy every benchmark/datasets/<folder> the gold references into the capsule."""
    wanted = sorted(set(re.findall(r"benchmark/datasets/([A-Za-z0-9_\-]+)", gold_text)))
    copied = []
    for folder in wanted:
        src = DATASETS_DIR / folder if DATASETS_DIR else None
        if src and src.exists():
            shutil.copytree(src, capsule / "benchmark" / "datasets" / folder, dirs_exist_ok=True)
            copied.append(folder)
    return copied, wanted


def _seed_capsule_with_gold(capsule: Path, pred_rel: str, gold_name: str) -> None:
    """Copy the gold output into the capsule as the prediction (perfect agent)."""
    src = GOLD_RESULTS_DIR / gold_name
    if not src.exists():
        raise FileNotFoundError(f"gold output missing: {src}")
    dst = capsule / "pred_results" / pred_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _gold_program_for(eval_name: str) -> Path:
    """Resolve a task's gold program via the CSV map (fallback: strip _eval)."""
    gold_base = EVAL_TO_GOLD.get(eval_name) or (
        eval_name[:-5] if eval_name.endswith("_eval") else eval_name
    )
    return GOLD_PROG_DIR / f"{gold_base}.py"


def brute_test_one(eval_name: str, base_packages=None, run_ver: bool = True) -> dict:
    """Run VER (execute gold) + SR (eval), or SR-only seed mode; return a record."""
    eval_path = EVAL_DIR / f"{eval_name}.py"
    gold_prog = _gold_program_for(eval_name)
    record = {"task": eval_name, "ver": "skipped", "sr": None, "msg": ""}

    if not eval_path.exists():
        record["msg"] = f"eval script not found: {eval_path}"
        return record
    pred_rel, gold_name = _parse_io_paths(eval_path.read_text(encoding="utf-8", errors="ignore"))
    if not pred_rel:
        record["msg"] = "could not parse pred_results path from eval script"
        return record

    capsule = Path(tempfile.mkdtemp(prefix=f"brute_{eval_name}_"))
    sandbox = None
    try:
        do_ver = run_ver and gold_prog.exists() and DATASETS_DIR is not None
        if do_ver:
            copied, wanted = _copy_input_data(gold_prog.read_text(encoding="utf-8", errors="ignore"), capsule)
            missing = set(wanted) - set(copied)
            if missing:
                do_ver = False
                record["ver"] = f"skipped (missing data {sorted(missing)})"

        if do_ver:
            shutil.copy2(gold_prog, capsule / gold_prog.name)
            sandbox = ExecutionSandbox(capsule, base_packages=base_packages)
            ver_ok, ver_msg = sandbox.run_generated_code(
                script_path=capsule / gold_prog.name,
                script_name=gold_prog.name,
                expected_output=pred_rel,
                timeout=900,
            )
            record["ver"] = "PASS" if ver_ok else "FAIL"
            if not ver_ok:
                record["sr"] = False  # VER failed -> SR is False (SAB semantics)
                record["msg"] = ver_msg[:160]
                return record
        else:
            if not gold_name:
                record["msg"] = "seed mode needs a gold_results path in the eval script"
                return record
            _seed_capsule_with_gold(capsule, pred_rel, gold_name)
            if not record["ver"].startswith("skipped"):
                record["ver"] = "skipped (seed mode)"
            sandbox = ExecutionSandbox(capsule, base_packages=base_packages)

        try:
            success, msg = sandbox.run_eval_script(eval_path, visual_judge_path=None, timeout=300)
            record["sr"] = bool(success)
            record["msg"] = msg[:120]
        except EvalInfraError as e:
            record["sr"] = None  # excluded, not a failure
            record["msg"] = f"EXCLUDED (infra): {e}"
    except Exception as e:
        record["msg"] = f"harness error: {e}"
    finally:
        if sandbox is not None:
            sandbox.cleanup()
        shutil.rmtree(capsule, ignore_errors=True)
    return record


def run_brute_test(eval_names: list[str], base_packages=None, run_ver: bool = True) -> list[dict]:
    """Run the gold check over the tasks, streaming each result, then summarize."""
    total = len(eval_names)
    base_label = "light" if base_packages is not None else "full (BASIC_PACKAGES)"
    print("=" * 84)
    print(f"BRUTE GOLD-SOLUTION CHECK — execute gold (VER) then run eval (SR) — {total} task(s)")
    print(f"datasets: {DATASETS_DIR or 'NOT FOUND (SR-seed only)'} | base: {base_label}")
    print("=" * 84)

    records = []
    for i, name in enumerate(eval_names, 1):
        r = brute_test_one(name, base_packages=base_packages, run_ver=run_ver)
        records.append(r)
        sr = "—(excluded)" if r["sr"] is None else ("PASS" if r["sr"] else "FAIL")
        print(f"[{i}/{total}] {r['task']:<38} VER={r['ver']:<22} SR={sr:<12} {r['msg'][:44]}")

    evaluated = [r for r in records if r["sr"] is not None]
    excluded = [r for r in records if r["sr"] is None]
    ver_ran = [r for r in records if r["ver"] in ("PASS", "FAIL")]
    ver_pass = sum(1 for r in ver_ran if r["ver"] == "PASS")
    sr_pass = sum(1 for r in evaluated if r["sr"])

    print("-" * 84)
    ver_line = f"{ver_pass}/{len(ver_ran)} ({ver_pass/len(ver_ran)*100:.0f}%)" if ver_ran else "none executed"
    sr_pct = (sr_pass / len(evaluated) * 100) if evaluated else 0.0
    print(f"VER: {ver_line} | SR: {sr_pass}/{len(evaluated)} ({sr_pct:.0f}%) | excluded (infra): {len(excluded)}")
    print("=" * 84)
    return records


if __name__ == "__main__":
    argv = sys.argv[1:]
    run_ver = "--seed-only" not in argv
    base = LIGHT_BASE if "--light" in argv else None
    tasks = [a for a in argv if a not in FLAGS] or ALL_TASKS

    if not tasks:
        print("No tasks: benchmark CSV not found and none given on the command line.")
        sys.exit(2)
    if "--list" in argv:
        print("\n".join(tasks))
        print(f"total: {len(tasks)}")
        sys.exit(0)

    results = run_brute_test(tasks, base_packages=base, run_ver=run_ver)
    evaluated = [r for r in results if r["sr"] is not None]
    ver_ran = [r for r in results if r["ver"] in ("PASS", "FAIL")]
    ok = (
        bool(evaluated) and all(r["sr"] for r in evaluated)
        and all(r["ver"] == "PASS" for r in ver_ran)
    )
    print("RESULT:", "GOLD SCORES VER=SR=100% ✓" if ok else "SOME TASKS DID NOT PASS ✗")
    sys.exit(0 if ok else 1)
