"""Policy-BT verifier (E44-E48 architecture) — offline tests, mocked LLM.

Covers the plan §6 acceptance surface with NO network and NO real keys:
- digest determinism (golden re-run) + id_order_match vs in-workspace test
  inputs + the E26 figure-dominance rule;
- reexec: LOCAL ./venv at the task-sandbox root, whole-process-group
  timeout kill (no orphans), ledger fields, stale-vs-fresh artifact flag,
  original workspace untouched (before/after manifest);
- policy: AST screen allowlist/banlist, truncated-JSON rejection (E48),
  token-cap escalation + cache-busting repair roles, runtime smoke;
- verdicts: win-rate math, BT fit sanity (hand-checked ordering,
  tie = 2 x 0.5), both-orders antisymmetry -> tie + warning, crash rung;
- visual: per-pair prompt assembly (labeled sides), both-orders mirror,
  disagreement -> tie + warning, one-side-no-figures rung, retry/self-heal
  with a provider that returns empty once, per-gen cap fallback to the
  code policy (+ its WARNING line);
- gradient: elimination-first fold ordering, refiner schema contract;
- registry: restart-survivable persistence + arrival ordering;
- evaluator: short-circuit paths, registry accumulation across evaluate()
  calls, full stage-line order (the §6b SMOKE), JSON-lines sink, verbose
  per-pair lines, cumulative spend line;
- config plumbing: defaults, JSON round-trip, CLI overrides.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from sources.core.failure_fingerprint import (  # noqa: F401,E402 — primes the
    compute_failure_fingerprint,  # core<->evaluators circular import chain
)
from sources.evaluators.policy_bt_verifier import (  # noqa: E402
    MAIN_STANZA,
    PolicyBTVerifierEvaluator,
    ReexecRunner,
    TaskRegistry,
    VisualChannel,
    ensure_policy,
    fold_gradient,
    is_figure_dominant,
    refine_gradient,
    run_policy,
    smoke_test,
    static_screen,
    task_key,
)
from sources.evaluators.policy_bt_verifier import (  # noqa: E402
    digest as digest_mod,
)
from sources.evaluators.policy_bt_verifier import (  # noqa: E402
    evaluator as ev_mod,
)
from sources.evaluators.policy_bt_verifier import (  # noqa: E402
    gradient as gradient_mod,
)
from sources.evaluators.policy_bt_verifier import (  # noqa: E402
    metrics as metrics_mod,
)
from sources.evaluators.policy_bt_verifier import (  # noqa: E402
    policy as policy_mod,
)
from sources.evaluators.policy_bt_verifier import (  # noqa: E402
    recon as recon_mod,
)
from sources.evaluators.policy_bt_verifier import (  # noqa: E402
    reexec as reexec_mod,
)
from sources.evaluators.policy_bt_verifier import (  # noqa: E402
    verdicts as verdicts_mod,
)
from sources.evaluators.policy_bt_verifier import (  # noqa: E402
    visual as visual_mod,
)
from sources.evaluators.policy_bt_verifier.logging_utils import (  # noqa: E402
    StageLogger,
)
from sources.evaluators.policy_bt_verifier.registry import (  # noqa: E402
    code_verdict_key,
    visual_key,
)

GOAL = (
    "Train a classifier on compounds.csv and write per-compound predicted "
    "probabilities to pred_results/predictions.csv with columns "
    "(smiles, probability); report the achieved AUC in metrics.json."
)

RECON_JSON = {
    "schema_map": "deliverable is pred_results/predictions.csv (smiles, probability)",
    "deliverable_contract": {
        "expected_paths": ["pred_results/predictions.csv"],
        "expected_schema": "columns smiles, probability; one row per input",
        "format_notes": "csv",
    },
    "primary_metrics": [],
    "tie_policy": "tie when both sides produce equal rows",
    "coverage_footer": "pred_results/predictions.csv rows+columns (grading bars unknown)",
}

REFINER_JSON = {
    "refined_gradient": "## Verified issues\n- example issue",
    "verified_issues": [{"issue": "i", "evidence": "e", "confidence": 0.9}],
    "likely_explanations": [{"hypothesis": "h", "basis": "b", "confidence": 0.5}],
    "missing_files": ["pred_results/predictions.csv"],
    "unverified_from_raw": [],
    "corrections": [],
}

_TIE_POLICY_BODY = (
    "import json\n"
    "def compare(dA, dB):\n"
    "    return {\"winner\": \"tie\", \"confidence\": 0.5, \"rungs\": [],\n"
    "            \"deciding\": [], \"feedback_winner\": \"tie - no decisive evidence\",\n"
    "            \"feedback_looser\": \"tie - no decisive evidence\"}\n"
)

_DECISIVE_POLICY_BODY = (
    "import json\n"
    "def compare(dA, dB):\n"
    "    a = len(dA.get('deliverables') or [])\n"
    "    b = len(dB.get('deliverables') or [])\n"
    "    if a == b:\n"
    "        return {\"winner\": \"tie\", \"confidence\": 0.5, \"rungs\": [],\n"
    "                \"deciding\": [], \"feedback_winner\": \"both sides equal\",\n"
    "                \"feedback_looser\": \"both sides equal\"}\n"
    "    w = 'A' if a > b else 'B'\n"
    "    hi, lo = max(a, b), min(a, b)\n"
    "    return {\"winner\": w, \"confidence\": 0.9, \"rungs\": [],\n"
    "            \"deciding\": ['deliverables'],\n"
    "            \"feedback_winner\": f\"{hi} deliverables vs {lo}\",\n"
    "            \"feedback_looser\": f\"{lo} deliverables vs {hi}\"}\n"
)


class _FakeLog:
    """StageLogger stand-in that records stage/warn/detail calls."""

    def __init__(self):
        self.stages: list[str] = []
        self.warns: list[str] = []
        self.details: list[str] = []
        self.spend_usd = 0.0

    def stage(self, stage, message="", **fields):
        self.stages.append(stage)

    def warn(self, text):
        self.warns.append(text)

    def detail(self, text):
        self.details.append(text)


# ------------------------------------------------------------- fixtures ----


def _seed_workspace(ws: Path, n_rows: int = 3, with_figs: bool = False) -> None:
    """A tiny but REAL workspace: input CSVs + main script + outputs."""
    ws.mkdir(parents=True, exist_ok=True)
    smiles = ["CCO", "NCC", "c1ccccc1", "CC(=O)O"][:n_rows]
    (ws / "compounds_test.csv").write_text(
        "smiles,tox\n" + "\n".join(f"{s},{i % 2}" for i, s in enumerate(smiles)) + "\n",
        encoding="utf-8",
    )
    (ws / "train.csv").write_text(
        "smiles,tox\n" + "\n".join(f"{s}C,{i % 2}" for i, s in enumerate(smiles)) + "\n",
        encoding="utf-8",
    )
    rows = "\n".join(f"{s},{0.2 + 0.15 * i}" for i, s in enumerate(smiles))
    rows_list = [f"{s},{0.2 + 0.15 * i}" for i, s in enumerate(smiles)]
    (ws / "main.py").write_text(
        "from pathlib import Path\n"
        "Path('pred_results').mkdir(exist_ok=True)\n"
        f"rows = {rows_list!r}\n"
        "Path('pred_results/predictions.csv').write_text("
        "chr(10).join(['smiles,probability'] + rows) + chr(10))\n"
        "print('AUC = 0.83 (validation split)')\n",
        encoding="utf-8",
    )
    pred = ws / "pred_results"
    pred.mkdir(exist_ok=True)
    (pred / "predictions.csv").write_text(
        f"smiles,probability\n{rows}\n", encoding="utf-8"
    )
    (ws / "output.log").write_text(
        "training finished\nAUC = 0.83 (validation split)\n", encoding="utf-8"
    )
    if with_figs:
        visual_mod_test_png(pred / "figure1.png", 8, 6)


def visual_mod_test_png(path: Path, w: int = 8, h: int = 6) -> None:
    """Write a real 1-bit PNG (no matplotlib needed)."""
    from PIL import Image

    Image.new("1", (w * 10, h * 10), 1).save(path)


def _wf_dir(tmp_path: Path, uuid: str, goal: str = GOAL) -> Path:
    """Workflow folder with state_result + genotype so _load works."""
    d = tmp_path / "workflows" / uuid
    d.mkdir(parents=True, exist_ok=True)
    (d / "state_result.json").write_text(
        json.dumps({"goal": goal, "answers": ["ok"]}), encoding="utf-8"
    )
    (d / f"workflow_genotype_{uuid}.py").write_text("print('workflow')\n")
    return d


def _make_evaluator(tmp_path: Path, **attrs) -> PolicyBTVerifierEvaluator:
    """Build a PolicyBTVerifierEvaluator without BaseEvaluator.__init__."""
    v = PolicyBTVerifierEvaluator.__new__(PolicyBTVerifierEvaluator)
    v.memory_dir = tmp_path / "memory"
    v.workflow_dir = tmp_path / "workflows"
    v.workspace_dir = tmp_path / "workspace"
    for d in (v.memory_dir, v.workflow_dir, v.workspace_dir):
        d.mkdir(parents=True, exist_ok=True)
    v._runner_temp_root = tmp_path / "verifier_tmp"
    v._runner_temp_root.mkdir(parents=True, exist_ok=True)
    v.logger = logging.getLogger("test-policy-bt")
    v.reward_mode = "win_rate"
    v.scorer_timeout_s = 30
    v.visual_channel = True
    v.refiner = True
    v.repair_rounds = 3
    v.max_pairs = 0
    v.reexec_timeout_s = 30
    v.venv_reuse = True
    v.visual_tiebreak_code = False
    v.visual_max_pairs_per_gen = 0
    v.verbose = False
    v._session_tag = "test-session"  # N9: set by __init__ in production
    v.recon_refresh_gap = 2
    v._gens_since_refresh = 0
    v.model_pricing = {}
    v._vision_llm_config = types.SimpleNamespace(
        model="moonshotai/kimi-k3",
        provider="openrouter",
        temperature=0.0,
        reasoning_effort="low",
        max_tokens=1024,
        openrouter_provider=None,
        openrouter_quantizations=None,
        key="",
    )
    v.llm_config = types.SimpleNamespace(
        model="deepseek/deepseek-v4.1-flash",
        provider="openrouter",
        temperature=0.0,
        reasoning_effort="low",
        max_tokens=12000,
        openrouter_provider=None,
        openrouter_quantizations=None,
        key="",
    )
    for k, val in attrs.items():
        setattr(v, k, val)
    return v


class _FakeReexecRunner:
    """ReexecRunner stand-in: real file copy, synthetic ledger."""

    def __init__(self, sandbox_root, timeout_s=900, venv_reuse=True, log=None):
        self.task_sandbox = Path(sandbox_root)
        self.runs_dir = self.task_sandbox / "runs"
        self.venv_path = self.task_sandbox / "venv"
        self.timeout_s = timeout_s

    def run_copy_path(self, uuid):
        return self.runs_dir / uuid

    def run(self, workspace, uuid):
        dst = self.run_copy_path(uuid)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(workspace, dst)
        fresh = {
            "pred_results/predictions.csv": {"bytes": 64, "sha256": "0" * 64}
        }
        return {
            "uuid": uuid,
            "exit_code": 0,
            "wall_time_s": 0.2,
            "timed_out": False,
            "error_signature": [],
            "log_tail": "AUC = 0.83",
            "fresh_artifacts": fresh,
            "produced_pred_results": True,
            "source_manifest_intact": True,
            "n_files_copied": len(list(dst.rglob("*"))),
            "venv_created": False,
            "packages_installed": 0,
        }


class _ScriptedLLM:
    """Dispatches canned JSON by agent role; records every call."""

    def __init__(self, recon=RECON_JSON, policy=_DECISIVE_POLICY_BODY,
                 refiner=REFINER_JSON, first_policy_raw=None,
                 text_extract=None):
        self.recon = recon
        self.text_extract = text_extract or {}
        self.policy = policy
        self.refiner = refiner
        self.first_policy_raw = first_policy_raw
        self.calls: list[tuple[str, int]] = []
        self.policy_writes = 0

    def __call__(self, uuid, role, prompt, max_tokens, log):
        self.calls.append((role, max_tokens))
        if role == "recon":
            return self.recon
        if role == "recon_refresh":
            return self.recon
        if role.startswith("policy_write"):
            self.policy_writes += 1
            if self.first_policy_raw is not None and self.policy_writes == 1:
                return self.first_policy_raw
            return {"policy_py": self.policy}
        if role.startswith("policy_bt_refiner"):
            return self.refiner
        if role == "metric_text_extract":
            return self.text_extract
        raise AssertionError(f"unexpected llm role: {role}")

_JSON_POLICY_BODY = (
    "import json\n"
    "def compare(dA, dB):\n"
    "    a = (dA.get('json_metrics') or {}).get('metrics.json', {})"
    ".get('auc', 0.0)\n"
    "    b = (dB.get('json_metrics') or {}).get('metrics.json', {})"
    ".get('auc', 0.0)\n"
    "    if a == b:\n"
    "        return {\"winner\": \"tie\", \"confidence\": 0.5, \"rungs\": [],\n"
    "                \"deciding\": [], \"feedback_winner\": \"equal\",\n"
    "                \"feedback_looser\": \"equal\"}\n"
    "    w = 'A' if a > b else 'B'\n"
    "    return {\"winner\": w, \"confidence\": 0.9, \"rungs\": [],\n"
    "            \"deciding\": ['json_metrics[metrics.json].auc'],\n"
    "            \"feedback_winner\": f\"auc {max(a, b):.2f} vs {min(a, b):.2f}\",\n"
    "            \"feedback_looser\": f\"auc {min(a, b):.2f} vs {max(a, b):.2f}\"}\n"
)

_BRITTLE_JSON_POLICY_BODY = (
    "import json\n"
    "def compare(dA, dB):\n"
    "    if dB.get('json_metrics'):\n"
        # KeyError whenever one side reports json metrics and the other
        # does not — a branch-only error invisible to a self-compare
    "        return dA['json_metrics']['metrics.json']['auc']\n"
    "    return {\"winner\": \"tie\", \"confidence\": 0.5, \"rungs\": [],\n"
    "            \"deciding\": [], \"feedback_winner\": \"tie\",\n"
    "            \"feedback_looser\": \"tie\"}\n"
)


class _SeqPolicyLLM(_ScriptedLLM):
    """_ScriptedLLM returning a SEQUENCE of policy bodies (write order)."""

    def __init__(self, policies, **kw):
        super().__init__(**kw)
        self.policies = list(policies)

    def __call__(self, uuid, role, prompt, max_tokens, log):
        self.calls.append((role, max_tokens))
        if role == "recon":
            return self.recon
        if role.startswith("policy_write"):
            self.policy_writes += 1
            body = self.policies.pop(0) if self.policies else self.policy
            return {"policy_py": body}
        if role.startswith("policy_bt_refiner"):
            return self.refiner
        if role == "metric_text_extract":
            return {}
        raise AssertionError(f"unexpected llm role: {role}")


def _seed_workspace(ws: Path, n_rows: int = 3, with_figs: bool = False) -> None:
    """A tiny but REAL workspace: input CSVs + main script + outputs."""
    ws.mkdir(parents=True, exist_ok=True)
    smiles = ["CCO", "NCC", "c1ccccc1", "CC(=O)O"][:n_rows]
    (ws / "compounds_test.csv").write_text(
        "smiles,tox\n" + "\n".join(f"{s},{i % 2}" for i, s in enumerate(smiles)) + "\n",
        encoding="utf-8",
    )
    (ws / "train.csv").write_text(
        "smiles,tox\n" + "\n".join(f"{s}C,{i % 2}" for i, s in enumerate(smiles)) + "\n",
        encoding="utf-8",
    )
    rows = "\n".join(f"{s},{0.2 + 0.15 * i}" for i, s in enumerate(smiles))
    rows_list = [f"{s},{0.2 + 0.15 * i}" for i, s in enumerate(smiles)]
    (ws / "main.py").write_text(
        "from pathlib import Path\n"
        "Path('pred_results').mkdir(exist_ok=True)\n"
        f"rows = {rows_list!r}\n"
        "Path('pred_results/predictions.csv').write_text("
        "chr(10).join(['smiles,probability'] + rows) + chr(10))\n"
        "print('AUC = 0.83 (validation split)')\n",
        encoding="utf-8",
    )
    pred = ws / "pred_results"
    pred.mkdir(exist_ok=True)
    (pred / "predictions.csv").write_text(
        f"smiles,probability\n{rows}\n", encoding="utf-8"
    )
    (ws / "output.log").write_text(
        "training finished\nAUC = 0.83 (validation split)\n", encoding="utf-8"
    )
    if with_figs:
        visual_mod_test_png(pred / "figure1.png", 8, 6)


def test_digest_deterministic_and_id_order_match(tmp_path: Path):
    ws = tmp_path / "ws"
    _seed_workspace(ws)
    ledger = {
        "exit_code": 0,
        "wall_time_s": 1.0,
        "timed_out": False,
        "error_signature": [],
        "fresh_artifacts": {"pred_results/predictions.csv": {"bytes": 64}},
        "produced_pred_results": True,
        "log_tail": "AUC = 0.83",
    }
    d1 = digest_mod.build_digest("u1", ws, ledger)
    d2 = digest_mod.build_digest("u1", ws, ledger)
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)

    tbl = d1["tables"][0]
    idm = tbl["id_order_match"]
    assert idm is not None and idm["match"] is True
    assert idm["pred_col"] == "smiles"

    # shuffled predictions break the order match
    (ws / "pred_results" / "predictions.csv").write_text(
        "smiles,probability\nc1ccccc1,0.5\nCCO,0.7\nNCC,0.4\nCC(=O)O,0.9\n",
        encoding="utf-8",
    )
    d3 = digest_mod.build_digest("u1", ws, ledger)
    assert d3["tables"][0]["id_order_match"]["match"] is False

    # deliverables carry the stale-artifact flag from the ledger fresh set
    by_path = {dv["path"]: dv for dv in d1["deliverables"]}
    assert by_path["pred_results/predictions.csv"]["produced_this_run"] is True
    assert by_path["compounds_test.csv"]["produced_this_run"] is False
    # regex scraping removed (operator directive 2026-10-10): log_metrics
    # is empty; text metrics come from the LLM extractor instead
    assert d1["log_metrics"] == []


def test_digest_bootstrap_and_figure_dominance(tmp_path: Path):
    ws = tmp_path / "ws"
    _seed_workspace(ws, with_figs=True)
    d = digest_mod.build_digest("u1", ws, None)
    assert d["exec"]["exit_code"] is None
    assert d["exec"]["bootstrap"] is None

    data = {"deliverables": [{"path": "pred_results/p.csv", "kind": "csv"},
                             {"path": "pred_results/q.csv", "kind": "csv"},
                             {"path": "pred_results/dens.npy", "kind": "npy"},
                             {"path": "pred_results/fig.png", "kind": "png"}]}
    assert is_figure_dominant(data) is False  # charge_density class
    assert is_figure_dominant({"deliverables": []}) is False



def test_digest_booleans_are_flags_and_coordinates_are_shape_markers(
    tmp_path: Path,
):
    """P0.1/P0.2: booleans never become 1.0 metrics; nested arrays
    (coordinates) become {shape, len} markers, not per-element scalars
    whose key names could pattern-match as metric evidence (HP gen 4)."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "results.json").write_text(
        json.dumps(
            {
                "optimality": {
                    "globally_optimal": True,
                    "minimum_energy": -2.0,
                },
                "example_optimal_conformation_coordinates": [
                    [1.5, 2.5],
                    [3.5, 4.5],
                    [4.5, 5.5],
                ],
                "saw_counts": [376, 94, 16],
                "plain_index_list": [1, 2, 3],
                "n_saws": 16268,
            }
        ),
        encoding="utf-8",
    )
    d = digest_mod.build_digest("u1", ws, None)
    jm = d["json_metrics"]["results.json"]
    assert "optimality.globally_optimal" not in jm  # never a metric
    # nested array -> shape/len marker, no [i][j] coordinate leaves
    assert jm["example_optimal_conformation_coordinates"] == {
        "shape": [3, 2],
        "len": 3,
    }
    assert not any("][0]" in k or "][1]" in k for k in jm)
    # 1-D numeric list under a metric-like key keeps its elements
    assert jm["saw_counts[0]"] == 376.0 and jm["saw_counts[2]"] == 16.0
    # 1-D numeric list under a NON-metric key drops its elements
    assert not any(k.startswith("plain_index_list") for k in jm)
    # flags live in their own block, preserved as booleans
    assert d["json_flags"]["results.json"]["optimality.globally_optimal"] is True
    assert all(isinstance(v, bool) for v in d["json_flags"]["results.json"].values())


def test_reexec_local_venv_ledger_and_source_untouched(tmp_path: Path):
    ws_root = tmp_path / "task" / "ws"
    ws = ws_root / "gen1"
    _seed_workspace(ws)
    (ws / "stale.csv").write_text("old\n", encoding="utf-8")
    old = time.time() - 3600
    os.utime(ws / "stale.csv", (old, old))

    sandbox = tmp_path / "verifier" / "policy_bt" / task_key(GOAL)
    runner = ReexecRunner(sandbox, timeout_s=60, venv_reuse=True)
    before = digest_mod_reexec_manifest(ws)
    ledger = runner.run(ws, "gen1")

    # LOCAL ./venv INSIDE the task sandbox — never a parent-folder venv
    assert runner.venv_path == sandbox / "venv"
    assert (sandbox / "venv" / "bin" / "python").exists()
    assert not (ws_root / "venv").exists() and not (ws / "venv").exists()

    assert ledger["exit_code"] == 0
    assert ledger["wall_time_s"] >= 0
    assert ledger["source_manifest_intact"] is True
    assert before == digest_mod_reexec_manifest(ws)

    fresh = ledger["fresh_artifacts"]
    assert "pred_results/predictions.csv" in fresh  # written by this run
    assert "stale.csv" not in fresh  # pre-existing -> STALE
    assert "main.py" not in fresh  # inputs copied before the run are STALE


def digest_mod_reexec_manifest(ws: Path):
    from sources.evaluators.policy_bt_verifier.reexec import frozen_manifest

    return frozen_manifest(ws)


def test_reexec_timeout_kills_process_group_no_orphans(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    marker = f"policy_bt_orphan_probe_{os.getpid()}"
    (ws / "main.py").write_text(
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', "
        f"'import time; time.sleep(120)  # {marker}'])\n"
        "print('spawned child', flush=True)\n"
        "time.sleep(120)\n",
        encoding="utf-8",
    )
    log = _FakeLog()
    sandbox = tmp_path / "sb"
    runner = ReexecRunner(sandbox, timeout_s=6, venv_reuse=False, log=log)
    t0 = time.monotonic()
    ledger = runner.run(ws, "gen-timeout")
    wall = time.monotonic() - t0
    assert ledger["timed_out"] is True
    assert ledger["killed_pgid"] is not None
    assert wall < 30  # killed promptly, not after the child's 120s
    # the kill logged loudly (plan §6b failure paths)
    assert any("timed out" in w and "process group" in w for w in log.warns)
    # no orphaned grandchild survived the group kill
    probe = subprocess.run(
        ["pgrep", "-f", marker], capture_output=True, text=True
    )
    assert probe.stdout.strip() == "", f"orphan survived: {probe.stdout}"


def test_reexec_no_main_script_ledger(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "data.csv").write_text("a\n1\n", encoding="utf-8")
    runner = ReexecRunner(tmp_path / "sb2", timeout_s=10)
    ledger = runner.run(ws, "gen-x")
    assert ledger["launch_error"] == "no_main_script"
    assert ledger["source_manifest_intact"] is True


# ------------------------------------------------------------- (3) policy ----


def test_contract_commands_from_run_sh(tmp_path: Path):
    """P1.1: run.sh python lines parse in order; pip lines are skipped."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "run.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd \"$(dirname \"$0\")\"\n"
        "pip install -r requirement.txt\n"
        "python3 hp_exact_enumeration.py\n"
        "python helper.py --mode fast\n"
        "echo done\n",
        encoding="utf-8",
    )
    source, cmds = reexec_mod.contract_commands(ws)
    assert source == "run.sh"
    assert cmds == [
        ("hp_exact_enumeration.py", []),
        ("helper.py", ["--mode", "fast"]),
    ]
    # README is the secondary contract; nothing anywhere -> no contract
    ws2 = tmp_path / "ws2"
    ws2.mkdir()
    (ws2 / "README.md").write_text(
        "Run:\n\n    python analysis.py out.csv\n", encoding="utf-8"
    )
    source2, cmds2 = reexec_mod.contract_commands(ws2)
    assert source2 == "readme" and cmds2 == [("analysis.py", ["out.csv"])]
    ws3 = tmp_path / "ws3"
    ws3.mkdir()
    assert reexec_mod.contract_commands(ws3) == ("", [])


def test_reexec_runs_run_sh_not_largest_py(tmp_path: Path):
    """P1.1 integration: run.sh decides the program (the HP gen-3 stale
    largest-.py bug); ledger records source + commands; fresh artifacts
    come from the CONTRACT program."""
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    (ws / "run.sh").write_text(
        "pip install -r requirement.txt\npython3 deliverable.py\n",
        encoding="utf-8",
    )
    (ws / "requirement.txt").write_text("# stdlib only\n", encoding="utf-8")
    (ws / "deliverable.py").write_text(
        "from pathlib import Path\n"
        "Path('hp_min_energy_results.json').write_text('{\"E\": -2}')\n",
        encoding="utf-8",
    )
    (ws / "stale_big.py").write_text("x = 1\n" * 200, encoding="utf-8")
    runner = ReexecRunner(tmp_path / "sb", timeout_s=60, venv_reuse=False)
    ledger = runner.run(ws, "g1")
    assert ledger["main_script_source"] == "shell_run_sh"
    assert ledger["main_script"] == "deliverable.py"
    assert ledger["commands"] == [{"script": "deliverable.py", "args": []}]
    assert ledger["install_source"] == "requirement.txt"  # P1.2 singular
    assert ledger["exit_code"] == 0
    assert "hp_min_energy_results.json" in ledger["fresh_artifacts"]


def test_reexec_shell_preserves_pipes_and_redirects(tmp_path: Path):
    """Shell execution: run.sh pipes/redirects execute as written.

    The python-argv extraction dropped ``| tee`` so tee'd files never
    refreshed on re-execution and were reported stale. ``bash run.sh``
    preserves the full pipeline (operator directive 2026-10-10).
    """
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    (ws / "run.sh").write_text(
        "python3 gen.py | tee run_output.txt\n"
        "python3 gen.py > redirected.txt\n",
        encoding="utf-8",
    )
    (ws / "gen.py").write_text(
        "print('Minimum energy found: -9')\n", encoding="utf-8"
    )
    runner = ReexecRunner(tmp_path / "sb", timeout_s=60, venv_reuse=False)
    ledger = runner.run(ws, "g1")
    assert ledger["main_script_source"] == "shell_run_sh"
    assert ledger["exit_code"] == 0
    # both shell-produced files must be FRESH (mtime after t0_wall)
    assert "run_output.txt" in ledger["fresh_artifacts"]
    assert "redirected.txt" in ledger["fresh_artifacts"]
    assert "Minimum energy found: -9" in ledger["log_tail"]


def test_metric_resolution_via_llm_text_extraction():
    """LLM text extraction replaces regex scraping (operator directive:
    no regex, no brittle patterns). A metric printed in the program
    output resolves through the llm_extract callable."""
    from sources.evaluators.policy_bt_verifier import metrics as mm

    digest = {
        "exec": {"log_tail": ["Minimum energy found: -9",
                              "Contacts in optimal: 9"]},
        "json_metrics": {},
        "log_metrics": [],
        "deliverables": [],
    }
    primary = [
        {"id": "min_energy", "key": "minimum energy",
         "aliases": [{"file": "results.json", "key": "minimum_energy"}]},
    ]
    seen_prompts: list[str] = []

    def fake_extract(text_lines, metric_keys, llm_json=None):
        # The resolver hands us the raw text lines and the alias keys
        seen_prompts.append("\n".join(text_lines))
        assert "minimum_energy" in metric_keys
        # simulate the LLM reading "Minimum energy found: -9"
        assert any("-9" in ln for ln in text_lines)
        return {"minimum_energy": -9.0}

    out = mm.resolve_primary_values(digest, primary, llm_extract=fake_extract)
    assert out["min_energy"]["status"] == "resolved"
    assert out["min_energy"]["value"] == -9.0
    assert out["min_energy"]["source"] == "llm:minimum_energy"
    assert seen_prompts  # the extractor actually received the text

    # without llm_extract the same digest resolves nothing (no regex path)
    out2 = mm.resolve_primary_values(digest, primary)
    assert out2["min_energy"]["status"] == "missing"

def test_scan_imports_remap_and_local_dirs(tmp_path: Path):
    """P1.3: sklearn/PIL map to their PyPI names; local dirs and stdlib
    are skipped; the union scan covers every top-level .py."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "main.py").write_text(
        "import sklearn\nfrom PIL import Image\n", encoding="utf-8"
    )
    (ws / "util.py").write_text("import utils\n", encoding="utf-8")
    (ws / "utils").mkdir()
    (ws / "utils" / "__init__.py").write_text("", encoding="utf-8")
    assert reexec_mod.scan_imports(ws / "main.py") == [
        "pillow",
        "scikit-learn",
    ]
    assert reexec_mod.scan_imports_all(ws) == ["pillow", "scikit-learn"]


def test_digest_contract_files_and_unified_infra(tmp_path: Path):
    """P1.4/P1.5: goal-named files recorded (any suffix, presence from
    the snapshot); infra (marker_*, requirement.txt, run.sh) never a
    deliverable — but contract records keep them visible."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "run.sh").write_text("python3 solve.py\n", encoding="utf-8")
    (ws / "requirement.txt").write_text("numpy\n", encoding="utf-8")
    (ws / "marker_abc123.py").write_text("x=1\n", encoding="utf-8")
    (ws / "marker_def456.txt").write_text("m\n", encoding="utf-8")
    (ws / "data.csv").write_text("a\n1\n", encoding="utf-8")
    goal = (
        "Write results to pred_results/out.csv; provide run.sh and "
        "requirement.txt; see README.md for details."
    )
    d = digest_mod.build_digest("u1", ws, None, goal)
    cf = {c["path"]: c for c in d["contract_files"]}
    assert cf["run.sh"]["present"] is True
    assert cf["run.sh"]["head"].startswith("python3 solve.py")
    assert cf["requirement.txt"]["present"] is True
    assert cf["README.md"]["present"] is False  # the honest miss
    assert cf["pred_results/out.csv"]["present"] is False
    paths = [dv["path"] for dv in d["deliverables"]]
    assert "run.sh" not in paths and "requirement.txt" not in paths
    assert not any(p.startswith("marker_") for p in paths)
    assert "data.csv" in paths


def test_refiner_preview_lists_contract_and_full_tree(tmp_path: Path):
    """P1.6: preview carries the goal-contract records + a FULL listing
    so the refiner can never infer absence from the filtered inventory."""
    assert "NOT absent from the workspace" in gradient_mod.REFINER_PROMPT
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "run.sh").write_text("python3 solve.py\n", encoding="utf-8")
    (ws / "solve.py").write_text("print('hi')\n", encoding="utf-8")
    digest = digest_mod.build_digest("u1", ws, None, "provide run.sh")
    pv = gradient_mod.build_preview(ws, digest, None)
    assert "goal-contract files" in pv
    assert "run.sh | True" in pv
    assert "FULL top-level file listing" in pv
    assert "solve.py |" in pv  # .py IS in the full listing (never a deliverable)


def test_static_screen_allowlist_and_banlist():
    assert static_screen(_TIE_POLICY_BODY) == []
    assert static_screen("import sys\ndef compare(dA, dB): pass")
    errs = static_screen("import sys\ndef compare(dA, dB): pass")
    assert any("banned import: sys" in e for e in errs)
    errs = static_screen(
        "def compare(dA, dB):\n    return open('x').read()\n"
    )
    assert any("banned name: open" in e for e in errs)
    errs = static_screen("x = 1\n")
    assert any("missing def compare" in e for e in errs)
    errs = static_screen("def compare(dA, dB): return eval('1')\n")
    assert any("banned name: eval" in e for e in errs)


def test_truncated_json_never_becomes_policy():
    truncated = '{"policy_py": "import json\ndef compare(dA, dB): ...'  # cut
    assert policy_mod.policy_source(truncated) == policy_mod.TRUNCATED_SENTINEL
    assert policy_mod.policy_source("raw unparseable text") == (
        policy_mod.TRUNCATED_SENTINEL
    )
    # and the sentinel itself fails the screen -> drives the repair loop
    assert static_screen(policy_mod.TRUNCATED_SENTINEL)


def test_ensure_policy_escalation_and_cache_busting():
    llm = _ScriptedLLM(first_policy_raw='{"policy_py": "import json\ndef c')
    llm_json = lambda prompt, mt, role: llm("u", role, prompt, mt, None)  # noqa: E731
    digests = [{"deliverables": [{"path": "p"}]}, {"deliverables": []}]
    src, meta = ensure_policy(
        GOAL, RECON_JSON, digests, llm_json, repair_rounds=3, log=_FakeLog()
    )
    assert src is not None and "def compare" in src
    assert meta["smoke"] == "pass"
    assert meta["escalations"] == 1
    # attempt 1 at 12000; escalation at 32768 with a DISTINCT cache-busting role
    assert llm.calls[0] == ("policy_write_v3", 12000)
    assert llm.calls[1][1] == 32768
    assert llm.calls[1][0].endswith("_32768")
    assert meta["attempts"] == 1


def test_ensure_policy_repair_feeds_screen_errors():
    bad = "import sys\ndef compare(dA, dB): return {}\n"
    prompts: list[str] = []

    def llm_json(prompt, mt, role):
        prompts.append(prompt)
        if "fix" not in role:
            return {"policy_py": bad}
        return {"policy_py": _DECISIVE_POLICY_BODY}

    src, meta = ensure_policy(
        GOAL, RECON_JSON, [{}], llm_json, repair_rounds=2, log=None
    )
    assert src is not None
    assert meta["repairs"] == 1 and meta["attempts"] == 2
    # REAL repair plumbing: the exact screen error is fed back under a
    # REPAIR REQUIRED banner (cache-busting role is exercised by "fix"
    # routing above — a same-prompt cache hit could not repair)
    assert "banned import: sys" in prompts[1]
    assert "REPAIR REQUIRED" in prompts[1]


def test_bt_fit_ridge_sign_converges():
    """P4.1: the ridge term must SHRINK theta (Newton on the MAP
    objective). The old ``g += RIDGE * th`` inverted the sign, Newton
    diverged (theta ~ +/-6e4..6e6) and every offline BT rating
    saturated; one match A>B now converges to a finite MAP optimum
    (|theta| ~ 0.34, gap ~ 0.67), and the cyclic mixed record stays
    finite and sub-unit in magnitude."""
    import math

    th = verdicts_mod.bt_fit_weighted([0], [1], [1.0], 2)
    assert th[0] > th[1]
    assert abs(th[0]) < 1.5 and abs(th[1]) < 1.5  # converges, no blow-up
    gap = th[0] - th[1]
    assert 0.1 < gap < 1.0  # MAP gap ~0.67 for one win, ridge 1.0
    # cyclic 0>1, 1>0 + a 3rd-player split: finite, sub-unit magnitudes
    th2 = verdicts_mod.bt_fit_weighted([0, 1, 0], [1, 0, 1], [1.0, 1.0, 1.0], 2)
    assert all(math.isfinite(t) for t in th2)
    assert all(abs(t) < 1.0 for t in th2)
    # weighted acceptance example from the review: stays well-behaved
    th3 = verdicts_mod.bt_fit_weighted([0, 1, 0], [1, 0, 1], [1, 1, 1], 2)
    assert th3[0] == pytest.approx(th2[0])


def test_run_policy_and_smoke_both_orders():
    full = _DECISIVE_POLICY_BODY + MAIN_STANZA
    dA = {"deliverables": [{"path": "a"}, {"path": "b"}]}
    dB = {"deliverables": [{"path": "a"}]}
    v = run_policy(full, dA, dB)
    assert v["winner"] == "A" and "error" not in v
    v_rev = run_policy(full, dB, dA)
    assert v_rev["winner"] == "B"
    ok, err = smoke_test(full, [dA, dB])
    assert ok and err is None
    # antisymmetry violation surfaces as a smoke failure
    bad_policy = (
        "import json\n"
        "def compare(dA, dB):\n"
        "    return {\"winner\": \"A\", \"confidence\": 0.9, \"rungs\": [],\n"
        "            \"deciding\": [], \"feedback_winner\": \"x\",\n"
        "            \"feedback_looser\": \"y\"}\n"
    ) + MAIN_STANZA
    ok, err = smoke_test(bad_policy, [dA, dB])
    assert not ok and "antisymmetry" in err


def test_smoke_covers_every_distinct_pair_branch_only_errors():
    """P0.3: a policy that only crashes on ONE digest shape (HP gen 4's
    KeyError('saw_count') on every gen 1-3 pair) must fail the smoke —
    the old first-two-digests smoke passed it."""
    branchy = (
        "import json\n"
        "def compare(dA, dB):\n"
        "    if (dA.get('tables') or []) and not (dB.get('tables') or []):\n"
        "        return dA['saw_count']  # KeyError: branch-only error\n"
        "    return {\"winner\": \"tie\", \"confidence\": 0.5, \"rungs\": [],\n"
        "            \"deciding\": [], \"feedback_winner\": \"x\",\n"
        "            \"feedback_looser\": \"y\"}\n"
    ) + MAIN_STANZA
    digests = [
        {},
        {"deliverables": [{"path": "x"}]},
        {"tables": [{"path": "pred_results/p.csv"}]},
    ]
    # the old smoke ran only the FIRST TWO digests: both orders of
    # ({}, {deliverables}) never take the crashing branch
    ok_old_style, _ = smoke_test(branchy, digests[:2])
    assert ok_old_style
    ok, err = smoke_test(branchy, digests)
    assert not ok and "KeyError" in err and "saw_count" in err


def test_run_policy_non_dict_output_is_error_not_crash():
    """P0.6: a policy returning None (or a list) on one branch yields a
    tie with an error field — never an exception out of the pair path."""
    none_policy = (
        "import json\n"
        "def compare(dA, dB):\n"
        "    if dB.get('deliverables'):\n"
        "        return None\n"
        "    return {\"winner\": \"tie\", \"confidence\": 0.5, \"rungs\": [],\n"
        "            \"deciding\": [], \"feedback_winner\": \"\",\n"
        "            \"feedback_looser\": \"\"}\n"
    ) + MAIN_STANZA
    v = run_policy(none_policy, {}, {"deliverables": [{"path": "a"}]})
    assert "error" in v and "non-dict" in v["error"]
    verdict = verdicts_mod.code_pair_verdict(
        run_policy,
        none_policy,
        "uA",
        {},
        "uB",
        {"deliverables": [{"path": "a"}]},
        log=_FakeLog(),
    )
    assert verdict["winner"] == "tie" and verdict.get("error")


# ------------------------------------------------------------ (4) verdicts ----


def test_win_rate_math_and_first_generation():
    r = verdicts_mod.win_rate_reward(["win", "win", "tie", "loss"])
    assert r["reward"] == pytest.approx((2 + 0.5) / 4)
    assert (r["wins"], r["losses"], r["ties"]) == (2, 1, 1)
    r0 = verdicts_mod.win_rate_reward([])
    assert r0["reward"] == 0.5 and r0["n_pairs"] == 0


def test_bt_fit_transitivity_and_tie_half_weights():
    uuids = ["A", "B", "C"]
    recs = [
        {"a": "A", "b": "B", "winner": "A"},
        {"a": "B", "b": "C", "winner": "B"},
        {"a": "A", "b": "C", "winner": "A"},
    ]
    out = verdicts_mod.bradley_terry_reward(uuids, recs)
    ratings = out["ratings"]
    assert ratings["A"] > ratings["B"] > ratings["C"]
    assert out["pick"] == "A"
    # a tie is two half-weight matches -> identical ratings
    tie = verdicts_mod.bradley_terry_reward(
        ["A", "B"], [{"a": "A", "b": "B", "winner": "tie"}]
    )
    assert tie["ratings"]["A"] == pytest.approx(tie["ratings"]["B"], abs=1e-6)
    # hand-computed 2-player sanity: single win -> positive gap
    th = verdicts_mod.bt_fit_weighted([0], [1], [1.0], 2)
    assert th[0] > th[1]


def test_code_pair_verdict_mirrored_and_disagreement():
    full = _DECISIVE_POLICY_BODY + MAIN_STANZA
    dA = {"deliverables": [{"path": "a"}, {"path": "b"}]}
    dB = {"deliverables": [{"path": "a"}]}
    v = verdicts_mod.code_pair_verdict(
        run_policy, full, "uuidA", dA, "uuidB", dB, log=None
    )
    assert v["winner"] == "uuidA" and v["mirrored"] is True

    calls = {"n": 0}

    def always_first_arg(src, x, y):
        calls["n"] += 1
        return {
            "winner": "A",
            "confidence": 0.9,
            "feedback_winner": "x",
            "feedback_looser": "y",
        }

    log = _FakeLog()
    v = verdicts_mod.code_pair_verdict(
        always_first_arg, full, "uuidA", dA, "uuidB", dB, log=log
    )
    assert v["winner"] == "tie" and v["mirrored"] is False
    assert any("antisymmetry disagreement" in w for w in log.warns)


def test_crash_rung_rules():
    clean = {"exit_code": 0, "timed_out": False, "error_signature": [],
             "fresh_artifacts": {"a": 1}}
    crash = {"exit_code": 1, "timed_out": False,
             "error_signature": ["ModuleNotFoundError"], "fresh_artifacts": {}}
    assert verdicts_mod.crash_rung("u1", clean, "u2", clean) is None
    v = verdicts_mod.crash_rung("u1", clean, "u2", crash)
    assert v["winner"] == "u1" and v["rung"] == "one_side_crashed"
    v = verdicts_mod.crash_rung("u1", crash, "u2", dict(crash))
    assert v["winner"] == "tie"  # identical signatures tie (chem clintox)
    other = dict(crash, error_signature=["AssertionError"])
    v = verdicts_mod.crash_rung("u1", crash, "u2", other)
    assert v["winner"] == "tie"  # different but equal partial evidence
    partial = dict(crash, error_signature=["AssertionError"],
                   fresh_artifacts={"a": 1, "b": 2})
    v = verdicts_mod.crash_rung("u1", partial, "u2", crash)
    assert v["winner"] == "u1"
    # identical signatures dominate partial evidence (chem clintox rule)
    partial_same_sig = dict(crash, fresh_artifacts={"a": 1, "b": 2})
    v = verdicts_mod.crash_rung("u1", partial_same_sig, "u2", crash)
    assert v["winner"] == "tie"

# ------------------------------------------------------------- (5) visual ----


def _png(tmp_path: Path, name: str, size: int = 200) -> Path:
    p = tmp_path / name
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * size)
    return p


def test_visual_prompt_assembly_labels_both_sides():
    prompt = visual_mod.build_pair_prompt(GOAL, ["a1.png"], ["b1.png", "b2.png"])
    assert "CANDIDATE A FIGURES" in prompt and "- a1.png" in prompt
    assert "CANDIDATE B FIGURES" in prompt and "- b2.png" in prompt
    assert "winner" in prompt and "feedback_looser" in prompt


def test_visual_retry_once_and_self_heal_cache(tmp_path: Path):
    fa, fb = [_png(tmp_path, "a.png"), _png(tmp_path, "b.png")]
    store: list[dict] = []
    roles: list[str] = []
    responses = iter([
        "",  # first order, first attempt: empty (billable-but-zero-byte)
        json.dumps({"winner": "A", "confidence": 0.9,
                    "figure_notes": "A clearer",
                    "feedback_winner": "A shows labeled axes",
                    "feedback_looser": "B lacks axis labels"}),
        json.dumps({"winner": "B", "confidence": 0.9,  # mirrored order
                    "figure_notes": "A clearer",
                    "feedback_winner": "A shows labeled axes",
                    "feedback_looser": "B lacks axis labels"}),
    ])

    def vision_call(prompt, figs_a, figs_b, max_tokens, role):
        roles.append(role)
        return next(responses)

    ch = VisualChannel(
        vision_call,
        cache_get=lambda k: next(
            (e for e in store if e.get("k") == k and not e.get("parse_fail")),
            None,
        ),
        cache_put=lambda k, e: store.append({"k": k, **e}),
        log=_FakeLog(),
    )
    v = ch.judge_pair(GOAL, "uA", [fa], "uB", [fb])
    # empty-first-order retried; mirrored B-side served by a fresh call too
    assert v is not None and v["winner"] == "uA" and v["mirrored"] is True

    # P3.1: the retry after the parse fail carried a DIFFERENT role
    # (attempt + max_tokens) so the provider cache cannot re-serve the
    # failed attempt verbatim.
    assert roles == ["r1_600", "r2_800", "r1_600"]
    # a later identical pair is served entirely from cache (self-heal scan
    # skips the stored parse-fail entry and prefers the healed one)
    responses2 = iter(["should-not-be-called"])
    ch2 = VisualChannel(
        lambda p, fa, fb, m, r: next(responses2),
        lambda k: next(
            (e for e in store if e.get("k") == k and not e.get("parse_fail")), None
        ),
        lambda k, e: store.append({"k": k, **e}),
        log=_FakeLog(),
    )
    v2 = ch2.one_order(GOAL, "uA", [fa], "uB", [fb])
    assert v2 is not None and v2["winner"] == "A"


def test_visual_disagreement_ties_with_warning(tmp_path: Path):
    fa, fb = [_png(tmp_path, "a.png"), _png(tmp_path, "b.png")]
    flip = {"winner": "A", "confidence": 0.9}
    calls = {"n": 0}

    def vision_call(prompt, figs_a, figs_b, max_tokens, role):
        calls["n"] += 1
        return json.dumps(flip)  # same answer in both orders -> not mirrored

    log = _FakeLog()
    ch = VisualChannel(vision_call, lambda k: None, lambda k, e: None, log=log)
    v = ch.judge_pair(GOAL, "uA", [fa], "uB", [fb])
    assert v["winner"] == "tie" and v["mirrored"] is False
    assert any("antisymmetry disagreement" in w for w in log.warns)
    assert calls["n"] == 2


def test_figure_rung_one_side_no_figures():
    """P3.4: the rung decides on DIGEST figure counts, not uploads."""
    v = visual_mod.figure_rung("uA", 2, "uB", 0)
    assert v["winner"] == "uA" and v["rung"] == "one_side_no_figures"
    assert visual_mod.figure_rung("uA", 0, "uB", 0) is None


def test_visual_cap_falls_to_code_policy_with_warning(tmp_path):
    ev = _make_evaluator(tmp_path)
    ev.visual_max_pairs_per_gen = 1
    registry = TaskRegistry.load(ev._runner_temp_root, GOAL)
    log = _FakeLog()
    runner = _FakeReexecRunner(tmp_path / "sb")
    _seed_workspace(ev.workspace_dir, with_figs=True)
    ledger = runner.run(ev.workspace_dir, "genA")
    digest_a = digest_mod.build_digest("genA", runner.run_copy_path("genA"), ledger)
    # predecessor with REAL figure files on disk (both sides must have
    # figures, else the figure rung decides before the cap applies)
    ws_b = tmp_path / "ws_b"
    _seed_workspace(ws_b, with_figs=True)
    ledger_b = runner.run(ws_b, "genB")
    digest_b = digest_mod.build_digest("genB", runner.run_copy_path("genB"), ledger_b)
    assert visual_mod.select_figures(digest_b, runner.run_copy_path("genB"))
    visual = ev._make_visual_channel("genA", GOAL, registry, log)
    pred = {"uuid": "genB", "digest": digest_b, "ledger": ledger_b}
    verdict = ev._pair_verdict(
        "genA", GOAL, digest_a, ledger, runner.run_copy_path("genA"),
        runner, pred, _TIE_POLICY_BODY + MAIN_STANZA, "test_sha_123",
        visual, allow_visual=False, registry=registry, log=log,
    )
    assert verdict["channel"] == "code_policy"
    assert any("budget cap" in w for w in log.warns)


# ------------------------------------------------------------ (6) gradient ----


def test_gradient_fold_ordering_losses_ties_wins():
    recs = [
        {"other": "g1", "rel": "won", "feedback": "W1", "confidence": 0.9},
        {"other": "g2", "rel": "lost", "feedback": "L2", "confidence": 0.7},
        {"other": "g3", "rel": "tie", "feedback": "", "confidence": 0.5},
        {"other": "g4", "rel": "lost", "feedback": "L4", "confidence": 0.95},
        {"other": "g5", "rel": "won", "feedback": "W5", "confidence": 0.8},
    ]
    text, meta = fold_gradient(
        "gen", recs, coverage_footer="cover (grading bars unknown)"
    )
    assert meta["footer_appended"] is True
    assert text.index("L4") < text.index("L2")  # confidence-sorted losses
    assert text.index("L2") < text.index("UNRESOLVED")
    assert text.index("g3") < text.index("KEEP")
    assert "NOT ASSESSED (grading bars unknown): cover" in text
    # no coverage footers INSIDE the pairwise feedback (v2b lesson)
    for seg in ("L4", "L2", "W1", "W5"):
        assert "not assessed" not in text[text.index(seg):text.index(seg) + len(seg) + 10]


def test_refiner_contract_keys_and_no_solutions():
    assert "Never propose solutions" in gradient_mod.REFINER_PROMPT
    llm_json = lambda p, m, r: dict(REFINER_JSON)  # noqa: E731
    out = refine_gradient(llm_json, "gen", "RAW", "PREVIEW")
    for key in gradient_mod.REFINER_KEYS:
        assert key in out
    assert refine_gradient(lambda p, m, r: "junk", "gen", "RAW", "P") is None
    assert gradient_mod.refined_text("RAW", None) == "RAW"
    # P1.7: the refined text KEEPS the fold header + decisive-loss block
    fold = (
        "GEN g4 — verdicts vs predecessors (3 pairs: 3W/0L/0T):\n"
        "\n"
        "WHAT TO FIX FIRST (decisive losses):\n"
        "- none: no decisive losses against any predecessor.\n"
        "\n"
        "KEEP (what beats predecessors):\n"
        "- vs g1: auc 0.9 vs 0.7\n"
    )
    final = gradient_mod.refined_text(fold, REFINER_JSON)
    assert final.startswith("GEN g4 — verdicts vs predecessors (3 pairs: 3W/0L/0T):")
    assert "WHAT TO FIX FIRST" in final
    assert REFINER_JSON["refined_gradient"] in final


# ------------------------------------------------------------ (7) registry ----


def test_registry_persistence_and_arrival_order(tmp_path: Path):
    reg = TaskRegistry.load(tmp_path, GOAL)
    reg.record_generation("u1", {"uuid": "u1"}, reward=0.5)
    reg.record_generation("u2", {"uuid": "u2"}, reward=1.0)
    reg.set_verdict(code_verdict_key("sha", "u1", "u2"), {"winner": "u1"})
    reg.add_visual_entry(visual_key("pv", "u1", "u2", "AB", "ih"), {"winner": "A"})
    reg.set_recon(RECON_JSON, recon_mod.RECON_PROMPT_VERSION)
    reg.save()

    reg2 = TaskRegistry.load(tmp_path, GOAL)  # restart-survivable
    assert [g["uuid"] for g in reg2.generations] == ["u1", "u2"]
    assert reg2.previous_generations("u2")[0]["uuid"] == "u1"
    assert reg2.previous_generations("u1") == []
    assert reg2.get_verdict(code_verdict_key("sha", "u1", "u2"))["winner"] == "u1"
    assert reg2.get_recon(recon_mod.RECON_PROMPT_VERSION) == RECON_JSON
    assert reg2.rewards() == {"u1": 0.5, "u2": 1.0}
    # upsert keeps arrival position
    reg2.record_generation("u1", {"uuid": "u1"}, reward=0.9)
    assert [g["uuid"] for g in reg2.generations] == ["u1", "u2"]
    assert reg2.generation("u1")["reward"] == 0.9
    # self-heal scan sees parse-fail entries but they never satisfy a hit
    reg2.add_visual_entry(visual_key("pv", "u1", "u3", "AB", "ih"), {"parse_fail": True})
    hits = reg2.visual_entries(visual_key("pv", "u1", "u3", "AB", "ih"))
    assert len(hits) == 1 and hits[0].get("parse_fail")


def test_task_key_stability():
    assert task_key(GOAL) == task_key(GOAL)
    assert task_key(GOAL) != task_key(GOAL + " ")
    assert len(task_key(GOAL)) == 16


# ------------------------------------------------- (8) evaluator + logging ----


def test_short_circuit_paths(tmp_path: Path, capsys):
    ev = _make_evaluator(tmp_path)
    uuid = "deadgen0001"
    _wf_dir(tmp_path, uuid)
    # no state_result -> short circuit (folder has state_result though; wipe it)
    (tmp_path / "workflows" / uuid / "state_result.json").write_text("{}", encoding="utf-8")
    out = ev.evaluate(uuid)
    assert out["overall_score"] == 0.0
    assert out["abstracted_textual_gradient"]
    assert out["abstractec_textual_gradient"] == out["abstracted_textual_gradient"]
    assert out["skipped_reason"]
    saved = json.loads(
        (tmp_path / "workflows" / uuid / "state_result.json").read_text()
    )
    assert saved["evaluation"]["verifier"]["overall_score"] == 0.0


def _run_generation(ev, llm, uuid, n_rows=3):
    _wf_dir(ev.workflow_dir.parent, uuid)
    _seed_workspace(ev.workspace_dir, n_rows=n_rows)
    ev._llm_json = llm
    return ev.evaluate(uuid)


def test_evaluate_accumulates_generations_and_stage_order(
    tmp_path: Path, monkeypatch, capsys
):
    monkeypatch.setattr(ev_mod, "ReexecRunner", _FakeReexecRunner)
    ev = _make_evaluator(tmp_path, verbose=True)
    llm = _ScriptedLLM()

    out1 = _run_generation(ev, llm, "aaaaaaaa-1111")
    assert out1["n_pairs"] == 0 and out1["overall_score"] == 0.5  # neutral prior
    assert out1["policy_smoke"] == "pass"
    assert "abstractec_textual_gradient" in out1

    # evolve: more deliverables -> decisive policy must beat predecessor
    (ev.workspace_dir / "pred_results" / "extra.csv").write_text(
        "smiles,probability\nCCO,0.5\n", encoding="utf-8"
    )
    out2 = _run_generation(ev, llm, "aaaaaaaa-2222")
    assert out2["n_pairs"] == 1 and out2["n_wins"] == 1
    assert out2["overall_score"] == 1.0

    # registry persisted the predecessor (tier-1 resolution)
    reg = TaskRegistry.load(ev._runner_temp_root, GOAL)
    assert [g["uuid"] for g in reg.generations] == [
        "aaaaaaaa-1111", "aaaaaaaa-2222",
    ]
    assert reg.policy is not None  # policy cached (composite sha)
    assert "_" in reg.policy["sha"]  # composite: POLICY_SHA_reconVersion

    # ---- §6b stage lines: capture stdout, assert order + JSON sink ----
    captured = capsys.readouterr().out
    tag1 = "aaaaaaaa-1111"[-8:]
    tag2 = "aaaaaaaa-2222"[-8:]  # P5.1: log tags are uuid SUFFIXES
    stage_lines = [
        ln for ln in captured.splitlines() if ln.startswith(f"[policy_bt][{tag2}]")
    ]
    stages_seen = [
        ln.split("]", 2)[2].split()[0] for ln in stage_lines if "]" in ln
    ]
    for stage in ("REEXEC", "DIGEST", "RECON", "POLICY", "PAIRS", "REWARD",
                  "GRADIENT", "DONE"):
        assert stage in stages_seen, f"missing stage {stage}: {stages_seen}"
    order = {s: stages_seen.index(s) for s in
             ("REEXEC", "DIGEST", "RECON", "POLICY", "PAIRS", "REWARD",
              "GRADIENT", "DONE")}
    assert order["REEXEC"] < order["DIGEST"] < order["RECON"] < order["POLICY"]
    assert order["POLICY"] < order["PAIRS"] < order["REWARD"] < order["GRADIENT"]
    assert order["GRADIENT"] < order["DONE"]
    assert any(f"(gen {tag2} vs {tag1})" in ln for ln in stage_lines)
    # uuid — uuid[:8] is the DATE in Mimosa uuids and collides all day)
    tkey = task_key(GOAL)
    led = ev._runner_temp_root / "policy_bt" / tkey / (
        "policy_bt_ledger_aaaaaaaa-2222.json"
    )
    assert led.exists()
    # verbose mode: per-pair detail lines present

    # JSON-lines sink parses and carries stage + uuid
    sink = ev.workflow_dir / "aaaaaaaa-2222" / "verifier_policy_bt.log"
    lines = [ln for ln in sink.read_text().splitlines() if ln.strip()]
    records = [json.loads(ln) for ln in lines]
    assert all("stage" in r and "uuid" in r for r in records)
    assert {r["stage"] for r in records} >= {
        "REEXEC", "DIGEST", "RECON", "POLICY", "PAIRS", "REWARD", "DONE"
    }
    # one ledger file per generation (full-uuid names never collide)
    led1 = (
        ev._runner_temp_root / "policy_bt" / tkey
        / "policy_bt_ledger_aaaaaaaa-1111.json"
    )
    assert led1.exists() and led.exists()
    # gradient + report artifacts alongside the hybrid convention
    wf_dir = ev.workflow_dir / "aaaaaaaa-2222"
    assert (wf_dir / "textual_gradient.txt").exists()
    assert (wf_dir / "evaluation.txt").exists()
    assert "1W/0L/0T" in (wf_dir / "evaluation.txt").read_text()


def test_evaluate_bradley_terry_mode(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ev_mod, "ReexecRunner", _FakeReexecRunner)
    ev = _make_evaluator(tmp_path, reward_mode="bradley_terry")
    llm = _ScriptedLLM()
    _run_generation(ev, llm, "bbbbbbbb-1111")
    (ev.workspace_dir / "pred_results" / "extra.csv").write_text(
        "smiles,probability\nCCO,0.5\n", encoding="utf-8"
    )
    out = _run_generation(ev, llm, "bbbbbbbb-2222")
    assert out["reward_mode"] == "bradley_terry"
    assert out["bt_ratings"]["bbbbbbbb-2222"] > out["bt_ratings"]["bbbbbbbb-1111"]
    assert 0.0 <= out["overall_score"] <= 1.0


def test_evaluate_excludes_unscoreable_predecessor_with_warning(
    tmp_path: Path, monkeypatch, capsys
):
    monkeypatch.setattr(ev_mod, "ReexecRunner", _FakeReexecRunner)
    ev = _make_evaluator(tmp_path)
    llm = _ScriptedLLM()
    _run_generation(ev, llm, "cccccccc-1111")
    # simulate a volatile /tmp: predecessor digest exists in registry, so
    # fabricate the opposite case — strip digests AND hide snapshots
    reg = TaskRegistry.load(ev._runner_temp_root, GOAL)
    for g in reg.generations:
        g["digest"] = None
    reg.save()
    out = _run_generation(ev, llm, "cccccccc-2222")
    assert out["n_excluded_predecessors"] == 1
    assert out["n_pairs"] == 0


def test_cached_policy_resmoked_and_rewritten_on_new_digest_shape(
    tmp_path: Path, monkeypatch, capsys
):
    """P0.4: a cached policy is re-validated against every stored digest
    shape each generation; a gen-1 policy that only ever self-compared
    and crashes on gen-2's shape falls to the rewrite path."""
    monkeypatch.setattr(ev_mod, "ReexecRunner", _FakeReexecRunner)
    ev = _make_evaluator(tmp_path)
    llm = _SeqPolicyLLM([_BRITTLE_JSON_POLICY_BODY])
    _run_generation(ev, llm, "dddddddd-1111")  # no metrics.json -> clean smoke
    assert llm.policy_writes == 1
    # gen 2 introduces a json_metrics file: the brittle cached policy
    # KeyErrors on the cross pair -> re-smoke fails -> rewrite happens
    (ev.workspace_dir / "metrics.json").write_text(
        json.dumps({"auc": 0.7}), encoding="utf-8"
    )
    out2 = _run_generation(ev, llm, "dddddddd-2222")
    assert llm.policy_writes == 2  # rewrite, not cache hit
    assert "failed the re-smoke" in capsys.readouterr().out
    assert out2["pairs"][0]["channel"] == "code_policy"  # not no_policy
    assert out2["policy_smoke"] == "pass"


def test_errored_pair_verdict_never_cached(tmp_path: Path, monkeypatch):
    """P0.5: a policy that errors once produces a decisive verdict on the
    next evaluation — the errored tie is not a permanent cached verdict."""
    ev = _make_evaluator(tmp_path)
    registry = TaskRegistry.load(ev._runner_temp_root, GOAL)
    log = _FakeLog()
    runner = _FakeReexecRunner(tmp_path / "sb")
    _seed_workspace(ev.workspace_dir)
    ledger = runner.run(ev.workspace_dir, "genA")
    digest_a = digest_mod.build_digest(
        "genA", runner.run_copy_path("genA"), ledger
    )
    ws_b = tmp_path / "ws_b"
    _seed_workspace(ws_b)
    (ws_b / "pred_results" / "extra.csv").write_text(
        "smiles,probability\nCCO,0.5\n", encoding="utf-8"
    )
    ledger_b = runner.run(ws_b, "genB")
    digest_b = digest_mod.build_digest(
        "genB", runner.run_copy_path("genB"), ledger_b
    )
    pred = {"uuid": "genB", "digest": digest_b, "ledger": ledger_b}
    real_run = ev_mod.policy_mod.run_policy
    policy = _DECISIVE_POLICY_BODY + MAIN_STANZA
    calls = {"n": 0}

    def flaky(src, dA, dB, timeout=60):  # P2.10: run_fn receives timeout
        calls["n"] += 1
        if calls["n"] <= 2:  # both orders of the first pair error
            return {"error": "KeyError: 'saw_count'", "wall_ms": 1}
        return real_run(src, dA, dB, timeout=timeout)


    monkeypatch.setattr(ev_mod.policy_mod, "run_policy", flaky)
    v1 = ev._pair_verdict(
        "genA", GOAL, digest_a, ledger, runner.run_copy_path("genA"),
        runner, pred, policy, "sha_x", None, False, registry, log,
    )
    assert v1["winner"] == "tie" and v1.get("error")
    key = code_verdict_key("sha_x", "genA", "genB")
    assert registry.get_verdict(key) is None  # NOT cached (P0.5)
    monkeypatch.setattr(ev_mod.policy_mod, "run_policy", real_run)
    v2 = ev._pair_verdict(
        "genA", GOAL, digest_a, ledger, runner.run_copy_path("genA"),
        runner, pred, policy, "sha_x", None, False, registry, log,
    )
    assert v2["winner"] == "genB" and "error" not in v2  # decisive on retry


def test_weak_evidence_guard_classification():
    """P0.7 unit: which decisive verdicts count as weak evidence."""
    primary = [{"field": "json_metrics", "key": "minimum_energy",
                "direction": "lower"}]
    # gen-4 style: self-reported metric outside the primary metrics
    weak = {
        "deciding": ["json_metrics[r.json].optimality.globally_optimal"],
        "confidence": 0.9,
    }
    assert ev_mod._decisive_weak_evidence(weak, primary) is True
    # the primary metric itself is strong evidence
    strong = {
        "deciding": ["json_metrics[hp_min_energy_results.json]."
                     "optimality.minimum_energy"],
        "confidence": 0.9,
    }
    assert ev_mod._decisive_weak_evidence(strong, primary) is False
    # low confidence on a self-reported rung is weak even for the key
    low_conf = dict(strong, confidence=0.5)
    assert ev_mod._decisive_weak_evidence(low_conf, primary) is True
    # structural/exec evidence is never weak
    structural = {"deciding": ["exec.exit_code"], "confidence": 1.0}
    assert ev_mod._decisive_weak_evidence(structural, primary) is False
    # string confidence must not raise (P2.2 preview)
    str_conf = dict(strong, confidence="high")
    assert ev_mod._decisive_weak_evidence(str_conf, primary) is True


def test_weak_evidence_reward_cap_integration(
    tmp_path: Path, monkeypatch, capsys
):
    """P0.7: all-decisive-on-unverifiable-self-reported-metrics caps the
    reward at 0.75 with reward_weak_evidence=true; a primary-metric win
    still reaches 1.0."""
    monkeypatch.setattr(ev_mod, "ReexecRunner", _FakeReexecRunner)

    # no primary metrics declared -> every decisive pair is weak -> cap
    recon_weak = dict(RECON_JSON, primary_metrics=[])
    ev = _make_evaluator(tmp_path / "noprimary")
    llm = _ScriptedLLM(recon=recon_weak, policy=_JSON_POLICY_BODY)
    _run_generation(ev, llm, "eeeeeeee-1111")
    (ev.workspace_dir / "metrics.json").write_text(
        json.dumps({"auc": 0.7}), encoding="utf-8"
    )
    ev.evaluate("eeeeeeee-1111")
    (ev.workspace_dir / "metrics.json").write_text(
        json.dumps({"auc": 0.9}), encoding="utf-8"
    )
    out2 = _run_generation(ev, llm, "eeeeeeee-2222")
    assert out2["n_wins"] == 1  # decisive on json_metrics
    assert out2["overall_score"] == 0.75  # capped (P0.7)
    assert out2["reward_weak_evidence"] is True
    assert out2["win_rate"] == 1.0  # the raw win-rate stays visible
    assert "weak-evidence guard" in capsys.readouterr().out

    # auc declared as a primary metric -> the same win is NOT capped
    recon_strong = dict(
        RECON_JSON,
        primary_metrics=[{"field": "json_metrics", "key": "auc",
                          "direction": "higher"}],
    )
    ev2 = _make_evaluator(tmp_path / "primary")
    llm2 = _ScriptedLLM(recon=recon_strong, policy=_JSON_POLICY_BODY)
    _run_generation(ev2, llm2, "ffffffff-1111")
    (ev2.workspace_dir / "metrics.json").write_text(
        json.dumps({"auc": 0.7}), encoding="utf-8"
    )
    ev2.evaluate("ffffffff-1111")
    (ev2.workspace_dir / "metrics.json").write_text(
        json.dumps({"auc": 0.9}), encoding="utf-8"
    )
    out4 = _run_generation(ev2, llm2, "ffffffff-2222")
    assert out4["overall_score"] == 1.0
    assert out4["reward_weak_evidence"] is False

# ---------------------------------------- (8b) N1/N2 + engine contract ----


def test_n1_loss_first_pair_returns_scores(tmp_path: Path, monkeypatch):
    """N1 (2026-10-08 audit): the loss branch set neither ``n_decisive``
    nor ``feedback`` — a generation whose FIRST pair is a loss raised
    UnboundLocalError out of evaluate() and aborted the campaign."""
    monkeypatch.setattr(ev_mod, "ReexecRunner", _FakeReexecRunner)
    ev = _make_evaluator(tmp_path)
    llm = _ScriptedLLM()
    # predecessor with MORE deliverables than the next generation
    _wf_dir(tmp_path, "lossgen01-1111")
    _seed_workspace(ev.workspace_dir)
    (ev.workspace_dir / "pred_results" / "extra.csv").write_text(
        "smiles,probability\nCCO,0.5\n", encoding="utf-8"
    )
    ev._llm_json = llm
    ev.evaluate("lossgen01-1111")
    (ev.workspace_dir / "pred_results" / "extra.csv").unlink()
    out = _run_generation(ev, llm, "lossgen01-2222")  # FIRST pair = loss
    assert out["n_losses"] == 1 and out["n_wins"] == 0
    assert out["overall_score"] == 0.0
    # the loss branch now supplies the LOOSER feedback (this gen lost)
    assert out["pairs"][0]["outcome"] == "lost"
    assert "deliverables" in out["pairs"][0]["feedback"]
    # engine contract §2.5: outcomes round-trip in the normalised spelling
    assert out["pair_outcomes"] == {"lossgen01-1111": "lost"}
    assert out["pair_outcomes"] == {
        p["prev_uuid"]: p["outcome"] for p in out["pairs"]
    }


def test_n1_mixed_win_and_weak_loss_cap(tmp_path: Path, monkeypatch):
    """N1 mixed case: a generation that WINS one predecessor and LOSES
    another must count both as decisive — pre-fix, n_decisive ignored
    losses so the weak-evidence guard (n_weak == n_decisive) could never
    fire on a loss-bearing generation."""
    monkeypatch.setattr(ev_mod, "ReexecRunner", _FakeReexecRunner)
    ev = _make_evaluator(tmp_path)
    # auc is NOT in the recon primary metrics -> every decisive auc pair
    # is weak evidence (P0.7), and every pair here is decisive.
    llm = _ScriptedLLM(policy=_JSON_POLICY_BODY)

    def gen(uuid: str, auc: float):
        _wf_dir(tmp_path, uuid)
        _seed_workspace(ev.workspace_dir)
        (ev.workspace_dir / "metrics.json").write_text(
            json.dumps({"auc": auc}), encoding="utf-8"
        )
        ev._llm_json = llm
        return ev.evaluate(uuid)

    gen("mixgen001-1111", 0.7)   # predecessor, auc 0.7
    out2 = gen("mixgen001-2222", 0.9)   # beats gen1 (win)
    out = gen("mixgen001-3333", 0.5)  # loses to BOTH, FIRST pair is a loss
    assert out["n_wins"] == 0 and out["n_losses"] == 2
    assert out["n_ties"] == 0
    # losses now count as decisive -> the all-weak guard fires
    assert out["reward_weak_evidence"] is True
    assert out["n_weak_evidence_pairs"] == 2
    assert out["win_rate"] == 0.0
    # pair_outcomes: the mixed vocabulary maps to won/lost/tie
    assert out["pair_outcomes"] == {
        "mixgen001-1111": "lost",
        "mixgen001-2222": "lost",
    }
    assert out2["pair_outcomes"] == {"mixgen001-1111": "won"}
    # and the engine's StopController consumes it unchanged: the loser
    # never becomes champion, losses advance the plateau
    from sources.core.stopping import StopController

    ctl = StopController(mode="plateau", min_iter=2, patience=1, max_iter=8,
                         verifier_kind="policy_bt")
    ctl.observe("mixgen001-1111", 0, 0.5, {}, failed=False)
    ctl.observe("mixgen001-2222", 1, 1.0,
                out2["pair_outcomes"], failed=False)
    assert ctl.champion == "mixgen001-2222"
    ctl.observe("mixgen001-3333", 2, 0.0, out["pair_outcomes"], failed=False)
    assert ctl.champion == "mixgen001-2222"  # lost -> incumbent stays
    assert ctl.since_change == 1

def test_n2_freshness_count_decisions_are_weak():
    """N2: HP 2026-10-08 — the gen-4 sweep was decided ONLY by
    ``deliverables[].produced_this_run`` counts ("3 vs 2" fresh files,
    science unchanged; deciding key verbatim from the run's registry).
    Freshness counts are weak evidence unless the deciding key names the
    goal contract or a recon primary metric."""
    freshness = {
        "deciding": ["deliverables[].produced_this_run"],
        "confidence": 0.9,
        "rungs": [{"name": "R2_fresh_deliverables", "verdict": "A"}],
    }
    # pre-N2 this was False (structural evidence, confidence 0.9)
    assert ev_mod._decisive_weak_evidence(freshness, []) is True
    # goal-contract presence stays objective — never weak
    contract = {
        "deciding": ["contract_files[run.sh].present"],
        "confidence": 0.6,
    }
    assert ev_mod._decisive_weak_evidence(contract, []) is False
    # a freshness decision on a recon primary-metric carrier is objective
    primary = [{"field": "json_metrics", "key": "auc", "direction": "higher"}]
    carrier = {
        "deciding": ["deliverables[results.json].produced_this_run.auc"],
        "confidence": 0.9,
    }
    assert ev_mod._decisive_weak_evidence(carrier, primary) is False
    # mixed rungs (freshness + a value determinant) keep the legacy path
    mixed = {
        "deciding": ["deliverables[].produced_this_run", "exec.exit_code"],
        "confidence": 0.9,
    }
    assert ev_mod._decisive_weak_evidence(mixed, []) is False


def test_n2_prompt_forbids_freshness_count_wins():
    """N2: the V3 policy prompt must tell the LLM that a raw COUNT of
    produced_this_run deliverables is never decisive evidence (the
    prompt hash — and thus the verdict cache — keys on this text)."""
    r2 = policy_mod.V3_PROMPT
    assert "R2 deliverable freshness" in r2
    assert "COUNT difference" in r2
    assert "never decisive" in r2




def test_config_defaults_and_json_roundtrip():
    from config import Config

    c = Config()
    assert c.verifier_kind == "hybrid"  # operator decision 1: opt-in
    assert c.policy_bt_reward == "win_rate"
    assert c.policy_bt_refiner is True  # operator decision 2
    assert c.policy_bt_reexec_timeout_s == 900
    assert c.policy_bt_visual_channel is True
    assert c.policy_bt_visual_tiebreak_code is False
    assert c.policy_bt_visual_max_pairs_per_gen == 0
    assert c.policy_bt_verbose is False

    c2 = Config()
    c2.from_json(
        {
            "verifier_kind": "policy_bt",
            "policy_bt_reward": "bradley_terry",
            "policy_bt_refiner": False,
            "policy_bt_reexec_timeout_s": 300,
            "policy_bt_visual_max_pairs_per_gen": 4,
            "policy_bt_verbose": True,
        }
    )
    assert c2.verifier_kind == "policy_bt"
    assert c2.policy_bt_reward == "bradley_terry"
    assert c2.policy_bt_refiner is False
    assert c2.policy_bt_reexec_timeout_s == 300
    assert c2.policy_bt_visual_max_pairs_per_gen == 4
    assert c2.policy_bt_verbose is True
    assert "policy_bt_reward" in Config().jsonify()


def test_cli_flags_apply_overrides():
    import argparse

    import main as main_mod
    from config import Config

    parser = argparse.ArgumentParser()
    main_mod.add_config_arguments(parser, Config())
    args = parser.parse_args(
        [
            "--verifier_kind", "policy_bt",
            "--policy_bt_reward", "bradley_terry",
            "--policy_bt_scorer_timeout_s", "90",
            "--no-policy_bt_visual_channel",
            "--no-policy_bt_refiner",
            "--policy_bt_reexec_timeout_s", "600",
            "--policy_bt_visual_tiebreak_code",
            "--policy_bt_verbose",
        ]
    )
    # the full CLI parser defines --max_evolve_iterations elsewhere
    args.__dict__.setdefault("max_evolve_iterations", None)
    c = Config()
    main_mod.apply_config_overrides(args, c)
    assert c.verifier_kind == "policy_bt"
    assert c.policy_bt_reward == "bradley_terry"
    assert c.policy_bt_scorer_timeout_s == 90
    assert c.policy_bt_visual_channel is False
    assert c.policy_bt_refiner is False
    assert c.policy_bt_reexec_timeout_s == 600
    assert c.policy_bt_visual_tiebreak_code is True
    assert c.policy_bt_verbose is True


def test_factory_selects_policy_bt(tmp_path: Path):
    from sources.evaluators.evaluator import WorkflowEvaluator

    cfg = types.SimpleNamespace(
        memory_dir=str(tmp_path / "mem"),
        workflow_dir=str(tmp_path / "wf"),
        model_pricing={},
        reasoning_effort="low",
        judge_model="openrouter/deepseek/deepseek-v4.1-flash",
        max_tokens=1024,
        openrouter_provider_for=lambda m: None,
        openrouter_quantizations_for=lambda m: None,
        verifier_kind="policy_bt",
        workspace_dir=str(tmp_path / "ws"),
        vision_judge_model=None,
        temp_dir=str(tmp_path / "tmp"),
    )
    for d in ("mem", "wf", "ws", "tmp"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    we = WorkflowEvaluator(cfg)
    assert isinstance(we.verifier_evaluator, PolicyBTVerifierEvaluator)


# ------------------------------------------------------------ (10) logging ----


def test_stage_logger_stdout_json_and_spend(tmp_path: Path, capsys):
    sink = tmp_path / "wf" / "u1" / "verifier_policy_bt.log"
    log = StageLogger("20261008_101010_abcdef01", sink=sink, verbose=False)
    log.stage("REEXEC", "sandbox=copy(3 files)", exit=0, fresh=1)
    log.add_cost(0.01)
    log.add_cost(None)
    log.detail("verbose-only line")
    log.warn("venv provisioning failed")
    log.stage("DONE", f"reward=0.5 total_llm={log.spend_line()}")
    out = capsys.readouterr().out
    tag = "20261008_101010_abcdef01"[-8:]
    assert f"[policy_bt][{tag}] REEXEC   sandbox=copy(3 files)" in out
    assert f"[policy_bt][{tag}] WARNING venv provisioning failed" in out
    assert "verbose-only line" not in out  # non-verbose suppresses detail
    assert log.spend_usd == pytest.approx(0.01)
    assert f"total_llm=${log.spend_usd:.4f}" in out
    records = [json.loads(ln) for ln in sink.read_text().splitlines()]
    assert records[0]["stage"] == "REEXEC" and records[0]["uuid"] == "20261008_101010_abcdef01"
    assert records[-1]["stage"] == "DONE"
    assert any(r["stage"] == "WARNING" for r in records)
    # P5.1: tags are uuid SUFFIXES — two same-day generations differ
    log_a = StageLogger("20261008_101010_aaaaaaaa")
    log_b = StageLogger("20261008_111111_bbbbbbbb")
    assert log_a.uuid8 == "aaaaaaaa" and log_b.uuid8 == "bbbbbbbb"
    # verbose mode emits the detail line
    log2 = StageLogger("abcdef01", sink=None, verbose=True)
    log2.detail("per-pair line")
    assert "DETAIL" in capsys.readouterr().out


# ======================================================================
# Incremental recon refresh + policy rewrite (2026-10-06)
# ======================================================================

class TestReconRefresh:
    """Recon versioning, trigger points, and policy rewrite on config change."""

    def test_config_hash_stable_and_different(self):
        """Same config → same hash; different config → different hash."""
        from sources.evaluators.policy_bt_verifier.evaluator import (
            PolicyBTVerifierEvaluator,
        )
        cfg_a = {"primary_metrics": [{"key": "auc"}], "tie_policy": "x"}
        cfg_b = {"primary_metrics": [{"key": "auc"}], "tie_policy": "x"}
        cfg_c = {"primary_metrics": [{"key": "rmse"}], "tie_policy": "x"}
        h1 = PolicyBTVerifierEvaluator._config_hash(cfg_a)
        h2 = PolicyBTVerifierEvaluator._config_hash(cfg_b)
        h3 = PolicyBTVerifierEvaluator._config_hash(cfg_c)
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 16

    def test_registry_recon_versioning(self, tmp_path):
        """set_recon stores version; get_recon_version returns it."""
        reg = TaskRegistry.load(tmp_path, "test-goal")
        assert reg.get_recon_version() == ""
        reg.set_recon({"key": "val"}, "recon-v1", recon_version="abc123")
        assert reg.get_recon_version() == "abc123"
        assert reg.get_recon("recon-v1") == {"key": "val"}
        reg.set_recon({"key": "new"}, "recon-v1", recon_version="def456")
        assert reg.get_recon_version() == "def456"
        assert reg.get_recon("recon-v1") == {"key": "new"}

    def test_refresh_gap_default(self):
        """Default gap is 2 generations between schema-triggered refreshes."""
        from config import Config
        c = Config()
        assert c.policy_bt_recon_refresh_gap == 2

    def test_composite_policy_sha_includes_config_hash(self):
        """Verdict cache keys use prompt_sha + recon_version, not bare POLICY_SHA."""
        # verified in the evaluator code: effective_policy_sha =
        # POLICY_SHA + "_" + recon_version[:8]
        from sources.evaluators.policy_bt_verifier import policy as pol
        assert "_" not in pol.POLICY_SHA  # the base sha has no underscore
        # the composite is built in evaluate(), tested via integration below

    def test_policy_rewrite_on_config_change(self, tmp_path):
        """When recon refresh produces a different config, the policy record
        is replaced and its sha changes (old verdicts stay cached under the
        old sha)."""
        reg = TaskRegistry.load(tmp_path, "test-goal")
        # Simulate first-gen policy
        reg.set_policy({"sha": "base123", "source": "old", "meta": {}})
        assert reg.get_policy("base123") is not None
        assert reg.get_policy("base123_v2") is None
        # Simulate refresh: new sha
        reg.set_policy({"sha": "base123_v2", "source": "new", "meta": {}})
        # Old sha no longer matches (single policy slot), new one does
        assert reg.get_policy("base123") is None
        assert reg.get_policy("base123_v2") is not None
        assert reg.get_policy("base123_v2")["source"] == "new"

    def test_no_refresh_between_triggers(self):
        """Generations 1, 2, 4, 5, 7-10, 12+ do NOT trigger refresh."""
        triggers = {3, 6, 11}
        for gen in [1, 2, 4, 5, 7, 8, 9, 10, 12, 15, 20]:
            assert gen not in triggers, f"gen {gen} should not trigger"
        for gen in [3, 6, 11]:
            assert gen in triggers, f"gen {gen} should trigger"


# ======================================================================
# Phase 2 (2026-10-08): N4-N13 + P2/P3/P5 regression tests
# ======================================================================


def test_n4_goal_file_regex_tokens():
    """N4: no bogus contract tokens — the policy treats present:false as
    decisive, so a bogus token is a bogus contract violation."""
    import tempfile
    from pathlib import Path as _P

    ws = _P(tempfile.mkdtemp())  # never read; contract_files checks
    goal = (
        "write results.json/.txt and metrics.json; skip data.jsonl; "
        "docs at https://example.com/run.sh; output pred_results/out.csv"
    )
    paths = [c["path"] for c in dm_contract(goal, ws)]
    assert "results.json" in paths          # split, not merged with /.txt
    assert not any("/." in p for p in paths)
    assert not any(p.endswith(".jsonl") for p in paths)  # data.jsonl -> no token
    assert not any("//" in p for p in paths)  # URL skipped
    assert "pred_results/out.csv" in paths


def dm_contract(goal, ws):
    return digest_mod.contract_files(goal, ws)


def test_n5_run_sh_flags_chains_redirects(tmp_path: Path):
    """N5: leading flags, chain splitting, redirect cuts, -m modules."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "run.sh").write_text(
        "python a.py > out.txt && python b.py; python -u c.py --mode fast 2>&1\n"
        "FOO=1 uv run python -m pkg.mod --flag value\n"
        "python3 -c 'print(1)'\n",
        encoding="utf-8",
    )
    source, cmds = reexec_mod.contract_commands(ws)
    assert source == "run.sh"
    assert cmds == [
        ("a.py", []),
        ("b.py", []),
        ("-u c.py", ["--mode", "fast"]),
        ("-m pkg.mod", ["--flag", "value"]),
    ]


def test_p24_p25_crash_rung_bootstrap_and_classes():
    """P2.4: bootstrap ledger defers to the channel; P2.5: timeout vs
    exit-1 are different classes (partial evidence decides, not a
    confidence-1.0 identical-signature tie)."""
    boot = {"bootstrap": "snapshot:x", "exit_code": None}
    crash = {"exit_code": 1, "timed_out": False, "error_signature": [],
             "fresh_artifacts": {}}
    assert verdicts_mod.crash_rung("u1", boot, "u2", crash) is None
    assert verdicts_mod.crash_rung("u1", crash, "u2", boot) is None
    timeout = {"exit_code": None, "timed_out": True,
               "error_signature": [], "fresh_artifacts": {}}
    v = verdicts_mod.crash_rung("u1", timeout, "u2", crash)
    assert v["winner"] == "tie"  # different classes, equal evidence
    assert v["rung"] == "both_crashed_different_signature_equal_evidence"
    # P2.8: a mutated source manifest is its own crash class
    mutated = dict(crash, source_manifest_intact=False,
                   fresh_artifacts={"a": 1})
    v = verdicts_mod.crash_rung("u1", mutated, "u2", crash)
    assert v["winner"] == "u1"  # more partial evidence wins


def test_p29_registry_atomic_save(tmp_path: Path):
    """P2.9: save() leaves no .tmp file and survives reload."""
    reg = TaskRegistry.load(tmp_path, "atomic-goal")
    reg.record_generation("u1", {"uuid": "u1"}, reward=0.5, session="s1")
    reg.save()
    assert not (tmp_path / f"policy_bt_registry_{task_key('atomic-goal')}.json.tmp").exists()
    reg2 = TaskRegistry.load(tmp_path, "atomic-goal")
    assert reg2.generation("u1")["session"] == "s1"


def test_n10_smoke_max_pairs_cap():
    """N10: max_pairs caps executed pairs on cache-hit re-smokes."""
    full = _TIE_POLICY_BODY + MAIN_STANZA
    calls = {"n": 0}

    def run(src, dA, dB):
        calls["n"] += 1
        return {"winner": "tie"}

    digests = [{"i": i} for i in range(4)]  # 6 pairs + self = 7
    ok, _ = smoke_test(full, digests, run_fn=run)
    assert ok and calls["n"] == 14
    calls["n"] = 0
    ok, _ = smoke_test(full, digests, run_fn=run, max_pairs=3)
    assert ok and calls["n"] == 6  # 3 pairs * both orders


def test_n12_screen_bans_reflection_and_dunders():
    """N12: the classic static-banlist escape hatches fail the screen."""
    bad = (
        "def compare(dA, dB):\n"
        "    x = getattr(__builtins__, '__imp' + 'ort__')('os')\n"
        "    return {'winner': 'tie'}\n"
    )
    errs = static_screen(bad)
    assert any("reflective" in e for e in errs)
    assert any("dunder" in e for e in errs)
    errs = static_screen("def compare(dA, dB):\n    y = (1).__class__\n    return {}\n")
    assert any("dunder attribute" in e for e in errs)
    assert static_screen(_TIE_POLICY_BODY) == []


def test_n13_string_confidence_folds():
    """N13: string confidence never raises in the loss sort."""
    recs = [
        {"other": "g1", "rel": "lost", "feedback": "L1", "confidence": "high"},
        {"other": "g2", "rel": "lost", "feedback": "L2", "confidence": 0.9},
    ]
    text, _meta = fold_gradient("gen", recs)
    assert "L1" in text and "L2" in text


def test_p55_id_order_match_honesty(tmp_path: Path):
    """P5.5: None when not evaluable; strict test-file token; equal length."""
    import pandas as pd

    ws = tmp_path
    pred = pd.DataFrame({"smiles": ["CCO", "NCC"]})
    assert dm_id_order(pred, ws) is None  # no test table anywhere
    (ws / "latest.csv").write_text("smiles\nCCO\nNCC\n", encoding="utf-8")
    assert dm_id_order(pred, ws) is None  # latest.csv is NOT a test file
    (ws / "test_in.csv").write_text("smiles\nCCO\nNCC\n", encoding="utf-8")
    assert dm_id_order(pred, ws)["match"] is True
    # length mismatch is a mismatch, not a prefix match
    (ws / "test_in.csv").write_text("smiles\nCCO\nNCC\nc1ccccc1\n", encoding="utf-8")
    assert dm_id_order(pred, ws)["match"] is False


def dm_id_order(pred, ws):
    return digest_mod.id_order_match(pred, ws)


def test_p56_json_metrics_relpath_priority_and_oversize(tmp_path: Path):
    """P5.6: relpath keys, metric-priority cap, oversized top-level scalars."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "pred_results").mkdir()
    (ws / "pred_results" / "metrics.json").write_text('{"auc": 0.9}', encoding="utf-8")
    (ws / "metrics.json").write_text('{"auc": 0.1}', encoding="utf-8")
    big = {"auc": 0.5, "n_rows": 10, "config": {"seed": 1}}
    big["points"] = [{"x": i, "y": i} for i in range(500)]  # many leaves
    (ws / "big.json").write_text(json.dumps(big), encoding="utf-8")
    (ws / "huge.json").write_text(json.dumps(dict(big, blob="x" * 70_000)), encoding="utf-8")
    d = digest_mod.build_digest("u", ws, None)
    assert d["json_metrics"]["pred_results/metrics.json"]["auc"] == 0.9
    assert d["json_metrics"]["metrics.json"]["auc"] == 0.1  # relpath, no collision
    jm_big = d["json_metrics"]["big.json"]
    assert len(jm_big) <= digest_mod.MAX_JSON_KEYS
    assert any("auc" in k for k in jm_big)  # metric-like keys survive the cap
    # oversized file: top-level scalars only, no deep leaves
    jm_huge = d["json_metrics"]["huge.json"]
    assert jm_huge["auc"] == 0.5 and not any(k.startswith("points") for k in jm_huge)


def test_p57_venv_fingerprint_rebuilds_stale_venv(tmp_path: Path):
    """P5.7: a venv with a stale fingerprint file is rebuilt."""
    sandbox = tmp_path / "sb"
    runner = ReexecRunner(sandbox, timeout_s=30, venv_reuse=True)
    assert runner._ensure_venv() is True  # created
    assert runner._ensure_venv() is False  # fingerprint matches -> reuse
    fp_file = sandbox / "venv" / ".mimosa_venv_fp"
    fp_file.write_text("stale-fingerprint\n", encoding="utf-8")
    assert runner._ensure_venv() is True  # rebuilt, never trusted
    assert fp_file.read_text().strip() == runner.venv_fingerprint()


# ======================================================================
# Metric ledger (review 2026-10-09 §4.1) — one source of truth for
# primary-metric values; replaces the fuzzy lookup + fold regex.
# ======================================================================


def _pm(mid, key, direction="higher", role="objective", aliases=None):
    """One normalized primary_metrics entry (ensure_shape output)."""
    return {
        "id": mid,
        "key": key,
        "direction": direction,
        "role": role,
        "aliases": aliases or [],
        "bounds": None,
        "integer": False,
        "convention": "",
    }


# --- frozen json_metrics of the REAL 10-08 HP-folding digests (registry
# policy_bt_registry_82c9c2036442baba.json; mined from the recon-refresh
# prompts of the run) — the key-drift the review documents ---
_DRIFT_CF812205 = {
    "results.json": {
        "contact_count": 2.0,
        "contacts": {"len": 2, "shape": [2, 2]},
        "length": 10.0,
        "min_energy": -2.0,
        "num_optimal_conformations_mod_D4_translation_reversal": 47.0,
        "num_raw_optimal_walks_first_step_plus_x": 94.0,
        "one_optimal_path": {"len": 10, "shape": [10, 2]},
    }
}
_DRIFT_396C629F = {
    "independent_check.json": {
        "independent_min_energy": -2.0,
        "independent_optimal_shapes_modulo_d4": 47.0,
        "independent_raw_optimal_walks": 376.0,
        "independent_total_saws": 16268.0,
        "n": 10.0,
        "reported_conformation_validation.energy": -2.0,
        "reported_conformation_validation.number_of_contacts": 2.0,
    },
    "independent_check_product.json": {
        "minimum_energy": -2.0,
        "raw_optimal_walks": 376.0,
        "reported_energy": -2.0,
        "total_saws": 16268.0,
    },
    "results.json": {
        "minimum_energy": -2.0,
        "n": 10.0,
        "one_optimal_conformation.number_of_h_h_contacts": 2.0,
        "optimal_conformations_modulo_symmetry": 47.0,
        "raw_optimal_walks": 376.0,
        "total_self_avoiding_walks_enumerated": 16268.0,
    },
}
_DRIFT_4FC65D93 = {
    "results.json": {
        "minimum_energy": -2.0,
        "optimal_conformation.coordinates": {"len": 10, "shape": [10, 2]},
        "hh_contacts": {"len": 2, "shape": [2, 2]},
        "raw_saw_count.count": 16268.0,
        "raw_optimal_walk_count.count": 376.0,
        "distinct_optimal_shapes.count": 47.0,
    }
}

# recon v1-style metrics: ids only, NO aliases (the pre-schema recon)
_DRIFT_METRICS_V1 = [
    _pm("min_energy", "results.json.minimum_energy", direction="lower"),
    _pm("hh_contacts", "results.json.optimal_conformation.hh_contact_count"),
    _pm("n_optimal", "results.json.number_of_optimal_conformations",
        role="invariant"),
    _pm("total_saws", "results.json.total_self_avoiding_walks_enumerated",
        role="invariant"),
]


class TestMetricLedger:
    """resolve_primary_values: alias/freshness/echo preferences, ambiguity."""

    def test_alias_hit_beats_lexical(self):
        """Test 1: an exact alias wins even when another key overlaps more
        tokens lexically (without the alias the lookup is ambiguous)."""
        digest = {
            "json_metrics": {
                "results.json": {"minimum_energy": -2.0},
                "report.json": {"min_energy_run_value": -99.0},
            },
            "log_metrics": [],
        }
        with_alias = _pm(
            "min_energy", "minimum energy", direction="lower",
            aliases=[{"file": "results.json", "key": "minimum_energy"}],
        )
        r = metrics_mod.resolve_primary_values(digest, [with_alias])
        assert r["min_energy"]["status"] == "resolved"
        assert r["min_energy"]["value"] == -2.0
        assert r["min_energy"]["source"] == "results.json:minimum_energy"
        # same digest, no alias: strict lexical finds TWO different values
        # → ambiguous (never guessed) — the alias is what disambiguates.
        r2 = metrics_mod.resolve_primary_values(
            digest, [_pm("min_energy", "minimum energy", direction="lower")]
        )
        assert r2["min_energy"]["status"] == "ambiguous"

    def test_freshness_preference(self):
        """Test 2: the same alias key in a fresh and a stale file — the
        fresh (produced_this_run) value wins."""
        digest = {
            "json_metrics": {
                "old_results.json": {"minimum_energy": -3.0},
                "results.json": {"minimum_energy": -2.0},
            },
            "log_metrics": [],
            "deliverables": [
                {"path": "old_results.json", "produced_this_run": False},
                {"path": "results.json", "produced_this_run": True},
            ],
        }
        m = _pm(
            "min_energy", "minimum energy", direction="lower",
            aliases=[
                {"file": "old_results.json", "key": "minimum_energy"},
                {"file": "results.json", "key": "minimum_energy"},
            ],
        )
        r = metrics_mod.resolve_primary_values(digest, [m])
        assert r["min_energy"]["status"] == "resolved"
        assert r["min_energy"]["value"] == -2.0
        assert r["min_energy"]["source"] == "results.json:minimum_energy"
        assert r["min_energy"]["fresh"] is True

    def test_verification_echoes_skipped(self):
        """Test 3: independent_verify_result.json echoes lose to the
        solver's own results.json; lexical never picks an echo at all."""
        digest = {
            "json_metrics": {
                "independent_verify_result.json": {"minimum_energy": -9.0},
                "results.json": {"minimum_energy": -2.0},
            },
            "log_metrics": [],
        }
        m = _pm(
            "min_energy", "minimum energy", direction="lower",
            aliases=[
                {"file": "independent_verify_result.json",
                 "key": "minimum_energy"},
                {"file": "results.json", "key": "minimum_energy"},
            ],
        )
        r = metrics_mod.resolve_primary_values(digest, [m])
        assert r["min_energy"]["value"] == -2.0  # non-verification file wins
        assert r["min_energy"]["source"] == "results.json:minimum_energy"
        # lexical-only: the echo file is excluded entirely → missing
        only_echo = {
            "json_metrics": {
                "independent_verify_result.json": {"minimum_energy": -9.0},
            },
            "log_metrics": [],
        }
        r2 = metrics_mod.resolve_primary_values(
            only_echo, [_pm("min_energy", "minimum energy", direction="lower")]
        )
        assert r2["min_energy"]["status"] == "missing"

    def test_ambiguity_reported_never_guessed(self):
        """Test 4: two candidates with different values → ambiguous with
        both listed; same value twice → resolved."""
        digest = {
            "json_metrics": {
                "results.json": {
                    "auc_validation": 0.71,
                    "auc_test": 0.93,
                    "auc_seed_a": 0.5,
                    "auc_seed_b": 0.5,
                },
            },
            "log_metrics": [],
        }
        diff = metrics_mod.resolve_primary_values(
            digest, [_pm("auc", "auc")]
        )
        assert diff["auc"]["status"] == "ambiguous"
        assert diff["auc"]["value"] is None
        sources = {c["source"] for c in diff["auc"]["candidates"]}
        assert sources == {
            "results.json:auc_validation", "results.json:auc_test",
            "results.json:auc_seed_a", "results.json:auc_seed_b",
        }
        same = metrics_mod.resolve_primary_values(
            digest, [_pm("auc_seed", "auc seed")]
        )
        assert same["auc_seed"]["status"] == "resolved"
        assert same["auc_seed"]["value"] == 0.5

    def test_real_drift_regression_10_08(self):
        """Test 5 (real-drift fixture): on the frozen 10-08 digests with
        the legacy (alias-less) recon, hh_contacts never resolves to 47,
        n_optimal never to 2, total_saws never to 376 — every wrong value
        the old fuzzy matcher produced is unreachable (missing/ambiguous
        instead)."""
        cases = {
            "cf812205": _DRIFT_CF812205,
            "396c629f": _DRIFT_396C629F,
            "4fc65d93": _DRIFT_4FC65D93,
        }
        for name, jm in cases.items():
            r = metrics_mod.resolve_primary_values(
                {"json_metrics": jm, "log_metrics": []}, _DRIFT_METRICS_V1
            )
            # positive control: the min-energy quantity resolves everywhere
            assert r["min_energy"]["status"] == "resolved", name
            assert r["min_energy"]["value"] == -2.0, name
            for mid, wrong in (
                ("hh_contacts", 47.0),
                ("n_optimal", 2.0),
                ("total_saws", 376.0),
            ):
                assert not (
                    r[mid]["status"] == "resolved"
                    and r[mid]["value"] == wrong
                ), f"{name}: {mid} resolved to the wrong quantity {wrong}"
                assert r[mid]["value"] is None, (name, mid)

    def test_recon_refresh_re_resolves(self):
        """Test 6: a metric missing under recon v1 (no alias) resolves
        after recon v2 adds the alias for the drifted key."""
        digest = {"json_metrics": _DRIFT_4FC65D93, "log_metrics": []}
        r1 = metrics_mod.resolve_primary_values(digest, _DRIFT_METRICS_V1)
        assert r1["total_saws"]["status"] == "missing"
        v2 = [
            _pm(
                "total_saws", "total SAWs", role="invariant",
                aliases=[{"file": "results.json",
                          "key": "raw_saw_count.count"}],
            ),
        ]
        r2 = metrics_mod.resolve_primary_values(digest, v2)
        assert r2["total_saws"]["status"] == "resolved"
        assert r2["total_saws"]["value"] == 16268.0

    def test_log_metrics_searched_in_lexical_fallback(self):
        """The lexical fallback also scrapes log_metrics (values that
        never landed in a JSON file)."""
        digest = {
            "json_metrics": {},
            "log_metrics": [
                {"metric": "auc", "value": 0.83, "context": "validation",
                 "raw_line": "AUC = 0.83", "source": "output.log"},
            ],
        }
        r = metrics_mod.resolve_primary_values(digest, [_pm("auc", "auc")])
        assert r["auc"]["status"] == "resolved"
        assert r["auc"]["value"] == 0.83
        assert r["auc"]["source"] == "output.log:auc"

    def test_recon_ensure_shape_normalizes_new_fields(self):
        """ensure_shape gives every primary_metrics entry the §4.1 fields
        with safe defaults (legacy cached configs keep working)."""
        cfg = recon_mod.ensure_shape({
            "primary_metrics": [
                {"key": "results.json.minimum_energy", "direction": "lower"},
                {"id": "auc", "key": "auc", "direction": "sideways",
                 "role": "weird", "aliases": [{"key": "metrics.json:auc"},
                                             {"file": "m.json"},
                                             "junk"],
                 "bounds": [0, 1], "integer": True},
            ],
            "tie_policy": "x",
        })
        [legacy, full] = cfg["primary_metrics"]
        assert legacy["id"] == "results_json_minimum_energy"
        assert legacy["direction"] == "lower"
        assert legacy["role"] == "objective"
        assert legacy["aliases"] == [] and legacy["bounds"] is None
        assert full["direction"] == "higher"  # invalid → neutral default
        assert full["role"] == "objective"
        assert full["aliases"] == [{"file": "", "key": "metrics.json:auc"}]
        assert full["bounds"] == [0.0, 1.0]
        assert full["integer"] is True

    def test_recon_prompt_strips_ledger_fields(self):
        """build_prompt never sends resolver output back to the recon LLM."""
        digest = {"uuid": "u", "primary_values": {"auc": {"status": "ok"}},
                  "primary_values_version": "v1", "json_metrics": {}}
        prompt = recon_mod.build_prompt("goal", [digest])
        assert "primary_values" not in prompt
        assert "slot_1" in prompt


# ======================================================================
# Fold from data (review §4.2) — METRICS from the ledger, no regex
# ======================================================================


class TestFoldMetrics:
    """fold_gradient(ledger=...) renders measured ties from the data."""

    LEDGER_METRICS = [
        _pm("min_energy", "minimum energy", direction="lower",
            aliases=[{"file": "results.json", "key": "minimum_energy"}]),
        _pm("hh_contacts", "hh contacts",
            aliases=[{"file": "results.json", "key": "hh_contact_count"}]),
        _pm("total_saws", "total SAWs", role="invariant"),
    ]

    @staticmethod
    def _pv(status, value=None, candidates=None):
        return {"status": status, "value": value, "source": None,
                "fresh": None, "candidates": candidates or []}

    def _ledger(self, current, preds):
        return {"current": current, "preds": preds,
                "metrics": self.LEDGER_METRICS}

    def test_real_tie_string_produces_metrics_section(self):
        """Test 7 (real tie regression): the HP 10-08 tie feedback plus a
        tied ledger → METRICS section names min_energy and says tied; the
        old regex missed exactly this single-digit -2 vs -2 case."""
        recs = [
            {"prev_uuid": "20261008_190432_32ed726e", "outcome": "tie",
             "feedback": "minimum_energy -2 vs -2; requirement.txt fresh "
                         "True vs True; run.sh fresh True vs True. Goal "
                         "requires: save a requirement.txt and a run.sh to "
                         "reproduce run."},
        ]
        current = {
            "min_energy": self._pv("resolved", -2.0),
            "hh_contacts": self._pv("resolved", 2.0),
            "total_saws": self._pv("resolved", 16268.0),
        }
        text, meta = fold_gradient(
            "gen", recs, "Assess minimum_energy (grading bars unknown)",
            ledger=self._ledger(current, {"p1": dict(current)}),
        )
        assert "METRICS (measured, vs 1 predecessor):" in text
        assert (
            "- min_energy (lower is better, objective): -2 here; "
            "predecessors -2 ×1 → tied with all" in text
        )
        assert "NOT ASSESSED" not in text  # everything resolved → no footer
        assert meta["tied"] == ["min_energy", "hh_contacts", "total_saws"]
        assert meta["footer_appended"] is False

    def test_non_metric_numbers_never_claim_ties(self):
        """Test 8: tie feedback full of numbers ('fresh deliverables 12
        vs 12') with NO ledger values makes no tied claim — numbers in
        prose are not measured metrics."""
        recs = [
            {"other": "g1", "rel": "tie",
             "feedback": "fresh deliverables 12 vs 12"},
        ]
        current = {m["id"]: self._pv("missing") for m in self.LEDGER_METRICS}
        text, meta = fold_gradient(
            "gen", recs, "Assess minimum_energy (grading bars unknown)",
            ledger=self._ledger(current, {"p1": dict(current)}),
        )
        assert "tied" not in text.replace("NOT ASSESSED", "")
        assert meta["tied"] == []
        # nothing resolved → the ledger cannot cover → footer stays
        assert "NOT ASSESSED (grading bars unknown)" in text
        assert meta["footer_appended"] is True
        # each missing metric is declared with its reason
        assert "NOT ASSESSED: min_energy — missing in this generation" in text

    def test_ambiguous_metrics_listed_with_reason(self):
        """Test 9: current-gen ambiguous metrics land under NOT ASSESSED
        with their candidates."""
        recs = [{"other": "g1", "rel": "tie", "feedback": "x"}]
        current = {
            "min_energy": self._pv("resolved", -2.0),
            "hh_contacts": self._pv("resolved", 2.0),
            "total_saws": self._pv(
                "ambiguous",
                candidates=[
                    {"source": "results.json:raw_optimal_walk_count.count",
                     "value": 376.0},
                    {"source": "results.json:raw_saw_count.count",
                     "value": 16268.0},
                ],
            ),
        }
        preds = {
            "p1": {
                "min_energy": self._pv("resolved", -2.0),
                "hh_contacts": self._pv("resolved", 2.0),
                "total_saws": self._pv("resolved", 16268.0),
            },
        }
        text, meta = fold_gradient(
            "gen", recs, "Assess minimum_energy (grading bars unknown)",
            ledger=self._ledger(current, preds),
        )
        line = next(
            ln for ln in text.splitlines()
            if ln.startswith("NOT ASSESSED: total_saws")
        )
        assert "ambiguous in this generation" in line
        assert "raw_optimal_walk_count.count=376" in line
        assert "raw_saw_count.count=16268" in line
        assert meta["not_assessed"] == ["total_saws"]

    def test_refined_text_keeps_metrics_block(self):
        """Test 10: the refiner cannot drop measured values — the METRICS
        block survives refinement like the header does."""
        recs = [{"other": "g1", "rel": "tie", "feedback": "f"}]
        current = {
            "min_energy": self._pv("resolved", -2.0),
            "hh_contacts": self._pv("resolved", 2.0),
            "total_saws": self._pv("resolved", 16268.0),
        }
        raw, _ = fold_gradient(
            "gen", recs, "",
            ledger=self._ledger(current, {"p1": dict(current)}),
        )
        final = gradient_mod.refined_text(raw, REFINER_JSON)
        assert final.startswith("GEN gen —")
        assert "METRICS (measured" in final
        assert "predecessors -2 ×1 → tied with all" in final
        assert REFINER_JSON["refined_gradient"] in final

    def test_footer_meta_matches_text(self):
        """Test 11 (S5): meta['footer_appended'] equals whether the
        footer line is in the text — for every combination."""
        recs_tie = [{"other": "g1", "rel": "tie", "feedback": "f"}]
        resolved = {
            "min_energy": self._pv("resolved", -2.0),
            "hh_contacts": self._pv("resolved", 2.0),
            "total_saws": self._pv("resolved", 16268.0),
        }
        for recs, footer in (
            (recs_tie, "footer text"),
            (recs_tie, ""),
            ([], "footer text"),
        ):
            text, meta = fold_gradient("gen", recs, footer)
            has_line = "NOT ASSESSED (grading bars unknown):" in text
            assert meta["footer_appended"] == has_line, (recs, footer)
        # with a covering ledger the footer is dropped on ties
        text, meta = fold_gradient(
            "gen", recs_tie, "footer text",
            ledger=self._ledger(resolved, {"p1": dict(resolved)}),
        )
        assert meta["footer_appended"] is False
        assert "NOT ASSESSED (grading bars unknown):" not in text
        # empty-pairs generation: the footer appears even with a covering
        # ledger (nothing was assessed at all)
        text, meta = fold_gradient(
            "gen", [], "footer text",
            ledger=self._ledger(resolved, {"p1": dict(resolved)}),
        )
        assert meta["footer_appended"] is True
        assert "NOT ASSESSED (grading bars unknown):" in text

    def test_pred_values_aggregated_with_missing_counts(self):
        """Predecessor values aggregate as counts with missing/ambiguous
        tallies; the tie verdict tolerates unresolved predecessors."""
        recs = [{"other": "g1", "rel": "tie", "feedback": "f"}]
        current = {
            "min_energy": self._pv("resolved", -2.0),
            "hh_contacts": self._pv("resolved", 2.0),
            "total_saws": self._pv("resolved", 16268.0),
        }
        preds = {
            "p1": dict(current),
            "p2": {
                "min_energy": self._pv("resolved", -2.0),
                "hh_contacts": self._pv("missing"),
                "total_saws": self._pv("ambiguous"),
            },
            "p3": {
                "min_energy": self._pv("resolved", -3.0),
                "hh_contacts": self._pv("resolved", 2.0),
                "total_saws": self._pv("resolved", 16268.0),
            },
        }
        text, meta = fold_gradient(
            "gen", recs, "", ledger=self._ledger(current, preds)
        )
        assert "METRICS (measured, vs 3 predecessors):" in text
        assert (
            "- min_energy (lower is better, objective): -2 here; "
            "predecessors -3 ×1; -2 ×2" in text
        )
        assert (
            "- hh_contacts (higher is better, objective): 2 here; "
            "predecessors 2 ×2; missing ×1 → tied" in text
        )
        assert "hh_contacts" in meta["tied"]
        assert (
            "- total_saws (higher is better, invariant): 16268 here; "
            "predecessors 16268 ×2; ambiguous ×1 → tied" in text
        )


# ======================================================================
# Policy contract (review §4.1) — the policy reads primary_values
# ======================================================================


class TestPolicyLedgerContract:
    """V3_PROMPT names the ledger; POLICY_SHA tracks the prompt."""

    def test_prompt_names_primary_values_and_sha_tracks(self):
        """Test 19: the prompt contract is the ledger, and POLICY_SHA is
        the prompt hash — any prompt change invalidates verdict caches."""
        import hashlib

        assert "primary_values" in policy_mod.V3_PROMPT
        expected = hashlib.sha256(
            (policy_mod.V3_PROMPT + recon_mod.RECON_PROMPT_VERSION).encode()
        ).hexdigest()[:12]
        assert expected == policy_mod.POLICY_SHA
        # the hash is sensitive to prompt content (bumps automatically)
        mutated = hashlib.sha256(
            (policy_mod.V3_PROMPT + "x" + recon_mod.RECON_PROMPT_VERSION)
            .encode()
        ).hexdigest()[:12]
        assert mutated != policy_mod.POLICY_SHA

    def test_policy_reads_ledger_with_missing_metrics(self):
        """Test 20: a ledger-reading policy smoke-passes on digests where
        some metrics are missing (the branch-only-error class of 10-07)."""
        src = (
            "import json\n"
            "def _val(d, mid):\n"
            "    return (d.get('primary_values') or {}).get(mid, {})"
            ".get('value')\n"
            "def compare(dA, dB):\n"
            "    a = _val(dA, 'min_energy')\n"
            "    b = _val(dB, 'min_energy')\n"
            "    if a is None or b is None:\n"
            "        return {'winner': 'tie', 'confidence': 0.5, 'rungs': [],\n"
            "                'deciding': [],\n"
            "                'feedback_winner': 'min_energy not comparable "
            "(missing or ambiguous)',\n"
            "                'feedback_looser': 'min_energy not comparable "
            "(missing or ambiguous)'}\n"
            "    if a == b:\n"
            "        return {'winner': 'tie', 'confidence': 0.5, 'rungs': [],\n"
            "                'deciding': ['primary_values[min_energy]'],\n"
            "                'feedback_winner': f'min_energy {a} vs {b} "
            "(tied)',\n"
            "                'feedback_looser': f'min_energy {a} vs {b} "
            "(tied)'}\n"
            "    w = 'A' if a < b else 'B'\n"
            "    return {'winner': w, 'confidence': 0.9, 'rungs': [],\n"
            "            'deciding': ['primary_values[min_energy]'],\n"
            "            'feedback_winner': f'min_energy {min(a, b)} vs "
            "{max(a, b)}',\n"
            "            'feedback_looser': f'min_energy {min(a, b)} vs "
            "{max(a, b)}'}\n"
        ) + MAIN_STANZA
        dA = {"primary_values": {"min_energy": {"status": "resolved",
                                                "value": -2.0}}}
        dB = {"primary_values": {"min_energy": {"status": "missing",
                                                "value": None}}}
        v_ab = policy_mod.run_policy(src, dA, dB, timeout=30)
        assert "error" not in v_ab, v_ab
        assert v_ab["winner"] == "tie"  # one side unresolved → not comparable
        assert v_ab["feedback_looser"].startswith("min_energy not comparable")
        v_ba = policy_mod.run_policy(src, dB, dA, timeout=30)
        assert "error" not in v_ba and v_ba["winner"] == "tie"
        # both resolved and equal → tie with the tied values stated
        v_eq = policy_mod.run_policy(src, dA, dict(dA), timeout=30)
        assert v_eq["winner"] == "tie"
        assert "-2.0 vs -2.0" in v_eq["feedback_looser"]
        # smoke through the real smoke path: distinct digests, both orders
        ok, err = policy_mod.smoke_test(
            src, [dA, dB], timeout=30, max_pairs=3
        )
        assert ok, err


def test_evaluate_wires_metric_ledger_into_scores(tmp_path, monkeypatch):
    """Evaluator wiring: the digest carries primary_values under the recon
    version; scores expose the compact metric_summary; gen-2 re-resolves
    the predecessor and its gradient carries the measured METRICS block."""
    monkeypatch.setattr(ev_mod, "ReexecRunner", _FakeReexecRunner)
    ev = _make_evaluator(tmp_path)
    recon = dict(
        RECON_JSON,
        primary_metrics=[
            {"id": "auc", "key": "AUC reported in metrics.json or logs",
             "direction": "higher", "role": "objective",
             "aliases": [{"file": "metrics.json", "key": "auc"}]},
        ],
    )
    llm = _ScriptedLLM(recon=recon, text_extract={"auc": 0.83})
    out = _run_generation(ev, llm, "cccccccc-1111")
    assert out["metric_summary"]["auc"]["status"] == "resolved"
    assert out["metric_summary"]["auc"]["value"] == 0.83
    reg = TaskRegistry.load(ev._runner_temp_root, GOAL)
    g1 = reg.generation("cccccccc-1111")
    assert g1["digest"]["primary_values"]["auc"]["value"] == 0.83
    assert g1["digest"]["primary_values_version"] == reg.get_recon_version()
    # the ledger resolves via LLM text extraction (the seeded workspace
    # prints "AUC = 0.83"; metrics.json does not exist) — regex-free path
    assert g1["digest"]["primary_values"]["auc"]["source"] == "llm:auc"
    out2 = _run_generation(ev, llm, "cccccccc-2222")
    assert out2["metric_summary"]["auc"]["value"] == 0.83
    grad = (
        ev.workflow_dir / "cccccccc-2222" / "textual_gradient.txt"
    ).read_text()
    assert "METRICS (measured" in grad
    assert "auc" in grad
