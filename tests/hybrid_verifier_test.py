"""Hybrid verifier (E19/E19b) — offline tests with mocked LLM/subprocess.

Covers the acceptance surface of the hybrid-verifier change-set:
- claim-extraction parsing/validation (generic-claim drop, count contract);
- scorer-script execution + one-JSON-line contract + bounded repair loop
  (mocked subprocess; one real WorkflowRunner execution for the
  pinned-subprocess contract);
- per-task registry persistence + zero-variance filter across
  "generations" + budgeted refinement;
- reward computation (mean-fallback first generation; mean_diff vs
  sign_sum vs escalation);
- gradient assembly + the ``abstracted_textual_gradient`` key fix;
- empty-run short-circuit parity (synthetic claim, reward 0, non-empty
  gradient);
- config plumbing (default hybrid, legacy escape) + facade wiring +
  deprecation banners.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from config import Config  # noqa: E402
from sources.core.failure_fingerprint import (  # noqa: E402,F401 — primes the
    compute_failure_fingerprint,  # core<->evaluators circular import chain,
)  # same as the other verifier tests in this suite.
from sources.evaluators import evaluator as evaluator_mod  # noqa: E402
from sources.evaluators.base import extract_json_payload  # noqa: E402
from sources.evaluators.hybrid_verifier import (  # noqa: E402
    ClaimsEvidenceLayer,
    HybridVerifierEvaluator,
    TaskRegistry,
    aggregation,
)
from sources.evaluators.hybrid_verifier import claims as claims_mod  # noqa: E402
from sources.evaluators.hybrid_verifier import inventory as inv_mod  # noqa: E402
from sources.evaluators.hybrid_verifier import scorers as scorers_mod  # noqa: E402
from sources.evaluators.hybrid_verifier.layers import LayerContext  # noqa: E402
from sources.evaluators.hybrid_verifier.registry import task_key  # noqa: E402

GOAL = (
    "Train a classifier on compounds.csv and write per-compound predicted "
    "probabilities to pred_results/predictions.csv with columns "
    "(smiles, probability); report the achieved AUC in metrics.json."
)


# ---------------------------------------------------------------- helpers ----


def _claim(
    cid: str,
    statement: str = None,
    rule: str = None,
    stage: str = "result",
    tidx: int = None,
) -> dict:
    return {
        "id": cid,
        "category": "completeness",
        "stage": stage,
        "temporal_index": (
            tidx if tidx is not None else int(cid[1:]) if cid[1:].isdigit() else 0
        ),
        "statement": statement or f"claim {cid} statement about row counts",
        "target": "pred_results/*.csv",
        "scoring_rule": rule
        or "fraction of input rows with a valid probability in [0,1] in the predictions csv (count rows; absent -> 0)",
    }


def _claims_json(n: int, lo: int = 8, hi: int = 12) -> str:
    claims = [_claim(f"C{i}") for i in range(1, n + 1)]
    return json.dumps({"claims": claims})


def _scorer_body(cid: str, score: float) -> str:
    return (
        "import json, os\n"
        f"print(json.dumps({{'claim_id': '{cid}', 'score': {score}, "
        "'evidence': 'measured 10 rows, 8 valid'}))\n"
    )

_CLEAN_EXACT_NAME_VERDICT = json.dumps(
    [{"claim_id": "C1", "lenient": False, "line_evidence": "", "reason": "exact"}]
)


class _FakeRunner:
    """Script executor mock: runs the scorer body with real Python."""

    def __init__(self, scores: dict[str, float] | None = None, fail_first: int = 0):
        self.scores = scores or {}
        self.fail_first = fail_first
        self.calls: list[str] = []

    def __call__(self, workspace: Path, code: str, execution_id: str):
        self.calls.append(execution_id)
        # Honor a body that prints JSON via the launcher constants.
        import subprocess

        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(workspace),
        )
        return scorers_mod.ExecOutcome(
            rc=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )


class _ScriptedRunner:
    """Executor mock that never spawns a process (pure contract tests)."""

    def __init__(self, outcomes: list[scorers_mod.ExecOutcome]):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, workspace: Path, code: str, execution_id: str):
        self.calls += 1
        return (
            self.outcomes.pop(0)
            if self.outcomes
            else scorers_mod.ExecOutcome(rc=0, stdout="{}")
        )


class _LLM:
    """Queue-based LLM stub: each call pops the next canned response."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[tuple[str, str]] = []

    def __call__(self, uuid: str, agent: str, prompt: str) -> str:
        self.prompts.append((agent, prompt))
        if not self.responses:
            raise AssertionError(f"unexpected LLM call: {agent}")
        return self.responses.pop(0)


class _StaticLLM:
    """Returns one canned response to every call; counts the calls."""

    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def __call__(self, uuid: str, agent: str, prompt: str) -> str:
        self.calls += 1
        return self.response


class _ScreenLLM:
    """Mock for the batched LLM content screens (generic / smuggled /
    exact-name).

    Parses the claims out of the screen prompt's CLAIMS (JSON) block (the
    exact-name screen's CLAIMS + SCRIPTS block carries claim_id/script
    pairs) and answers with a strict JSON array: "generic" per
    *generic_when* (predicate on the parsed claim), "smuggled" per
    *smuggled* (id -> (violating_field, violating_value)), "lenient" per
    *lenient* (id -> (line_evidence, reason)). Any other prompt fails
    the test loudly.
    """

    def __init__(self, generic_when=None, smuggled=None, lenient=None):
        self.generic_when = generic_when or (lambda c: False)
        self.smuggled = smuggled or {}
        self.lenient = lenient or {}
        self.calls: list[str] = []
        self.prompts: list[str] = []

    def __call__(self, uuid: str, agent: str, prompt: str) -> str:
        self.calls.append(agent)
        self.prompts.append(prompt)
        lines = prompt.splitlines()
        if "GENERIC-CLAIM SCREEN" in prompt:
            claims = json.loads(lines[lines.index("CLAIMS (JSON):") + 1])
            return json.dumps(
                [
                    {
                        "id": c["id"],
                        "generic": bool(self.generic_when(c)),
                        "reason": "stub verdict",
                    }
                    for c in claims
                ]
            )
        if "SMUGGLED-ANSWER SCREEN" in prompt:
            claims = json.loads(lines[lines.index("CLAIMS (JSON):") + 1])
            out = []
            for c in claims:
                hit = self.smuggled.get(c["id"])
                out.append(
                    {
                        "id": c["id"],
                        "smuggled": hit is not None,
                        "violating_field": hit[0] if hit else None,
                        "violating_value": hit[1] if hit else None,
                        "reason": "stub verdict",
                    }
                )
            return json.dumps(out)
        if "EXACT-NAME SCORER SCREEN" in prompt:
            pairs = json.loads(
                lines[lines.index("CLAIMS + SCRIPTS (JSON):") + 1]
            )
            return json.dumps(
                [
                    {
                        "claim_id": p["claim_id"],
                        "lenient": p["claim_id"] in self.lenient,
                        "line_evidence": self.lenient.get(
                            p["claim_id"], ("", "")
                        )[0],
                        "reason": self.lenient.get(
                            p["claim_id"], ("", "exact equality")
                        )[1],
                    }
                    for p in pairs
                ]
            )
        raise AssertionError(f"unexpected LLM call: {agent}")


def _make_evaluator(
    tmp_path: Path,
    llm,
    runner,
    pairwise_mode: str = "temporal",
    num_claims: int = 10,
    refinement_rounds: int = 2,
) -> HybridVerifierEvaluator:
    """Build a HybridVerifierEvaluator without BaseEvaluator.__init__."""
    v = HybridVerifierEvaluator.__new__(HybridVerifierEvaluator)
    v.memory_dir = tmp_path / "memory"
    v.workflow_dir = tmp_path / "workflows"
    v.workspace_dir = tmp_path / "workspace"
    for d in (v.memory_dir, v.workflow_dir, v.workspace_dir):
        d.mkdir(parents=True, exist_ok=True)
    v._runner_temp_root = tmp_path / "verifier_tmp"
    v._runner_temp_root.mkdir(parents=True, exist_ok=True)
    v.logger = logging.getLogger("test-hybrid-verifier")
    v.num_claims = num_claims
    v.refinement_rounds = refinement_rounds
    v.scorer_timeout_s = 60
    v.pairwise_mode = pairwise_mode
    v.gen_parallelism = 4
    v.llm_config = types.SimpleNamespace(temperature=0.2)
    v._llm_text = llm
    v._execute_scorer_script = runner
    v._claims_layer = ClaimsEvidenceLayer(
        llm_text=llm,
        run_script=runner,
        logger=v.logger,
        num_claims=num_claims,
        refinement_rounds=refinement_rounds,
        scorer_timeout_s=60,
        gen_parallelism=4,
        digest_max_files=8,
    )
    v.layers = [v._claims_layer]
    return v


def _seed_workspace(ws: Path) -> None:
    (ws / "pred_results").mkdir(parents=True, exist_ok=True)
    (ws / "pred_results" / "predictions.csv").write_text(
        "smiles,probability\nCCO,0.7\nNCC,0.4\n", encoding="utf-8"
    )
    (ws / "compounds.csv").write_text("smiles\nCCO\nNCC\n", encoding="utf-8")


def _wf_info(goal: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        goal=goal,
        state_result={"answers": ["ok"]},
        code="print('workflow')",
        uuid="u1",
    )


# ---------------------------------------------- (1) claim extraction parsing ----


def test_validate_claims_enforces_count_and_shape():
    obj = json.loads(_claims_json(5))
    _, err = claims_mod.validate_claims(obj, lo=8, hi=12)
    assert err is not None and "8-12" in err
    obj = json.loads(_claims_json(10))
    out, err = claims_mod.validate_claims(obj, lo=8, hi=12)
    assert err is None and len(out) == 10
    assert all(c["category"] in claims_mod.CLAIM_CATEGORIES for c in out)
    # duplicate ids are rejected
    obj["claims"][1]["id"] = "C1"
    _, err = claims_mod.validate_claims(obj, lo=8, hi=12)
    assert err is not None and "duplicate" in err


def test_next_claim_id_and_dedupe():
    claims = [_claim("C9"), _claim("C12")]
    assert claims_mod.next_claim_id(claims) == "C13"
    dupes = [_claim("C1"), _claim("C1")]
    fixed = claims_mod.dedupe_claim_ids(dupes, claims)
    ids = [c["id"] for c in fixed]
    assert ids == ["C1", "C13"]  # second collision re-ided past C12
    assert not ({*ids} & {"C9", "C12"})  # never reuses an existing id
    assert len(set(ids)) == len(ids)


def test_validate_claims_drops_generic_claims():
    screen = _ScreenLLM(
        generic_when=lambda c: "file exists" in c["statement"]
        or "runs without error" in c["statement"]
    )
    claims = [_claim(f"C{i}") for i in range(1, 10)]
    claims[0]["statement"] = "predictions.csv file exists in the workspace"
    claims[0]["scoring_rule"] = "score 1 if the file exists else 0"
    out, err = claims_mod.validate_claims(
        {"claims": claims}, lo=8, hi=12, llm_text=screen
    )
    # the generic claim was dropped; the remaining 8 still satisfy the floor
    assert err is None and len(out) == 8
    assert all("file exists" not in c["statement"] for c in out)
    # the screen ran as ONE batched call over all 9 claims, not per claim
    assert screen.calls == ["hybrid_screen_generic"]
    # a fully-generic batch falls under the floor -> repairable error
    for c in claims:
        c["statement"] = "predictions.csv file exists in the workspace"
        c["scoring_rule"] = "score 1 if the file exists else 0"
    _, err2 = claims_mod.validate_claims(
        {"claims": claims}, lo=8, hi=12, llm_text=screen
    )
    assert err2 is not None and "got 0" in err2
    # a script-runs claim is generic even with count vocabulary in the rule
    claims2 = [_claim(f"C{i}") for i in range(1, 11)]
    claims2[3]["statement"] = "the script runs without error"
    claims2[3]["scoring_rule"] = "count of successful runs, fraction -> 0..1"
    out2, err2 = claims_mod.validate_claims(
        {"claims": claims2}, lo=8, hi=12, llm_text=screen
    )
    assert err2 is None and len(out2) == 9
    assert all("runs without error" not in c["statement"] for c in out2)


def test_is_generic_claim_batch_llm_screen():
    """The generic screen is ONE batched LLM call: verdicts on unknown ids
    are ignored and a failed judge call degrades open (nothing flagged)."""
    generic = dict(
        _claim("C1"),
        statement="output is valid csv present in the workspace",
        scoring_rule="file exists check",
    )
    solid = _claim("C2")
    screen = _ScreenLLM(generic_when=lambda c: "file exists" in c["scoring_rule"])
    flagged = claims_mod.is_generic_claim([generic, solid], screen, uuid="u1")
    assert flagged == {"C1": "stub verdict"}
    assert screen.calls == ["hybrid_screen_generic"]  # one batched call
    # the prompt carried every claim's fields to the judge
    assert "claim C2 statement about row counts" in screen.prompts[0]
    assert "file exists check" in screen.prompts[0]
    # hallucinated ids never leak into the verdict
    bogus = _StaticLLM('[{"id": "C99", "generic": true, "reason": "rogue"}]')
    assert claims_mod.is_generic_claim([solid], bogus) == {}
    # a judge that never produces JSON degrades open after one repair round
    garbage = _StaticLLM("not json at all")
    assert claims_mod.is_generic_claim([solid], garbage) == {}
    assert garbage.calls == 2


# -------------------------------------- (2) scorer contract + repair loop ----


def test_parse_scorer_output_contract():
    ok = scorers_mod.parse_scorer_output(
        '{"claim_id": "C1", "score": 0.75, "evidence": "8/10 rows valid"}\n'
    )
    assert ok == {"claim_id": "C1", "score": 0.75, "evidence": "8/10 rows valid"}
    # clamped and non-finite handling
    assert (
        scorers_mod.parse_scorer_output(
            '{"claim_id": "C1", "score": 5, "evidence": "x"}'
        )["score"]
        == 1.0
    )
    assert (
        scorers_mod.parse_scorer_output(
            '{"claim_id": "C1", "score": NaN, "evidence": "x"}'
        )
        is None
    )
    assert scorers_mod.parse_scorer_output("not json at all") is None
    assert scorers_mod.parse_scorer_output("") is None
    assert (
        scorers_mod.parse_scorer_output(
            '{"claim_id": "C1", "score": true, "evidence": "x"}'
        )
        is None
    )


def test_static_violations_catches_write_and_network_apis():
    bad_net = "import requests\nprint(1)"
    assert any("requests" in v for v in scorers_mod.static_violations(bad_net))
    bad_write = "import pandas as pd\ndf.to_csv('out.csv')"
    assert any("to_csv" in v for v in scorers_mod.static_violations(bad_write))
    bad_open = "f = open('x.csv', 'w')"
    assert any("write-mode" in v for v in scorers_mod.static_violations(bad_open))
    banned_json = scorers_mod.static_violations(
        "import json\njson.dump({}, open('o','w'))"
    )
    assert banned_json and "dump" in banned_json[0]


def test_launcher_injects_constants(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    code = scorers_mod.build_launcher(
        ws,
        _claim("C1"),
        "import json\nprint(json.dumps({'id': CLAIM['id'], 'root': WORKSPACE_ROOT[:3]}))",
    )
    assert "WORKSPACE_ROOT" in code and "CLAIM" in code
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["id"] == "C1"


def test_scorer_build_repairs_on_bad_output(tmp_path: Path):
    """Non-JSON stdout triggers exactly one repair with feedback."""
    llm = _LLM(
        [
            "```python\nimport json\nprint('debug to stdout')\n```",  # attempt 1 body
            _CLEAN_EXACT_NAME_VERDICT,  # screen: clean
            "```python\nimport json\nprint(json.dumps({'claim_id': 'C1', 'score': 0.5, 'evidence': 'half ok'}))\n```",
            _CLEAN_EXACT_NAME_VERDICT,  # screen: clean
        ]
    )
    runner = _ScriptedRunner(
        [
            scorers_mod.ExecOutcome(rc=0, stdout="debug to stdout", stderr=""),
            scorers_mod.ExecOutcome(
                rc=0,
                stdout='{"claim_id": "C1", "score": 0.5, "evidence": "half ok"}',
            ),
        ]
    )
    v = _make_evaluator(tmp_path, llm, runner)
    reg = TaskRegistry.load(v._runner_temp_root, GOAL, logger=v.logger)
    reg.seed_claims([_claim("C1")])
    ctx = LayerContext(
        uuid="u1",
        goal=GOAL,
        workspace=v.workspace_dir,
        registry=reg,
        generation_index=0,
    )
    scores = v._claims_layer.collect(ctx)
    assert scores[0].score == 0.5
    assert runner.calls == 2
    # repair prompt carried the failure feedback
    repair_prompt = next(p for a, p in llm.prompts if "_repair" in a)
    assert "previous scoring script failed" in repair_prompt.lower()
    assert "debug to stdout" in repair_prompt


def test_scorer_build_exhausts_attempts_and_soft_fails(tmp_path: Path):
    llm = _LLM(["```python\nimport json\nprint('nope')\n```"] * 3)
    runner = _ScriptedRunner(
        [scorers_mod.ExecOutcome(rc=0, stdout="nope")] * 3,
    )
    v = _make_evaluator(tmp_path, llm, runner)
    reg = TaskRegistry.load(v._runner_temp_root, GOAL, logger=v.logger)
    reg.seed_claims([_claim("C1")])


def test_full_pipeline_real_subprocess(tmp_path: Path):
    """End-to-end: extraction -> scorer build -> real WorkflowRunner exec.

    Uses the evaluator's real ``_execute_scorer_script`` (pinned
    ``sys.executable`` subprocess, cwd = workspace) so the
    pinned-subprocess contract itself is exercised.
    """
    _seed_workspace(tmp_path / "workspace")

    class _PromptKeyedLLM:
        """Extraction returns 8 claims; scorers key off the claim in the prompt."""

        def __call__(self, uuid: str, agent: str, prompt: str) -> str:
            if (
                "GENERIC-CLAIM SCREEN" in prompt
                or "SMUGGLED-ANSWER SCREEN" in prompt
                or "EXACT-NAME SCORER SCREEN" in prompt
            ):
                return "[]"  # batched screens: every claim clean
            if "verification rubric" in prompt:
                return _claims_json(8)
            if "FORMAT DIGESTS" in prompt:
                return json.dumps(
                    {
                        "digests": [
                            {
                                "path": "pred_results/predictions.csv",
                                "digest": "columns smiles,probability; comma-delimited; floats 2dp",
                            }
                        ]
                    }
                )
            import re

            m = re.search(r'"id":\s*"(C\d+)"', prompt)
            cid = m.group(1) if m else "C1"
            n = int(cid[1:])
            return f"```python\n{_scorer_body(cid, 0.5 + 0.05 * n)}\n```"

    llm = _PromptKeyedLLM()
    v = HybridVerifierEvaluator.__new__(HybridVerifierEvaluator)
    v.memory_dir = tmp_path / "memory"
    v.workflow_dir = tmp_path / "workflows"
    v.workspace_dir = tmp_path / "workspace"
    for d in (v.memory_dir, v.workflow_dir):
        d.mkdir(parents=True, exist_ok=True)
    v._runner_temp_root = tmp_path / "verifier_tmp"
    v._runner_temp_root.mkdir(parents=True, exist_ok=True)
    v.logger = logging.getLogger("test-hybrid-real")
    v.num_claims = 10
    v.refinement_rounds = 2
    v.scorer_timeout_s = 60
    v.pairwise_mode = "temporal"
    v.gen_parallelism = 4
    v.llm_config = types.SimpleNamespace(temperature=0.2)
    v._llm_text = llm
    v._execute_scorer_script = lambda ws, code, eid: (
        HybridVerifierEvaluator._execute_scorer_script(v, ws, code, eid)
    )
    v._claims_layer = ClaimsEvidenceLayer(
        llm_text=v._llm_text,
        run_script=v._execute_scorer_script,
        logger=v.logger,
    )
    v.layers = [v._claims_layer]
    v._load_workflow_data = lambda uuid: _wf_info(GOAL)  # type: ignore[method-assign]

    result = v.evaluate("u1")
    assert result["n_claims"] == 8
    assert result["n_surviving"] == 8  # single observation: all stay alive
    assert result["n_pairs"] == 0
    assert result["reward_fallback"] == "mean_claim"
    expected_mean = sum(0.5 + 0.05 * i for i in range(1, 9)) / 8
    assert result["overall_score"] == pytest.approx(expected_mean, abs=1e-3)
    # artifacts
    state = json.loads((v.workflow_dir / "u1" / "state_result.json").read_text())
    verifier = state["evaluation"]["verifier"]
    assert verifier["overall_score"] == pytest.approx(expected_mean, abs=1e-3)
    # BOTH gradient keys, correct one primary
    assert verifier["abstracted_textual_gradient"].strip()
    assert (
        verifier["abstracted_textual_gradient"]
        == verifier["abstractec_textual_gradient"]
    )
    grad = (v.workflow_dir / "u1" / "textual_gradient.txt").read_text()
    assert "WHAT TO FIX FIRST" in grad and "Dead claims report" in grad
    report = (v.workflow_dir / "u1" / "evaluation.txt").read_text()
    assert "Hybrid Verifier Evaluation" in report
    assert "per-workspace" not in report  # not the refinement prompt
    # registry persisted with scorer scripts cached
    reg = TaskRegistry.load(v._runner_temp_root, GOAL)
    assert len(reg.claims) == 8
    assert all(c["scorer"] for c in reg.claims)
    assert reg.generations[0]["scores"]["C1"] == pytest.approx(0.55)
    # format digests computed once and cached in the registry
    assert "pred_results/predictions.csv" in reg.digests


# ------------------------- (3) registry + variance filter across generations ----


def test_claim_stats_zero_variance_semantics():
    assert aggregation.claim_stats([])["drop_reason"] == "all_scorer_fail"
    single = aggregation.claim_stats([0.7])
    assert not single["dropped"]  # live guard: 1 obs cannot discriminate yet
    flat = aggregation.claim_stats([0.5, 0.5, 0.5])
    assert flat["dropped"] and flat["drop_reason"] == "zero_variance"
    var = aggregation.claim_stats([0.2, 0.8])
    assert not var["dropped"] and var["variance"] > 0


def test_claim_stats_crash_is_not_measured_constant():
    """E37 R3: a crashed scorer (None) must not read as 'all scores equal 0'."""
    # every observation crashed -> could-not-verify, not zero-variance
    crashed = aggregation.claim_stats([None, None, None])
    assert crashed["dropped"] and crashed["drop_reason"] == "all_scorer_fail"
    # >= 2 MEASURED equal observations (crashes excluded) are still flat
    flat = aggregation.claim_stats([None, 0.0, 0.0])
    assert flat["dropped"] and flat["drop_reason"] == "zero_variance"
    # one crash + one real score: not enough measured evidence to prune
    guard = aggregation.claim_stats([None, 0.5])
    assert not guard["dropped"]


def test_registry_excludes_legacy_crash_zero_from_variance_filter(tmp_path: Path):
    """E37 R3 (pains_brenk): legacy registries recorded 0.0 with scorer-crash
    evidence; treating those as measurements pruned the correctness signal
    as zero-variance. The crash evidence must disqualify the observation."""
    reg = TaskRegistry.load(tmp_path, GOAL)
    reg.seed_claims([_claim("C1")])
    reg.record_generation("u0", {"C1": 0.0}, {"C1": "scorer failed: exit 1"}, reward=0.0)
    reg.record_generation("u1", {"C1": 0.0}, {"C1": "scorer raised: ValueError"}, reward=0.0)
    reg.record_generation("u2", {"C1": 0.6}, {"C1": "6 of 10 rows valid"}, reward=0.6)
    # only the MEASURED observations feed the variance filter
    assert reg.observed_scores("C1") == [0.6]
    assert not aggregation.claim_stats(reg.observed_scores("C1"))["dropped"]
    # a genuine measured 0 is kept
    reg.record_generation("u3", {"C1": 0.0}, {"C1": "0 of 10 rows valid"}, reward=0.0)
    assert reg.observed_scores("C1") == [0.6, 0.0]


def test_registry_all_crashes_drop_as_all_scorer_fail(tmp_path: Path):
    """Every observation a crash -> the claim is reported unverifiable, not flat."""
    reg = TaskRegistry.load(tmp_path, GOAL)
    reg.seed_claims([_claim("C1")])
    reg.record_generation("u0", {"C1": 0.0}, {"C1": "scorer failed: exit 1"}, reward=0.0)
    reg.record_generation("u1", {"C1": None}, {"C1": "scorer failed: exit 1"}, reward=0.0)
    stats = aggregation.claim_stats(reg.observed_scores("C1"))
    assert stats["dropped"] and stats["drop_reason"] == "all_scorer_fail"


def test_registry_persistence_and_upsert(tmp_path: Path):
    reg = TaskRegistry.load(tmp_path, GOAL)
    reg.seed_claims([_claim("C1"), _claim("C2")])
    reg.record_generation("u1", {"C1": 0.5, "C2": 0.2}, {"C1": "ev1"}, reward=0.5)
    reg.record_generation("u1", {"C1": 0.9}, {"C1": "ev1b"}, reward=0.9)  # upsert
    reg.merge_inventory(
        {"a.csv": {"bytes": 1, "kind": "csv", "signal": "1 rows x 2 cols"}}
    )
    reg.save()
    loaded = TaskRegistry.load(tmp_path, GOAL)
    assert len(loaded.generations) == 1
    assert loaded.generations[0]["scores"]["C1"] == 0.9
    assert loaded.inventory["a.csv"]["count"] == 1
    assert loaded.n_workspaces == 1
    assert loaded.observed_scores("C1") == [0.9]
    assert loaded.previous_generations("u1") == []


def test_zero_variance_drops_claim_and_triggers_replacement(tmp_path: Path):
    """Gen1 scores 0.5 on C1..C8; gen2 repeats 0.5 on C1 -> C1 dies + replaced."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    claims = [_claim(f"C{i}") for i in range(1, 5)]
    llm = _LLM(
        [
            # gen2 scorer build is skipped: scripts come from the registry below.
        ]
    )
    runner = _ScriptedRunner([])
    v = _make_evaluator(tmp_path, llm, runner, num_claims=4)
    reg = TaskRegistry.load(v._runner_temp_root, GOAL, logger=v.logger)
    reg.seed_claims(claims)
    # gen1: every claim scored 0.5 with a cached scorer script
    for c in reg.claims:
        c["scorer"] = _scorer_body(c["id"], 0.5)
    reg.record_generation(
        "u0",
        {c["id"]: 0.5 for c in reg.claims},
        {c["id"]: "flat" for c in reg.claims},
        reward=0.5,
    )
    reg.save()

    # gen2 execution: C1 stays 0.5 (dead), C2..C4 vary (alive)
    def run(ws_, code, eid):
        for c in ["C1", "C2", "C3", "C4", "C5"]:
            if c in eid:
                score = 0.8 if c == "C5" else (0.5 if c == "C1" else 0.9)
                return scorers_mod.ExecOutcome(
                    rc=0,
                    stdout=json.dumps(
                        {"claim_id": c, "score": score, "evidence": f"{c} at {score}"}
                    ),
                )
        raise AssertionError(eid)

    class _RefineLLM:
        """Refine prompt -> one replacement claim; scorer prompt -> body."""

        def __init__(self):
            self.prompts: list[tuple[str, str]] = []

        def __call__(self, uuid: str, agent: str, prompt: str) -> str:
            if "GENERIC-CLAIM SCREEN" in prompt or "SMUGGLED-ANSWER SCREEN" in prompt:
                return "[]"  # batched screens: every claim clean
            self.prompts.append((agent, prompt))
            if "FORMAT DIGESTS" in prompt:
                return json.dumps({"digests": []})
            if "refining" in prompt:
                return json.dumps(
                    {
                        "claims": [
                            _claim(
                                "C5",
                                rule="count of AUC metric lines in metrics.json, "
                                "fraction of expected 1 -> 0..1",
                            )
                        ]
                    }
                )
            return f"```python\n{_scorer_body('C5', 0.8)}\n```"

    llm2 = _RefineLLM()
    v2 = _make_evaluator(tmp_path, llm2, run, num_claims=4)
    v2._load_workflow_data = lambda uuid: _wf_info(GOAL)  # type: ignore[method-assign]
    result = v2.evaluate("u1")
    # C1 dead (zero variance) -> replaced by C5 within the same evaluation
    assert result["n_replacements"] == 1
    assert any(s["id"] == "C5" for s in result["claim_summary"])
    reg2 = TaskRegistry.load(v2._runner_temp_root, GOAL)
    assert reg2.claim("C1") is None or reg2.claim("C1")["state"] == "dead"
    assert reg2.claim("C5") is not None
    # refinement prompt mentioned the dead claim's outcome
    refine_prompt = next(p for a, p in llm2.prompts if "refining" in p)
    assert "NON-DISCRIMINATIVE" in refine_prompt or "scored equally" in refine_prompt


# ---------------------------------------------- (4) reward modes + fallback ----


def _ladder(*specs) -> list[dict]:
    """Build a temporal ladder from (id, stage) specs."""
    out = []
    for i, (cid, stage) in enumerate(specs):
        out.append(_claim(cid, stage=stage, tidx=i))
    return out


def test_temporal_operator_exact_case():
    """A fails a script-stage claim B passes, A dominates EVERY result
    claim -> B WINS (script-stage failure dominates result advantage)."""
    ladder = _ladder(
        ("S1", "script"), ("L1", "log"), ("R1", "result"), ("R2", "result")
    )
    a = {"S1": 0.1, "L1": 0.5, "R1": 1.0, "R2": 1.0}  # A fails script, aces results
    b = {"S1": 0.9, "L1": 0.5, "R1": 0.2, "R2": 0.1}  # B passes script, poor results
    r = aggregation.win_rate_reward(a, [b], ladder, mode="temporal")
    assert r["wins"] == 0 and r["losses"] == 1
    detail = r["pairs"][0]["detail"]
    assert detail["stage"] == "script" and detail["claim_id"] == "S1"
    # mirrored: A passes script, B fails -> A wins despite worse results
    r2 = aggregation.win_rate_reward(b, [a], ladder, mode="temporal")
    assert r2["wins"] == 1


def test_temporal_stage_tie_falls_through():
    """Equal script pass-counts -> decided at the log stage."""
    ladder = _ladder(
        ("S1", "script"), ("S2", "script"), ("L1", "log"), ("R1", "result")
    )
    # script stage: both pass 2/2 -> tie; log: A passes, B fails
    a = {"S1": 0.9, "S2": 0.8, "L1": 0.7, "R1": 0.5}
    b = {"S1": 0.9, "S2": 0.8, "L1": 0.2, "R1": 0.5}
    r = aggregation.win_rate_reward(a, [b], ladder, mode="temporal")
    assert r["wins"] == 1
    assert r["pairs"][0]["detail"]["stage"] == "log"
    # stage pass-counts equal but scores differ -> stage mean-diff tiebreak
    a2 = {"S1": 0.9, "S2": 0.6, "L1": 0.5, "R1": 0.5}  # 2 passes, weaker margins
    b2 = {"S1": 0.8, "S2": 0.7, "L1": 0.5, "R1": 0.5}  # 2 passes
    r2 = aggregation.win_rate_reward(a2, [b2], ladder, mode="temporal")
    # script dm = 0.1-0.1 = 0.0? 0.9+0.6=1.5 vs 0.8+0.7=1.5 -> log 0.5=0.5, result equal
    assert r2["ties"] == 1 and r2["pairs"][0]["detail"]["kind"] == "all_equal"


def test_temporal_strict_vs_temporal_difference():
    """Strict decides at the FIRST differing claim; temporal decides on
    stage pass-counts — the two modes can disagree on the same pair."""
    ladder = _ladder(("S1", "script"), ("S2", "script"), ("R1", "result"))
    a = {"S1": 0.9, "S2": 0.3, "R1": 0.5}  # S2 below pass line -> 1 script pass
    b = {"S1": 0.6, "S2": 0.6, "R1": 0.5}  # both pass            -> 2 script passes
    strict = aggregation.win_rate_reward(a, [b], ladder, mode="temporal_strict")
    assert strict["wins"] == 1  # first differing claim S1: 0.9 > 0.6 wins outright
    assert strict["pairs"][0]["detail"]["claim_id"] == "S1"
    # temporal: script pass-counts 1 vs 2 -> B WINS at the script stage even
    # though strict handed the pair to A on the first claim's margin
    temporal = aggregation.win_rate_reward(a, [b], ladder, mode="temporal")
    assert temporal["losses"] == 1
    detail = temporal["pairs"][0]["detail"]
    assert detail["stage"] == "script" and detail["claim_id"] == "S2"
    assert temporal["pairs"][0]["outcome"] != strict["pairs"][0]["outcome"]


def test_temporal_excludes_dead_claims():
    """Claims not in the comparison ladder (dead/dropped) never decide."""
    ladder = _ladder(("S1", "script"), ("R1", "result"))
    a = {"S1": 0.9, "R1": 0.2, "DEAD1": 0.0}
    b = {"S1": 0.9, "R1": 0.9, "DEAD1": 1.0}
    r = aggregation.win_rate_reward(a, [b], ladder, mode="temporal")
    # DEAD1 is not in the ladder: script ties, R1 decides for B
    assert r["losses"] == 1
    assert r["pairs"][0]["detail"]["claim_id"] == "R1"


def test_win_rate_flat_modes_still_available():
    now = {"A": 0.6, "B": 0.4, "C": 0.9}
    prev = {"A": 0.5, "B": 0.6, "C": 0.9}
    ladder = _ladder(("A", "script"), ("B", "log"), ("C", "result"))
    r_sign = aggregation.win_rate_reward(now, [prev], ladder, mode="sign_sum")
    assert r_sign["ties"] == 1 and r_sign["reward"] == 0.5
    r_md = aggregation.win_rate_reward(now, [prev], ladder, mode="mean_diff")
    assert r_md["losses"] == 1 and r_md["reward"] == 0.0
    r_esc = aggregation.win_rate_reward(now, [prev], ladder, mode="escalation")
    assert r_esc["losses"] == 1
    two_prevs = [prev, dict.fromkeys(["A", "B", "C"], 0.0)]
    r = aggregation.win_rate_reward(now, two_prevs, ladder, mode="mean_diff")
    assert r["reward"] == 0.5 and r["n_pairs"] == 2
    with pytest.raises(ValueError):
        aggregation.win_rate_reward(now, [prev], ladder, mode="bogus")
    assert "temporal" in aggregation.PAIRWISE_MODES
    assert aggregation.DEFAULT_PAIRWISE_MODE == "temporal"


def test_first_generation_mean_fallback():
    ladder = _ladder(("A", "script"), ("B", "result"))
    r = aggregation.win_rate_reward({"A": 0.4, "B": 0.8}, [], ladder)
    assert r["fallback"] == "mean_claim" and r["reward"] == pytest.approx(0.6)
    # claims scored on neither side are excluded (k_eff)
    d, dm, k = aggregation.decisive(
        {"A": 0.5, "B": None}, {"A": None, "B": 0.5}, ["A", "B"]
    )
    assert k == 0 and d == 0 and dm == 0.0


# --------------------------------------------- (5) gradient + key contract ----


def test_gradient_v5_decisive_first_no_elimination_point():
    """V5 (E35): decisive losses lead; NO elimination-point framing;
    execution facts + visual evidence sections present; every comparative
    row goal-anchored."""
    from sources.evaluators.hybrid_verifier import gradient as g

    claims = [
        _claim("V1", stage="visual", tidx=1),
        _claim("S1", stage="script", tidx=0),
        _claim("L1", stage="log", tidx=1),
        _claim("R1", stage="result", tidx=2),
    ]
    text = g.build_gradient(
        uuid="u1",
        goal=GOAL,
        now_scores={"V1": 0.2, "S1": 0.1, "L1": 0.9, "R1": 0.9},
        evidence={
            "V1": "figure shows the wrong plot type",
            "S1": "no read of compounds.csv found in code",
            "L1": "loss 0.42 final",
            "R1": "10/10 rows",
        },
        claims=claims,
        surviving=["V1", "S1", "L1", "R1"],
        pair_records=[
            {
                "prev_uuid": "u0",
                "outcome": "loss",
                "detail": {
                    "kind": "stage",
                    "stage": "visual",
                    "claim_id": "V1",
                    "now_passes": 0,
                    "prev_passes": 1,
                    "now": 0.2,
                    "prev": 0.9,
                },
                "prev_scores": {"V1": 0.9, "S1": 0.9, "L1": 0.5, "R1": 0.2},
            }
        ],
        reward=0.0,
        win_rate=0.0,
        mean_score=0.525,
        dead_claims=[
            {
                "id": "C3",
                "stage": "log",
                "statement": "flat",
                "drop_reason": "zero_variance",
            }
        ],
        exec_facts={"status": "divergent", "runtime_s": 48.3, "cap": 0.5},
        stages=claims_mod.FIGURE_STAGES,
    )
    # Section order: losses -> wins -> execution facts -> visual -> dead
    assert (
        text.index("## 1. WHAT TO FIX FIRST")
        < text.index("## 2. WHAT ALREADY WORKS")
        < text.index("## 3. Execution facts")
        < text.index("## 4. Visual evidence")
        < text.index("## 5. Dead claims report")
    )
    # NO elimination-point framing (E19c measured it worse: 0.553 vs 0.672)
    assert "ELIMINATION POINT" not in text
    assert "ELIMINATED at stage" not in text
    # decisive lost claims lead; visual loss ranks before the script loss
    fix_block = text.split("## 2.")[0]
    assert fix_block.index("[V1]") < fix_block.index("[S1]")
    # goal-anchored comparative rows with measured evidence (E32)
    assert "REQUIREMENT (from the task goal)" in fix_block
    assert "measured evidence: no read of compounds.csv found" in fix_block
    # execution facts from the gate
    assert "status: divergent" in text and "capped at 0.5" in text
    # visual evidence section renders the figure judgement
    assert "visual score: 0.200" in text and "wrong plot type" in text
    assert "zero_variance" in text
    # gate-disabled rendering keeps the section informative
    bare = g.build_gradient(
        uuid="u2",
        goal=GOAL,
        now_scores={"S1": 1.0},
        evidence={"S1": "ok"},
        claims=[_claim("S1", stage="script", tidx=0)],
        surviving=["S1"],
        pair_records=[],
        reward=1.0,
        win_rate=1.0,
        mean_score=1.0,
        dead_claims=[],
        exec_facts=None,
    )
    assert "gate disabled" in bare
    assert "non-figure deliverable" in bare
    report = g.build_evaluation_report(
        "u1",
        {
            "n_claims": 4,
            "n_surviving": 4,
            "n_dropped": 1,
            "n_scored": 4,
            "n_scorer_failures": 0,
            "overall_score": 0.0,
            "pairwise_mode": "temporal",
            "mean_claim_score": 0.525,
            "win_rate": 0.0,
            "n_pairs": 1,
            "n_wins": 0,
            "n_losses": 1,
            "n_ties": 0,
            "n_replacements": 0,
        },
        claims,
        {"V1": 0.2, "S1": 0.1, "L1": 0.9, "R1": 0.9},
        {"S1": "no read of input"},
        ["V1", "S1", "L1", "R1"],
        [
            {
                "prev_uuid": "u0",
                "outcome": "loss",
                "detail": {"kind": "stage", "stage": "script", "claim_id": "S1"},
            }
        ],
    )
    # evaluation.txt renders stage headers in the given (visual-first) order
    assert report.index("--- stage: VISUAL ---") < report.index("--- stage: SCRIPT ---")
    assert report.index("--- stage: SCRIPT ---") < report.index("--- stage: LOG ---")
    assert report.index("--- stage: LOG ---") < report.index("--- stage: RESULT ---")

# ------------------------------------------------- (6) short-circuit parity ----


def test_gradient_surfaces_persistently_failing_claims():
    """E37 R4 (clintox C16): a claim failing for EVERY generation has no
    decisive loss (both sides of every comparison fail), so it never
    entered the gradient. It must surface in its own section between the
    decisive losses and the decisive wins."""
    from sources.evaluators.hybrid_verifier import gradient as g

    claims = [
        _claim("C16", stage="result", tidx=0),
        _claim("R1", stage="result", tidx=1),
    ]
    pair_records = [
        {
            "prev_uuid": "u0",
            "outcome": "tie",
            "prev_scores": {"C16": 0.333, "R1": 0.9},
            "d": 0,
            "dm": 0.0,
        },
        {
            "prev_uuid": "u1",
            "outcome": "tie",
            "prev_scores": {"C16": 0.2, "R1": 0.1},
            "d": 0,
            "dm": 0.0,
        },
    ]
    text = g.build_gradient(
        uuid="u2",
        goal=GOAL,
        now_scores={"C16": 0.333, "R1": 0.9},
        evidence={
            "C16": "Expected columns [FDA_APPROVED, CT_TOX], "
            "found [FDA_APPROVED_prob, CT_TOX_prob]",
            "R1": "10/10 rows",
        },
        claims=claims,
        surviving=["C16", "R1"],
        pair_records=pair_records,
        reward=0.5,
        win_rate=0.5,
        mean_score=0.6,
        dead_claims=[],
    )
    # the section sits between the decisive losses and the decisive wins
    assert (
        text.index("## 1. WHAT TO FIX FIRST")
        < text.index("## 1b. PERSISTENTLY FAILING CLAIMS")
        < text.index("## 2. WHAT ALREADY WORKS")
    )
    block = text.split("## 1b.")[1].split("\n", 1)[1].split("## 2.")[0]
    assert "[C16]" in block
    assert "failing for 3 consecutive generations" in block
    assert "NEVER passed" in block
    assert "Expected columns [FDA_APPROVED, CT_TOX]" in block
    # R1 passed in gen u0 -> not persistently failing
    assert "[R1]" not in block

    # a claim with no predecessor yet (first generation) never qualifies
    first = g.build_gradient(
        uuid="u0",
        goal=GOAL,
        now_scores={"C16": 0.1},
        evidence={"C16": "wrong columns"},
        claims=claims[:1],
        surviving=["C16"],
        pair_records=[],
        reward=0.1,
        win_rate=0.1,
        mean_score=0.1,
        dead_claims=[],
    )
    assert "## 1b." in first
    assert first.split("## 1b.")[1].split("\n", 1)[1].split("## 2.")[0].strip() == "(none)"

    # crashed observations (None) cannot prove "never passed"
    crashed = g.build_gradient(
        uuid="u2",
        goal=GOAL,
        now_scores={"C16": 0.1},
        evidence={"C16": "wrong columns"},
        claims=claims[:1],
        surviving=["C16"],
        pair_records=[
            {"prev_uuid": "u0", "outcome": "tie", "prev_scores": {"C16": None}}
        ],
        reward=0.1,
        win_rate=0.1,
        mean_score=0.1,
        dead_claims=[],
    )
    assert crashed.split("## 1b.")[1].split("\n", 1)[1].split("## 2.")[0].strip() == "(none)"

def test_gradient_dead_claims_report_marks_unverifiable():
    """E37 R3: an all-crash claim drops as all_scorer_fail and the dead
    report says it could not be verified."""
    from sources.evaluators.hybrid_verifier import gradient as g

    text = g.build_gradient(
        uuid="u1",
        goal=GOAL,
        now_scores={"C1": 0.5},
        evidence={"C1": "ok"},
        claims=[_claim("C1")],
        surviving=["C1"],
        pair_records=[],
        reward=0.5,
        win_rate=0.5,
        mean_score=0.5,
        dead_claims=[
            {"id": "C9", "stage": "result", "statement": "flat",
             "drop_reason": "all_scorer_fail"},
            {"id": "C8", "stage": "log", "statement": "equal",
             "drop_reason": "zero_variance"},
        ],
    )
    dropped = text.split("## 5. Dead claims report")[1]
    assert "could not be verified" in dropped
    assert "[C9]" in dropped and "all_scorer_fail" in dropped
    assert "zero_variance" in dropped


def test_config_defaults_and_overrides():
    c = Config()
    assert c.verifier_kind == "hybrid"
    assert c.hybrid_verifier_num_claims == 10
    assert c.hybrid_verifier_refinement_rounds == 2
    assert c.hybrid_verifier_scorer_timeout_s == 60
    assert c.hybrid_verifier_pairwise_mode == "temporal"
    assert c.hybrid_verifier_digest_max_files == 8
    # E35 v3 knobs default ON (backward-compatible upgrades)
    assert c.hybrid_verifier_visual_rung is True
    assert c.hybrid_verifier_execution_gate is True
    c.from_json(
        {
            "verifier_kind": "legacy",
            "hybrid_verifier_num_claims": 6,
            "hybrid_verifier_pairwise_mode": "temporal_strict",
            "hybrid_verifier_digest_max_files": 4,
            "hybrid_verifier_visual_rung": False,
            "hybrid_verifier_execution_gate": False,
        }
    )
    assert c.verifier_kind == "legacy"
    assert c.hybrid_verifier_num_claims == 6
    assert c.hybrid_verifier_pairwise_mode == "temporal_strict"
    assert c.hybrid_verifier_digest_max_files == 4
    assert c.hybrid_verifier_visual_rung is False
    assert c.hybrid_verifier_execution_gate is False
    for k in (
        "verifier_kind",
        "hybrid_verifier_num_claims",
        "hybrid_verifier_refinement_rounds",
        "hybrid_verifier_scorer_timeout_s",
        "hybrid_verifier_pairwise_mode",
        "hybrid_verifier_digest_max_files",
        "hybrid_verifier_visual_rung",
        "hybrid_verifier_execution_gate",
    ):
        assert k in c.jsonify()


def test_cli_flags_apply_overrides():
    import argparse

    import main as main_mod

    parser = argparse.ArgumentParser()
    main_mod.add_config_arguments(parser, Config())
    args = parser.parse_args(
        [
            "--verifier_kind",
            "legacy",
            "--hybrid_verifier_num_claims",
            "7",
            "--hybrid_verifier_refinement_rounds",
            "3",
            "--hybrid_verifier_scorer_timeout_s",
            "90",
            "--hybrid_verifier_pairwise_mode",
            "temporal_strict",
            "--hybrid_verifier_digest_max_files",
            "5",
            "--no-hybrid_verifier_visual_rung",
            "--no-hybrid_verifier_execution_gate",
        ]
    )
    # the full CLI parser defines --max_evolve_iterations elsewhere
    args.__dict__.setdefault("max_evolve_iterations", None)
    c = Config()
    main_mod.apply_config_overrides(args, c)
    assert c.verifier_kind == "legacy"
    assert c.hybrid_verifier_num_claims == 7
    assert c.hybrid_verifier_refinement_rounds == 3
    assert c.hybrid_verifier_scorer_timeout_s == 90
    assert c.hybrid_verifier_pairwise_mode == "temporal_strict"
    assert c.hybrid_verifier_digest_max_files == 5
    assert c.hybrid_verifier_visual_rung is False
    assert c.hybrid_verifier_execution_gate is False
    # ...and the positive flags restore them
    args2 = parser.parse_args(
        ["--hybrid_verifier_visual_rung", "--hybrid_verifier_execution_gate"]
    )
    args2.__dict__.setdefault("max_evolve_iterations", None)
    main_mod.apply_config_overrides(args2, c)
    assert c.hybrid_verifier_visual_rung is True
    assert c.hybrid_verifier_execution_gate is True


def test_short_circuit_flags_artifacts_when_workspace_has_them(tmp_path: Path):
    v = _make_evaluator(tmp_path, _LLM([]), _ScriptedRunner([]))
    _seed_workspace(v.workspace_dir)
    v._load_workflow_data = lambda uuid: types.SimpleNamespace(  # type: ignore[method-assign]
        goal=GOAL,
        state_result={},
        code="",
    )
    result = v.evaluate("dead-run-2")
    assert result["overall_score"] == 0.0  # reward still 0: execution failed
    assert result["claim_summary"][0]["score"] == 1.0  # artifacts ARE present




def test_facade_defaults_to_hybrid(monkeypatch, tmp_path):
    """WorkflowEvaluator instantiates the hybrid verifier by default."""

    class _Stub:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(evaluator_mod, "GenericEvaluator", _Stub)
    monkeypatch.setattr(evaluator_mod, "ScenarioEvaluator", _Stub)
    monkeypatch.setattr(
        evaluator_mod,
        "HybridVerifierEvaluator",
        lambda config, workspace_dir=None: ("HYBRID", workspace_dir),
    )
    import builtins

    opened = []
    real_import = builtins.__import__

    def spy_import(name, *a, **k):
        if name.endswith((".verifier", ".legacy_verifier")):
            opened.append(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", spy_import)
    config = types.SimpleNamespace(
        memory_dir=str(tmp_path / "m"),
        workflow_dir=str(tmp_path / "w"),
        model_pricing={},
        reasoning_effort="low",
        judge_model="x/y",
        max_tokens=8,
        verifier_kind="hybrid",
        openrouter_provider_for=lambda m: None,
        openrouter_quantizations_for=lambda m: None,
    )
    facade = evaluator_mod.WorkflowEvaluator(config)
    assert facade.verifier_kind == "hybrid"
    assert facade.verifier_evaluator[0] == "HYBRID"
    assert opened == []  # legacy module never imported on the default path


def test_facade_legacy_escape_hatch(monkeypatch, tmp_path):
    class _Stub:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(evaluator_mod, "GenericEvaluator", _Stub)
    monkeypatch.setattr(evaluator_mod, "ScenarioEvaluator", _Stub)
    captured = {}

    def _fake_legacy(config, workspace_dir=None):
        captured["legacy"] = True
        return _Stub()

    import sources.evaluators.legacy_verifier as legacy_pkg

    monkeypatch.setattr(legacy_pkg, "VerifierEvaluator", _fake_legacy)
    # the facade does `from .legacy_verifier import VerifierEvaluator`
    # lazily — the from-import binds the attribute at call time, so the
    # patch on the package above is what the facade picks up:
    monkeypatch.setattr(
        "sys.modules",
        {**sys.modules},
        raising=False,
    )
    config = types.SimpleNamespace(
        memory_dir=str(tmp_path / "m"),
        workflow_dir=str(tmp_path / "w"),
        model_pricing={},
        reasoning_effort="low",
        judge_model="x/y",
        max_tokens=8,
        verifier_kind="legacy",
        openrouter_provider_for=lambda m: None,
        openrouter_quantizations_for=lambda m: None,
    )
    facade = evaluator_mod.WorkflowEvaluator(config)
    assert facade.verifier_kind == "legacy"
    assert isinstance(facade.verifier_evaluator, _Stub) or captured.get("legacy")


def test_legacy_modules_deprecated_but_importable():
    import importlib

    for mod_name in (
        "sources.evaluators.legacy_verifier.verifier",
        "sources.evaluators.legacy_verifier.verifier_claims",
        "sources.evaluators.legacy_verifier.verifier_claim_sources",
        "sources.evaluators.legacy_verifier.verifier_per_claim",
        "sources.evaluators.legacy_verifier.verifier_workspace",
    ):
        mod = importlib.import_module(mod_name)
        assert "[DEPRECATED 2026-09-24]" in mod.__doc__


# ------------------------------------------------------- (8) inventory ----


def test_inventory_signals_and_union_render(tmp_path: Path):
    ws = tmp_path / "ws"
    (ws / "pred_results").mkdir(parents=True)
    (ws / "pred_results" / "predictions.csv").write_text(
        "smiles,probability\nCCO,0.7\nNCC,0.4\n", encoding="utf-8"
    )
    # tiny valid PNG with IHDR 8x6
    import struct

    png = (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + struct.pack(">II", 8, 6)
        + b"\x08\x06\x00\x00\x00"
    )
    (ws / "figure.png").write_bytes(png)
    inv = inv_mod.scan_workspace(ws)
    assert inv["pred_results/predictions.csv"]["signal"] == "3 rows x 2 cols"
    assert inv["figure.png"]["signal"] == "8x6 px"
    assert inv_mod.workspace_has_artifacts(inv)
    union = inv_mod.render_union_inventory(
        {
            "a.csv": {"count": 2, "signal": "5 rows x 3 cols"},
            "b.txt": {"count": 1, "signal": ""},
        },
        n_workspaces=3,
        current=["a.csv"],
    )
    assert "present in 2/3 workspaces" in union
    assert "(present in the workspace being scored)" in union
    previews = inv_mod.deliverable_previews(inv, ws)
    assert "predictions.csv" in previews and "rows total" in previews


def test_task_key_stability():
    assert task_key("goal A") == task_key("goal A")
    assert task_key("goal A") != task_key("goal B")
    assert len(task_key("x")) == 16


def test_json_payload_extraction_tolerates_fences():
    assert extract_json_payload('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json_payload('noise {"a": [1,2]} noise') == '{"a": [1,2]}'


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ------------------------------------------- (9) temporal ladder validation ----


def test_ladder_claim_validation_and_ordering():
    obj = {
        "claims": [
            {
                "id": "C1",
                "stage": "script",
                "temporal_index": 2,
                "statement": "code reads compounds.csv",
                "target": "*.py",
                "scoring_rule": "count of input reads in code, fraction -> 0..1",
            },
            {
                "id": "C2",
                "stage": "result",
                "temporal_index": 5,
                "statement": "pred columns match original names",
                "target": "pred_results/*.csv",
                "scoring_rule": "count of columns matching the original names -> 0..1",
            },
            {
                "id": "C3",
                "stage": "log",
                "temporal_index": 1,
                "statement": "loss decreases over epochs",
                "target": "*.log",
                "scoring_rule": "fraction of epoch pairs with decreasing loss -> 0..1",
            },
        ]
    }
    out, err = claims_mod.validate_claims(obj, 3, 3)
    assert err is None
    assert [c["stage"] for c in out] == ["script", "log", "result"]
    # unknown stage coerces to result; missing temporal_index keeps order
    obj["claims"][0]["stage"] = "weird"
    out2, err2 = claims_mod.validate_claims(obj, 3, 3)
    assert err2 is None and out2[2]["stage"] == "result"
    assert isinstance(out2[2]["temporal_index"], int)


def test_registry_migrates_untagged_claims(tmp_path: Path):
    """Legacy registry JSON without stage fields loads with result-stage defaults."""
    from sources.evaluators.hybrid_verifier.registry import task_key as tk

    path = tmp_path / f"hybrid_registry_{tk(GOAL)}.json"
    legacy = {
        "version": "hybrid-v1-e19b-20260924",
        "goal": GOAL,
        "claims": [
            {
                "id": "C1",
                "category": "completeness",
                "statement": "s1",
                "target": "t",
                "scoring_rule": "count rows fraction",
                "scorer": "print(1)",
                "state": "alive",
                "drop_reason": None,
                "added_round": 0,
            },
            {
                "id": "C2",
                "category": "other",
                "statement": "s2",
                "target": "t",
                "scoring_rule": "count rows fraction",
                "scorer": None,
                "state": "alive",
                "drop_reason": None,
                "added_round": 0,
            },
        ],
        "generations": [],
        "inventory": {},
        "refinements": {},
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")
    reg = TaskRegistry.load(tmp_path, GOAL)
    assert [c["stage"] for c in reg.claims] == ["result", "result"]
    assert [c["temporal_index"] for c in reg.claims] == [0, 1]


# ------------------------------------------------ (10) format digest cache ----


def test_format_digest_cached_once_per_task(tmp_path: Path):
    from sources.evaluators.hybrid_verifier.layers import (
        ClaimsEvidenceLayer,
        LayerContext,
    )

    ws = tmp_path / "workspace"
    _seed_workspace(ws)
    calls = []

    def llm(agent, prompt):
        calls.append(agent)
        assert "FORMAT DIGESTS" in prompt
        return json.dumps(
            {
                "digests": [
                    {
                        "path": "pred_results/predictions.csv",
                        "digest": "columns smiles,probability; comma; 2dp floats",
                    }
                ]
            }
        )

    reg = TaskRegistry.load(tmp_path, GOAL)
    context = LayerContext(
        uuid="u1", goal=GOAL, workspace=ws, registry=reg, generation_index=0
    )
    layer = ClaimsEvidenceLayer(
        llm_text=lambda *a: llm(a[1], a[2]),
        run_script=lambda *a: scorers_mod.ExecOutcome(rc=0, stdout="{}"),
        logger=logging.getLogger("test-digest"),
    )
    block1 = layer._ensure_digests(context)
    block2 = layer._ensure_digests(context)  # cached: no second LLM call
    assert block1 == block2 and "smiles,probability" in block1
    assert calls == ["hybrid_format_digest"]
    assert "pred_results/predictions.csv" in reg.digests


def test_digest_sampling_deterministic_and_bounded(tmp_path: Path):
    from sources.evaluators.hybrid_verifier import digest as digest_mod

    ws = tmp_path / "ws"
    ws.mkdir()
    big = ws / "train.log"
    body = "".join(f"epoch {i} loss {1.0 - i * 0.01:.3f}\n" for i in range(500))
    big.write_text(body, encoding="utf-8")
    s1 = digest_mod.sample_sections(big, slice_bytes=200)
    s2 = digest_mod.sample_sections(big, slice_bytes=200)
    assert s1 == s2  # deterministic
    assert "--- head" in s1 and "--- middle" in s1 and "--- tail" in s1
    assert len(s1) <= 3 * (200 + 64)  # bounded bytes per slice + labels
    # digest JSON parsing
    parsed = digest_mod.parse_digests(
        '```json\n{"digests": [{"path": "a.csv", "digest": "cols x,y"}]}\n```'
    )
    assert parsed == {"a.csv": "cols x,y"}
    assert digest_mod.parse_digests("garbage") == {}


def test_scorer_prompt_carries_ladder_and_digests():
    claim = _claim("S1", stage="script", tidx=0)
    prompt = scorers_mod.scorer_prompt(
        GOAL, claim, "(union)", digests_block="- pred.csv: columns smiles,probability"
    )
    assert 'rung is stage "script"' in prompt
    assert "TEMPORAL LADDER" in prompt
    assert "columns smiles,probability" in prompt
    assert "_pred" in prompt  # operator policy examples cited
    bare = scorers_mod.scorer_prompt(GOAL, claim, "(union)")
    assert "Format digests" not in bare


# ---------------------------------------------------------------------------
# Bradley–Terry reward (aggregation.bradley_terry_reward)
# ---------------------------------------------------------------------------


def test_bradley_terry_orders_strengths():
    claims = [_claim("C1", stage="result")]
    # Nested total order under mean_diff: g2 > g1 > g0.
    g0 = {"C1": 0.2}
    g1 = {"C1": 0.5}
    g2 = {"C1": 0.8}
    top = aggregation.bradley_terry_reward({"C1": 1.0}, [g0, g1, g2], claims)
    mid = aggregation.bradley_terry_reward(g1, [g0, g1, g2], claims)
    low = aggregation.bradley_terry_reward(g0, [g0, g1, g2], claims)
    assert top["reward"] > 0.5 > low["reward"]
    assert top["reward"] > mid["reward"] > low["reward"]
    # Strengths are centered (mean zero) and ordered.
    s = top["strengths"]
    assert abs(sum(s)) < 1e-6
    assert s[0] == max(s)


def test_bradley_terry_single_opponent_and_ties():
    claims = [_claim("C1", stage="result")]
    win = aggregation.bradley_terry_reward({"C1": 0.9}, [{"C1": 0.1}], claims)
    loss = aggregation.bradley_terry_reward({"C1": 0.1}, [{"C1": 0.9}], claims)
    assert 0.5 < win["reward"] < 1.0
    assert 0.0 < loss["reward"] < 0.5
    # All-ties: equal vectors -> equal strengths, reward exactly 0.5.
    tie = aggregation.bradley_terry_reward({"C1": 0.4}, [{"C1": 0.4}], claims)
    assert abs(tie["reward"] - 0.5) < 1e-6
    assert tie["win_rate"] == 0.5


def test_bradley_terry_perfect_separation_stays_finite():
    # Current gen beats everyone; ridge keeps beta finite -> reward < 1.
    claims = [_claim("C1", stage="script"), _claim("C2", stage="result")]
    prevs = [{"C1": 0.1, "C2": 0.1}, {"C1": 0.2, "C2": 0.2}]
    res = aggregation.bradley_terry_reward(
        {"C1": 1.0, "C2": 1.0}, prevs, claims, mode="mean_diff"
    )
    assert 0.5 < res["reward"] < 1.0
    assert all(abs(b) < 10.0 for b in res["strengths"])


def test_bradley_terry_deterministic_and_first_gen_fallback():
    claims = [_claim("C1", stage="result")]
    a = aggregation.bradley_terry_reward({"C1": 0.7}, [{"C1": 0.3}], claims)
    b = aggregation.bradley_terry_reward({"C1": 0.7}, [{"C1": 0.3}], claims)
    assert a == b
    fb = aggregation.bradley_terry_reward({"C1": 0.6}, [], claims)
    assert fb["fallback"] == "mean_claim"
    assert abs(fb["reward"] - 0.6) < 1e-9


def test_bradley_terry_temporal_mode_respects_ladder():
    # A script-stage failure must lose even with a result-stage advantage.
    claims = [
        _claim("S1", stage="script", tidx=0),
        _claim("R1", stage="result", tidx=0),
    ]
    loser = {"S1": 0.0, "R1": 1.0}  # fails the script rung
    winner = {"S1": 1.0, "R1": 0.0}  # passes script, weak results
    res = aggregation.bradley_terry_reward(winner, [loser], claims, mode="temporal")
    assert res["reward"] > 0.5
    assert res["win_rate"] == 1.0


def test_config_reward_default_and_roundtrip():
    cfg = Config()
    assert cfg.hybrid_verifier_reward == "win_rate"
    cfg.hybrid_verifier_reward = "bradley_terry"
    data = cfg.jsonify()
    assert data["hybrid_verifier_reward"] == "bradley_terry"
    roundtrip = Config()
    roundtrip.from_json(data)
    assert roundtrip.hybrid_verifier_reward == "bradley_terry"

# ---------------------------------------------------------------------------
# E35 PRIME v3: T1 firewall (E29)
# ---------------------------------------------------------------------------


def test_t1_firewall_rejects_numeric_answer_keys():
    """The E29 smuggle class is flagged by the batched LLM screen: a
    specific numeric constant the goal does not state (energy -2,
    coverage 16268, optimum 376, measured 0.239448)."""
    goal = "Plot the charge density heatmap for the slab."
    smugglers = [
        {"statement": "the minimum energy equals -2",
         "target": "out.txt", "scoring_rule": "score 1 if value == -2 else 0"},
        {"statement": "coverage count is 16268",
         "target": "c.csv", "scoring_rule": "count rows == 16268"},
        {"statement": "optimum count 376",
         "target": "o.csv", "scoring_rule": "count == 376"},
        {"statement": "measured value 0.239448",
         "target": "m.json", "scoring_rule": "value matches 0.239448"},
    ]
    claims = [{**c, "id": f"C{i}"} for i, c in enumerate(smugglers, 1)]
    screen = _ScreenLLM(
        smuggled={
            f"C{i}": ("statement", v)
            for i, v in enumerate(("-2", "16268", "376", "0.239448"), 1)
        }
    )
    why = claims_mod.t1_violations(claims, goal, screen, uuid="u1")
    assert set(why) == {"C1", "C2", "C3", "C4"} and all(why.values())
    assert why["C1"] == ["statement:-2"]
    # ONE batched call over the whole claim set, prompt carrying goal + claims
    assert screen.calls == ["hybrid_screen_t1"]
    assert goal in screen.prompts[0]
    assert "the minimum energy equals -2" in screen.prompts[0]
    # a number the goal itself states is verifiable, not smuggled
    ok = {"id": "C1", "statement": "value equals 16268", "target": "x",
          "scoring_rule": "== 16268"}
    clean = _ScreenLLM()
    assert claims_mod.t1_violations(
        [ok], "the coverage must be 16268", clean
    ) == {"C1": []}


def test_t1_firewall_passes_structural_constants():
    """Structural constants pass the LLM screen clean: [0,1]/[-1,1]
    bands, the -1 sentinel, small counts, round bin/grid counts,
    hyphenated words, bracketed range bands, RGB components of
    goal-named colors."""
    goal = "Plot the charge density difference heatmap."
    structural = [
        {"statement": "probabilities lie in [0,1]",
         "target": "p.csv", "scoring_rule": "fraction of values in [0,1]"},
        {"statement": "missing values encoded as -1",
         "target": "p.csv", "scoring_rule": "count of -1 sentinels"},
        {"statement": "correlation in [-1,1]",
         "target": "m.json", "scoring_rule": "value within [-1,1]"},
        {"statement": "top-20 frames annotated",
         "target": "f/", "scoring_rule": "count of top-20 frame labels"},
        {"statement": "10-frame rolling average used",
         "target": "s.py", "scoring_rule": "count of 10-frame window calls"},
        {"statement": "histogram uses 250 bins",
         "target": "h.png", "scoring_rule": "count bins == 250"},
        {"statement": "pChEMBL values within [-2, 15]",
         "target": "x.csv", "scoring_rule": "fraction within [-2, 15]"},
        {"statement": "atoms drawn in orange",
         "target": "fig.png", "scoring_rule": "orange RGB (255, 165, 0)"},
        {"statement": "5-fold cross-validation used",
         "target": "s.py", "scoring_rule": "count of 5-fold splits"},
    ]
    claims = [{**c, "id": f"C{i}"} for i, c in enumerate(structural, 1)]
    screen = _ScreenLLM()
    why = claims_mod.t1_violations(claims, goal, screen)
    assert why == {f"C{i}": [] for i in range(1, len(structural) + 1)}
    assert screen.calls == ["hybrid_screen_t1"]  # one batched call
    # a judge failure degrades open: every claim stays clean
    garbage = _StaticLLM("no json here")
    assert claims_mod.t1_violations(claims, goal, garbage) == {
        f"C{i}": [] for i in range(1, len(structural) + 1)
    }


# ---------------------------------------------------------------------------
# E35 PRIME v3: content lever (E30)
# ---------------------------------------------------------------------------


def test_extract_prompt_demands_content_lever():
    prompt = claims_mod.extract_claims_prompt(GOAL, "(union)", "(prev)", 8, 12)
    # >=2 core_computation: the goal's computation end-to-end
    assert 'AT LEAST 2 claims of category "core_computation"' in prompt
    # the SELF-CHECK mandate: the model judges its own claims (operator
    # mandate — content judgment via the LLM, batched post-screens follow)
    assert 'output a "self_check" field' in prompt
    assert '"self_check": {"generic": false, "smuggled_answer": false' in prompt
    assert "trivially satisfy" in prompt and "smuggled_answer" in prompt
    assert "goal's computation END-TO-END" in prompt
    assert "goal-stated parameters" in prompt
    assert "a bare import" in prompt and "not identity" in prompt
    # >=2 method_identity: exact named method; lookalike=0; import != identity
    assert 'AT LEAST 2 claims of category "method_identity"' in prompt
    assert "lookalike substitute must score 0" in prompt
    assert "import" in prompt and "identity" in prompt
    # the E16 key_missing fidelity families are demanded too
    for fam in ("output_schema", "prediction_sanity", "deliverable_path"):
        assert f'"{fam}"' in prompt
    # the new categories are accepted by validation
    assert "core_computation" in claims_mod.CLAIM_CATEGORIES
    assert "method_identity" in claims_mod.CLAIM_CATEGORIES


def test_extraction_retries_on_t1_and_composition_then_accepts():
    """The extractor re-issues with feedback when the model smuggles an
    answer key or ignores the mandate, then accepts the compliant set."""
    import tempfile as _tf

    goal = "Predict the bulk modulus for the provided structures."
    smuggled = dict(_claim("C1"), statement="energy minimum equals -7.5")
    compliant = [_claim(f"C{i+2}") for i in range(9)]  # C2..C10; C1 = smuggler
    for c in compliant[:2]:
        c["category"] = "core_computation"
    for c in compliant[2:4]:
        c["category"] = "method_identity"
    weak = [_claim(f"C{i+1}") for i in range(9)]
    weak[1]["category"] = "core_computation"  # only 1: mandate unmet
    responses = [
        json.dumps({"claims": [smuggled, *compliant[:7]]}),  # extract 1
        "[]",  # generic screen 1: clean
        json.dumps(
            [
                {
                    "id": "C1",
                    "smuggled": True,
                    "violating_field": "statement",
                    "violating_value": "-7.5",
                    "reason": "goal never states -7.5",
                }
            ]
        ),  # T1 screen 1: C1 smuggles an answer key
        json.dumps({"claims": weak}),  # extract 2 (composition shortfall)
        "[]",  # generic screen 2: clean
        json.dumps({"claims": compliant}),  # extract 3
        "[]",  # generic screen 3: clean
        "[]",  # T1 screen 3: clean
    ]
    llm = _LLM(responses)
    with _tf.TemporaryDirectory() as td:
        v = _make_evaluator(Path(td), llm, _ScriptedRunner([]), num_claims=10)
        (ws := Path(td) / "ws").mkdir()
        reg = TaskRegistry.load(Path(td), goal, logger=v.logger)
        reg.merge_inventory(inv_mod.scan_workspace(ws))
        ctx = LayerContext(
            uuid="u1", goal=goal, workspace=ws, registry=reg, generation_index=0
        )
        err = v._claims_layer._extract(ctx)
        assert err is None
        assert len(reg.claims) == 9
        assert all(c["statement"] != smuggled["statement"] for c in reg.claims)
        # 3 extraction calls, each followed by the batched screens
        # (generic inside validate_claims, T1 after it) = 8 calls total
        assert len(llm.prompts) == 8
        extract_prompts = [
            p for _, p in llm.prompts
            if "TEMPORALLY-ORDERED verification rubric" in p
        ]
        assert len(extract_prompts) == 3
        # the retries carried the feedback
        assert "T1 FIREWALL" in extract_prompts[1]
        assert "core_computation" in extract_prompts[2]


def test_figure_task_detection_threshold():
    import tempfile as _tf

    with _tf.TemporaryDirectory() as td:
        ws = _figure_workspace(Path(td))
        inv = inv_mod.scan_workspace(ws)
        # results-dir pool is 2 pngs / 2 files -> figure task
        assert inv_mod.is_figure_task(inv) is True
        # a CSV deliverable task is not
        ws2 = Path(td) / "ws2"
        (ws2 / "pred_results").mkdir(parents=True)
        (ws2 / "pred_results" / "pred.csv").write_text("a\n1\n")
        assert inv_mod.is_figure_task(inv_mod.scan_workspace(ws2)) is False
        # mixed results dir below the 50% image share (2 png / 5 files)
        ws3 = Path(td) / "ws3"
        (ws3 / "pred_results").mkdir(parents=True)
        (ws3 / "pred_results" / "a.png").write_bytes(_png_bytes())
        (ws3 / "pred_results" / "b.png").write_bytes(_png_bytes())
        for name in ("c.csv", "d.csv", "e.csv"):
            (ws3 / "pred_results" / name).write_text("a\n1\n")
        assert inv_mod.is_figure_task(inv_mod.scan_workspace(ws3)) is False
        # empty workspace is not a figure task
        assert inv_mod.is_figure_task({}) is False


# ---------------------------------------------------------------------------
# E35 PRIME v3: visual rung (E26) — detection + mocked kimi scoring
# ---------------------------------------------------------------------------


def _png_bytes(w: int = 8, h: int = 6) -> bytes:
    import struct

    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + struct.pack(">II", w, h)
        + b"\x08\x06\x00\x00\x00"
    )


FIGURE_GOAL = (
    "Plot the planar-averaged charge density difference and its cumulative "
    "integral to pred_results/charge.png using orange markers."
)


def _figure_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / "pred_results").mkdir(parents=True)
    (ws / "pred_results" / "charge.png").write_bytes(_png_bytes())
    (ws / "pred_results" / "integral.png").write_bytes(_png_bytes(4, 4))
    (ws / "input.dat").write_text("x,y\n1,2\n", encoding="utf-8")
    return ws


class _Kimi:
    """Queue-based vision stub: records calls, pops canned responses."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(self, uuid: str, agent: str, prompt: str, images):
        self.calls.append((agent, [p.name for p in images]))
        if not self.responses:
            raise AssertionError(f"unexpected vision call: {agent}")
        return self.responses.pop(0)



def test_visual_layer_freezes_claims_then_scores_each_generation():
    import tempfile as _tf

    from sources.evaluators.hybrid_verifier.layers import VisualEvidenceLayer

    with _tf.TemporaryDirectory() as td:
        ws = _figure_workspace(Path(td))
        reg = TaskRegistry(Path(td), FIGURE_GOAL)
        reg.merge_inventory(inv_mod.scan_workspace(ws))
        kimi = _Kimi(
            [
                json.dumps(
                    {
                        "claims": [
                            {"id": "V1", "statement": "figure shows the planar-averaged density",
                             "rubric": "1.0 if both curves; 0.5 if one; 0.0 otherwise"},
                            {"id": "V2", "statement": "axis labels present and readable",
                             "rubric": "1.0 if labeled; 0.0 if not"},
                            {"id": "V3", "statement": "orange markers used",
                             "rubric": "1.0 if orange; 0.0 otherwise"},
                        ]
                    }
                ),
                json.dumps(
                    {
                        "scores": {
                            "V1": {"score": 1.0, "evidence": "two curves visible"},
                            "V2": {"score": 0.5, "evidence": "x label only"},
                            "V3": {"score": 0.0, "evidence": "blue markers"},
                        }
                    }
                ),
            ]
        )
        layer = VisualEvidenceLayer(vision_call=kimi)
        ctx = LayerContext(
            uuid="u1", goal=FIGURE_GOAL, workspace=ws,
            registry=reg, generation_index=0,
        )
        scores = layer.collect(ctx)
        assert [(s.claim_id, s.score) for s in scores] == [
            ("V1", 1.0), ("V2", 0.5), ("V3", 0.0),
        ]
        assert scores[0].category == "visual"
        # extraction grounded with <=2 sample figures, BEFORE scoring
        assert kimi.calls[0] == ("hybrid_visual_extract", ["charge.png", "integral.png"])
        assert kimi.calls[1][0] == "hybrid_visual_score"
        assert kimi.calls[1][1] == ["charge.png", "integral.png"]  # <=3 largest
        # claims frozen into the registry BEFORE the scoring call happened
        frozen = [(c["id"], c["stage"], c["category"]) for c in reg.claims]
        assert frozen == [
            ("V1", "visual", "visual"),
            ("V2", "visual", "visual"),
            ("V3", "visual", "visual"),
        ]
        # second generation: no re-extraction, exactly one scoring call
        # against the SAME frozen rubric (same criteria both sides)
        kimi.responses.append(
            json.dumps({"scores": {c: {"score": 0.9, "evidence": "ok"}
                                   for c in ("V1", "V2", "V3")}})
        )
        kimi.calls.clear()
        s2 = layer.collect(ctx)
        assert [c[0] for c in kimi.calls] == ["hybrid_visual_score"]
        assert all(s.score == 0.9 for s in s2)


def test_visual_layer_no_figures_scores_zero_and_non_figure_inactive():
    import tempfile as _tf

    from sources.evaluators.hybrid_verifier.layers import VisualEvidenceLayer

    with _tf.TemporaryDirectory() as td:
        # figure task registry (images in the union) but current workspace
        # produced no figures -> every visual claim scores 0.0
        ws = _figure_workspace(Path(td))
        reg = TaskRegistry(Path(td), FIGURE_GOAL)
        reg.merge_inventory(inv_mod.scan_workspace(ws))
        kimi = _Kimi(
            [json.dumps({"claims": [
                {"id": "V1", "statement": "two curves", "rubric": "1.0 if two"},
                {"id": "V2", "statement": "labels", "rubric": "1.0 if labels"},
                {"id": "V3", "statement": "orange", "rubric": "1.0 if orange"},
            ]})]
        )
        layer = VisualEvidenceLayer(vision_call=kimi)
        ctx = LayerContext(
            uuid="u1", goal=FIGURE_GOAL, workspace=ws,
            registry=reg, generation_index=0,
        )
        assert layer.collect(ctx)  # extraction + scoring both happen here
        empty_ws = Path(td) / "empty"
        (empty_ws / "pred_results").mkdir(parents=True)
        (empty_ws / "pred_results" / "log.txt").write_text("no figure\n")
        ctx2 = LayerContext(
            uuid="u2", goal=FIGURE_GOAL, workspace=empty_ws,
            registry=reg, generation_index=1,
        )
        kimi.calls.clear()
        s2 = layer.collect(ctx2)
        assert [(x.claim_id, x.score, x.evidence) for x in s2] == [
            ("V1", 0.0, "no figure produced"),
            ("V2", 0.0, "no figure produced"),
            ("V3", 0.0, "no figure produced"),
        ]
        assert kimi.calls == []  # zero is measured, not judged
        # non-figure task: the layer stays dormant entirely
        csv_ws = Path(td) / "csv"
        (csv_ws / "pred_results").mkdir(parents=True)
        (csv_ws / "pred_results" / "pred.csv").write_text("a\n1\n")
        reg2 = TaskRegistry(Path(td), "csv task goal")
        reg2.merge_inventory(inv_mod.scan_workspace(csv_ws))
        ctx3 = LayerContext(
            uuid="u3", goal="csv task", workspace=csv_ws,
            registry=reg2, generation_index=0,
        )
        assert layer.collect(ctx3) == []
        assert reg2.claims == []


def test_claims_layer_never_scores_visual_claims():
    """The claims pipeline builds no Python scorers for visual claims."""
    import tempfile as _tf

    with _tf.TemporaryDirectory() as td:
        reg = TaskRegistry(Path(td), GOAL)
        text = [_claim("C1"), _claim("C2")]
        for c in text:
            c["scorer"] = _scorer_body(c["id"], 0.5)
        visual = {
            "id": "V1", "stage": "visual", "temporal_index": 0,
            "category": "visual", "statement": "two curves",
            "target": "figs", "scoring_rule": "1.0 if two curves",
            "scorer": None, "state": "alive", "drop_reason": None,
            "added_round": 0,
        }
        reg.seed_claims(text)
        reg.append_claims([visual])
        assert reg.claim("V1") is not None
        ws = Path(td) / "ws"
        _seed_workspace(ws)

        class _RecordingRunner:
            calls: list[str] = []

            def __call__(self, workspace, code, execution_id):
                type(self).calls.append(execution_id)
                for c in ("C1", "C2"):
                    if c in execution_id:
                        return scorers_mod.ExecOutcome(
                            rc=0,
                            stdout=json.dumps(
                                {"claim_id": c, "score": 0.5, "evidence": "ok"}
                            ),
                        )
                raise AssertionError(execution_id)

        runner = _RecordingRunner()
        layer = ClaimsEvidenceLayer(
            llm_text=_LLM([]), run_script=runner, logger=None, num_claims=2
        )
        ctx = LayerContext(
            uuid="u1", goal=GOAL, workspace=ws, registry=reg, generation_index=0
        )
        scores = layer.collect(ctx)
        # only the text claims were executed; V1 was never handed a scorer
        assert sorted(s.claim_id for s in scores) == ["C1", "C2"]
        assert all("V1" not in eid for eid in runner.calls)
        # and the dead-claim replacement loop skips visual claims too
        reg.mark_dead("V1", "zero_variance")
        extra = layer.revise(ctx, [c["id"] for c in reg.dead_claims()])
        assert extra == []  # no refinement LLM call was attempted


def test_execution_gate_caps_by_status():
    from sources.evaluators.hybrid_verifier.layers import (
        GATE_CAPS,
        ExecutionGateLayer,
    )

    assert GATE_CAPS == {
        "crash": 0.0,
        "no_entry_script": 0.0,
        "timeout": 0.0,
        "divergent": 0.5,
        "clean_recover": None,
    }
    for status, expected in GATE_CAPS.items():
        gate = ExecutionGateLayer(status_lookup=lambda u, s=status: (s, 7.0))
        ctx = types.SimpleNamespace(uuid="u1")
        assert gate.gate(ctx, []) == expected
        assert gate.last_facts == {"status": status, "runtime_s": 7.0, "cap": expected}
        assert gate.collect(ctx) == []  # the gate scores no claims
    # no evidence -> no cap (never punish without re-execution data)
    miss = ExecutionGateLayer(status_lookup=lambda u: (None, None))
    assert miss.gate(types.SimpleNamespace(uuid="unknown"), []) is None
    # lookup failure degrades to no cap, never crashes
    def _boom(u):
        raise RuntimeError("ledger unreadable")

    bad = ExecutionGateLayer(status_lookup=_boom)
    assert bad.gate(types.SimpleNamespace(uuid="u"), []) is None


def test_execution_gate_wired_into_evaluate(tmp_path: Path):
    """A divergent re-execution caps an otherwise winning generation."""
    from sources.evaluators.hybrid_verifier.layers import ExecutionGateLayer

    claims = [_claim(f"C{i}") for i in range(1, 5)]
    llm = _LLM([])
    runner = _ScriptedRunner([])
    v = _make_evaluator(tmp_path, llm, runner, num_claims=4)
    reg = TaskRegistry.load(v._runner_temp_root, GOAL, logger=v.logger)
    reg.seed_claims(claims)
    for c in reg.claims:
        c["scorer"] = _scorer_body(c["id"], 0.9)
    reg.record_generation(
        "u0",
        {c["id"]: 0.1 for c in reg.claims},
        {c["id"]: "weak" for c in reg.claims},
        reward=0.1,
    )
    reg.save()

    def run(ws_, code, eid):
        for c in ["C1", "C2", "C3", "C4"]:
            if c in eid:
                return scorers_mod.ExecOutcome(
                    rc=0,
                    stdout=json.dumps(
                        {"claim_id": c, "score": 0.9, "evidence": "strong"}
                    ),
                )
        raise AssertionError(eid)

    v2 = _make_evaluator(tmp_path, _LLM([]), run, num_claims=4)
    v2._load_workflow_data = lambda uuid: _wf_info(GOAL)  # type: ignore[method-assign]
    gate = ExecutionGateLayer(status_lookup=lambda u: ("divergent", 31.2))
    v2.layers = [v2._claims_layer, gate]
    result = v2.evaluate("u1")
    # raw win-rate is a clean 1.0 (every claim beats u0) but the gate caps 0.5
    assert result["win_rate"] == 1.0
    assert result["overall_score"] == 0.5
    assert result["execution_gate"] == {
        "status": "divergent", "runtime_s": 31.2, "cap": 0.5,
    }
    assert "capped at 0.5" in result["abstracted_textual_gradient"]
    # crash caps at 0.0 even for the same winning scores
    v3 = _make_evaluator(tmp_path, _LLM([]), run, num_claims=4)
    v3._load_workflow_data = lambda uuid: _wf_info(GOAL)  # type: ignore[method-assign]
    v3.layers = [
        v3._claims_layer,
        ExecutionGateLayer(status_lookup=lambda u: ("crash", None)),
    ]
    result3 = v3.evaluate("u2")
    # u2 ties with u1 (identical scores) and beats u0 -> raw 0.75; the
    # crash status still floors the reward at 0.0
    assert result3["win_rate"] == 0.75 and result3["overall_score"] == 0.0
    assert result3["execution_gate"]["cap"] == 0.0




# ---------------------------------------------------------------------------
# E35 PRIME v3: configurable stage order (visual-first for figure tasks)
# ---------------------------------------------------------------------------


def test_stage_order_visual_first_configurable():
    ladder = [
        _claim("V1", stage="visual", tidx=1),
        _claim("S1", stage="script", tidx=1),
        _claim("R1", stage="result", tidx=1),
    ]
    a = {"V1": 0.1, "S1": 0.9, "R1": 1.0}  # fails the visual rung only
    b = {"V1": 0.9, "S1": 0.1, "R1": 0.0}  # fails everything else
    # figure-task order: the visual stage decides FIRST (E26 visual-early)
    w, detail = aggregation.temporal_winner(a, b, ladder, "temporal", aggregation.FIGURE_STAGES)
    assert w == -1 and detail["stage"] == "visual" and detail["claim_id"] == "V1"
    r = aggregation.win_rate_reward(a, [b], ladder, stages=aggregation.FIGURE_STAGES)
    assert r["losses"] == 1 and r["pairs"][0]["detail"]["stage"] == "visual"
    bt = aggregation.bradley_terry_reward(
        a, [b], ladder, mode="temporal", stages=aggregation.FIGURE_STAGES
    )
    assert bt["win_rate"] == 0.0
    # standard order: visual claims never decide; script does
    w2, detail2 = aggregation.temporal_winner(a, b, ladder, "temporal")
    assert w2 == 1 and detail2["stage"] == "script"
    # sorting honours the order too
    ordered = claims_mod.sort_claims_temporally(
        [ladder[2], ladder[0], ladder[1]], claims_mod.FIGURE_STAGES
    )
    assert [c["id"] for c in ordered] == ["V1", "S1", "R1"]
    assert [c["id"] for c in claims_mod.sort_claims_temporally(
        [ladder[2], ladder[0], ladder[1]], claims_mod.STAGES
    )] == ["S1", "R1", "V1"]
    assert claims_mod.FIGURE_STAGES == ("visual", "script", "log", "result")


# ---------------------------------------------------------------------------
# E35 PRIME v3: layer wiring + backward compatibility
# ---------------------------------------------------------------------------


def _v3_config(tmp_path: Path, **overrides):
    cfg = types.SimpleNamespace(
        memory_dir=str(tmp_path / "m"),
        workflow_dir=str(tmp_path / "w"),
        model_pricing={},
        reasoning_effort="low",
        judge_model="openai/gpt-4o-mini",
        max_tokens=2048,
        workspace_dir=str(tmp_path / "ws"),
        temp_dir=str(tmp_path / "t"),
        openrouter_provider_for=lambda m: None,
        openrouter_quantizations_for=lambda m: None,
        vision_judge_model="openrouter/moonshotai/kimi-k3",
        verifier_kind="hybrid",
    )
    for k, val in overrides.items():
        setattr(cfg, k, val)
    return cfg


def test_evaluator_layer_wiring_and_knobs(tmp_path: Path):
    from sources.core.llm_provider import LLMConfig

    v = HybridVerifierEvaluator(_v3_config(tmp_path), workspace_dir=tmp_path / "ws")
    assert [getattr(x, "name", "?") for x in v.layers] == [
        "visual", "claims", "execution_gate",
    ]
    assert v.visual_rung is True and v.execution_gate is True
    # vision transport: kimi-k3 at temperature 0 (E26 convention)
    assert isinstance(v._vision_llm_config, LLMConfig)
    assert v._vision_llm_config.temperature == 0.0
    assert v._vision_llm_config.model == "moonshotai/kimi-k3"
    # backward compatibility: both v3 knobs off -> legacy layer set
    v2 = HybridVerifierEvaluator(
        _v3_config(
            tmp_path / "b",
            hybrid_verifier_visual_rung=False,
            hybrid_verifier_execution_gate=False,
        ),
        workspace_dir=tmp_path / "ws2",
    )
    assert [getattr(x, "name", "?") for x in v2.layers] == ["claims"]
    assert v2.visual_rung is False and v2.execution_gate is False
    # a missing vision model still constructs; vision calls raise at use
    v3 = HybridVerifierEvaluator(
        _v3_config(tmp_path / "c", vision_judge_model=None),
        workspace_dir=tmp_path / "ws3",
    )
    assert v3._vision_llm_config is None


def test_evaluate_uses_figure_stage_order_for_figure_tasks(tmp_path: Path):
    """End-to-end: a figure-task workspace arms the visual rung, orders the
    ladder visual-first and renders visual evidence in the gradient."""
    from sources.evaluators.hybrid_verifier.layers import VisualEvidenceLayer

    ws = tmp_path / "ws"
    (ws / "pred_results").mkdir(parents=True)
    (ws / "pred_results" / "charge.png").write_bytes(_png_bytes())
    (ws / "pred_results" / "integral.png").write_bytes(_png_bytes(4, 4))
    figure_goal = (
        "Plot training loss and validation loss vs epoch to "
        "pred_results/curves.png."
    )
    kimi = _Kimi(
        [
            json.dumps({"claims": [
                {"id": "V1", "statement": "two loss curves present",
                 "rubric": "1.0 if both; 0.5 if one; 0.0 otherwise"},
                {"id": "V2", "statement": "epoch axis labeled",
                 "rubric": "1.0 if labeled"},
                {"id": "V3", "statement": "legend distinguishes the curves",
                 "rubric": "1.0 if legend"},
            ]}),
            json.dumps({"scores": {
                "V1": {"score": 1.0, "evidence": "both curves"},
                "V2": {"score": 1.0, "evidence": "epoch labeled"},
                "V3": {"score": 0.0, "evidence": "no legend"},
            }}),
        ]
    )
    claims = [_claim(f"C{i}") for i in range(1, 5)]
    llm = _LLM([])
    runner = _ScriptedRunner([])
    v = _make_evaluator(tmp_path, llm, runner, num_claims=4)
    v.workspace_dir = ws
    visual = VisualEvidenceLayer(vision_call=kimi, logger=v.logger)
    v.layers = [visual, v._claims_layer]
    reg = TaskRegistry.load(v._runner_temp_root, figure_goal, logger=v.logger)
    reg.seed_claims(claims)
    for c in reg.claims:
        c["scorer"] = _scorer_body(c["id"], 0.5)
    reg.save()
    v._load_workflow_data = lambda uuid: _wf_info(figure_goal)  # type: ignore[method-assign]
    result = v.evaluate("u1")
    assert result["figure_task"] is True
    assert result["stage_order"] == ["visual", "script", "log", "result"]
    assert result["execution_gate"] is None  # gate layer not wired here
    assert any(
        s["id"].startswith("V") for s in result["claim_summary"]
    )
    grad = result["abstracted_textual_gradient"]
    assert "## 4. Visual evidence" in grad and "no legend" in grad
    assert "## 3. Execution facts" in grad  # still rendered (no gate data)
    reg2 = TaskRegistry.load(v._runner_temp_root, figure_goal)
    assert {c["stage"] for c in reg2.claims} >= {"visual", "result"}


# --------------------------------------- (N3) exact-name contract screen ----


def test_scorer_prompt_demands_exact_name_matching():
    """N3: the scorer prompt must forbid substring/case-insensitive
    matching of exact column/file names and demand set equality, using
    ONLY generic (non-benchmark) example names."""
    claim = _claim(
        "C8",
        statement="the output CSV has exactly the columns smiles, "
        "FDA_APPROVED and CT_TOX",
        rule="fraction of the three exact column names present in the "
        "header (exact equality, no suffixes)",
    )
    prompt = scorers_mod.scorer_prompt(GOAL, claim, "pred_results/out.csv")
    assert "EXACT equality" in prompt
    assert 'set(df.columns) == {"gene_id", "expression_value", "p_value"}' in prompt
    assert "PRED_score" in prompt
    assert "does NOT satisfy" in prompt
    # no benchmark-derived example names leak into the prompt TEMPLATE
    # (the claim's own text above is task input, not a prompt example)
    template = prompt.replace(json.dumps(claim, indent=1), "<CLAIM>")
    assert "FDA_APPROVED" not in template and "CT_TOX" not in template
    # the repair prompt no longer coaches fuzzy matching for exact names
    repair = scorers_mod.repair_prompt("feedback", "previous code")
    assert "EXACT set equality" in repair
    assert "PRED_score is NOT PRED" in repair
    assert "FDA_APPROVED" not in repair


_CLINTOX_CLAIM = _claim(
    "C8",
    statement="the predictions CSV header contains exactly the columns "
    "smiles, FDA_APPROVED and CT_TOX",
    rule="fraction of required exact column names FDA_APPROVED, CT_TOX, "
    "smiles present in the CSV header (exact names, no suffix variants)",
    tidx=0,
)

_SUBSTRING_SCORER = """import json
import pandas as pd
df = pd.read_csv("pred_results/out.csv")
cols = [c.lower() for c in df.columns]
ok = sum(1 for name in ["smiles", "fda_approved", "ct_tox"]
         if any(name in c for c in cols))
print(json.dumps({"claim_id": "C8", "score": ok / 3, "evidence": "cols"}))
"""


_EXACT_SCORER = """import json
import pandas as pd
df = pd.read_csv("pred_results/out.csv")
expected = {"smiles", "FDA_APPROVED", "CT_TOX"}
score = 1.0 if set(df.columns) == expected else len(expected & set(df.columns)) / 3
print(json.dumps({"claim_id": "C8", "score": score, "evidence": "exact set"}))
"""


def test_exact_name_screen_is_one_batched_llm_call():
    """The exact-name screen is ONE batched LLM judgment over the task's
    claim/script pairs (never a regex): lenient verdicts come back keyed
    by claim id with the offending line, every pair rides the same call,
    unknown ids are ignored and a failed judge call degrades open."""
    screen = _ScreenLLM(
        lenient={
            "C8": (
                "any(name in c for c in cols)",
                "case-insensitive substring membership",
            )
        }
    )
    flagged = scorers_mod.exact_name_violations(
        [_CLINTOX_CLAIM], {"C8": _SUBSTRING_SCORER}, screen, uuid="u1"
    )
    assert flagged == {
        "C8": "case-insensitive substring membership — line: "
        "any(name in c for c in cols)"
    }
    assert screen.calls == ["hybrid_screen_exact_name"]  # one batched call
    # the prompt carried BOTH the claim's fields and the script body
    assert "EXACT-NAME SCORER SCREEN" in screen.prompts[0]
    assert "pred_results/out.csv" in screen.prompts[0]

    # every pair goes in the SAME call; clean pairs come back unflagged
    both = _ScreenLLM(
        lenient={"C8": ("x in c", "substring membership, never ==")}
    )
    flagged2 = scorers_mod.exact_name_violations(
        [_CLINTOX_CLAIM, _claim("C9")],
        {"C8": _SUBSTRING_SCORER, "C9": _EXACT_SCORER},
        both,
        uuid="u1",
    )
    assert flagged2 == {"C8": "substring membership, never == — line: x in c"}
    assert both.calls == ["hybrid_screen_exact_name"]

    # verdicts on ids outside the screened set are ignored
    bogus = _StaticLLM('[{"claim_id": "C99", "lenient": true, "reason": "rogue"}]')
    assert scorers_mod.exact_name_violations(
        [_CLINTOX_CLAIM], {"C8": _EXACT_SCORER}, bogus
    ) == {}
    # a judge that never produces JSON degrades open after one repair round
    garbage = _StaticLLM("not json at all")
    assert scorers_mod.exact_name_violations(
        [_CLINTOX_CLAIM], {"C8": _SUBSTRING_SCORER}, garbage
    ) == {}
    assert garbage.calls == 2


def test_exact_name_screen_skips_missing_scripts_structural_stays_pure():
    """A claim with no generated script is never screened (no judge call),
    and the structural static screen passes a lenient matcher untouched —
    content judgment is the LLM screen's job, never static analysis."""
    silent = _StaticLLM("[1]")
    assert scorers_mod.exact_name_violations([_CLINTOX_CLAIM], {}, silent) == {}
    assert silent.calls == 0
    assert scorers_mod.static_violations(_SUBSTRING_SCORER) == []


def test_build_scorer_regenerates_on_exact_name_violation(tmp_path: Path):
    """The screen rides the existing repair loop: attempt 1 (substring
    matcher) is rejected on the judge's lenient verdict, the repair prompt
    carries the exact-equality feedback, and attempt 2 (set equality) is
    kept."""
    llm = _LLM(
        [
            f"```python\n{_SUBSTRING_SCORER}\n```",
            json.dumps(
                [
                    {
                        "claim_id": "C8",
                        "lenient": True,
                        "line_evidence": "any(name in c for c in cols)",
                        "reason": "case-insensitive substring match",
                    }
                ]
            ),
            f"```python\n{_EXACT_SCORER}\n```",
            json.dumps(
                [
                    {
                        "claim_id": "C8",
                        "lenient": False,
                        "line_evidence": "",
                        "reason": "set equality",
                    }
                ]
            ),
        ]
    )
    runner = _ScriptedRunner(
        [scorers_mod.ExecOutcome(
            rc=0,
            stdout='{"claim_id": "C8", "score": 1.0, "evidence": "exact set"}',
        )]
    )
    v = _make_evaluator(tmp_path, llm, runner)
    reg = TaskRegistry.load(v._runner_temp_root, GOAL, logger=v.logger)
    reg.seed_claims([_CLINTOX_CLAIM])
    ctx = LayerContext(
        uuid="u1",
        goal=GOAL,
        workspace=v.workspace_dir,
        registry=reg,
        generation_index=0,
    )
    script, telemetry, parsed = v._claims_layer._build_scorer(ctx, _CLINTOX_CLAIM)
    assert parsed is not None and parsed["score"] == 1.0
    assert "set(df.columns) == expected" in script
    assert telemetry["attempts"][0]["parse"] == "static_violation"
    assert any(
        "lenient matching" in msg and "EXACT equality" in msg
        for msg in telemetry["attempts"][0]["static_violations"]
    )
    # the repair prompt fed back the exact-equality requirement
    repair_prompt = next(p for a, p in llm.prompts if "_repair" in a)
    assert "exact" in repair_prompt.lower() and "lenient" in repair_prompt.lower()
    assert runner.calls == 1  # only the repaired script ever executed

# ------------------------------------ (R4-fix) pruned persistent failures ----


def test_gradient_surfaces_pruned_persistent_failures():
    """R4-fix: a claim that scored a CONSTANT FAILING value (0.3 for three
    generations) is zero-variance-pruned and never reaches the surviving
    claim loop — it must still surface in section 1b with the pruned
    annotation, otherwise the workflow stops hearing about it."""
    from sources.evaluators.hybrid_verifier import gradient as g

    dead = dict(
        _claim("C13", stage="result", tidx=0),
        state="dead",
        drop_reason="zero_variance",
    )
    claims = [dead, _claim("R1", stage="result", tidx=1)]
    pair_records = [
        {"prev_uuid": f"u{i}", "outcome": "tie",
         "prev_scores": {"C13": 0.3, "R1": 0.9}, "d": 0, "dm": 0.0}
        for i in range(3)
    ]
    text = g.build_gradient(
        uuid="u3",
        goal=GOAL,
        now_scores={"C13": 0.3, "R1": 0.9},
        evidence={"C13": "AUC undefined on degenerate labels", "R1": "ok"},
        claims=claims,
        surviving=["R1"],  # C13 was pruned before the gradient saw it
        pair_records=pair_records,
        reward=0.5,
        win_rate=0.5,
        mean_score=0.6,
        dead_claims=[{**dead, "drop_reason": "zero_variance"}],
    )
    block = text.split("## 1b.")[1].split("\n", 1)[1].split("## 2.")[0]
    assert "[C13]" in block
    assert "failing for 4 consecutive generations" in block
    assert "pruned as zero-variance — was failing constantly before pruning" in block
    assert "AUC undefined on degenerate labels" in block
    assert "[R1]" not in block  # passing claim, pruned or not, stays out

    # a constant-PASSING pruned claim (the clintox C8 false positive) is a
    # scorer bug, not a persistent failure — it must NOT enter section 1b
    false_positive = dict(
        _claim("C8", stage="result", tidx=0),
        state="dead",
        drop_reason="zero_variance",
    )
    passing_records = [
        {"prev_uuid": f"u{i}", "outcome": "tie",
         "prev_scores": {"C8": 1.0}, "d": 0, "dm": 0.0}
        for i in range(3)
    ]
    fp_text = g.build_gradient(
        uuid="u3", goal=GOAL,
        now_scores={"C8": 1.0}, evidence={"C8": "lenient header match"},
        claims=[false_positive], surviving=[],
        pair_records=passing_records, reward=1.0, win_rate=1.0,
        mean_score=1.0, dead_claims=[false_positive],
    )
    fp_block = fp_text.split("## 1b.")[1].split("\n", 1)[1].split("## 2.")[0]
    assert fp_block.strip() == "(none)"


def test_gradient_pruned_claim_with_crashed_scorer_does_not_crash():
    """A pruned (zero-variance) claim whose scorer crashed this generation
    has NO measured score, so "never passed" cannot be asserted for it: the
    "None disqualifies" rule must cover pruned claims too.

    Regression (clintox, 2026-09-28): the pruned branch formatted the None
    score with "{s:.3f}" and raised
    TypeError: unsupported format string passed to NoneType.__format__,
    which aborted the whole ScienceAgentBench evaluation row.
    """
    from sources.evaluators.hybrid_verifier import gradient as g

    dead = dict(
        _claim("C16", stage="result", tidx=0),
        state="dead",
        drop_reason="zero_variance",
    )
    pair_records = [
        {"prev_uuid": f"u{i}", "outcome": "tie",
         "prev_scores": {"C16": 0.2}, "d": 0, "dm": 0.0}
        for i in range(2)
    ]
    text = g.build_gradient(
        uuid="u2",
        goal=GOAL,
        now_scores={"C16": None},  # scorer crashed this generation
        evidence={"C16": "scorer raised: boom"},
        claims=[dead],
        surviving=[],
        pair_records=pair_records,
        reward=0.0,
        win_rate=0.0,
        mean_score=0.0,
        dead_claims=[dead],
    )
    block = text.split("## 1b.")[1].split("\n", 1)[1].split("## 2.")[0]
    assert block.strip() == "(none)"
    assert "[C16]" not in block
    # the claim is still reported as unscored in the all-claims transcript
    assert "NOT SCORED" in text


def test_registry_exposes_dead_claim_score_history(tmp_path: Path):
    """The registry keeps a pruned claim's per-generation measurements —
    the gradient's persistent-failure verdict for dead claims depends on
    them surviving the pruning."""
    reg = TaskRegistry.load(tmp_path, GOAL)
    reg.seed_claims([_claim("C13")])
    for i, uuid in enumerate(["u0", "u1", "u2"]):
        reg.record_generation(
            uuid, {"C13": 0.3}, {"C13": "measured 0.3"}, reward=0.3
        )
    reg.mark_dead("C13", "zero_variance")
    reg.save()

    reloaded = TaskRegistry.load(tmp_path, GOAL)
    history = reloaded.claim_score_history("C13")
    assert [h["score"] for h in history] == [0.3, 0.3, 0.3]
    assert [h["uuid"] for h in history] == ["u0", "u1", "u2"]
    assert all(h["evidence"] == "measured 0.3" for h in history)
    assert reloaded.claim_score_history("C13", exclude_uuid="u1") == [
        history[0], history[2]
    ]
    # crashed scorers are not measurements — same rule as observed_scores
    reg.record_generation("u3", {"C13": 0.0}, {"C13": "scorer raised: boom"})
    assert len(reg.claim_score_history("C13")) == 3


# ------------------------------- (N9) unmeasured-workspace degradation ----


class _AlwaysFailRunner:
    """Executor mock: every scorer build/execution attempt crashes."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, ws, code, eid):
        self.calls += 1
        return scorers_mod.ExecOutcome(rc=1, stdout="", stderr="Traceback: boom")


class _ExtractOnlyLLM:
    """Extraction returns 8 claims; every screen passes; digests soft-fail."""

    def __call__(self, uuid: str, agent: str, prompt: str) -> str:
        if (
            "GENERIC-CLAIM SCREEN" in prompt
            or "SMUGGLED-ANSWER SCREEN" in prompt
            or "EXACT-NAME SCORER SCREEN" in prompt
        ):
            return "[]"
        if "verification rubric" in prompt:
            return _claims_json(8)
        if "FORMAT DIGESTS" in prompt:
            return json.dumps({"digests": []})
        raise AssertionError(f"unexpected LLM call: {agent}")


def test_unmeasured_artifacts_get_neutral_prior_not_zero(tmp_path: Path):
    """N9 (E41 phonon gen5): every scorer fails on a workspace that produced
    real artifacts -> the reward is a flagged neutral prior, not a fake 0.0
    that buries possibly-SR-true generations."""
    _seed_workspace(tmp_path / "workspace")
    v = _make_evaluator(tmp_path, _ExtractOnlyLLM(), _AlwaysFailRunner())
    v._load_workflow_data = lambda uuid: _wf_info(GOAL)  # type: ignore[method-assign]

    result = v.evaluate("u1")
    assert result["n_scored"] == 0
    assert result["overall_score"] == pytest.approx(0.5)
    assert result["reward_fallback"] == "unmeasured_prior"
    assert result["skipped_reason"] == "all_scorers_failed"
    assert "MEASUREMENT FAILURE" in result["abstracted_textual_gradient"]
    state = json.loads((v.workflow_dir / "u1" / "state_result.json").read_text())
    assert state["evaluation"]["verifier"]["reward_fallback"] == "unmeasured_prior"


def test_unmeasured_empty_workspace_stays_zero(tmp_path: Path):
    """Empty workspace -> nothing was produced; 0.0 stays the honest reward."""
    v = _make_evaluator(tmp_path, _ExtractOnlyLLM(), _AlwaysFailRunner())
    v._load_workflow_data = lambda uuid: _wf_info(GOAL)  # type: ignore[method-assign]

    result = v.evaluate("u1")
    assert result["n_scored"] == 0
    assert result["overall_score"] == 0.0
    assert result["reward_fallback"] == "mean_claim"
    assert "MEASUREMENT FAILURE" not in result["abstracted_textual_gradient"]


def test_gate_cap_survives_unmeasured_workspace(tmp_path: Path):
    """E24 crash cap + all scorers failed -> the cap's verdict stands (the
    re-execution DID measure the crash); no unmeasured_prior marker."""
    _seed_workspace(tmp_path / "workspace")
    v = _make_evaluator(tmp_path, _ExtractOnlyLLM(), _AlwaysFailRunner())
    v._load_workflow_data = lambda uuid: _wf_info(GOAL)  # type: ignore[method-assign]

    class _CrashGate:
        name = "crash-gate"
        last_facts = {"status": "crash", "runtime_s": None, "cap": 0.0}

        def collect(self, context):
            return []

        def gate(self, context, collected):
            return 0.0

    v.layers = [v._claims_layer, _CrashGate()]
    result = v.evaluate("u1")
    assert result["n_scored"] == 0
    assert result["overall_score"] == 0.0
    assert result["reward_fallback"] != "unmeasured_prior"


def test_gradient_unmeasured_note_renders_conditionally():
    from sources.evaluators.hybrid_verifier.gradient import build_gradient

    kwargs = dict(
        uuid="u1",
        goal="g",
        now_scores={"C1": None},
        evidence={"C1": ""},
        claims=[_claim("C1")],
        surviving=[],
        pair_records=[],
        reward=0.5,
        win_rate=0.5,
        mean_score=0.0,
        dead_claims=[],
    )
    assert "MEASUREMENT FAILURE" not in build_gradient(**kwargs)
    noted = build_gradient(**kwargs, unmeasured=True)
    assert "MEASUREMENT FAILURE" in noted
    assert "NEUTRAL PRIOR" in noted
