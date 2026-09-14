"""Verifier error-recovery loop, pass-despite-error guard, last-resort scoring.

Covers the recovery change-set with mocked LLM/sandbox calls only:
- ``_run_verifier_with_recovery`` iterates up to ``_RECOVERY_MAX_ATTEMPTS``
  executions, stops early on a trustworthy pass/fail, gives up after the
  cap, accumulates feedback history, and never re-vets the same package;
- ``_demote_pass_despite_error`` distrusts printed verdicts that ship with
  hard exception markers (stderr crash, or pass + load-failure salvage in
  details) while leaving quoted-target failures and honest fails alone;
- ``_aggregate`` scores post-recovery executable errors as 0.0 (numerator
  AND denominator) like visual errors, keeps ``n_error`` telemetry, adds
  ``n_error_scored_zero`` / ``n_recovery_attempts``, and still never fires
  the hard-fail cap on errors;
- the generation/regen prompts carry the NO-SALVAGE-ON-LOAD-FAILURE rule
  and the unavailable-package instruction.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(_REPO_ROOT))

# Prime the import chain (verifier <-> cli circular import), same as the
# other verifier tests in this suite.
from sources.core.failure_fingerprint import (  # noqa: E402
    compute_failure_fingerprint,  # noqa: F401
)
from sources.evaluators import verifier as verifier_mod  # noqa: E402
from sources.evaluators import verifier_per_claim as vpc  # noqa: E402
from sources.evaluators.verifier import VerifierEvaluator  # noqa: E402
from sources.evaluators.verifier_per_claim import (  # noqa: E402
    RECOVERY_PROMPT_RULES,
    VERIFIER_PROMPT_RULES,
    _VerifierPerClaimMixin,
)

# ---------- helpers ----------------------------------------------------------


def _make_evaluator(tmp_path: Path) -> VerifierEvaluator:
    """Build a VerifierEvaluator with paths rooted at tmp_path; LLMs mocked."""
    v = VerifierEvaluator.__new__(VerifierEvaluator)
    v.workspace_dir = tmp_path / "workspace"
    v.workspace_dir.mkdir(parents=True, exist_ok=True)
    v._runner_temp_root = tmp_path / "verifier_tmp"
    v._runner_temp_root.mkdir(parents=True, exist_ok=True)
    v.workflow_dir = tmp_path / "workflows"
    v.workflow_dir.mkdir(parents=True, exist_ok=True)
    v.logger = logging.getLogger("test-verifier-recovery")
    v._DEFAULT_CLAIM_IMPORTANCE = 5
    v._HARD_FAIL_IMPORTANCE = VerifierEvaluator._HARD_FAIL_IMPORTANCE
    v.hard_fail_cap = verifier_mod._HARD_FAIL_CAP
    v._workspace_files = set()
    v.preview_per_claim_cap = 2000
    return v


def _claim(cid: str = "boom", importance: int = 5) -> dict:
    return {
        "id": cid,
        "importance": importance,
        "description": f"description of {cid}",
        "likely_relevant_files": [],
    }


def _exec(status: str, details: str = "", stderr: str = "") -> dict:
    return {
        "status": status,
        "actual": None,
        "details": details,
        "raw_stdout": "",
        "raw_stderr": stderr,
        "exit_status": "ok",
    }


def _scored(cid: str, importance: int, status: str, kind: str = "executable",
            score: float | None = None, attempts: int | None = None) -> dict:
    return {
        "claim": {"id": cid, "importance": importance, "likely_relevant_files": []},
        "status": status,
        "score": (1.0 if status == "pass" else 0.0) if score is None else score,
        "verifier_kind": kind,
        "details": "",
        **({"recovery_attempts": attempts} if attempts is not None else {}),
    }


# ---------- (1) aggregation: last-resort scoring -----------------------------


def test_exhausted_executable_error_scores_zero_with_telemetry(tmp_path: Path):
    """Post-loop errors enter the weighted mean as 0, with new counters."""
    v = _make_evaluator(tmp_path)
    result = v._aggregate([
        _scored("ok", 10, "pass", attempts=1),
        _scored("boom", 10, "error", attempts=vpc._RECOVERY_MAX_ATTEMPTS),
    ])
    assert result["overall_score"] == pytest.approx(0.5)
    assert result["n_error"] == 1
    assert result["n_error_scored_zero"] == 1
    assert result["n_recovery_attempts"] == 1 + vpc._RECOVERY_MAX_ATTEMPTS
    assert result["n_scored"] == 2


def test_hard_fail_cap_still_ignores_zero_scored_errors(tmp_path: Path):
    """Zero-scored errors are measurement failures, not refutations: no cap."""
    v = _make_evaluator(tmp_path)
    result = v._aggregate([
        _scored("must_have", 10, "error"),
        _scored("ok", 5, "pass"),
    ])
    assert result["hard_fail_capped"] is False
    # (0*10 + 1*5) / (10+5): the error drags the mean down but caps nothing.
    assert result["overall_score"] == pytest.approx(1 / 3, abs=1e-3)


def test_all_claims_errored_is_zero_not_skipped(tmp_path: Path):
    """Every claim erroring yields overall 0.0, not the old skip marker."""
    v = _make_evaluator(tmp_path)
    result = v._aggregate([
        _scored("a", 10, "error"),
        _scored("b", 8, "error"),
    ])
    assert result["overall_score"] == 0.0
    assert "skipped_reason" not in result
    assert result["n_error_scored_zero"] == 2


# ---------- (2) the retry loop ------------------------------------------------


class _RunScript:
    """Scripted ``_run_verifier`` stand-in returning queued results."""

    def __init__(self, results: list[dict], record: list[str]) -> None:
        self.results = list(results)
        self.record = record

    def __call__(self, uuid: str, claim_id: str, code: str) -> dict:
        self.record.append(code)
        if not self.results:
            raise AssertionError("unexpected extra _run_verifier call")
        return self.results.pop(0)


def test_recovery_stops_on_first_trustworthy_pass(tmp_path: Path):
    """Two crashes then a pass: loop stops, attempts + actions recorded."""
    v = _make_evaluator(tmp_path)
    ran: list[str] = []
    v._run_verifier = _RunScript([
        _exec("error", details="boom", stderr="TypeError: bad slice"),
        _exec("error", details="boom again", stderr="IndexError: out of range"),
        _exec("pass", details="checked"),
    ], ran)
    regens: list[dict] = []

    def regen(uuid, claim, prev_spec, exec_result, **kw):
        regens.append(kw)
        return {"executable": True, "code": f"regen#{len(regens)}"}

    v._regenerate_verifier_with_feedback = regen  # type: ignore[method-assign]

    spec, result = v._run_verifier_with_recovery("u", _claim(), {"executable": True, "code": "v0"})

    assert result["status"] == "pass"
    assert result["recovery_attempts"] == 3
    assert len(regens) == 2
    assert [c for c in ran] == ["v0", "regen#1", "regen#2"]
    assert result["recovery_exhausted"] is False
    assert any("regenerated script" in a for a in result["recovery_actions"])


def test_recovery_gives_up_after_attempt_cap(tmp_path: Path):
    """Persistent code bugs stop at ``_RECOVERY_MAX_ATTEMPTS`` executions."""
    v = _make_evaluator(tmp_path)
    n_runs = {"n": 0}

    def always_error(uuid, claim_id, code):
        n_runs["n"] += 1
        return _exec("error", details="still broken", stderr="ValueError: nope")

    v._run_verifier = always_error  # type: ignore[method-assign]

    def regen(uuid, claim, prev_spec, exec_result, **kw):
        return {"executable": True, "code": f"regen{n_runs['n']}"}

    v._regenerate_verifier_with_feedback = regen  # type: ignore[method-assign]

    spec, result = v._run_verifier_with_recovery("u", _claim(), {"executable": True, "code": "v0"})

    assert n_runs["n"] == vpc._RECOVERY_MAX_ATTEMPTS
    assert result["status"] == "error"
    assert result["recovery_attempts"] == vpc._RECOVERY_MAX_ATTEMPTS
    assert result["recovery_exhausted"] is True
    assert "recovery loop exhausted" in result["details"]
    assert any("exhausted" in a for a in result["recovery_actions"])


def test_recovery_accumulates_history_in_regen_prompt(tmp_path: Path):
    """Each regen sees the failure history (kind + action) of prior attempts."""
    v = _make_evaluator(tmp_path)
    v._run_verifier = _RunScript([
        _exec("error", details="d1", stderr="NameError: x"),
        _exec("error", details="d2", stderr="TypeError: y"),
        _exec("pass"),
    ], [])
    captured: list[dict] = []

    def regen(uuid, claim, prev_spec, exec_result, **kw):
        captured.append(kw)
        return {"executable": True, "code": f"r{len(captured)}"}

    v._regenerate_verifier_with_feedback = regen  # type: ignore[method-assign]

    v._run_verifier_with_recovery("u", _claim(), {"executable": True, "code": "v0"})

    assert len(captured) == 2
    second = captured[1]
    assert second["history"] is not None
    assert [h["kind"] for h in second["history"]] == ["code_bug", "code_bug"]
    assert second["history"][0]["action"] == "regenerated script with failure feedback"


def test_import_failure_vets_once_and_never_reinstalls(tmp_path: Path):
    """Same missing module: one LLM vet, one failed install, no retries."""
    v = _make_evaluator(tmp_path)
    vet_calls = {"n": 0}
    installs: list[list[str]] = []
    v._run_verifier = _RunScript([
        _exec("error", stderr="Traceback (most recent call last):\nModuleNotFoundError: No module named 'rdkit'"),
        _exec("error", stderr="ModuleNotFoundError: No module named 'rdkit'"),
        _exec("error", stderr="ModuleNotFoundError: No module named 'rdkit'"),
        _exec("pass", details="rewritten without rdkit"),
    ], [])

    def vet(uuid, claim, stderr):
        vet_calls["n"] += 1
        return ["rdkit"]

    def install(packages):
        installs.append(list(packages))
        return False  # install always fails

    v._llm_packages_needed_for_claim = vet  # type: ignore[method-assign]
    v._sandbox_install_packages = install  # type: ignore[method-assign]

    regens: list[dict] = []

    def regen(uuid, claim, prev_spec, exec_result, **kw):
        regens.append(kw)
        return {"executable": True, "code": f"r{len(regens)}"}

    v._regenerate_verifier_with_feedback = regen  # type: ignore[method-assign]

    spec, result = v._run_verifier_with_recovery("u", _claim(), {"executable": True, "code": "v0"})

    assert vet_calls["n"] == 1, "the same missing module must be vetted once"
    assert installs == [["rdkit"]], "a failed install must not be retried"
    assert result["status"] == "pass"
    # Every regen after the failed install names rdkit as unavailable.
    assert all("rdkit" in r["unavailable_modules"] for r in regens)
    assert regens[0]["unavailable_modules"] == ["rdkit"]


def test_successful_install_extends_available_imports(tmp_path: Path):
    """An installed package is offered to the next regen prompt."""
    v = _make_evaluator(tmp_path)
    v._run_verifier = _RunScript([
        _exec("error", stderr="ModuleNotFoundError: No module named 'pint'"),
        _exec("error", stderr="TypeError: still bad"),
        _exec("pass"),
    ], [])
    v._llm_packages_needed_for_claim = lambda *a, **k: ["pint"]  # type: ignore[assignment]
    v._sandbox_install_packages = lambda pkgs: True  # type: ignore[method-assign]
    regens: list[dict] = []

    def regen(uuid, claim, prev_spec, exec_result, **kw):
        regens.append(kw)
        return {"executable": True, "code": f"r{len(regens)}"}

    v._regenerate_verifier_with_feedback = regen  # type: ignore[method-assign]

    v._run_verifier_with_recovery("u", _claim(), {"executable": True, "code": "v0"})

    assert regens[0]["extra_available"] == ["pint"]
    assert regens[0]["unavailable_modules"] == []


# ---------- (3) pass-despite-error guard --------------------------------------


def test_guard_demotes_pass_with_import_marker_in_details():
    """The rdkit salvage: pass printed after catching a load failure."""
    result = _VerifierPerClaimMixin._demote_pass_despite_error(
        _exec("pass", details="direct unpickle failed "
              "(ModuleNotFoundError: No module named 'rdkit'); "
              "pickle GLOBAL refs=['rdkit.Chem.rdchem.Mol']")
    )
    assert result["status"] == "error"
    assert "not trusted" in result["details"]


def test_guard_demotes_stderr_crash_even_when_status_fail():
    """Verifier's own traceback in stderr invalidates a printed fail too."""
    result = _VerifierPerClaimMixin._demote_pass_despite_error(
        _exec("fail", details="",
              stderr="Traceback (most recent call last):\n"
                     "  File \"verify_x.py\", line 3, in <module>\n"
                     "FileNotFoundError: missing.json")
    )
    assert result["status"] == "error"


def test_guard_ignores_quoted_target_traceback_in_details():
    """A check quoting the TARGET script's crash in details is legitimate."""
    result = _VerifierPerClaimMixin._demote_pass_despite_error(
        _exec("fail",
              details="Script exited with code 1: Traceback (most recent call last):",
              stderr="Traceback (most recent call last):\nValueError: boom")
    )
    assert result["status"] == "fail"


def test_guard_ignores_honest_fail_with_import_marker():
    """A fail that reports a genuinely missing package is a legitimate fail."""
    result = _VerifierPerClaimMixin._demote_pass_despite_error(
        _exec("fail", details="importlib.util.find_spec('rdkit') returned None")
    )
    assert result["status"] == "fail"


def test_guard_leaves_clean_results_untouched():
    clean = _exec("pass", details="orange_pixels=163419")
    assert _VerifierPerClaimMixin._demote_pass_despite_error(clean) is clean
    clean_fail = _exec("fail", details="median mismatch")
    assert _VerifierPerClaimMixin._demote_pass_despite_error(clean_fail) is clean_fail


def test_scripted_pass_with_module_error_in_stderr_retried_then_zero(tmp_path: Path):
    """End-to-end: persistent salvage-pass becomes error and scores 0."""
    v = _make_evaluator(tmp_path)
    salvage = _exec(
        "pass",
        details="pickle GLOBAL refs=['rdkit.Chem.rdchem.Mol']",
        stderr="ModuleNotFoundError: No module named 'rdkit'",
    )
    v._run_verifier = _RunScript([salvage] * vpc._RECOVERY_MAX_ATTEMPTS, [])
    v._llm_packages_needed_for_claim = lambda *a, **k: []  # type: ignore[assignment]
    regens: list[dict] = []

    def regen(uuid, claim, prev_spec, exec_result, **kw):
        regens.append(kw)
        return {"executable": True, "code": f"r{len(regens)}"}

    v._regenerate_verifier_with_feedback = regen  # type: ignore[method-assign]

    spec, result = v._run_verifier_with_recovery("u", _claim(), {"executable": True, "code": "v0"})

    assert result["status"] == "error"
    assert result["recovery_exhausted"] is True
    assert any("demoted" in a for a in result["recovery_actions"])
    scored = v._score_executable(_claim(), spec, result)
    assert scored["score"] == 0.0
    aggregate = v._aggregate([_scored("boom", 10, "error")])
    assert aggregate["overall_score"] == 0.0


# ---------- (4) prompt rules ---------------------------------------------------


def test_verifier_rules_carry_no_salvage_rule():
    """Both rule blocks forbid metadata salvage on load failure."""
    squashed_rules = " ".join(VERIFIER_PROMPT_RULES.split())
    assert "NO SALVAGE ON LOAD FAILURE" in squashed_rules
    assert "pickle GLOBAL-opcode inspection" in squashed_rules
    squashed_recovery = " ".join(RECOVERY_PROMPT_RULES.split())
    assert "NO SALVAGE ON LOAD FAILURE" in squashed_recovery
    assert "FAILURE HISTORY" in squashed_recovery


def test_generation_prompt_embeds_no_salvage_rule(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    prompt = v._build_verifier_prompt(_claim(), "file.csv\t10B", "goal text")
    assert "NO SALVAGE ON LOAD FAILURE" in prompt


def test_regen_prompt_names_unavailable_packages_and_history(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    history = [
        {"attempt": 1, "kind": "import",
         "stderr_tail": "ModuleNotFoundError: No module named 'rdkit'",
         "action": "install failed; marked unavailable"},
        {"attempt": 2, "kind": "code_bug", "stderr_tail": "TypeError: x",
         "action": "regenerated script with failure feedback"},
    ]
    prompt = v._build_regen_prompt(
        _claim(), {"executable": True, "code": "old"},
        _exec("error", details="d", stderr="ModuleNotFoundError: No module named 'rdkit'"),
        history=history,
        unavailable_modules=["rdkit"],
        extra_available=["biopython"],
    )
    assert "UNAVAILABLE PACKAGES: rdkit" in prompt
    assert "Do NOT import them" in " ".join(prompt.split())
    assert "FAILURE HISTORY" in prompt
    assert "attempt 1: failed as import" in prompt
    assert "biopython" in prompt  # extra available appended to allow-list
