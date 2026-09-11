"""Change-set A guards: result-focused claim rules, no source parsing.

Regression tests for the result-focused-claims change-set:
- the check-writer rules no longer instruct AST / source parsing;
- the reproducibility claim source (Source D) is no longer registered;
- the file-selector fallback never returns code files.
"""

from __future__ import annotations

import inspect
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(_REPO_ROOT))

# Prime the import chain the same way tests/verifier_claim_cache_test.py does
# (failure_fingerprint implicitly resolves the verifier <-> cli circular import).
from sources.core.failure_fingerprint import compute_failure_fingerprint  # noqa: E402, F401
from sources.evaluators import verifier_claim_sources  # noqa: E402
from sources.evaluators import verifier_per_claim  # noqa: E402
from sources.evaluators.verifier_claim_sources import SOURCES  # noqa: E402
from sources.evaluators.verifier_per_claim import (  # noqa: E402
    VERIFIER_PROMPT_RULES,
    _VerifierPerClaimMixin,
)


def test_verifier_rules_have_no_ast_parsing_instructions():
    """The check-writer rules must not teach AST / source parsing."""
    banned_phrases = (
        "parse the script",
        "ast.walk",
        "ast.Call",
        "ast.ImportFrom",
        "stdlib ``ast``",
        "walk the AST",
    )
    for phrase in banned_phrases:
        assert phrase not in VERIFIER_PROMPT_RULES, f"banned phrase present: {phrase!r}"
    # The replacement evidence rule is present.
    assert "RESULT artefacts" in VERIFIER_PROMPT_RULES
    assert "never fall back to source parsing" in VERIFIER_PROMPT_RULES


def test_verifier_language_block_bans_source_parsing():
    """The language block in the verifier prompt bans source parsing."""
    source = inspect.getsource(
        _VerifierPerClaimMixin._build_verifier_prompt
    )
    assert "parse workflow scripts with the stdlib" not in source
    assert "not with `ast`, not with regex" in source


def test_claim_source_d_not_registered():
    """The reproducibility source (D) is removed from the registry."""
    labels = [s.label for s in SOURCES]
    assert "d" not in labels
    assert not hasattr(verifier_claim_sources, "_build_source_d")
    assert len(SOURCES) == 5
    # Every remaining source still renders the shared rules block.
    ctx = verifier_claim_sources.ClaimContext(
        goal="dummy goal",
        workspace_listing="(no files)",
        target_min=1,
        target_max=2,
    )
    for s in SOURCES:
        assert "<short_slug>" in s.build(ctx), f"source {s.label} missing rules block"


def test_claim_rules_block_forbids_source_files():
    """likely_relevant_files must point at RESULT artefacts, never source."""
    rules = verifier_claim_sources._CLAIM_RULES_BLOCK
    assert "Do NOT list workflow source files" in rules
    assert "a claim only verifiable by reading source code" in rules


def test_verifier_select_files_fallback_excludes_code(tmp_path):
    """Empty/error selections must never fall back to code files."""
    for name in ("solution.py", "src/run.R", "pred_results/out.csv", "report.md"):
        f = tmp_path / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")

    v = _VerifierPerClaimMixin.__new__(_VerifierPerClaimMixin)
    v.workspace_dir = tmp_path
    v.logger = logging.getLogger("test-verifier")
    eligible = ["solution.py", "src/run.R", "pred_results/out.csv", "report.md"]
    v._eligible_workspace_files = lambda: list(eligible)  # type: ignore[assignment]
    # Judge errors out -> fallback, not "everything".
    v._call_judge_for_json = lambda *a, **k: (None, "judge exploded")  # type: ignore[assignment]
    picked = v._llm_select_files("uuid1", {"id": "c1"}, "narration")
    assert picked == ["pred_results/out.csv", "report.md"]

    # Judge returns an empty pick -> fallback again, still no code files.
    # (_validate_workspace_paths comes from the workspace mixin at runtime.)
    from sources.evaluators.verifier_workspace import _VerifierWorkspaceMixin  # noqa: E402

    v._validate_workspace_paths = (  # type: ignore[assignment]
        _VerifierWorkspaceMixin._validate_workspace_paths.__get__(v)
    )
    v._call_judge_for_json = lambda *a, **k: ({"files": []}, None)  # type: ignore[assignment]
    picked = v._llm_select_files("uuid2", {"id": "c2"}, "narration")
    assert picked == ["pred_results/out.csv", "report.md"]

    # Only code files eligible -> empty fallback, never code.
    v._eligible_workspace_files = lambda: ["solution.py", "src/run.R"]  # type: ignore[assignment]
    picked = v._llm_select_files("uuid3", {"id": "c3"}, "narration")
    assert picked == []


def test_verifier_rules_forbid_data_object_substitution():
    """No-substitution amendment: never stand in a different data object.

    The check-writer must treat a claim as NOT executable when the specific
    object the claim refers to (fitted training data, intermediate table,
    fitted model) is absent from the workspace — never approximate it with
    a raw input file or another stand-in (DILI raw-vs-fitted false fail).
    """
    head = "NEVER substitute a different data object for a claim's target"
    tail = "do not approximate with a raw input file or any other stand-in"
    # Present in the shared rules constant (whitespace-squashed: the prompt
    # text is hard-wrapped across lines).
    squashed_rules = " ".join(VERIFIER_PROMPT_RULES.split())
    assert head in squashed_rules
    assert tail in squashed_rules
    # Mirrored in the language block of the verifier-generation prompt.
    lang_block = inspect.getsource(_VerifierPerClaimMixin._build_verifier_prompt)
    squashed_lang = " ".join(lang_block.split())
    assert head in squashed_lang
    assert tail in squashed_lang
    # The regen/recovery prompt re-embeds the shared rules constant, so it
    # inherits the no-substitution rule automatically.
    regen_src = inspect.getsource(_VerifierPerClaimMixin._build_regen_prompt)
    assert "{VERIFIER_PROMPT_RULES}" in regen_src
