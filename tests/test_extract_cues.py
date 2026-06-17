"""Tests for ``_extract_cues`` in ``grounding``.

Covers the four cue-extraction cases the literature retriever depends on so
that dataset-canonical papers surface alongside generic background.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(_REPO_ROOT))

from sources.core.evaluators.grounding import _extract_cues  # noqa: E402


def test_csv_preview_block_yields_header_and_identifiers() -> None:
    """A goal containing a CSV preview block surfaces header + caps tokens."""
    goal = (
        "Build a model and save predictions. Preview:\n"
        "smiles,FDA_APPROVED,CT_TOX\n"
        "CCO,1,0\n"
    )
    cues = _extract_cues(goal)
    assert cues["header_line"] == ["smiles,FDA_APPROVED,CT_TOX"]
    assert "FDA_APPROVED" in cues["identifiers"]
    assert "CT_TOX" in cues["identifiers"]


def test_goal_with_no_preview_yields_empty_header() -> None:
    """An identifier-free, preview-free goal produces empty cue lists."""
    goal = "Write a program that computes the mean of a list of numbers."
    cues = _extract_cues(goal)
    assert cues["header_line"] == []
    assert cues["identifiers"] == []


def test_truncated_preview_keeps_visible_columns() -> None:
    """A CSV-like header followed by ``...`` is still surfaced verbatim."""
    goal = (
        "Output a file with this header:\n"
        "smiles,FDA_APPROVED,CT_TOX,...\n"
        "(rest truncated)\n"
    )
    cues = _extract_cues(goal)
    assert cues["header_line"] == ["smiles,FDA_APPROVED,CT_TOX,..."]
    assert "FDA_APPROVED" in cues["identifiers"]


def test_mixed_case_identifiers_are_extracted_in_first_seen_order() -> None:
    """Backticked, all-caps, and CamelCase identifiers all surface once each."""
    goal = (
        "Use `RandomForestRegressor` on the SMILES column and the "
        "FDA_APPROVED label. Compare to RandomForestRegressor without ECFP."
    )
    cues = _extract_cues(goal)
    ids = cues["identifiers"]
    assert "RandomForestRegressor" in ids
    assert "SMILES" in ids
    assert "FDA_APPROVED" in ids
    assert "ECFP" in ids
    assert ids.count("RandomForestRegressor") == 1


if __name__ == "__main__":
    test_csv_preview_block_yields_header_and_identifiers()
    test_goal_with_no_preview_yields_empty_header()
    test_truncated_preview_keeps_visible_columns()
    test_mixed_case_identifiers_are_extracted_in_first_seen_order()
    print("test_extract_cues: ok")
