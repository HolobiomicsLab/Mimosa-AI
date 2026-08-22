"""Grounding must be observable, not assumed.

Every failure path in ``query_perspicacite`` returns ``None`` and each caller
substitutes a "no relevant scientific context" string. A run in which every
grounding call failed therefore produces byte-identical artifacts to a fully
grounded one. ``grounding_stats()`` is what tells the two apart, and it is
folded into ``run_metrics.json``.
"""

import pytest

from sources.utils import perspicacite_client as pc


@pytest.fixture(autouse=True)
def _clean_ledger():
    pc.reset_grounding_stats()
    before = dict(pc._SETTINGS)
    yield
    pc.reset_grounding_stats()
    pc._SETTINGS.update(before)


def test_no_attempts_reports_none_rather_than_a_fake_perfect_score():
    stats = pc.grounding_stats()
    assert stats["attempts"] == 0
    assert stats["grounded"] == 0
    assert stats["hit_rate"] is None


def test_all_failures_are_distinguishable_from_success(monkeypatch):
    monkeypatch.setattr(pc, "_read_cache", lambda *a, **k: None)
    monkeypatch.setattr(pc, "_query_perspicacite_streaming", lambda *a, **k: None)
    monkeypatch.setattr(pc, "_query_perspicacite_non_streaming", lambda *a, **k: None)

    assert pc.query_perspicacite("some goal") is None

    stats = pc.grounding_stats()
    assert stats["attempts"] == 1
    assert stats["grounded"] == 0
    assert stats["hit_rate"] == 0.0
    assert stats["by_outcome"] == {"failed": 1}


def test_successful_retrieval_counts_as_grounded(monkeypatch):
    monkeypatch.setattr(pc, "_read_cache", lambda *a, **k: None)
    monkeypatch.setattr(pc, "_write_cache", lambda *a, **k: None)
    monkeypatch.setattr(pc, "_query_perspicacite_streaming", lambda *a, **k: "context")

    assert pc.query_perspicacite("some goal") == "context"

    stats = pc.grounding_stats()
    assert stats["grounded"] == 1
    assert stats["hit_rate"] == 1.0
    assert stats["by_outcome"] == {"ok": 1}


def test_cache_hits_count_as_grounded(monkeypatch):
    monkeypatch.setattr(pc, "_read_cache", lambda *a, **k: "cached context")

    assert pc.query_perspicacite("some goal") == "cached context"

    stats = pc.grounding_stats()
    assert stats["grounded"] == 1
    assert stats["by_outcome"] == {"cache_hit": 1}


def test_mixed_outcomes_produce_a_partial_hit_rate(monkeypatch):
    monkeypatch.setattr(pc, "_read_cache", lambda *a, **k: None)
    monkeypatch.setattr(pc, "_write_cache", lambda *a, **k: None)
    answers = iter(["context", None, "context"])
    monkeypatch.setattr(
        pc, "_query_perspicacite_streaming", lambda *a, **k: next(answers)
    )
    monkeypatch.setattr(pc, "_query_perspicacite_non_streaming", lambda *a, **k: None)

    for goal in ("a", "b", "c"):
        pc.query_perspicacite(goal)

    stats = pc.grounding_stats()
    assert stats["attempts"] == 3
    assert stats["grounded"] == 2
    assert stats["hit_rate"] == pytest.approx(2 / 3, abs=1e-4)


def test_configure_sets_kb_scope_and_is_reported():
    pc.configure(kb_name="asb-paper-example", mode="basic", max_papers=3)
    assert pc._SETTINGS == {
        "kb_name": "asb-paper-example",
        "mode": "basic",
        "max_papers": 3,
    }
    stats = pc.grounding_stats()
    assert stats["kb_name"] == "asb-paper-example"
    assert stats["mode"] == "basic"


def test_configure_from_config_reads_the_config_fields():
    class _Cfg:
        perspicacite_kb_name = "kb-x"
        perspicacite_mode = "profound"
        perspicacite_max_papers = 9

    pc.configure_from_config(_Cfg())
    assert pc._SETTINGS["kb_name"] == "kb-x"
    assert pc._SETTINGS["mode"] == "profound"
    assert pc._SETTINGS["max_papers"] == 9


def test_config_without_the_fields_leaves_defaults_untouched():
    """Older configs must not break the client."""
    pc.configure_from_config(object())
    assert pc._SETTINGS["kb_name"] is None
    assert pc._SETTINGS["mode"] == "agentic"
