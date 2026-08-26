"""Tests for the task-definition view: prompt, task ref, grounding, card.

Fixtures mirror what the real writers produce — csv_mode's ``task_ref.json``
and ``original_task_*.txt`` (whose structured ASB tag exists in several
on-disk spellings), the orchestrator's ``run_metrics.json`` grounding block,
and an ASB corpus tree ``<corpus>/<challenge>/cards/<task_id>.json``.
"""

from __future__ import annotations

import json

import pytest

from app import taskdef
from app.settings import get_settings

RUN = "20260822_183340_88cdd1fe"

GROUNDING = {
    "attempts": 5, "grounded": 5, "hit_rate": 1.0, "enabled": True,
    "by_outcome": {"cache_hit": 2, "ok": 3}, "total_seconds": 435.3,
    "mean_seconds": 87.1, "kb_name": None, "mode": "agentic",
}

CARD = {
    "schema_version": "0.17.0",
    "task_id": "task_001",
    "research_question": "How many features survive filtering?",
    "data_accessions": [{"accession": "MSV000080492", "kind": "massive"}],
    "review_questions": ["Is the filter threshold justified?"],
}


@pytest.fixture()
def task_roots(tmp_path, monkeypatch):
    """One run dir with prompt + metrics, and a corpus with one card."""
    workflows = tmp_path / "workflows"
    corpus = tmp_path / "corpus"
    monkeypatch.setenv("MIMOSA_WORKFLOW_DIR", str(workflows))
    monkeypatch.setenv("MIMOSA_CORPUS_DIR", str(corpus))
    get_settings.cache_clear()

    run = workflows / RUN
    run.mkdir(parents=True)
    (run / f"original_task_{RUN}.txt").write_text(
        "Broader context:Reproduce experiments from 'Some study' "
        "[ASB benchmark, challenge p_iimn, task_001] --- Your task:Filter "
        "the feature table at three thresholds.",
        encoding="utf-8")
    (run / "run_metrics.json").write_text(
        json.dumps({"overall_score": 0.5, "grounding": GROUNDING}),
        encoding="utf-8")

    cards = corpus / "p_iimn" / "cards"
    cards.mkdir(parents=True)
    (cards / "task_001.json").write_text(json.dumps(CARD), encoding="utf-8")

    yield tmp_path
    get_settings.cache_clear()


# ── tag parsing (structural, never vocabulary-bound) ─────────────────────────

@pytest.mark.parametrize("text", [
    "[ASB benchmark, challenge p_iimn, task_001]",
    "[ASB benchmark p_iimn, task_001]",
    "prose before (ASB Metabolomics challenge p_iimn, task_001) prose after",
    "(ASB Metabolomics challenge: p_iimn, task: task_001)",
    "[ASB Metabolomics — challenge: p_iimn, task: task_001]",
])
def test_every_on_disk_tag_spelling_parses(text):
    assert taskdef.parse_task_tag(text) == {"challenge": "p_iimn",
                                            "task_id": "task_001"}


@pytest.mark.parametrize("text", [
    "no tag at all",
    "ASB benchmark challenge [q_x]",          # challenge only, no task id
    "reproduce (ASB) the experiments",        # bare marker, nothing to join
    "[ASB benchmark, challenge ../evil, task_001",  # unterminated bracket
])
def test_unparseable_text_yields_none_never_a_guess(text):
    assert taskdef.parse_task_tag(text) is None


# ── the assembled view ───────────────────────────────────────────────────────

def test_view_joins_prompt_tag_grounding_and_card(task_roots):
    view = taskdef.task_view(RUN)
    assert "three thresholds" in view["prompt"]  # full text, not truncated
    assert view["prompt_file"] == f"original_task_{RUN}.txt"
    assert view["task_ref"] == {"challenge": "p_iimn", "task_id": "task_001",
                                "csv_row": None, "source": "prompt_tag"}
    assert view["task_ref_absent_reason"] is None
    assert view["grounding"] == GROUNDING  # verbatim, nothing projected away
    assert view["card"] == CARD            # verbatim, nothing projected away
    assert view["card_absent_reason"] is None


def test_task_ref_json_wins_over_the_prompt_tag(task_roots):
    run = get_settings().workflow_dir / RUN
    (run / "task_ref.json").write_text(
        json.dumps({"challenge": "p_iimn", "task_id": "task_001", "csv_row": 4}),
        encoding="utf-8")
    ref = taskdef.task_view(RUN)["task_ref"]
    assert ref["source"] == "task_ref.json"
    assert ref["csv_row"] == 4


def test_absent_layers_carry_reasons_never_silent_nulls(task_roots, monkeypatch):
    run = get_settings().workflow_dir / RUN
    (run / f"original_task_{RUN}.txt").write_text("no tag here", encoding="utf-8")
    (run / "run_metrics.json").write_text(json.dumps({"overall_score": 1}),
                                          encoding="utf-8")
    monkeypatch.delenv("MIMOSA_CORPUS_DIR")
    get_settings.cache_clear()
    view = taskdef.task_view(RUN)
    assert view["task_ref"] is None
    assert "no structured ASB tag" in view["task_ref_absent_reason"]
    assert view["grounding"] is None
    assert "no grounding block" in view["grounding_absent_reason"]
    assert view["card"] is None
    assert view["card_absent_reason"] == "MIMOSA_CORPUS_DIR is not set"


def test_missing_card_and_missing_prompt_are_reasoned(task_roots):
    run = get_settings().workflow_dir / RUN
    (run / "task_ref.json").write_text(
        json.dumps({"challenge": "p_iimn", "task_id": "task_099", "csv_row": 0}),
        encoding="utf-8")
    (run / f"original_task_{RUN}.txt").unlink()
    view = taskdef.task_view(RUN)
    assert view["prompt"] is None
    assert view["prompt_absent_reason"] == "no original_task_*.txt in the run dir"
    assert view["card"] is None
    assert view["card_absent_reason"] == "card not found in corpus: p_iimn/cards/task_099.json"


def test_unsafe_path_components_never_reach_the_filesystem(task_roots):
    run = get_settings().workflow_dir / RUN
    (run / "task_ref.json").write_text(
        json.dumps({"challenge": "../secrets", "task_id": "task_001"}),
        encoding="utf-8")
    view = taskdef.task_view(RUN)
    assert view["card"] is None
    assert view["card_absent_reason"] == "task ref contains unsafe path components"


def test_newest_prompt_file_wins(task_roots):
    run = get_settings().workflow_dir / RUN
    (run / "original_task_20260823_000000_aaaa0000.txt").write_text(
        "newer prompt [ASB benchmark, challenge p_iimn, task_001]",
        encoding="utf-8")
    view = taskdef.task_view(RUN)
    assert view["prompt_file"] == "original_task_20260823_000000_aaaa0000.txt"
    assert view["prompt"].startswith("newer prompt")


def test_route_serves_the_view_behind_the_run_id_guard(task_roots):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        res = client.get(f"/api/runs/{RUN}/task")
        assert res.status_code == 200
        assert res.json()["task_ref"]["challenge"] == "p_iimn"
        assert client.get("/api/runs/20990101_000000_deadbeef/task").status_code == 404
        assert client.get("/api/runs/..%2Fescape/task").status_code == 404
