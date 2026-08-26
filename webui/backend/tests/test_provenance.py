"""Tests for the provenance view: ASTRA capsules + independent evaluations.

Fixtures mirror what the real writers produce — the transparency exporter's
``runs_capsule/<uuid>/{astra.yaml, universes/}`` and asb_eval's
``eval_astra.yaml`` (which stamps the run id into the document name and the
workspace input's description, never into the directory layout).
"""

from __future__ import annotations

import json

import pytest
import yaml

from app import provenance
from app.settings import get_settings

RUN = "20260822_183340_88cdd1fe"
SIBLING = "20260822_190000_aaaa1111"

ASTRA_DOC = {
    "version": "0.1",
    "name": f"Mimosa best run {RUN}",
    "description": "ASTRA export of the best-performing evolved workflow.",
    "decisions": {
        "param_sourcing": {
            "label": "Parameter specification source",
            "rationale": "Published parameters ensure reproducibility.",
            "default": "published",
            "options": {
                "published": {"label": "Peer-reviewed parameters"},
                "defaults": {"label": "Tool defaults"},
            },
            "model": "anthropic/claude-haiku-4-5",
        }
    },
    "outputs": [{"id": "report_md", "type": "file"}],
}

EVAL_DOC = {
    "version": "0.1",
    "name": f"ASB criteria evaluation of Mimosa run {RUN} on task_001",
    "inputs": [
        {"id": "asb_capsule", "type": "capsule", "source": "/corpus/p_iimn"},
        {"id": "run_artefacts", "type": "directory",
         "description": f"Workspace produced by Mimosa run {RUN}."},
    ],
    "evaluation": {
        "criteria_source": "p_iimn/cards/task_001.json :: direct_checks",
        "independent_of_subject": True,
        "summary": {"checks_total": 6, "checks_decided": 2, "passed": 1},
        "verdicts": [{"check_id": "c00", "status": "pass",
                      "method": "deterministic", "criterion": "row count"}],
        "workspace_flags": [{"artefact": "t.csv", "kind": "synthetic_ids"}],
        "judge": {
            "instrument": {"model": "m", "prompt_sha": "abc", "passes": 3},
            "summary": {"judged_total": 2, "judge_fail": 2},
            "refused": [],
            "verdicts": [],
        },
    },
}


@pytest.fixture()
def data_roots(tmp_path, monkeypatch):
    """Point every settings dir at a throwaway tree with one run + capsule."""
    workflows = tmp_path / "workflows"
    capsules = tmp_path / "runs_capsule"
    evals = tmp_path / "evaluations"
    monkeypatch.setenv("MIMOSA_WORKFLOW_DIR", str(workflows))
    monkeypatch.setenv("MIMOSA_CAPSULE_DIR", str(capsules))
    monkeypatch.setenv("MIMOSA_EVAL_DIR", str(evals))
    get_settings.cache_clear()

    for uuid, parents in ((RUN, []), (SIBLING, [RUN])):
        run = workflows / uuid
        run.mkdir(parents=True)
        (run / "run_metrics.json").write_text("{}", encoding="utf-8")
        (run / f"lineage_{uuid}.json").write_text(
            json.dumps({"uuid": uuid, "parents": parents,
                        "evolution_kind": "seed" if not parents else "mutation",
                        "iteration": 0 if not parents else 1}),
            encoding="utf-8")

    cap = capsules / RUN
    (cap / "universes").mkdir(parents=True)
    (cap / "astra.yaml").write_text(yaml.safe_dump(ASTRA_DOC), encoding="utf-8")
    (cap / "universes" / "best.yaml").write_text(
        yaml.safe_dump({"id": "best", "decisions": {"param_sourcing": "published"}}),
        encoding="utf-8")

    out = evals / "p_iimn" / "task_001"
    out.mkdir(parents=True)
    (out / "eval_astra.yaml").write_text(yaml.safe_dump(EVAL_DOC), encoding="utf-8")

    yield tmp_path
    get_settings.cache_clear()


def test_capsule_is_read_with_universes(data_roots):
    cap = provenance.read_astra_capsule(RUN)
    assert cap is not None
    assert "param_sourcing" in cap["decisions"]
    assert cap["universes"][0]["decisions"] == {"param_sourcing": "published"}


def test_run_without_capsule_points_at_the_family_best(data_roots):
    assert provenance.read_astra_capsule(SIBLING) is None
    view = provenance.provenance(SIBLING)
    assert view["astra"] is None
    assert view["family_capsules"] == [RUN]


def test_evaluations_match_on_the_run_id_stamped_in_the_document(data_roots):
    evs = provenance.find_evaluations(RUN)
    assert len(evs) == 1
    ev = evs[0]
    assert ev["independent_of_subject"] is True
    assert ev["summary"]["checks_total"] == 6
    assert ev["judge"]["instrument"]["passes"] == 3
    # a different run finds nothing — no cross-run bleed
    assert provenance.find_evaluations(SIBLING) == []


def test_everything_is_defensive_on_missing_or_junk_files(data_roots):
    junk = get_settings().capsule_dir / SIBLING
    junk.mkdir()
    (junk / "astra.yaml").write_text(":\nnot yaml [", encoding="utf-8")
    assert provenance.read_astra_capsule(SIBLING) is None
    (get_settings().eval_dir / "p_iimn" / "task_001" / "eval_astra.yaml").write_text(
        "[]", encoding="utf-8")  # parses to a list, not a dict
    assert provenance.find_evaluations(RUN) == []


def test_settings_defaults_derive_from_the_repo_not_a_developer_machine(monkeypatch):
    for var in ("MIMOSA_ROOT", "MIMOSA_WORKFLOW_DIR", "MIMOSA_WORKSPACE_DIR",
                "MIMOSA_CORPUS_DIR"):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    try:
        s = get_settings()
        import app.settings as settings_mod
        repo_root = __import__("pathlib").Path(settings_mod.__file__).resolve().parents[3]
        assert s.root == repo_root
        assert s.workspace_dir == repo_root / "workspace"
        assert "/Users/mlg" not in str(vars(s))
        assert s.corpus_dir is None  # unset by default — no honest in-repo default
    finally:
        get_settings.cache_clear()


def test_health_reports_every_data_root_with_an_exists_flag(data_roots, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("MIMOSA_CORPUS_DIR", str(data_roots / "corpus"))
    get_settings.cache_clear()
    with TestClient(app) as client:
        body = client.get("/api/health").json()
    assert body["capsule_dir_exists"] is True
    assert body["eval_dir_exists"] is True
    assert body["corpus_dir"].endswith("corpus")
    assert body["corpus_dir_exists"] is False  # named but not created
    # the pre-existing keys survive unchanged
    assert body["ok"] is True and "workflow_dir" in body and "run_count" in body


def test_route_serves_the_view(data_roots):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        res = client.get(f"/api/runs/{RUN}/provenance")
        assert res.status_code == 200
        body = res.json()
        assert body["astra"]["name"] == f"Mimosa best run {RUN}"
        assert len(body["evaluations"]) == 1
        assert client.get("/api/runs/20990101_000000_deadbeef/provenance").status_code == 404
