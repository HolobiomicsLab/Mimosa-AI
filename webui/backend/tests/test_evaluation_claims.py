"""Tests for parsing per-claim pass/fail detail from a verifier evaluation.txt."""

from __future__ import annotations

import pytest

from app import store

REPORT = """Verifier Evaluation
============================================================
Claims: 3  pass=1  fail=1  error=0  unsure=1  scored=3
Overall: 0.500, uncapped 0.500, hard_fail_capped=False)
  base_mean=0.500
[output_contains_hello] (importance=10; literal deliverable; named in goal) The output contains 'hello'.
  relevant_files: ['hello.py']
  kind=executable status=pass score=1.0
  details: Output contains 'hello': 'hello'

[confirms_hello] (importance=9; core methodology) The workflow verifies the string appears.
  relevant_files: ['hello.py', 'check.py']
  kind=executable status=fail score=0.0
  details: hello.py only prints 'hello' without verification.

[soft_quality] (importance=3; tangential) The code is reasonably documented.
  relevant_files: []
  kind=soft status=unsure score=0.5
  details: Could not determine documentation quality.
"""


@pytest.fixture()
def run_with_report(tmp_path, monkeypatch):
    monkeypatch.setenv("MIMOSA_WORKFLOW_DIR", str(tmp_path))
    store.get_settings.cache_clear()
    run = tmp_path / "20260706_120000_abcd1234"
    run.mkdir()
    (run / "evaluation.txt").write_text(REPORT, encoding="utf-8")
    yield run.name
    store.get_settings.cache_clear()


def test_parses_all_claims(run_with_report):
    claims = store.read_evaluation_claims(run_with_report)
    assert claims is not None
    assert [c["status"] for c in claims] == ["pass", "fail", "unsure"]


def test_claim_fields(run_with_report):
    fail = store.read_evaluation_claims(run_with_report)[1]
    assert fail["id"] == "confirms_hello"
    assert fail["importance"] == 9
    assert fail["rationale"] == "core methodology"
    assert fail["description"] == "The workflow verifies the string appears."
    assert fail["kind"] == "executable"
    assert fail["score"] == 0.0
    assert fail["relevant_files"] == ["hello.py", "check.py"]
    assert "only prints" in fail["details"]


def test_empty_relevant_files(run_with_report):
    soft = store.read_evaluation_claims(run_with_report)[2]
    assert soft["relevant_files"] == []
    assert soft["kind"] == "soft"


def test_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("MIMOSA_WORKFLOW_DIR", str(tmp_path))
    store.get_settings.cache_clear()
    (tmp_path / "20260706_120000_deadbeef").mkdir()
    assert store.read_evaluation_claims("20260706_120000_deadbeef") is None
    store.get_settings.cache_clear()


def test_non_claim_report_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("MIMOSA_WORKFLOW_DIR", str(tmp_path))
    store.get_settings.cache_clear()
    run = tmp_path / "20260706_120000_11112222"
    run.mkdir()
    (run / "evaluation.txt").write_text("Scenario passed: 4/5 assertions\n", encoding="utf-8")
    assert store.read_evaluation_claims(run.name) is None
    store.get_settings.cache_clear()
