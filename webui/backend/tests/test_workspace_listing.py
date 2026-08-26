"""Listing tests: the backend states truncation itself (no client cap mirror)."""

from __future__ import annotations

import pytest
from app import workspace
from app.settings import get_settings


@pytest.fixture()
def live_root(tmp_path, monkeypatch):
    """Settings pointed at a scratch live workspace with three files."""
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    for name in ("a.txt", "b.txt", "c.txt"):
        (ws_dir / name).write_text(name, encoding="utf-8")
    monkeypatch.setenv("MIMOSA_WORKSPACE_DIR", str(ws_dir))
    get_settings.cache_clear()
    yield ws_dir
    get_settings.cache_clear()


def test_walk_states_truncation_when_it_stops_at_the_cap(live_root):
    files, truncated = workspace._walk(live_root, max_files=2)
    assert len(files) == 2
    assert truncated is True


def test_walk_under_the_cap_is_stated_complete(live_root):
    files, truncated = workspace._walk(live_root, max_files=10)
    assert len(files) == 3
    assert truncated is False


def test_walk_exactly_at_the_cap_reads_truncated_conservatively(live_root):
    """Completeness is unknown at the cap — the producer must not claim it."""
    _, truncated = workspace._walk(live_root, max_files=3)
    assert truncated is True


def test_listing_response_carries_the_producer_stated_flag(live_root):
    listing = workspace.list_files("live")
    assert listing is not None
    assert listing["truncated"] is False
    assert len(listing["files"]) == 3
