"""Endpoint tests for live-workspace uploads: sanitization, collisions, delete."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.workspace as workspace_mod


@pytest.fixture()
def ws_client(tmp_path, monkeypatch):
    """TestClient with the live workspace pointed at a scratch directory."""
    ws_dir = tmp_path / "ws"
    monkeypatch.setenv("MIMOSA_WORKSPACE_DIR", str(ws_dir))
    from app.settings import get_settings

    get_settings.cache_clear()
    from app.main import app

    yield TestClient(app), ws_dir
    get_settings.cache_clear()


def _upload(client: TestClient, *files: tuple[str, bytes]):
    return client.post(
        "/api/workspace/upload",
        files=[("files", (name, body, "application/octet-stream")) for name, body in files],
    )


def test_upload_saves_into_workspace(ws_client):
    client, ws_dir = ws_client
    res = _upload(client, ("data.csv", b"a,b\n1,2\n"))
    assert res.status_code == 200
    assert res.json()["saved"] == [{"name": "data.csv", "size": 8, "kind": "data"}]
    assert (ws_dir / "data.csv").read_bytes() == b"a,b\n1,2\n"


def test_upload_collision_gets_numbered_suffix(ws_client):
    client, ws_dir = ws_client
    _upload(client, ("data.csv", b"first"))
    res = _upload(client, ("data.csv", b"second"), ("data.csv", b"third"))
    names = [f["name"] for f in res.json()["saved"]]
    assert names == ["data (1).csv", "data (2).csv"]
    assert (ws_dir / "data.csv").read_bytes() == b"first"
    assert (ws_dir / "data (2).csv").read_bytes() == b"third"


def test_upload_strips_path_components(ws_client):
    client, ws_dir = ws_client
    res = _upload(client, ("../../evil.txt", b"payload"))
    assert res.status_code == 200
    assert res.json()["saved"][0]["name"] == "evil.txt"
    assert (ws_dir / "evil.txt").is_file()
    assert not (ws_dir.parent / "evil.txt").exists()


# Control-char names are covered in test_safe_upload_name_strips_and_rejects:
# HTTP clients (httpx included) percent-escape them in multipart headers, so
# they can't reach the endpoint verbatim.
@pytest.mark.parametrize("bad_name", [".env", "..", ".hidden", " ", "a" * 300 + ".csv"])
def test_upload_rejects_unusable_names(ws_client, bad_name):
    client, ws_dir = ws_client
    res = _upload(client, (bad_name, b"x"), ("ok.txt", b"y"))
    assert res.status_code == 422
    assert not ws_dir.exists() or not any(ws_dir.iterdir())


def test_upload_over_size_cap_is_rejected_and_removed(ws_client, monkeypatch):
    client, ws_dir = ws_client
    monkeypatch.setattr(workspace_mod, "_UPLOAD_MAX_BYTES", 10)
    monkeypatch.setattr(workspace_mod, "_UPLOAD_CHUNK", 4)
    res = _upload(client, ("big.bin", b"x" * 32))
    assert res.status_code == 413
    assert not any(ws_dir.iterdir())  # no final file, no .uploading-* temp


def test_failed_batch_rolls_back_earlier_files(ws_client, monkeypatch):
    client, ws_dir = ws_client
    monkeypatch.setattr(workspace_mod, "_UPLOAD_MAX_BYTES", 10)
    res = _upload(client, ("good.txt", b"tiny"), ("big.bin", b"x" * 32))
    assert res.status_code == 413
    assert not any(ws_dir.iterdir())  # all-or-nothing: good.txt removed again


def test_upload_never_writes_through_dangling_symlink(ws_client):
    client, ws_dir = ws_client
    ws_dir.mkdir(parents=True, exist_ok=True)
    outside = ws_dir.parent / "outside"
    outside.mkdir()
    (ws_dir / "trap.csv").symlink_to(outside / "payload.py")
    res = _upload(client, ("trap.csv", b"print('pwned')"))
    assert res.status_code == 200
    assert res.json()["saved"][0]["name"] == "trap (1).csv"
    assert not (outside / "payload.py").exists()
    assert (ws_dir / "trap.csv").is_symlink()  # link left untouched


def test_safe_upload_name_strips_and_rejects():
    assert workspace_mod.safe_upload_name("dir/sub/data.csv") == "data.csv"
    assert workspace_mod.safe_upload_name("../../evil.txt") == "evil.txt"
    for bad in (None, "", ".", "..", ".env", "x" * 300, "a\tb.csv", "line\nbreak.csv"):
        assert workspace_mod.safe_upload_name(bad) is None


def test_unique_destination_numbers_past_existing(tmp_path):
    (tmp_path / "data.csv").touch()
    (tmp_path / "data (1).csv").touch()
    assert workspace_mod.unique_destination(tmp_path, "data.csv").name == "data (2).csv"


def test_delete_removes_only_workspace_files(ws_client):
    client, ws_dir = ws_client
    _upload(client, ("gone.txt", b"bye"))
    res = client.delete("/api/workspace/live/file", params={"path": "gone.txt"})
    assert res.status_code == 200 and not (ws_dir / "gone.txt").exists()
    assert client.delete("/api/workspace/live/file", params={"path": "gone.txt"}).status_code == 404


def test_delete_blocks_traversal(ws_client):
    client, ws_dir = ws_client
    ws_dir.mkdir(parents=True, exist_ok=True)
    outside = ws_dir.parent / "secret.txt"
    outside.write_text("keep me")
    res = client.delete("/api/workspace/live/file", params={"path": "../secret.txt"})
    assert res.status_code == 404
    assert outside.read_text() == "keep me"
