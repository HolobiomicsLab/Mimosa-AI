"""HTTP + WebSocket routes for the Observatory API.

Run endpoints are read-only; run ids are validated against the on-disk set
before any filesystem access, so path components can't escape the data roots.
The setup/launch endpoints and the live-workspace upload write to disk.
"""

from __future__ import annotations

import mimetypes
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import (
    APIRouter, Body, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.responses import FileResponse

from . import bridge, config_store, lineage, memory, provenance, store, workspace
from .launcher import launcher
from .live import hub

router = APIRouter(prefix="/api")

_RUN_ID_RE = re.compile(r"^(single_agent_)?\d{8}_\d{6}_[0-9a-f]{8}$")

_MEDIA = {
    "image": "image/png",
    "code": "text/plain; charset=utf-8",
    "markdown": "text/markdown; charset=utf-8",
    "text": "text/plain; charset=utf-8",
    "json": "application/json",
}


def _require_run(run_id: str) -> None:
    if not _RUN_ID_RE.match(run_id) or not store.run_path(run_id).is_dir():
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")


@router.get("/health")
def health() -> dict[str, Any]:
    s = store.workflow_dir()
    return {
        "ok": True,
        "workflow_dir": str(s),
        "workflow_dir_exists": s.is_dir(),
        "run_count": len(store.list_run_ids()),
    }


@router.get("/runs")
def get_runs() -> list[dict[str, Any]]:
    return store.list_runs()


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    _require_run(run_id)
    detail = store.run_detail(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="run not found")
    # Learning mode = this run evolved (has an evolution family) or is itself a
    # mutation/crossover. Used to hide expert evolution UI for plain task runs.
    fam = lineage.family_size(run_id)
    detail["family_size"] = fam
    detail["learning_mode"] = fam > 1 or detail.get("evolution_kind") in {"mutation", "crossover"}
    return detail


@router.get("/runs/{run_id}/tree")
def get_tree(run_id: str) -> dict[str, Any]:
    _require_run(run_id)
    result = lineage.tree(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="no lineage for run")
    return result


@router.get("/runs/{run_id}/provenance")
def get_provenance(run_id: str) -> dict[str, Any]:
    """The run's ASTRA capsule (decisions + universes) and every independent
    asb_eval evaluation capsule that names it. Empty sections are normal —
    only a family's best run has a capsule, and evaluations exist only after
    ``asb_eval`` has been run against the workspace."""
    _require_run(run_id)
    return provenance.provenance(run_id)


@router.get("/runs/{run_id}/series")
def get_series(run_id: str) -> dict[str, Any]:
    _require_run(run_id)
    return lineage.series(run_id)


@router.get("/runs/{run_id}/artifacts")
def get_artifacts(run_id: str) -> list[dict[str, Any]]:
    _require_run(run_id)
    return store.list_artifacts(run_id)


@router.get("/runs/{run_id}/artifacts/{name}")
def get_artifact(run_id: str, name: str) -> FileResponse:
    _require_run(run_id)
    resolved = store.artifact_path(run_id, name)
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"artifact '{name}' not found")
    path, kind = resolved
    return FileResponse(path, media_type=_MEDIA.get(kind, "application/octet-stream"))


@router.get("/runs/{run_id}/memory")
def get_memory(run_id: str) -> dict[str, Any]:
    _require_run(run_id)
    result = memory.list_memory(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="no memory for run")
    return result


@router.get("/runs/{run_id}/memory/timeline")
def get_timeline(run_id: str) -> dict[str, Any]:
    _require_run(run_id)
    result = memory.timeline(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="no memory for run")
    return result


@router.get("/runs/{run_id}/memory/step")
def get_step(
    run_id: str,
    agent: str = Query(..., pattern=r"^[A-Za-z0-9_.-]+$"),
    index: int = Query(..., ge=0),
) -> dict[str, Any]:
    _require_run(run_id)
    result = memory.step_detail(run_id, agent, index)
    if result is None:
        raise HTTPException(status_code=404, detail="step not found")
    return result


@router.get("/runs/{run_id}/memory/call/{name}")
def get_call(run_id: str, name: str) -> dict[str, Any]:
    _require_run(run_id)
    result = memory.call_detail(run_id, name)
    if result is None:
        raise HTTPException(status_code=404, detail="call not found")
    return result


@router.get("/archive")
def get_archive(limit: int | None = Query(None, ge=1, le=5000)) -> list[dict[str, Any]]:
    return lineage.qd_archive(limit)


@router.get("/workspace/scopes")
def get_scopes() -> dict[str, Any]:
    return workspace.list_scopes()


@router.get("/workspace/{scope}/files")
def get_workspace_files(scope: str) -> dict[str, Any]:
    result = workspace.list_files(scope)
    if result is None:
        raise HTTPException(status_code=404, detail=f"scope '{scope}' not found")
    return result


@router.get("/workspace/{scope}/file")
def get_workspace_file(scope: str, path: str = Query(...)) -> FileResponse:
    resolved = workspace.resolve_file(scope, path)
    if resolved is None:
        raise HTTPException(status_code=404, detail="file not found")
    target, kind = resolved
    media = _MEDIA.get(kind) or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return FileResponse(target, media_type=media)


@router.post("/workspace/upload")
def post_workspace_upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Store run-input files in the live workspace (the next launch reads them).

    All-or-nothing: an error response means nothing from this request persisted.
    """
    named = [(workspace.safe_upload_name(f.filename), f) for f in files]
    rejected = [f.filename or "(unnamed)" for name, f in named if name is None]
    if rejected:
        raise HTTPException(status_code=422, detail=f"unusable filename(s): {', '.join(rejected)}")
    try:
        saved = workspace.save_uploads([(name, f.file) for name, f in named])
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=422, detail=f"could not store upload: {exc}")
    return {"saved": saved}


@router.delete("/workspace/live/file")
def delete_workspace_file(path: str = Query(...)) -> dict[str, Any]:
    """Remove one file from the live workspace (undo for a mistaken upload)."""
    if not workspace.delete_live_file(path):
        raise HTTPException(status_code=404, detail="file not found")
    return {"deleted": path}


# ── Phase 2: setup & launch ──


def _clean_objective(payload: dict[str, Any]) -> str:
    objective = str(payload.get("objective", "")).strip()
    if len(objective) < 10:
        raise HTTPException(status_code=422, detail="objective must be at least 10 characters")
    return objective


@router.get("/setup")
def get_setup() -> dict[str, Any]:
    """Everything the setup page needs in one round-trip (no secret values)."""
    available, error = bridge.bridge_available()
    return {
        "config": config_store.editable_view(),
        "keys": config_store.key_status(),
        "presets": config_store.MODEL_PRESETS,
        "bridge": {"available": available, "error": error},
    }


@router.patch("/setup/config")
def patch_setup_config(patch: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        applied = config_store.update_config(patch)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"applied": applied, "config": config_store.editable_view()}


@router.post("/setup/keys")
def post_setup_key(payload: dict[str, str] = Body(...)) -> dict[str, Any]:
    try:
        saved = config_store.save_key(payload.get("name", ""), payload.get("value", ""))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {**saved, "keys": config_store.key_status()}


@router.get("/setup/mcp")
def get_setup_mcp() -> dict[str, Any]:
    return config_store.mcp_health()


@router.post("/assist/refine")
def post_refine(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """One round of the objective-clarifier LLM (runs in the Mimosa venv)."""
    return bridge.refine(_clean_objective(payload), payload.get("history") or [])


@router.post("/assist/classify")
def post_classify(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Goal-vs-task mode suggestion for an objective (LLM, Mimosa venv)."""
    return bridge.classify(_clean_objective(payload))


@router.get("/assist/objective-history")
def get_objective_history() -> dict[str, Any]:
    """Past objectives saved by the CLI/webui (newest first)."""
    return bridge.objective_history()


@router.post("/launches")
def post_launch(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    objective = _clean_objective(payload)
    mode = payload.get("mode", "task")
    if mode not in ("task", "goal"):
        raise HTTPException(status_code=422, detail="mode must be 'task' or 'goal'")
    result = launcher.launch(
        objective=objective,
        mode=mode,
        learn=bool(payload.get("learn", False)),
        judge=bool(payload.get("judge", True)),
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    if not result.get("ok"):
        raise HTTPException(status_code=503, detail=result.get("error"))
    return result


@router.get("/launches")
def get_launches() -> list[dict[str, Any]]:
    return launcher.list()


@router.get("/launches/{launch_id}")
def get_launch(launch_id: str) -> dict[str, Any]:
    status = launcher.status(launch_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"launch '{launch_id}' not found")
    return status


@router.post("/launches/{launch_id}/cancel")
def post_cancel_launch(launch_id: str) -> dict[str, Any]:
    status = launcher.cancel(launch_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"launch '{launch_id}' not found")
    return status


@router.websocket("/live")
async def live_ws(ws: WebSocket) -> None:
    await ws.accept()
    await hub.register(ws)
    try:
        while True:
            # We don't expect client messages; this keeps the socket open and
            # detects disconnects promptly.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        hub.unregister(ws)
