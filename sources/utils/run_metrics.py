"""Persist per-iteration evolution metrics to disk.

Three artifacts live next to the workflow runs:

- ``<workflow_dir>/<uuid>/run_metrics.json`` — one file per workflow.
- ``<workflow_dir>/qd_archive.jsonl`` — append-only QD validation trail.
- ``<workflow_dir>/variation_log.jsonl`` — append-only mutation-scope trail.

These helpers handle filesystem mechanics only; the schema is owned by
the callers in ``evolution_engine``.
"""

import json
from pathlib import Path


def write_run_metrics(workflow_folder: str | Path, metrics: dict) -> Path:
    """Write ``run_metrics.json`` inside ``workflow_folder``.

    Args:
        workflow_folder: Directory of the workflow run (created if missing).
        metrics: JSON-serialisable mapping; non-JSON values are coerced to
            ``str`` via ``json.dump(default=str)``.

    Returns:
        Path to the written ``run_metrics.json``.
    """
    folder = Path(workflow_folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "run_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return path


def append_jsonl(file_path: str | Path, entry: dict) -> Path:
    """Append ``entry`` as a single line to a JSONL file.

    Args:
        file_path: Target JSONL path (parents created if missing).
        entry: JSON-serialisable mapping; non-JSON values coerced via
            ``default=str``.

    Returns:
        Path that was written to.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return path


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        wf = root / "20260101_aaaaaaaa"
        out = write_run_metrics(wf, {"uuid": "x", "score": 0.5})
        assert json.loads(out.read_text())["score"] == 0.5

        log = root / "qd_archive.jsonl"
        append_jsonl(log, {"i": 0, "ok": True})
        append_jsonl(log, {"i": 1, "ok": False})
        lines = log.read_text().splitlines()
        assert [json.loads(line) for line in lines] == [
            {"i": 0, "ok": True},
            {"i": 1, "ok": False},
        ]
        print("run_metrics smoke: OK")
