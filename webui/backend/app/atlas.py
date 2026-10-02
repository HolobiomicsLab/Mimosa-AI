"""Project every run's QD behaviour descriptor onto a 2-D atlas.

Mimosa's Quality-Diversity search already embeds each evolved workflow as a
384-dim behaviour descriptor (``run_metrics.json`` → ``qd_descriptor``) — the
space the evolution engine itself explores. The atlas is a PCA projection of
those descriptors: each run becomes a point, parent→child links become trails,
and the picture is literally "where the search went", not an ad-hoc embedding
invented for display.

Runs without a descriptor (38 of 93 at the time of writing: crashed runs and
pre-QD snapshots) are reported in ``skipped`` rather than silently dropped.
Everything follows store.py's defensive contract — missing or malformed
metrics yield an empty atlas, never an exception.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np

from . import lineage, store

# A projection fitted on fewer points than this is geometry-free noise; the
# frontend shows a "not enough embedded runs" note instead.
MIN_POINTS = 3


def _components(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centred data, first two principal axes, and explained-variance ratios."""
    centred = x - x.mean(axis=0)
    # SVD-based PCA: stable for n_samples << n_features (55 runs x 384 dims).
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    var = s**2
    total = float(var.sum()) or 1.0
    return centred, vt[:2], var[:2] / total


def compute_atlas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The atlas for a set of run rows.

    Each row carries at least ``id`` and ``qd_descriptor``; ``score``,
    ``iteration``, ``evolution_kind``, ``parents``, ``cost``, ``family`` and
    ``started_at`` are passed through onto the projected point when present.
    Rows whose descriptor is missing, empty, or of a deviant dimensionality
    are skipped (listed with a reason) — one malformed run must not sink the
    picture.
    """
    usable: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    dims: dict[int, int] = {}
    for r in rows:
        q = r.get("qd_descriptor")
        if not isinstance(q, list) or not q:
            skipped.append({"id": str(r.get("id")), "reason": "no_descriptor"})
            continue
        dims[len(q)] = dims.get(len(q), 0) + 1
        usable.append(r)

    if usable:
        # The QD space has one native dimensionality; a stray vector of any
        # other length is a corrupt record, not a second space.
        native = max(dims, key=lambda d: dims[d])
        kept = []
        for r in usable:
            if len(r["qd_descriptor"]) == native:
                kept.append(r)
            else:
                skipped.append({"id": str(r.get("id")), "reason": "dimension_mismatch"})
        usable = kept

    if len(usable) < MIN_POINTS:
        return {"points": [], "edges": [], "skipped": skipped,
                "variance_explained": [], "n_dimensions": 0}

    x = np.asarray([r["qd_descriptor"] for r in usable], dtype=np.float64)
    centred, axes, ratios = _components(x)
    xy = centred @ axes.T

    ids = {str(r["id"]) for r in usable}
    points = []
    for r, (px, py) in zip(usable, xy):
        points.append({
            "id": str(r["id"]),
            "x": round(float(px), 5),
            "y": round(float(py), 5),
            "score": r.get("score"),
            "iteration": r.get("iteration"),
            "evolution_kind": r.get("evolution_kind"),
            "family": r.get("family"),
            "started_at": r.get("started_at"),
            "cost": r.get("cost"),
        })
    # Trails only between points that are both on the map.
    edges = [
        {"source": str(p), "target": str(r["id"])}
        for r in usable
        for p in (r.get("parents") or [])
        if str(p) in ids
    ]
    return {
        "points": points,
        "edges": edges,
        "skipped": skipped,
        "variance_explained": [round(float(v), 4) for v in ratios],
        "n_dimensions": int(x.shape[1]),
    }


_TOKEN = re.compile(r"[A-Za-z_]{2,}")


def tfidf_vectors(texts: list[str]) -> list[list[float]]:
    """L2-normalised TF-IDF over word/identifier tokens, one row per text.

    Exists because the QD behaviour descriptor turns out to embed the TASK,
    not the evolved workflow: within a family every mutation carries a
    byte-identical vector, so a family's trajectory through QD space has no
    extent by construction. The genotype space projects the evolved CODE
    instead — within-family drift is real there.
    """
    docs = [_TOKEN.findall(t.lower()) for t in texts]
    df: Counter[str] = Counter()
    tfs: list[Counter[str]] = []
    for toks in docs:
        c = Counter(toks)
        tfs.append(c)
        df.update(set(toks))
    vocab = sorted(df)
    idx = {w: i for i, w in enumerate(vocab)}
    n = len(docs)
    mat = np.zeros((n, len(vocab)), dtype=np.float64)
    for r, counts in enumerate(tfs):
        total = sum(counts.values()) or 1
        for w, k in counts.items():
            mat[r, idx[w]] = (k / total) * (math.log((1 + n) / (1 + df[w])) + 1.0)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).tolist()


def _genotype_text(run_id: str) -> str | None:
    path = store._find(store.run_path(run_id), "workflow_genotype_*.py")
    return store._read_text(path) if path else None


def load_rows() -> list[dict[str, Any]]:
    """One atlas row per run on disk, descriptor included when it exists."""
    fam = lineage.families()
    rows: list[dict[str, Any]] = []
    for run_id in store.list_run_ids():
        metrics = store.read_run_metrics(run_id) or {}
        rec = store.read_lineage(run_id) or {}
        rows.append({
            "id": run_id,
            "qd_descriptor": store.read_qd_descriptor(run_id),
            "score": store.overall_score(run_id, metrics or None),
            "iteration": rec.get("iteration", metrics.get("iteration")),
            "evolution_kind": rec.get("evolution_kind", metrics.get("evolution_kind") or "seed"),
            "parents": [p for p in rec.get("parents", []) if isinstance(p, str)],
            "family": fam.get(run_id),
            "started_at": rec.get("created_at") or store.parse_created_at(run_id),
            "cost": metrics.get("iteration_cost_usd"),
        })
    return rows


def atlas_view(space: str = "qd") -> dict[str, Any]:
    """The full-fleet atlas for one embedding space.

    ``qd``: Mimosa's own 384-dim behaviour descriptor (task-level — families
    coincide). ``genotype``: TF-IDF of each run's evolved workflow code
    (within-family drift visible). Same payload shape either way.
    """
    rows = load_rows()
    if space == "genotype":
        texts: list[str] = []
        with_code: list[dict[str, Any]] = []
        for r in rows:
            text = _genotype_text(r["id"])
            if text:
                texts.append(text)
                with_code.append(r)
            else:
                r["qd_descriptor"] = None
        if with_code:
            for r, vec in zip(with_code, tfidf_vectors(texts)):
                r["qd_descriptor"] = vec
    return compute_atlas(rows)
