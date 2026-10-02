"""compute_atlas: PCA over qd_descriptors, defensive on partial fleets."""

from __future__ import annotations

import math

from app.atlas import compute_atlas


def _row(rid: str, vec, **kw):
    return {"id": rid, "qd_descriptor": vec, **kw}


def test_projects_points_and_keeps_metadata():
    rows = [
        _row("a", [1.0, 0.0, 0.0], score=0.2, iteration=0, parents=[]),
        _row("b", [0.0, 1.0, 0.0], score=0.5, iteration=1, parents=["a"]),
        _row("c", [0.0, 0.0, 1.0], score=0.9, iteration=2, parents=["b"]),
        _row("d", [1.0, 1.0, 0.0], score=0.7, iteration=1, parents=["a"]),
    ]
    atlas = compute_atlas(rows)
    assert [p["id"] for p in atlas["points"]] == ["a", "b", "c", "d"]
    assert all(math.isfinite(p["x"]) and math.isfinite(p["y"]) for p in atlas["points"])
    assert atlas["points"][2]["score"] == 0.9
    assert {(e["source"], e["target"]) for e in atlas["edges"]} == {
        ("a", "b"), ("b", "c"), ("a", "d")
    }
    assert atlas["n_dimensions"] == 3
    assert len(atlas["variance_explained"]) == 2
    # PCA is centred: the point cloud has zero mean on both axes
    # (coordinates are rounded to 5 decimals, so the tolerance allows that)
    assert abs(sum(p["x"] for p in atlas["points"])) < 1e-3


def test_runs_without_descriptor_are_reported_not_dropped_silently():
    rows = [
        _row("a", [1.0, 0.0], score=0.1),
        _row("b", [0.0, 1.0], score=0.2),
        _row("c", [1.0, 1.0], score=0.3),
        _row("crashed", [], score=None),
        _row("legacy", None),
    ]
    atlas = compute_atlas(rows)
    assert len(atlas["points"]) == 3
    assert {s["id"] for s in atlas["skipped"]} == {"crashed", "legacy"}
    assert all(s["reason"] == "no_descriptor" for s in atlas["skipped"])


def test_deviant_dimensionality_is_a_corrupt_record_not_a_second_space():
    rows = [
        _row("a", [1.0, 0.0, 0.0]),
        _row("b", [0.0, 1.0, 0.0]),
        _row("c", [0.0, 0.0, 1.0]),
        _row("odd", [1.0, 2.0]),
    ]
    atlas = compute_atlas(rows)
    assert {p["id"] for p in atlas["points"]} == {"a", "b", "c"}
    assert atlas["skipped"] == [{"id": "odd", "reason": "dimension_mismatch"}]


def test_too_few_points_yields_an_empty_atlas_not_noise():
    atlas = compute_atlas([_row("a", [1.0, 0.0]), _row("b", [0.0, 1.0])])
    assert atlas["points"] == [] and atlas["edges"] == []
    assert len(atlas["skipped"]) == 0


def test_tfidf_separates_different_code_and_collapses_identical():
    from app.atlas import tfidf_vectors

    v = tfidf_vectors([
        "def annotate(spectra): return match(spectra)",
        "def annotate(spectra): return match(spectra)",
        "import pandas as pd\npd.read_csv('features.csv')",
    ])
    same = sum((a - b) ** 2 for a, b in zip(v[0], v[1]))
    diff = sum((a - b) ** 2 for a, b in zip(v[0], v[2]))
    assert same < 1e-12  # identical genotypes coincide
    assert diff > 0.1  # different code is far apart


def test_edges_only_between_mapped_points():
    rows = [
        _row("a", [1.0, 0.0, 0.1], parents=["ghost"]),
        _row("b", [0.0, 1.0, 0.2], parents=["a"]),
        _row("c", [0.5, 0.5, 0.9], parents=["b"]),
    ]
    atlas = compute_atlas(rows)
    assert {(e["source"], e["target"]) for e in atlas["edges"]} == {("a", "b"), ("b", "c")}
