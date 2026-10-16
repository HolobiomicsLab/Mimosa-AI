"""Genotype-embedding behaviour descriptor for the QD archive.

This Layer 2 helper turns a workflow's genotype source into the unit-norm
behaviour vector that the quality-diversity archive uses for novelty.
Backend selection lives in :mod:`sources.core.genotype_embedding`; this
file is only the QD-facing shim. :mod:`sources.core.selection` imports it
when it scores a candidate against the archive.
"""

from .genotype_embedding import embed_genotype


def genotype_embedding_descriptor(code: str | None) -> list[float] | None:
    """Return the unit-norm genotype embedding for QD novelty.

    The quality-diversity selection layer calls this with a workflow's
    genotype source when it scores novelty against the archive. It
    delegates to the embedding backend and flattens the vector to plain
    floats.

    Args:
        code (str | None): Workflow source code as a string; ``None``
            when the run carries no genotype.

    Returns:
        L2-normalized embedding as a plain ``list[float]``. ``None`` for
        degenerate inputs (missing source, empty string, or backend
        failure) so the selection layer can fall through to a neutral
        novelty signal — never to max-novel.
    """
    vec = embed_genotype(code)
    if vec is None:
        return None
    return [float(x) for x in vec.tolist()]


if __name__ == "__main__":
    assert genotype_embedding_descriptor(None) is None
    assert genotype_embedding_descriptor("") is None
    out = genotype_embedding_descriptor("def workflow(state): return state")
    if out is not None:
        norm = sum(x * x for x in out) ** 0.5
        assert abs(norm - 1.0) < 1e-4, f"expected unit norm, got {norm}"
        print(f"smoke OK: unit-norm descriptor of dim {len(out)}")
    else:
        print("smoke OK: degenerate path (backend unavailable)")
