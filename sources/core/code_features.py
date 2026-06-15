"""Genotype-embedding behaviour descriptor for the QD archive.

The descriptor must be orthogonal to fitness so the archive can separate
"different ways of being good" from "different ways of being mediocre".
This module exposes the workflow genotype's unit-norm code embedding as
the QD behaviour descriptor — two workflows whose generated code is
semantically similar collapse to the same point, so QD keeps only
genotypes that explored a different approach.

Backend selection lives in :mod:`sources.core.genotype_embedding`; this
file is only the QD-facing shim.
"""

from .genotype_embedding import embed_genotype


def genotype_embedding_descriptor(code: str | None) -> list[float] | None:
    """Return the unit-norm genotype embedding for QD novelty.

    Args:
        code: Workflow source code as a string.

    Returns:
        L2-normalised embedding as a plain ``list[float]``. ``None`` for
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
