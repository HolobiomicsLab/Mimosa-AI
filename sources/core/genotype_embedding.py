"""Pluggable code-genotype embedding for QD novelty.

Two workflows whose generated code is semantically similar collapse to
the same point in embedding space — keeping both is redundant. Ones that
explored different approaches land far apart. The QD novelty axis reads
"how different is this code" by mean cosine distance to prior genotypes
in this space.

Default backend: local ``all-MiniLM-L6-v2`` (sentence-transformers) —
no network at runtime, deterministic, free. Optional backend: OpenAI
``text-embedding-3-small`` when both
``MIMOSA_GENOTYPE_EMBEDDING_BACKEND=openai`` and ``OPENAI_API_KEY`` are
set. A degenerate input (``None``, empty, or a backend error) returns
``None`` so the selection layer can treat the missing signal as
"no novelty" (neutral) instead of max-novel — otherwise broken offspring
would be rewarded.
"""

import hashlib
import logging
import os
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_LOCAL_MODEL: str = "all-MiniLM-L6-v2"
_DEFAULT_OPENAI_MODEL: str = "text-embedding-3-small"


class _Embedder(Protocol):
    """Minimal interface for a text-to-unit-vector backend."""

    def encode(self, text: str) -> np.ndarray: ...


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` rescaled to unit L2 norm; zero vectors pass through."""
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec
    return vec / norm


class _LocalMiniLMEmbedder:
    """sentence-transformers all-MiniLM-L6-v2 backend."""

    def __init__(self, model_name: str = _DEFAULT_LOCAL_MODEL) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, token=False)

    def encode(self, text: str) -> np.ndarray:
        """Encode ``text`` and return an L2-normalised float32 vector."""
        raw = self._model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return _l2_normalize(np.asarray(raw, dtype=np.float32))


class _OpenAIEmbedder:
    """OpenAI text-embedding-3 backend; activated by env vars."""

    def __init__(self, model_name: str = _DEFAULT_OPENAI_MODEL) -> None:
        from openai import OpenAI
        self._client = OpenAI()
        self._model_name = model_name

    def encode(self, text: str) -> np.ndarray:
        """Encode ``text`` via the API and return an L2-normalised vector."""
        resp = self._client.embeddings.create(model=self._model_name, input=text)
        raw = np.asarray(resp.data[0].embedding, dtype=np.float32)
        return _l2_normalize(raw)


class GenotypeEmbedder:
    """Process-local genotype embedder with SHA-1 cache.

    Identical code text is embedded once per process, so the QD inner
    loop stays cheap even when archive refresh walks dozens of members.
    The backend is constructed lazily so importing this module never
    triggers model loading.
    """

    def __init__(self, backend: _Embedder | None = None) -> None:
        """Bind a backend; defer instantiation if none is supplied."""
        self._backend: _Embedder | None = backend
        self._cache: dict[str, np.ndarray] = {}

    def _ensure_backend(self) -> _Embedder:
        """Return the bound backend, instantiating the default on first call."""
        if self._backend is None:
            self._backend = _select_default_backend()
        return self._backend

    def embed(self, genotype: str | None) -> np.ndarray | None:
        """Embed a genotype source to a unit-norm vector.

        Args:
            genotype: Workflow source code; ``None``, empty, or
                non-string inputs are treated as degenerate.

        Returns:
            Unit-norm ``np.ndarray`` of shape ``(d,)`` on success; ``None``
            when the input is degenerate or the backend errors out.
        """
        if not isinstance(genotype, str) or not genotype:
            return None
        key = hashlib.sha1(genotype.encode("utf-8")).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            vec = self._ensure_backend().encode(genotype)
        except Exception as exc:
            logger.warning("genotype embed failed: %s", exc)
            return None
        if vec is None or vec.size == 0 or not np.isfinite(vec).all():
            return None
        self._cache[key] = vec
        return vec


def _select_default_backend() -> _Embedder:
    """Pick MiniLM (default) or OpenAI based on environment variables."""
    backend = os.environ.get("MIMOSA_GENOTYPE_EMBEDDING_BACKEND", "local").strip().lower()
    if backend == "openai" and os.environ.get("OPENAI_API_KEY"):
        try:
            return _OpenAIEmbedder()
        except Exception as exc:
            logger.warning("OpenAI embedder unavailable, falling back to MiniLM: %s", exc)
    return _LocalMiniLMEmbedder()


_default_embedder: GenotypeEmbedder | None = None


def get_default_embedder() -> GenotypeEmbedder:
    """Module-level singleton — instantiated lazily on first call."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = GenotypeEmbedder()
    return _default_embedder


def embed_genotype(genotype: str | None) -> np.ndarray | None:
    """Convenience wrapper around the module-level default embedder."""
    return get_default_embedder().embed(genotype)


def reset_default_embedder() -> None:
    """Drop the cached singleton — test helper, never used at runtime."""
    global _default_embedder
    _default_embedder = None


if __name__ == "__main__":
    class _StubBackend:
        """Deterministic stub: SHA-1 bytes → 20-dim float vector."""

        def encode(self, text: str) -> np.ndarray:
            digest = hashlib.sha1(text.encode("utf-8")).digest()
            arr = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            return _l2_normalize(arr)

    emb = GenotypeEmbedder(backend=_StubBackend())
    a = emb.embed("def x(): return 1")
    b = emb.embed("def x(): return 1")
    c = emb.embed("def y(): return 2")
    assert a is not None and b is not None and c is not None
    assert np.allclose(a, b), "cache should return identical vector"
    assert not np.allclose(a, c), "different code → different vector"
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5, "embedding must be unit norm"
    assert emb.embed(None) is None
    assert emb.embed("") is None
    assert emb.embed(123) is None  # type: ignore[arg-type]
    print("smoke OK: deterministic, cached, unit-norm embeddings")
