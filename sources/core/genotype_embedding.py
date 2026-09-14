"""Pluggable code-genotype embedding for QD novelty.

Two workflows whose generated code is semantically similar collapse to
the same point in embedding space — keeping both is redundant. Ones that
explored different approaches land far apart. The QD novelty axis reads
"how different is this code" by mean cosine distance to prior genotypes
in this space.

The local MiniLM backend embeds long workflow sources as chunked,
mean-pooled vectors (see :class:`_LocalMiniLMEmbedder`): the model truncates
every input at 256 word-pieces, so a single encode() call would only ever
read the leading boilerplate of a multi-thousand-character genotype.

Default backend: local ``all-MiniLM-L6-v2`` (sentence-transformers) —
no network at runtime, deterministic, free. Optional backend: OpenAI
``text-embedding-3-small`` when both
``MIMOSA_GENOTYPE_EMBEDDING_BACKEND=openai`` and ``OPENAI_API_KEY`` are
set. A degenerate input (``None``, empty, or a backend error) returns
``None`` so the selection layer can treat the missing signal as
"no novelty" (neutral) instead of max-novel — otherwise broken offspring
would be rewarded.
"""

import ast
import hashlib
import logging
import os
import re
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_LOCAL_MODEL: str = "all-MiniLM-L6-v2"
_DEFAULT_OPENAI_MODEL: str = "text-embedding-3-small"


def _hub_model_id(model_name: str) -> str:
    """Map a bare sentence-transformers name to its hub id."""
    return model_name if "/" in model_name else f"sentence-transformers/{model_name}"


def _cached_snapshot_path(model_name: str) -> str | None:
    """Return the local HF cache snapshot dir for ``model_name``, else None.

    ``snapshot_download(local_files_only=True)`` resolves purely from the
    on-disk cache and never issues an HTTP request.
    """
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(_hub_model_id(model_name), local_files_only=True)
    except Exception:
        return None


def load_sentence_transformer(model_name: str):
    """Load a SentenceTransformer, preferring the local HF cache."""
    from sentence_transformers import SentenceTransformer

    snapshot = _cached_snapshot_path(model_name)
    if snapshot is not None:
        try:
            return SentenceTransformer(snapshot)
        except Exception:
            pass  # partial/incompatible cache — fall through to hub load
    return SentenceTransformer(model_name, token=False)


class _Embedder(Protocol):
    """Minimal interface for a text-to-unit-vector backend."""

    def encode(self, text: str) -> np.ndarray: ...


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` rescaled to unit L2 norm; zero vectors pass through."""
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec
    return vec / norm


# MiniLM truncates every input at ``max_seq_length = 256`` word-pieces
# (~950 characters of code: measured 3.7 chars/token on real genotypes).
# Whole-workflow sources are 1,000-12,000 characters, so a single encode()
# call would embed only the leading boilerplate and the first ~150 words of
# the first agent prompt — everything that differentiates two workflows
# (later agents, routing, tool wiring) would be invisible. Chunks are
# therefore capped well below that budget.
_MAX_CHUNK_CHARS = 900


def _canonicalise_for_embedding(code: str) -> str:
    """Drop comment-only and blank lines before embedding.

    Novelty should track code structure and agent prompts, not commenting
    style or vertical whitespace. Comment *tails* on code lines are kept
    (removing them would risk joining tokens).
    """
    kept = [
        line
        for line in code.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return "\n".join(kept) if kept else code


def _split_oversized(text: str, max_chars: int) -> list[str]:
    """Split one oversized block on paragraph boundaries, then hard by lines."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(para) <= max_chars:
            current = para
            continue
        # Hard-split an oversized paragraph on line boundaries.
        lines, buf = para.splitlines(), ""
        for line in lines:
            if len(line) > max_chars:
                # Unbreakable line (no newlines at all): slice by characters.
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(
                    line[i : i + max_chars] for i in range(0, len(line), max_chars)
                )
                continue
            if len(buf) + len(line) + 1 > max_chars and buf:
                chunks.append(buf)
                buf = line
            else:
                buf = f"{buf}\n{line}" if buf else line
        current = buf
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def _split_top_level_blocks(code: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split source into top-level AST blocks, then cap each block's size.

    Each top-level statement (an agent-instruction string assignment, a
    ``SmolAgentFactory`` call, a routing block, ...) becomes its own chunk
    so every part of the workflow contributes to the mean-pooled embedding.
    Unparsable input falls back to whole-text windowing rather than raising.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _split_oversized(code, max_chars) or [code]
    lines = code.splitlines(keepends=True)
    blocks: list[str] = []
    for node in tree.body:
        try:
            start, end = node.lineno - 1, node.end_lineno  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - end_lineno always set in 3.8+
            break
        segment = "".join(lines[start:end]).strip()
        if segment:
            blocks.append(segment)
    if not blocks:
        return _split_oversized(code, max_chars) or [code]
    chunks: list[str] = []
    for block in blocks:
        chunks.extend(_split_oversized(block, max_chars))
    return chunks or [code]


class _LocalMiniLMEmbedder:
    """sentence-transformers all-MiniLM-L6-v2 backend.

    Encodes long sources as chunked mean-pooled embeddings: the text is
    canonicalised (comments/blank lines dropped), split into top-level
    blocks capped at ``_MAX_CHUNK_CHARS``, each chunk is embedded, and the
    per-chunk vectors are averaged and re-normalised. This keeps the whole
    workflow — every agent prompt and the wiring tail — inside the 256
    word-piece budget per chunk instead of silently truncating at the
    first ~950 characters.
    """

    def __init__(self, model_name: str = _DEFAULT_LOCAL_MODEL) -> None:
        self._model = load_sentence_transformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        """Chunk, embed, mean-pool, and L2-normalise ``text``."""
        canonical = _canonicalise_for_embedding(text)
        chunks = _split_top_level_blocks(canonical)
        raw = self._model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] > 1:
            arr = arr.mean(axis=0)
        return _l2_normalize(arr.reshape(-1))


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
    backend = (
        os.environ.get("MIMOSA_GENOTYPE_EMBEDDING_BACKEND", "local").strip().lower()
    )
    if backend == "openai" and os.environ.get("OPENAI_API_KEY"):
        try:
            return _OpenAIEmbedder()
        except Exception as exc:
            logger.warning(
                "OpenAI embedder unavailable, falling back to MiniLM: %s", exc
            )
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
