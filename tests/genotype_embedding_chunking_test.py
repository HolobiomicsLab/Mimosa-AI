"""Offline tests for chunked genotype embedding (no model download).

Covers the pure splitting/canonicalisation helpers and the mean-pooling
behaviour of :class:`_LocalMiniLMEmbedder` with a stubbed SentenceTransformer.
An integration test that loads the real MiniLM from the local HF cache is
skipped automatically when the cache is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sources.core.genotype_embedding import (  # noqa: E402
    _MAX_CHUNK_CHARS,
    _canonicalise_for_embedding,
    _LocalMiniLMEmbedder,
    _split_oversized,
    _split_top_level_blocks,
)

# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------


def test_canonicalise_drops_comment_and_blank_lines() -> None:
    code = (
        "# header comment\n\n"
        "import os  # trailing comment kept\n\n"
        "# another comment\n"
        "x = 1\n"
    )
    out = _canonicalise_for_embedding(code)
    assert "header comment" not in out
    assert "another comment" not in out
    assert "import os  # trailing comment kept" in out
    assert "x = 1" in out


def test_canonicalise_keeps_code_when_all_comments() -> None:
    code = "# only\n# comments\n"
    out = _canonicalise_for_embedding(code)
    assert out == code  # never returns empty text


# ---------------------------------------------------------------------------
# Block splitting
# ---------------------------------------------------------------------------


def test_split_top_level_blocks_one_chunk_per_statement() -> None:
    code = (
        'prompt_a = """You are agent A."""\n\n'
        'prompt_b = """You are agent B."""\n\n'
        'workflow.add_node("a", node_a)\n'
        "app = workflow.compile()\n"
    )
    chunks = _split_top_level_blocks(code)
    assert len(chunks) == 4
    assert chunks[0].startswith("prompt_a")
    assert chunks[-1] == "app = workflow.compile()"


def test_split_top_level_blocks_respects_char_cap() -> None:
    body = "\n".join(f"line_{i} = {i}" for i in range(200))  # ~2000 chars
    chunks = _split_top_level_blocks(f"block = 1\n{body}\n")
    assert all(len(c) <= _MAX_CHUNK_CHARS for c in chunks)
    assert len(chunks) > 1


def test_split_top_level_blocks_syntax_error_falls_back() -> None:
    code = "def broken(:\n    pass\n" * 5
    chunks = _split_top_level_blocks(code)
    assert chunks  # never raises, always returns something
    assert all(len(c) <= _MAX_CHUNK_CHARS for c in chunks)


def test_split_oversized_prefers_paragraph_boundaries() -> None:
    paras = ["word " * 20] * 30  # 30 paragraphs of ~100 chars
    text = "\n\n".join(paras)
    chunks = _split_oversized(text, _MAX_CHUNK_CHARS)
    assert len(chunks) > 1
    assert all(len(c) <= _MAX_CHUNK_CHARS for c in chunks)


def test_split_oversized_hard_splits_giant_paragraph() -> None:
    text = "x" * 5000  # single paragraph, no boundaries
    chunks = _split_oversized(text, _MAX_CHUNK_CHARS)
    assert all(len(c) <= _MAX_CHUNK_CHARS for c in chunks)
    assert "".join(c.replace("\n", "") for c in chunks).startswith("x")


def test_short_source_single_chunk() -> None:
    assert _split_top_level_blocks("x = 1") == ["x = 1"]


# ---------------------------------------------------------------------------
# Mean-pooling embedder (stubbed model)
# ---------------------------------------------------------------------------


class _StubModel:
    """Records inputs; returns one deterministic unit-ish vector per chunk."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        texts = list(texts)
        self.calls.append(texts)
        rows = []
        for t in texts:
            seed = abs(hash(t)) % 1000
            rng = np.random.default_rng(seed)
            rows.append(rng.normal(size=8))
        return np.asarray(rows, dtype=np.float32)


def _make_embedder() -> tuple[_LocalMiniLMEmbedder, _StubModel]:
    emb = _LocalMiniLMEmbedder.__new__(_LocalMiniLMEmbedder)  # skip model load
    emb._model = _StubModel()
    return emb, emb._model  # type: ignore[return-value]


def test_encode_short_input_single_chunk_and_unit_norm() -> None:
    emb, stub = _make_embedder()
    vec = emb.encode("x = 1")
    assert len(stub.calls) == 1 and len(stub.calls[0]) == 1
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-5)


def test_encode_long_input_is_chunked_and_mean_pooled() -> None:
    emb, stub = _make_embedder()
    code = "\n\n".join(
        f'prompt_{i} = """{"sentence about topic " + str(i) + ". " * 120}"""'
        for i in range(10)
    )
    vec = emb.encode(code)
    chunks = stub.calls[0]
    assert len(chunks) > 1
    assert all(len(c) <= _MAX_CHUNK_CHARS for c in chunks)
    # mean-pool: recompute expected vector from the recorded chunks
    rows = np.asarray(
        [np.random.default_rng(abs(hash(c)) % 1000).normal(size=8) for c in chunks],
        dtype=np.float32,
    ).mean(axis=0)
    expected = rows / np.linalg.norm(rows)
    assert np.allclose(vec, expected, atol=1e-6)
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-5)


def test_encode_canonicalises_before_chunking() -> None:
    emb, stub = _make_embedder()
    commented = "# big comment header\n" + "x = 1\n"
    emb.encode(commented)
    assert all("big comment header" not in c for c in stub.calls[0])


# ---------------------------------------------------------------------------
# Integration with the real MiniLM (only when cached locally)
# ---------------------------------------------------------------------------

_MINILM_CACHE = (
    Path.home()
    / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
)


@pytest.mark.skipif(not _MINILM_CACHE.exists(), reason="MiniLM not in local HF cache")
def test_real_minilm_full_vs_prefix_no_longer_identical() -> None:
    """After the fix, the tail of a long document must influence the embedding."""
    from sources.core.genotype_embedding import load_sentence_transformer

    model = load_sentence_transformer("all-MiniLM-L6-v2")
    emb = _LocalMiniLMEmbedder.__new__(_LocalMiniLMEmbedder)
    emb._model = model

    head = (
        "workflow = StateGraph(WorkflowState)\n\n"
        'prompt_a = """You are a chemistry agent. """\n'
    )
    tail_a = " ".join(f"agent alpha step {i}" for i in range(120))
    tail_b = " ".join(f"agent beta plot {i}" for i in range(120))
    doc_a = head + f'prompt_b = """{tail_a}"""\n'
    doc_b = head + f'prompt_b = """{tail_b}"""\n'

    va, vb = emb.encode(doc_a), emb.encode(doc_b)
    cos = float(va @ vb)
    # Same head, very different tails: before the fix both embeddings were
    # identical (cos == 1.0) because only the shared head was read.
    assert cos < 0.99, f"tail content must move the embedding (cos={cos:.4f})"
