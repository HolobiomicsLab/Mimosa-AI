"""Tests for ``_detect_language`` in ``verifier_per_claim``.

Covers the four buckets the verifier-gen prompt branches on so that an
R-only workspace does not silently get an AST-parsed-by-Python verifier.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(_REPO_ROOT))

from sources.core.evaluators.verifier_per_claim import _detect_language  # noqa: E402


def test_pure_python_listing_is_python() -> None:
    """All-``.py`` listing classifies as ``python``."""
    listing = "main.py\t512\nutils.py\t128\nREADME.md\t64"
    assert _detect_language(listing) == "python"


def test_mostly_r_listing_is_r() -> None:
    """``.R``-only listing (ignoring non-source files) classifies as ``r``."""
    listing = "pipeline.R\t1024\nhelpers.R\t256\ndata.csv\t2048"
    assert _detect_language(listing) == "r"


def test_empty_listing_is_unknown() -> None:
    """Empty listing yields ``unknown`` instead of guessing a default."""
    assert _detect_language("") == "unknown"


def test_mixed_listing_is_mixed() -> None:
    """A listing with both ``.py`` and ``.R`` files yields ``mixed``."""
    listing = "train.py\t256\nplot.R\t512\nREADME.md\t64"
    assert _detect_language(listing) == "mixed"


if __name__ == "__main__":
    test_pure_python_listing_is_python()
    test_mostly_r_listing_is_r()
    test_empty_listing_is_unknown()
    test_mixed_listing_is_mixed()
    print("test_detect_language: ok")
