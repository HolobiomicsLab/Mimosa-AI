"""Leakage warning shared by every gold-feedback artifact.

The gold-feedback verifier feeds the benchmark grader's own verdict
(computed against the gold reference) back into the evolution loop. Any run
that uses it is contaminated by the benchmark: it is an upper-bound research
control, never a reportable Mimosa benchmark score.
"""

from __future__ import annotations

LEAK_WARNING = (
    "ORACLE / BENCHMARK-LEAKING MODE (verifier_kind=gold): the benchmark "
    "grader's verdict against the gold reference is fed back into evolution. "
    "Results of this run are contaminated by the benchmark and must NEVER be "
    "reported as Mimosa benchmark scores (VER/SR/CBS)."
)


def leak_banner(width: int = 78) -> str:
    """Return the leakage warning framed as a multi-line banner.

    Args:
        width: Width of the ``!`` rule lines above and below the text.

    Returns:
        The banner text (rule, warning, rule), without a trailing newline.
    """
    rule = "!" * width
    return f"{rule}\n{LEAK_WARNING}\n{rule}"
