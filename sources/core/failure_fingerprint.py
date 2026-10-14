"""Failure-fingerprint behaviour descriptor for the QD archive.

[DEPRECATED] The archive no longer uses this descriptor as its diversity
metric; novelty now comes from genotype embeddings (see
:mod:`sources.core.genotype_embedding`). The readers stay because older
runs keep a fingerprint inside their ``state_result.json``. The legacy
verifier still computes the descriptor once per evaluation and stores it
with the run's scores.
"""

import json
from pathlib import Path
from typing import Any, Iterable

# Canonical claim-source order. The verifier emits ``"source_a"`` … through
# the registered extractor prompts; expected letters are kept here so
# the descriptor stays a fixed-length vector even if a source is silenced.
SOURCES: tuple[str, ...] = ("a", "b", "c", "d", "e", "f", "g")
DESCRIPTOR_DIM: int = len(SOURCES)

_NEUTRAL_PASS_RATE: float = 0.5
_PASS_STATUS: str = "pass"


def _source_letter(raw: Any) -> str | None:
    """Normalise a claim source label to its canonical letter (``a``-``g``).

    ``_pass_total_by_source`` calls it for every verifier claim, passing
    the raw ``claim["source"]`` value from the scored claim dicts.

    Args:
        raw: Claim source label from verifier output. Accepts
            ``"source_a"``, ``"a"``, ``"A"``, ``"SOURCE_A"``.

    Returns:
        The bare lowercase letter when recognised, else ``None`` for
        unknown or empty labels.
    """
    if not raw:
        return None
    s = str(raw).strip().lower()
    if s.startswith("source_"):
        s = s[len("source_"):]
    return s if (len(s) == 1 and s in SOURCES) else None


def _pass_total_by_source(
    per_claim: Iterable[dict[str, Any]],
) -> dict[str, tuple[int, int]]:
    """Tally ``(passes, total)`` per canonical source letter.

    :func:`compute_failure_fingerprint` calls it with the verifier's
    per-claim results. Non-pass statuses (``fail``, ``error``,
    ``unsure``) all count toward ``total`` but not ``passes``: a
    measurement error is still a non-pass, and quality scoring already
    penalises errors via ``quality_norm``.

    Args:
        per_claim: Iterable of per-claim scored dicts from the verifier;
            each carries ``status`` and ``claim["source"]``.

    Returns:
        Dict mapping each letter in ``SOURCES`` to its ``(passes,
        total)`` integer pair.
    """
    tally: dict[str, list[int]] = {s: [0, 0] for s in SOURCES}
    for c in per_claim:
        claim = c.get("claim") if isinstance(c, dict) else None
        letter = _source_letter((claim or {}).get("source"))
        if letter is None:
            continue
        tally[letter][1] += 1
        if c.get("status") == _PASS_STATUS:
            tally[letter][0] += 1
    return {s: (passes, total) for s, (passes, total) in tally.items()}


def compute_failure_fingerprint(
    per_claim: Iterable[dict[str, Any]],
) -> dict[str, list[float]]:
    """Return centered fingerprint, presence mask, and raw pass rates.

    The legacy verifier calls this after scoring all claims and stores
    the result under ``evaluation.verifier.failure_fingerprint`` in the
    run's ``state_result.json``.

    Args:
        per_claim: Iterable of per-claim scored dicts as produced by the
            verifier (each carries ``status`` and ``claim["source"]``).

    Returns:
        Dict with three fixed-length lists of length ``DESCRIPTOR_DIM``:
        ``vector`` (the centered descriptor used as the QD behaviour
        descriptor), ``presence_mask`` (1.0 when a source emitted at
        least one claim, 0.0 otherwise), and ``pass_rates`` (raw pass
        rate per source, with absent sources filled by the neutral 0.5).
    """
    counts = _pass_total_by_source(per_claim)
    raw: list[float] = []
    presence: list[float] = []
    for s in SOURCES:
        passes, total = counts[s]
        if total > 0:
            raw.append(passes / total)
            presence.append(1.0)
        else:
            raw.append(_NEUTRAL_PASS_RATE)
            presence.append(0.0)

    present_values = [r for r, p in zip(raw, presence) if p > 0.5]
    mean_present = (
        sum(present_values) / len(present_values) if present_values else 0.0
    )
    vector = [
        (r - mean_present) if p > 0.5 else 0.0
        for r, p in zip(raw, presence)
    ]
    return {
        "vector": vector,
        "presence_mask": presence,
        "pass_rates": raw,
    }


def failure_fingerprint_from_state_result(
    state_result: dict[str, Any] | None,
) -> list[float] | None:
    """Pull the centered vector from a persisted ``state_result`` dict.

    Archive and selection tooling call it when scoring stored runs. It
    walks ``evaluation.verifier.failure_fingerprint`` and checks the
    vector shape before returning it.

    Args:
        state_result: Parsed ``state_result.json`` content from disk, or
            ``None``.

    Returns:
        The centered vector as ``DESCRIPTOR_DIM`` floats, or ``None``
        when no usable fingerprint is persisted (the workflow
        short-circuited, the verifier has not run yet, or the field has
        the wrong shape). Callers should fall back to a neutral vector
        so distance lookups stay well-defined.
    """
    if not isinstance(state_result, dict):
        return None
    evaluation = state_result.get("evaluation")
    if not isinstance(evaluation, dict):
        return None
    verifier = evaluation.get("verifier")
    if not isinstance(verifier, dict):
        return None
    fp = verifier.get("failure_fingerprint")
    if not isinstance(fp, dict):
        return None
    vec = fp.get("vector")
    if not isinstance(vec, list) or len(vec) != DESCRIPTOR_DIM:
        return None
    try:
        return [float(x) for x in vec]
    except (TypeError, ValueError):
        return None


def failure_fingerprint_from_workflow_dir(
    workflow_dir: str | Path,
    uuid: str,
) -> list[float] | None:
    """Read the fingerprint vector from ``<workflow_dir>/<uuid>/state_result.json``.

    Convenience wrapper for callers that hold a workflow directory
    instead of a parsed ``state_result`` dict.

    Args:
        workflow_dir: Project workflow directory from config (parent of
            the ``<uuid>/`` folder).
        uuid: Workflow UUID to read.

    Returns:
        The centered fingerprint vector, or ``None`` when the file is
        missing, unreadable, or carries no fingerprint of the expected
        shape.
    """
    path = Path(workflow_dir) / uuid / "state_result.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return failure_fingerprint_from_state_result(state)


def neutral_fingerprint() -> list[float]:
    """Zero-vector fingerprint of length ``DESCRIPTOR_DIM``.

    Archive code uses it as the cold-start default (no archive yet, no
    verifier output yet) so distance computations stay well-defined and
    a cold run is treated as neither novel nor stale relative to itself.

    Returns:
        List of ``DESCRIPTOR_DIM`` zeros.
    """
    return [0.0] * DESCRIPTOR_DIM


if __name__ == "__main__":
    # Quality firewall: all-pass and all-fail must both yield zero profile.
    all_pass = [
        {"claim": {"source": "source_a"}, "status": "pass"},
        {"claim": {"source": "source_b"}, "status": "pass"},
        {"claim": {"source": "source_c"}, "status": "pass"},
    ]
    all_fail = [
        {"claim": {"source": "source_a"}, "status": "fail"},
        {"claim": {"source": "source_b"}, "status": "fail"},
        {"claim": {"source": "source_c"}, "status": "fail"},
    ]
    assert compute_failure_fingerprint(all_pass)["vector"] == [0.0] * DESCRIPTOR_DIM
    assert compute_failure_fingerprint(all_fail)["vector"] == [0.0] * DESCRIPTOR_DIM

    # Mixed: a passes, b fails => a positive, b negative, others zero.
    mixed = compute_failure_fingerprint([
        {"claim": {"source": "source_a"}, "status": "pass"},
        {"claim": {"source": "source_b"}, "status": "fail"},
    ])
    assert mixed["vector"][0] > 0.0 and mixed["vector"][1] < 0.0
    assert mixed["presence_mask"][:2] == [1.0, 1.0]
    assert mixed["presence_mask"][2:] == [0.0] * 5
    assert mixed["vector"][2:] == [0.0] * 5

    # Source label normalisation across forms.
    norm = compute_failure_fingerprint([
        {"claim": {"source": "A"}, "status": "pass"},
        {"claim": {"source": "SOURCE_B"}, "status": "fail"},
    ])
    assert norm["presence_mask"][0] == 1.0 and norm["presence_mask"][1] == 1.0

    # state_result projection round-trip.
    state = {"evaluation": {"verifier": {"failure_fingerprint": mixed}}}
    assert failure_fingerprint_from_state_result(state) == mixed["vector"]
    assert failure_fingerprint_from_state_result({}) is None
    assert failure_fingerprint_from_state_result(None) is None

    assert neutral_fingerprint() == [0.0] * DESCRIPTOR_DIM
    print("smoke OK: centered fingerprint, quality firewall holds")
