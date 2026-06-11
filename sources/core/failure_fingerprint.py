"""Failure-fingerprint behaviour descriptor for the QD archive.

Replaces the topology-based descriptor for novelty: instead of measuring
"different DAG shapes", we measure "different failure profiles" — two
workflows that fail in the same way are redundant regardless of how
different their structure looks; two that fail in different ways explore
different basins and should both survive.

Per source A-F, the descriptor uses the per-claim pass rate. Sources with
zero claims get a neutral value of ``0.5`` and presence-mask ``0``. The
final vector is then **centered**: the mean pass rate across present
sources is subtracted from every entry. This encodes the *profile shape*
of which sources fail relative to the others — NOT the overall quality
level. An all-pass run and an all-fail run both yield the zero profile;
quality already drives ``quality_norm`` in QD, and must not leak into
novelty or QD collapses back into greedy search.
"""

import json
from pathlib import Path
from typing import Any, Iterable

# Canonical claim-source order. The verifier emits ``"source_a"`` … through
# the six independent extractor prompts; expected letters are kept here so
# the descriptor stays a fixed-length vector even if a source is silenced.
SOURCES: tuple[str, ...] = ("a", "b", "c", "d", "e", "f")
DESCRIPTOR_DIM: int = len(SOURCES)

_NEUTRAL_PASS_RATE: float = 0.5
_PASS_STATUS: str = "pass"


def _source_letter(raw: Any) -> str | None:
    """Normalise a claim source label to its lowercase letter (``a``-``f``).

    Accepts ``"source_a"``, ``"a"``, ``"A"``, ``"SOURCE_A"``. Returns the
    bare letter when recognised, ``None`` otherwise.
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

    Non-pass statuses (``fail``, ``error``, ``unsure``) all count toward
    ``total`` but not ``passes``: a measurement error is still a non-pass
    from the optimiser's perspective, and quality already penalises errors
    via ``quality_norm``.
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

    Returns ``None`` when no usable fingerprint has been persisted (the
    workflow short-circuited, the verifier hasn't run yet, or the field
    has the wrong shape). Callers should fall back to a neutral vector
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

    Returns ``None`` when the file is missing, unreadable, or doesn't
    carry a fingerprint of the expected shape.
    """
    path = Path(workflow_dir) / uuid / "state_result.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return failure_fingerprint_from_state_result(state)


def neutral_fingerprint() -> list[float]:
    """Zero-vector fingerprint of length ``DESCRIPTOR_DIM``.

    Used as the cold-start default (no archive yet, no verifier output
    yet) so distance computations stay well-defined and the cold run is
    treated as neither novel nor stale relative to itself.
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
    assert mixed["presence_mask"][2:] == [0.0] * 4
    assert mixed["vector"][2:] == [0.0] * 4

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
