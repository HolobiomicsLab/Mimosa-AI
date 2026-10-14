"""Regularized Bradley–Terry ratings for the current task tournament."""

from __future__ import annotations

import math

import numpy as np

PRIOR_PRECISION = 1.0
FIT_TOLERANCE = 1e-10
MAX_FIT_STEPS = 100


def fit_ratings(generations: list[dict], pairs: list[dict]) -> dict[str, float]:
    """Fit current ratings for the evaluator from persisted pair outcomes.

    Args:
        generations: Generation records with UUID and execution eligibility.
        pairs: Pair records with a, b, and winner in a/b/tie vocabulary.

    Returns:
        UUID to probability of beating a neutral-strength opponent.
    """
    identifiers = [item["uuid"] for item in generations]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate tournament generation")
    ratings = _fit_strengths(identifiers, pairs)
    return {
        item["uuid"]: ratings[item["uuid"]]
        if item["execution"]["eligible"] else 0.0
        for item in generations
    }


def _fit_strengths(identifiers: list[str], pairs: list[dict]) -> dict[str, float]:
    """Fit centered strengths with a Gaussian prior to prevent infinite wins.

    Args:
        identifiers: Ordered UUID strings from the registry.
        pairs: Validated pair verdicts from the current policy version.

    Returns:
        Bounded per-UUID ratings; an unobserved generation has rating 0.5.
    """
    index = {identifier: position for position, identifier in enumerate(identifiers)}
    observations = _observations(index, pairs)
    strength = np.zeros(len(index))
    for _ in range(MAX_FIT_STEPS):
        gradient, curvature = _derivatives(strength, observations)
        step = np.linalg.solve(curvature, gradient) if len(index) else strength
        strength -= step
        if not len(step) or np.max(np.abs(step)) < FIT_TOLERANCE:
            return {key: _sigmoid(float(strength[value])) for key, value in index.items()}
    raise RuntimeError("Bradley–Terry fit did not converge")


def _observations(index: dict[str, int], pairs: list[dict]) -> list[tuple]:
    """Validate tournament edges before fitting; callers supply registry pairs.

    Args:
        index: UUID to strength-vector position.
        pairs: Pair records with one comparison per unordered UUID pair.

    Returns:
        Indexed observations with fractional wins for ties.
    """
    observations, seen = [], set()
    for pair in pairs:
        left, right = index[pair["a"]], index[pair["b"]]
        edge = tuple(sorted((left, right)))
        if left == right or edge in seen:
            raise ValueError("Repeated or self tournament edge")
        seen.add(edge)
        score = {"a": 1.0, "b": 0.0, "tie": 0.5}[pair["winner"]]
        observations.append((left, right, score))
    return observations


def _derivatives(strength: np.ndarray, observations: list[tuple]) -> tuple:
    """Compute logistic-loss derivatives for the fitting loop.

    Args:
        strength: Current log-strength vector.
        observations: Indexed left, right, fractional-win tuples.

    Returns:
        Gradient vector and positive-definite Hessian matrix.
    """
    gradient = PRIOR_PRECISION * strength.copy()
    curvature = PRIOR_PRECISION * np.eye(len(strength))
    for left, right, score in observations:
        probability = _sigmoid(float(strength[left] - strength[right]))
        gradient[left] += probability - score
        gradient[right] -= probability - score
        weight = probability * (1.0 - probability)
        curvature[left, left] += weight
        curvature[right, right] += weight
        curvature[left, right] -= weight
        curvature[right, left] -= weight
    return gradient, curvature


def _sigmoid(value: float) -> float:
    """Convert a fitted log strength into a finite probability.

    Args:
        value: Finite log strength from the fitting loop.

    Returns:
        Logistic probability without exponential overflow.
    """
    decay = math.exp(-abs(value))
    return 1.0 / (1.0 + decay) if value >= 0 else decay / (1.0 + decay)


if __name__ == "__main__":
    sample = [{"uuid": key, "execution": {"eligible": True}} for key in ("a", "b")]
    print(fit_ratings(sample, [{"a": "a", "b": "b", "winner": "a"}]))
