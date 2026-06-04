"""
Evolution Strategy
Supports two modes:
  1. Greedy : validates that the latest run improved over recent history.
  2. Open-ended : maintains a population archive, uses novelty + quality to decide which individuals survive
"""

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .code_features import extract_code_features

MAX_CHILDREN_PER_PARENT = 2

class SelectionStrategy(Enum):
    """Available selection strategies for the evolution loop."""
    GREEDY = "greedy"             # Accept only if strictly better (current behaviour)
    TOURNAMENT = "tournament"     # Probabilistic tournament selection
    NOVELTY = "novelty"           # Novelty search: reward behavioural diversity
    QUALITY_DIVERSITY = "qd"      # MAP-Elites style: novelty + quality combined


@dataclass
class PopulationMember:
    """A single individual in the evolution archive.

    Attributes:
        iteration: Iteration index at which this member was produced.
        reward: Capped reward used for greedy comparisons.
        cost: Monetary or compute cost spent to produce this member.
        uuid: Workflow UUID; ``None`` for members without on-disk artifacts.
        behaviour_descriptor: Fixed-length structural feature vector used
            for novelty distance.
        novelty_score: k-NN novelty against the rest of the archive.
        qd_score: Combined quality-diversity score.
        reward_uncapped: Base + info-bonus − cheat, with no hard-fail cap.
        created_at: Wall-clock timestamp of construction.
    """
    iteration: int
    reward: float
    cost: float
    uuid: str | None = None
    behaviour_descriptor: list[float] = field(default_factory=list)
    novelty_score: float = 0.0
    qd_score: float = 0.0        # combined quality-diversity score
    reward_uncapped: float = 0.0  # base+info-bonus−cheat, no hard-fail cap
    created_at: datetime = field(default_factory=datetime.now)


class SelectionPressure:
    """Evaluates whether a new run should be kept or discarded.

    In **greedy** mode (default) this behaves identically to the previous
    implementation — it checks whether the latest run beats the recent
    baseline by a minimum threshold.

    In **open-ended** modes it maintains a population archive and uses
    novelty / quality-diversity scoring so that low-performing but
    behaviourally novel runs can survive and potentially lead to better
    solutions later.
    """

    def __init__(
        self,
        min_improvement_threshold: float = 0.01,
        strategy: str | SelectionStrategy = SelectionStrategy.QUALITY_DIVERSITY,
        population_size: int = 25,
        novelty_k_neighbours: int = 10,
        novelty_weight: float = 0.4,
        admit_threshold: float = 0.3,
        max_children: int = MAX_CHILDREN_PER_PARENT,
    ) -> None:
        """Configure thresholds and the active selection strategy.

        Args:
            min_improvement_threshold: Minimum relative improvement for greedy mode (5% default).
            strategy: Selection strategy (greedy | tournament | novelty | qd).
            population_size: Max individuals kept in the archive (for open-ended modes).
            novelty_k_neighbours: k for k-nearest novelty calculation.
            novelty_weight: Weight of novelty vs quality in QD score (0 = pure quality, 1 = pure novelty).
            admit_threshold: Minimum qd_score for admission when greedy validity fails.
            max_children: Maximum offspring drawn from a single parent before
                its inverse-child-count weight pushes it below peers.
        """
        self.logger = logging.getLogger(__name__)
        self.min_improvement_threshold = min_improvement_threshold

        if isinstance(strategy, str):
            strategy = SelectionStrategy(strategy.lower())
        self.strategy = strategy

        self.population_size = population_size
        self.novelty_k = novelty_k_neighbours
        self.novelty_weight = novelty_weight
        self.admit_threshold = admit_threshold
        self.max_children = max_children

        # Population archive for open-ended modes
        self._archive: list[PopulationMember] = []
        # Count of offspring rejected by the admit gate (S2 telemetry)
        self._n_admit_rejected: int = 0

    def validate_survivor(
        self,
        baseline_runs: list[Any] | Any,
        new_runs: list[Any] | Any,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Validate whether the new run(s) represent a meaningful step forward.

        Dispatches to the strategy-specific validator based on
        ``self.strategy``.

        Args:
            baseline_runs: One or more previous runs (list or single IndividualRun).
            new_runs: One or more candidate runs (list or single IndividualRun).
            threshold: Override for ``min_improvement_threshold``.

        Returns:
            Dict with keys: ``valid``, ``relative_improvement``,
            ``absolute_improvement``, ``baseline_reward``, ``new_reward``,
            ``confidence``, ``threshold_used``, ``strategy``, ``validated_at``
            (open-ended modes additionally include ``novelty_score``,
            ``qd_score``, ``archive_size``, ``admit_rejected``,
            ``admit_rejected_total``).
        """
        threshold = threshold if threshold is not None else self.min_improvement_threshold

        # Normalise to lists
        baseline_list = baseline_runs if isinstance(baseline_runs, list) else [baseline_runs]
        new_list = new_runs if isinstance(new_runs, list) else [new_runs]

        if self.strategy == SelectionStrategy.GREEDY:
            return self._validate_greedy(baseline_list, new_list, threshold)
        elif self.strategy == SelectionStrategy.TOURNAMENT:
            return self._validate_tournament(baseline_list, new_list, threshold)
        elif self.strategy in (SelectionStrategy.NOVELTY, SelectionStrategy.QUALITY_DIVERSITY):
            return self._validate_open_ended(baseline_list, new_list, threshold)
        else:
            return self._validate_greedy(baseline_list, new_list, threshold)

    def select_parent(
        self,
        runs: list[Any],
        child_counts: dict[str, int] | None = None,
    ) -> Any:
        """Select a single parent for mutation.

        In greedy mode: always returns the best-scoring run from `runs`.
        In tournament mode: probabilistic tournament among a random subset of `runs`.
        In novelty/QD mode: samples a `PopulationMember` from `_archive`
            biased toward high QD-score, falling back to greedy over `runs`
            when the archive is empty (cold start).
        Callers driving from archive must rehydrate the chosen member's
        UUID into their domain object (e.g., WorkflowInfo).

        Args:
            runs: Candidate pool. May be a list of `PopulationMember` (archive
                draw) or any object with a `reward` attribute (greedy/tournament).
            child_counts: Optional ``{uuid: n_children_already}`` map used in
                QD/novelty mode to apply a ``1/(1+n_children)`` penalty so
                already-mined parents don't keep dominating the offspring stream.
                v2_evolution §7 leveraged-move #4.

        Returns:
            The chosen parent (a run object or a ``PopulationMember``), or
            ``None`` if neither `runs` nor the archive holds any candidates.
        """
        if not runs and not self._archive:
            return None

        if self.strategy == SelectionStrategy.GREEDY:
            return max(runs, key=lambda r: _safe_attr(r, "reward", 0.0))

        if self.strategy == SelectionStrategy.TOURNAMENT:
            k = min(3, len(runs))
            candidates = random.sample(runs, k)
            return max(candidates, key=lambda r: _safe_attr(r, "reward", 0.0))

        # Novelty / QD: archive-driven if populated.
        if self._archive:
            members = [c for c in (runs or []) if isinstance(c, PopulationMember)]

            if not members:
                members = self._archive

            eligible = [
                m for m in members
                if (child_counts or {}).get(getattr(m, "uuid", None) or "", 0) < self.max_children
            ]
            members = eligible or members
            weights = [
                max(m.qd_score, 0.01)
                / (1 + (child_counts or {}).get(getattr(m, "uuid", None) or "", 0))
                for m in members
            ]
            return random.choices(members, weights=weights, k=1)[0]

        # Cold start fallback
        return max(runs, key=lambda r: _safe_attr(r, "reward", 0.0)) if runs else None

    def select_parents(
        self,
        candidates: list[Any],
        n_parents: int = 2,
        crossover_rate: float = 0.3,
        child_counts: dict[str, int] | None = None,
    ) -> tuple[list[Any], bool]:
        """Select one or more parents from a candidate pool.

        Args:
            candidates: Pool of objects with a reward (or overall_score) attribute.
            n_parents:  Number of parents to pick when crossover fires (≥2).
            crossover_rate: Probability ∈ [0, 1] of choosing crossover over mutation.
            child_counts: Optional ``{uuid: n_children_already}`` map forwarded to
                `select_parent` to apply an inverse-child-count penalty.

        Returns:
            ``(parents, use_crossover)`` — the chosen parents and whether to
            apply crossover; ``use_crossover`` is ``False`` when fewer than two
            distinct parents could be drawn.
        """
        if not candidates:
            return [], False

        n_parents = max(n_parents, 2)
        # Decide crossover vs mutation
        do_crossover = (
            len(candidates) >= 2
            and random.random() < crossover_rate
        )
        if not do_crossover:
            parent = self.select_parent(candidates, child_counts=child_counts)
            return [parent], False
        selected: list[Any] = []
        pool = list(candidates)  # shallow copy so we can remove picked items

        for _ in range(min(n_parents, len(pool))):
            parent = self.select_parent(pool, child_counts=child_counts)
            if parent is None:
                break
            selected.append(parent)
            pool = [c for c in pool if c is not parent]
        # Safety: if we ended up with < 2, fall back to mutation
        if len(selected) < 2:
            return (
                selected or [self.select_parent(candidates, child_counts=child_counts)],
                False,
            )
        return selected, True

    @property
    def archive(self) -> list[PopulationMember]:
        """Read-only access to the population archive."""
        return list(self._archive)

    # ------------------------------------------------------------------
    # Greedy strategy (backward-compatible)
    # ------------------------------------------------------------------

    def _validate_greedy(
        self,
        baseline_list: list[Any],
        new_list: list[Any],
        threshold: float,
    ) -> dict[str, Any]:
        """Classic greedy validation: best-of-new must beat mean-of-baseline.

        Args:
            baseline_list: Previous runs whose mean reward forms the bar.
            new_list: Candidate run(s) being evaluated.
            threshold: Relative-improvement threshold for acceptance.

        Returns:
            Validation result dict from :meth:`_build_result`.
        """
        baseline_reward = _mean_reward(baseline_list)
        new_reward = _best_reward(new_list)

        absolute_improvement = new_reward - baseline_reward
        relative_improvement = absolute_improvement / max(abs(baseline_reward), 1e-6)
        is_valid = relative_improvement > threshold
        confidence = min(1.0, abs(relative_improvement) / max(threshold, 1e-6))

        result = self._build_result(
            is_valid, relative_improvement, absolute_improvement,
            baseline_reward, new_reward, confidence, threshold,
        )
        self._log_validation(is_valid, relative_improvement, baseline_reward, new_reward, confidence, threshold)
        return result

    # ------------------------------------------------------------------
    # Tournament strategy
    # ------------------------------------------------------------------

    def _validate_tournament(
        self,
        baseline_list: list[Any],
        new_list: list[Any],
        threshold: float,
    ) -> dict[str, Any]:
        """Tournament selection: new run wins with probability proportional
        to its advantage over a random baseline sample.

        Args:
            baseline_list: Previous runs used to sample a small baseline.
            new_list: Candidate run(s) being evaluated.
            threshold: Relative-improvement threshold for unconditional accept.

        Returns:
            Validation result dict from :meth:`_build_result`.
        """
        baseline_sample = random.sample(baseline_list, min(3, len(baseline_list)))
        baseline_reward = _mean_reward(baseline_sample)
        new_reward = _best_reward(new_list)

        absolute_improvement = new_reward - baseline_reward
        relative_improvement = absolute_improvement / max(abs(baseline_reward), 1e-6)

        # Probabilistic acceptance: always accept improvements,
        # accept regressions with probability that decays with magnitude
        if relative_improvement > threshold:
            is_valid = True
        elif relative_improvement > -threshold:
            # Near-neutral: 50% chance to keep (exploration)
            is_valid = random.random() < 0.5
        else:
            # Regression: small chance proportional to exp(-|delta|)
            accept_prob = math.exp(-abs(relative_improvement) * 10)
            is_valid = random.random() < accept_prob

        confidence = min(1.0, abs(relative_improvement) / max(threshold, 1e-6))

        result = self._build_result(
            is_valid, relative_improvement, absolute_improvement,
            baseline_reward, new_reward, confidence, threshold,
        )
        self._log_validation(is_valid, relative_improvement, baseline_reward, new_reward, confidence, threshold)
        return result

    # ------------------------------------------------------------------
    # Open-ended (novelty / quality-diversity)
    # ------------------------------------------------------------------

    def _validate_open_ended(
        self,
        baseline_list: list[Any],
        new_list: list[Any],
        threshold: float,
    ) -> dict[str, Any]:
        """Novelty / QD validation: admit to archive if the candidate is improving or behaviourally novel.

        QD weighting uses ``reward_uncapped`` (base + info_bonus − cheat).

        Args:
            baseline_list: Previous runs whose mean reward forms the bar.
            new_list: Candidate run(s) being evaluated.
            threshold: Relative-improvement threshold for greedy validity.

        Returns:
            Validation result dict from :meth:`_build_result`, extended with
            ``novelty_score``, ``qd_score``, ``archive_size``,
            ``admit_rejected`` and ``admit_rejected_total``.
        """
        baseline_reward = _mean_reward(baseline_list)
        new_reward = _best_reward(new_list)
        best_new = max(new_list, key=lambda r: _safe_attr(r, "reward", 0.0))
        new_reward_uncapped = _safe_attr(best_new, "reward_uncapped", 0.0) or new_reward

        descriptor = self._extract_behaviour_descriptor(best_new)
        novelty = self._compute_novelty(descriptor)

        quality_norm = min(new_reward_uncapped, 1.0)
        novelty_norm = min(novelty / max(self._novelty_range(), 1e-6), 1.0)
        qd_score = (1 - self.novelty_weight) * quality_norm + self.novelty_weight * novelty_norm

        absolute_improvement = new_reward - baseline_reward
        relative_improvement = absolute_improvement / max(abs(baseline_reward), 1e-6)
        is_valid = relative_improvement > threshold or qd_score > self.admit_threshold

        member = PopulationMember(
            iteration=_safe_attr(best_new, "iteration_count", 0),
            reward=new_reward,
            cost=_safe_attr(best_new, "cost", 0.0),
            uuid=_safe_attr(best_new, "current_uuid", None),
            behaviour_descriptor=descriptor,
            novelty_score=novelty,
            qd_score=qd_score,
            reward_uncapped=new_reward_uncapped,
        )
        admit_rejected = not self._try_admit(member, is_valid)

        confidence = min(1.0, qd_score / max(self.admit_threshold, 1e-6))

        result = self._build_result(
            is_valid, relative_improvement, absolute_improvement,
            baseline_reward, new_reward, confidence, threshold,
        )
        result["novelty_score"] = novelty
        result["qd_score"] = qd_score
        result["archive_size"] = len(self._archive)
        result["admit_rejected"] = admit_rejected
        result["admit_rejected_total"] = self._n_admit_rejected

        self._log_validation(is_valid, relative_improvement, baseline_reward, new_reward, confidence, threshold)
        if self.strategy in (SelectionStrategy.NOVELTY, SelectionStrategy.QUALITY_DIVERSITY):
            self.logger.info(
                f"Open-ended: novelty={novelty:.3f}, qd={qd_score:.3f}, "
                f"archive={len(self._archive)}/{self.population_size}"
            )
        return result

    def _extract_behaviour_descriptor(self, run: Any) -> list[float]:
        """Topology-based descriptor parsed from the workflow source.

        Reads ``run.code`` and returns a fixed-length vector of structural features.

        Args:
            run: Object exposing a ``code`` attribute with workflow source.

        Returns:
            Fixed-length feature vector used as a behaviour descriptor.
        """
        return extract_code_features(_safe_attr(run, "code", None))

    def _compute_novelty(self, descriptor: list[float]) -> float:
        """Compute novelty as mean distance to k-nearest archive members.

        Args:
            descriptor: Behaviour descriptor of the candidate.

        Returns:
            Mean Euclidean distance to the k nearest archive members; ``1.0``
            when the archive is empty (first individual is maximally novel).
        """
        if not self._archive:
            return 1.0  # First individual is maximally novel

        distances = [
            _euclidean(descriptor, m.behaviour_descriptor)
            for m in self._archive
        ]
        distances.sort()
        k = min(self.novelty_k, len(distances))
        return sum(distances[:k]) / k if k > 0 else 0.0

    def _novelty_range(self) -> float:
        """Estimate the typical novelty scale from the archive.

        Returns:
            Maximum positive ``novelty_score`` across the archive, or ``1.0``
            when the archive has fewer than two members.
        """
        if len(self._archive) < 2:
            return 1.0
        novelties = [m.novelty_score for m in self._archive if m.novelty_score > 0]
        return max(novelties) if novelties else 1.0

    def _try_admit(self, member: PopulationMember, is_valid: bool) -> bool:
        """Admit `member` to the archive when it passes the validity check.

        Args:
            member: Candidate population member.
            is_valid: Result of the upstream validity check.

        Returns:
            ``True`` if the member was added, ``False`` if rejected (and the
            internal rejection counter is incremented).
        """
        if not is_valid:
            self._n_admit_rejected += 1
            self.logger.info(
                f"ADMIT REJECTED (uncapped={member.reward_uncapped:.3f}, "
                f"qd={member.qd_score:.3f}, novelty={member.novelty_score:.3f}) — "
                f"total rejected={self._n_admit_rejected}"
            )
            return False
        self._add_to_archive(member)
        return True

    def _add_to_archive(self, member: PopulationMember) -> None:
        """Append member; evict the lowest-qd_score peer if over capacity.

        Args:
            member: The population member to admit.
        """
        self._archive.append(member)

        if len(self._archive) > self.population_size:
            weakest = min(self._archive, key=lambda m: m.qd_score)
            self._archive.remove(weakest)
            self.logger.debug(
                f"Evicted archive member (qd={weakest.qd_score:.3f}, "
                f"reward={weakest.reward:.3f}) — archive full"
            )

        self._refresh_member_metrics()

    def _refresh_member_metrics(self) -> None:
        """Recompute stored novelty + qd_score for every archive member."""
        n = len(self._archive)
        if n < 2:
            return
        # Pass 1: k-NN novelty against current archive peers.
        for m in self._archive:
            distances = sorted(
                _euclidean(m.behaviour_descriptor, o.behaviour_descriptor)
                for o in self._archive
                if o is not m
            )
            k = min(self.novelty_k, len(distances))
            m.novelty_score = sum(distances[:k]) / k if k > 0 else 0.0
        # Pass 2: renormalise qd_score against the current novelty range.
        novelty_range = max(
            (m.novelty_score for m in self._archive if m.novelty_score > 0),
            default=1.0,
        )
        for m in self._archive:
            quality_norm = min(max(m.reward_uncapped, 0.0), 1.0)
            novelty_norm = min(m.novelty_score / max(novelty_range, 1e-6), 1.0)
            m.qd_score = (
                (1 - self.novelty_weight) * quality_norm
                + self.novelty_weight * novelty_norm
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_result(
        self,
        is_valid: bool,
        relative_improvement: float,
        absolute_improvement: float,
        baseline_reward: float,
        new_reward: float,
        confidence: float,
        threshold: float,
    ) -> dict[str, Any]:
        """Assemble the common validation-result dictionary.

        Args:
            is_valid: Whether the candidate passed validation.
            relative_improvement: Reward delta relative to baseline.
            absolute_improvement: Raw reward delta.
            baseline_reward: Reward summary of the baseline runs.
            new_reward: Reward summary of the new runs.
            confidence: Bounded confidence score in [0, 1].
            threshold: Threshold used for the decision.

        Returns:
            Dict of standardised keys consumed by callers and downstream logs.
        """
        return {
            "valid": is_valid,
            "relative_improvement": relative_improvement,
            "absolute_improvement": absolute_improvement,
            "baseline_reward": baseline_reward,
            "new_reward": new_reward,
            "validated_at": datetime.now(),
            "confidence": confidence,
            "threshold_used": threshold,
            "strategy": self.strategy.value,
        }

    def _log_validation(
        self,
        is_valid: bool,
        relative_improvement: float,
        baseline_reward: float,
        new_reward: float,
        confidence: float,
        threshold: float,
    ) -> None:
        """Emit an info/warning log line summarising the validation outcome.

        Args:
            is_valid: Whether the candidate passed validation.
            relative_improvement: Reward delta relative to baseline.
            baseline_reward: Reward summary of the baseline runs.
            new_reward: Reward summary of the new runs.
            confidence: Bounded confidence score in [0, 1].
            threshold: Threshold used for the decision.
        """
        if is_valid:
            self.logger.info(
                f"✅ ACCEPTED ({self.strategy.value}): {relative_improvement:+.1%} "
                f"({baseline_reward:.3f} → {new_reward:.3f}) "
                f"[confidence: {confidence:.0%}]"
            )
        else:
            self.logger.warning(
                f"⚠️ REJECTED ({self.strategy.value}): {relative_improvement:+.1%} "
                f"({baseline_reward:.3f} → {new_reward:.3f}) "
                f"[below {threshold:.0%} threshold]"
            )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _safe_attr(obj: Any, attr: str, default: Any = 0.0) -> Any:
    """Safely get an attribute from an object, returning default if missing.

    Args:
        obj: Source object, possibly ``None``.
        attr: Attribute name to look up.
        default: Value returned when `obj` is ``None`` or `attr` is missing.

    Returns:
        Attribute value or `default`.
    """
    return getattr(obj, attr, default) if obj is not None else default


def _mean_reward(runs: list[Any]) -> float:
    """Mean reward across a list of runs.

    Args:
        runs: Iterable of run-like objects with a ``reward`` attribute.

    Returns:
        Arithmetic mean of available rewards; ``0.0`` when `runs` is empty.
    """
    rewards = [_safe_attr(r, "reward", 0.0) for r in runs if r is not None]
    return sum(rewards) / max(len(rewards), 1)


def _best_reward(runs: list[Any]) -> float:
    """Best (max) reward across a list of runs.

    Args:
        runs: Iterable of run-like objects with a ``reward`` attribute.

    Returns:
        Maximum reward seen, or ``0.0`` when `runs` is empty.
    """
    rewards = [_safe_attr(r, "reward", 0.0) for r in runs if r is not None]
    return max(rewards) if rewards else 0.0


def _euclidean(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two vectors of equal length.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Euclidean distance, or ``float("inf")`` when the vectors differ in
        length (used as a sentinel for incomparable descriptors).
    """
    if len(a) != len(b):
        return float("inf")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


if __name__ == "__main__":
    from types import SimpleNamespace

    sp = SelectionPressure(strategy="qd", population_size=50, novelty_k_neighbours=25, novelty_weight=0.4)

    seed = SimpleNamespace(
        reward=0.97, reward_uncapped=1.05, current_uuid="seed",
        iteration_count=1, cost=0.0, code="x=1",
    )
    sp._validate_open_ended([seed], [seed], threshold=0.01)
    assert len(sp._archive) == 1, sp._archive

    distinct = SimpleNamespace(
        reward=0.91, reward_uncapped=0.91, current_uuid="distinct",
        iteration_count=5, cost=0.0,
        code="\n".join(["def f():"] + ["    x = 'y' * 800"] * 6),
    )
    sp._validate_open_ended([seed], [distinct], threshold=0.01)
    assert len(sp._archive) == 2, f"distinct sibling rejected; archive={[m.uuid for m in sp._archive]}"

    print("smoke OK: distinct sibling admitted alongside higher-reward seed")