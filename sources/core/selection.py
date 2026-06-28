"""
Evolution Strategy
Supports two modes:
  1. Greedy : validates that the latest run improved over recent history.
  2. Open-ended : maintains a population archive, uses novelty + quality to decide which individuals survive

Novelty uses the genotype embedding from :mod:`sources.core.code_features`:
two workflows whose generated code is semantically close collapse onto
the same point and are redundant; ones that explored different
approaches land far apart and both survive. A length-penalty term in
``qd_score`` discourages runaway code growth without overriding ranking.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .code_features import genotype_embedding_descriptor

MAX_CHILDREN_PER_PARENT = 2

# Comparison-set modes for novelty.
NOVELTY_ARCHIVE_KNN = "archive_knn"
NOVELTY_PREVIOUS_N = "previous_n"


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
        behaviour_descriptor: Unit-norm genotype embedding used for the
            cosine-distance novelty signal. Empty list when no embedding
            could be produced (treated as neutral by the selection layer).
        genotype_chars: Raw character length of the workflow source —
            used for the length-penalty term in ``qd_score``.
        novelty_score: Mean cosine distance to the current comparison set.
        qd_score: Combined quality-diversity score with length penalty.
        reward_uncapped: Base reward with no hard-fail cap.
        created_at: Wall-clock timestamp of construction.
    """
    iteration: int
    reward: float
    cost: float
    uuid: str | None = None
    behaviour_descriptor: list[float] = field(default_factory=list)
    genotype_chars: int = 0
    novelty_score: float = 0.0
    qd_score: float = 0.0
    reward_uncapped: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


class SelectionPressure:
    """Evaluates whether a new run should be kept or discarded.

    In **greedy** mode (default) this behaves identically to the previous
    implementation — it checks whether the latest run beats the recent
    baseline by a minimum threshold.

    In **open-ended** modes it maintains a population archive and uses
    novelty / quality-diversity scoring so that low-performing but
    behaviourally novel runs can survive and potentially lead to better
    solutions later. Novelty is ``1 − cosine_similarity`` between unit-norm
    genotype embeddings.
    """

    def __init__(
        self,
        min_improvement_threshold: float = 0.01,
        strategy: str | SelectionStrategy = SelectionStrategy.QUALITY_DIVERSITY,
        population_size: int = 25,
        novelty_k_neighbours: int = 10,
        novelty_weight: float = 0.25,
        admit_threshold: float = 0.3,
        max_children: int = MAX_CHILDREN_PER_PARENT,
        novelty_comparison: str = NOVELTY_ARCHIVE_KNN,
        previous_n: int = 5,
        length_penalty_baseline_chars: int = 5000,
        length_penalty_lambda: float = 0.05,
    ) -> None:
        """Configure thresholds and the active selection strategy.

        Args:
            min_improvement_threshold: Minimum relative improvement for greedy mode (5% default).
            strategy: Selection strategy (greedy | tournament | novelty | qd).
            population_size: Max individuals kept in the archive (for open-ended modes).
            novelty_k_neighbours: k for k-nearest novelty calculation (archive_knn mode).
            novelty_weight: Weight of novelty vs quality in QD score (0 = pure quality, 1 = pure novelty).
            admit_threshold: Minimum qd_score for admission when greedy validity fails.
            max_children: Maximum offspring drawn from a single parent before
                its inverse-child-count weight pushes it below peers.
            novelty_comparison: Comparison-set mode — ``"archive_knn"`` (k-NN
                against the archive, default) or ``"previous_n"`` (mean
                distance to the N most recently produced genotypes).
            previous_n: Window size for ``"previous_n"`` mode.
            length_penalty_baseline_chars: Genotype size at which the length
                penalty starts to grow above zero.
            length_penalty_lambda: Strength of the length penalty term
                subtracted from ``qd_score``. Kept conservative so it only
                breaks near-ties.
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
        self.novelty_comparison = (
            novelty_comparison if novelty_comparison in (NOVELTY_ARCHIVE_KNN, NOVELTY_PREVIOUS_N)
            else NOVELTY_ARCHIVE_KNN
        )
        self.previous_n = max(1, int(previous_n))
        self.length_baseline_chars = max(1, int(length_penalty_baseline_chars))
        self.length_lambda = float(length_penalty_lambda)

        self._archive: list[PopulationMember] = []
        self._previous_descriptors: list[list[float]] = []
        self._n_admit_rejected: int = 0
        # Reset each call to _validate_open_ended so log lines surface the
        # eviction from *that* admission only, not a stale carryover.
        self._last_evicted_uuid: str | None = None

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
        In novelty/QD mode: samples a `PopulationMember` from `_archive` biased toward high QD-score

        Args:
            runs: Candidate pool. May be a list of `PopulationMember` (archive
                draw) or any object with a `reward` attribute (greedy/tournament).
            child_counts: Optional ``{uuid: n_children_already}`` map used in
                QD/novelty mode to apply a ``1/(1+n_children)`` penalty

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
        crossover_rate: float = 0.4,
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
        do_crossover = (
            len(candidates) >= 2
            and random.random() < crossover_rate
        )
        if not do_crossover:
            parent = self.select_parent(candidates, child_counts=child_counts)
            return [parent], False
        selected: list[Any] = []
        pool = list(candidates)

        for _ in range(min(n_parents, len(pool))):
            parent = self.select_parent(pool, child_counts=child_counts)
            if parent is None:
                break
            selected.append(parent)
            pool = [c for c in pool if c is not parent]
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

        QD weighting uses ``reward_uncapped`` (uncapped base reward),
        then subtracts a length-penalty term so runaway code growth costs
        ranking points without overriding it on real improvements.

        Args:
            baseline_list: Previous runs whose mean reward forms the bar.
            new_list: Candidate run(s) being evaluated.
            threshold: Relative-improvement threshold for greedy validity.

        Returns:
            Validation result dict from :meth:`_build_result`, extended with
            ``novelty_score``, ``qd_score``, ``length_penalty``,
            ``archive_size``, ``admit_rejected`` and ``admit_rejected_total``.
        """
        self._last_evicted_uuid = None

        baseline_reward = _mean_reward(baseline_list)
        new_reward = _best_reward(new_list)
        best_new = max(new_list, key=lambda r: _safe_attr(r, "reward", 0.0))
        new_reward_uncapped = _safe_attr(best_new, "reward_uncapped", 0.0) or new_reward

        descriptor = self._extract_behaviour_descriptor(best_new)
        genotype_chars = _genotype_chars(best_new)
        novelty = self._compute_novelty(descriptor)

        novelty_range = self._novelty_range()
        quality_norm = min(max(new_reward_uncapped, 0.0), 1.0)
        novelty_norm = min(novelty / max(novelty_range, 1e-6), 1.0)
        length_penalty = _length_penalty(genotype_chars, self.length_baseline_chars)
        qd_score = self._compose_qd_score(
            new_reward_uncapped, novelty, genotype_chars, novelty_range,
        )

        absolute_improvement = new_reward - baseline_reward
        relative_improvement = absolute_improvement / max(abs(baseline_reward), 1e-6)
        is_valid = relative_improvement > threshold or qd_score > self.admit_threshold

        member = PopulationMember(
            iteration=_safe_attr(best_new, "iteration_count", 0),
            reward=new_reward,
            cost=_safe_attr(best_new, "cost", 0.0),
            uuid=_safe_attr(best_new, "current_uuid", None),
            behaviour_descriptor=descriptor or [],
            genotype_chars=genotype_chars,
            novelty_score=novelty,
            qd_score=qd_score,
            reward_uncapped=new_reward_uncapped,
        )
        admit_rejected = not self._try_admit(member, is_valid)
        self._record_previous(descriptor)

        confidence = min(1.0, qd_score / max(self.admit_threshold, 1e-6))

        result = self._build_result(
            is_valid, relative_improvement, absolute_improvement,
            baseline_reward, new_reward, confidence, threshold,
        )
        result["novelty_score"] = novelty
        result["qd_score"] = qd_score
        result["quality_norm"] = quality_norm
        result["novelty_norm"] = novelty_norm
        result["length_penalty"] = length_penalty
        result["behaviour_descriptor"] = descriptor or []
        result["archive_size"] = len(self._archive)
        result["admit_rejected"] = admit_rejected
        result["admit_rejected_total"] = self._n_admit_rejected
        result["evicted_uuid"] = self._last_evicted_uuid

        self._log_validation(is_valid, relative_improvement, baseline_reward, new_reward, confidence, threshold)
        if self.strategy in (SelectionStrategy.NOVELTY, SelectionStrategy.QUALITY_DIVERSITY):
            self.logger.info(
                f"Open-ended: novelty={novelty:.3f}, qd={qd_score:.3f}, "
                f"len_pen={length_penalty:.3f}, "
                f"archive={len(self._archive)}/{self.population_size}"
            )
        return result

    def _extract_behaviour_descriptor(self, run: Any) -> list[float] | None:
        """Unit-norm genotype embedding used as the QD behaviour descriptor.

        Reads a pre-computed ``run.behaviour_descriptor`` first (used by
        tests injecting stub vectors); otherwise embeds ``run.code`` via
        the configured genotype embedder.

        Args:
            run: Object exposing ``code`` (workflow source) and optionally
                a pre-computed ``behaviour_descriptor``.

        Returns:
            Unit-norm descriptor as ``list[float]``, or ``None`` when no
            usable genotype was available — callers must treat the missing
            signal as neutral, never as max-novel.
        """
        precomputed = _safe_attr(run, "behaviour_descriptor", None)
        if isinstance(precomputed, list) and precomputed:
            return [float(x) for x in precomputed]
        code = _safe_attr(run, "code", None)
        return genotype_embedding_descriptor(code)

    def _compute_novelty(self, descriptor: list[float] | None) -> float:
        """Mean cosine distance to the active comparison set.

        ``archive_knn`` (default) uses the mean distance to the k-nearest
        archive members — the existing behaviour, generalised to cosine
        distance. ``previous_n`` uses the mean distance to the N most
        recently produced genotypes (lighter-weight mode that ignores
        eviction).

        Args:
            descriptor: Candidate descriptor; ``None`` / empty yields ``0.0``
                so a missing embedding contributes nothing to QD score.

        Returns:
            Non-negative novelty value in ``[0, 2]`` (cosine distance range
            for unit vectors). ``0.0`` for empty comparison sets so the
            cold start is neither novel nor stale.
        """
        if not descriptor:
            return 0.0
        peers = self._comparison_peers()
        if not peers:
            return 0.0
        distances = sorted(_cosine_distance(descriptor, p) for p in peers)
        if self.novelty_comparison == NOVELTY_PREVIOUS_N:
            return sum(distances) / len(distances)
        k = min(self.novelty_k, len(distances))
        return sum(distances[:k]) / k if k > 0 else 0.0

    def _comparison_peers(self) -> list[list[float]]:
        """Descriptors of the active comparison set, filtered to non-empty."""
        if self.novelty_comparison == NOVELTY_PREVIOUS_N:
            return [p for p in self._previous_descriptors if p]
        return [m.behaviour_descriptor for m in self._archive if m.behaviour_descriptor]

    def _record_previous(self, descriptor: list[float] | None) -> None:
        """Push ``descriptor`` onto the previous-N sliding window."""
        if not descriptor:
            return
        self._previous_descriptors.append(list(descriptor))
        if len(self._previous_descriptors) > self.previous_n:
            self._previous_descriptors.pop(0)

    def _novelty_range(self) -> float:
        """Estimate the typical novelty scale from the active comparison set.

        Returns:
            Maximum positive novelty seen across the current comparison
            members, or ``1.0`` when too small to estimate.
        """
        if self.novelty_comparison == NOVELTY_PREVIOUS_N:
            peers = self._comparison_peers()
            if len(peers) < 2:
                return 1.0
            pair_max = max(
                _cosine_distance(peers[i], peers[j])
                for i in range(len(peers)) for j in range(i + 1, len(peers))
            )
            return pair_max if pair_max > 0 else 1.0
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
        # Refresh metrics first so eviction uses up-to-date qd_scores (including
        # the newly-admitted member's novelty contribution to existing peers).
        self._refresh_member_metrics()

        if len(self._archive) > self.population_size:
            weakest = min(self._archive, key=lambda m: m.qd_score)
            self._archive.remove(weakest)
            self._last_evicted_uuid = getattr(weakest, "uuid", None)
            self.logger.debug(
                f"Evicted archive member (qd={weakest.qd_score:.3f}, "
                f"reward={weakest.reward:.3f}) — archive full"
            )

    def _refresh_member_metrics(self) -> None:
        """Recompute stored novelty + qd_score for every archive member."""
        if len(self._archive) < 2:
            return
        for m in self._archive:
            m.novelty_score = self._knn_novelty_against_peers(m)
        novelty_range = max(
            (m.novelty_score for m in self._archive if m.novelty_score > 0),
            default=1.0,
        )
        for m in self._archive:
            m.qd_score = self._compose_qd_score(
                m.reward_uncapped, m.novelty_score, m.genotype_chars, novelty_range,
            )

    def _knn_novelty_against_peers(self, m: PopulationMember) -> float:
        """k-NN cosine-distance novelty for ``m`` against the rest of the archive."""
        distances = sorted(
            _cosine_distance(m.behaviour_descriptor, o.behaviour_descriptor)
            for o in self._archive
            if o is not m and o.behaviour_descriptor
        )
        k = min(self.novelty_k, len(distances))
        return sum(distances[:k]) / k if k > 0 else 0.0

    def _compose_qd_score(
        self,
        reward_uncapped: float,
        novelty: float,
        genotype_chars: int,
        novelty_range: float,
    ) -> float:
        """``(1-w)·quality + w·novelty − λ·length_penalty`` in one place."""
        quality_norm = min(max(reward_uncapped, 0.0), 1.0)
        novelty_norm = min(novelty / max(novelty_range, 1e-6), 1.0)
        length_penalty = _length_penalty(genotype_chars, self.length_baseline_chars)
        return (
            (1 - self.novelty_weight) * quality_norm
            + self.novelty_weight * novelty_norm
            - self.length_lambda * length_penalty
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


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Cosine distance between two vectors of equal length.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        ``1 − cosine_similarity`` in ``[0, 2]``; ``1.0`` (neutral) when the
        vectors are empty or have mismatching shapes.
    """
    if not a or not b or len(a) != len(b):
        return 1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    cos = dot / (norm_a * norm_b)
    return 1.0 - max(min(cos, 1.0), -1.0)


def _length_penalty(chars: int, baseline_chars: int) -> float:
    """Clipped relative excess over the baseline genotype length.

    Args:
        chars: Genotype character count for the run.
        baseline_chars: Configured baseline above which the penalty grows.

    Returns:
        ``clip((chars − baseline) / baseline, 0, 1)``.
    """
    if baseline_chars <= 0:
        return 0.0
    raw = (chars - baseline_chars) / baseline_chars
    return max(0.0, min(1.0, raw))


def _genotype_chars(run: Any) -> int:
    """Character length of the run's workflow source code, ``0`` when missing."""
    code = _safe_attr(run, "code", None)
    return len(code) if isinstance(code, str) else 0


if __name__ == "__main__":
    from types import SimpleNamespace

    def _run(reward: float, descriptor: list[float], uuid: str, *, code_len: int = 100) -> SimpleNamespace:
        return SimpleNamespace(
            reward=reward, reward_uncapped=reward, current_uuid=uuid,
            iteration_count=1, cost=0.0,
            code="x" * code_len,
            behaviour_descriptor=descriptor,
        )

    sp = SelectionPressure(
        strategy="qd", population_size=50, novelty_k_neighbours=25,
        novelty_weight=0.4, length_penalty_baseline_chars=5000,
        length_penalty_lambda=0.05,
    )

    seed = _run(0.97, [0.6, 0.0, 0.0, 0.8], "seed")
    sp._validate_open_ended([seed], [seed], threshold=0.01)
    assert len(sp._archive) == 1, sp._archive

    # Distinct genotype embedding → admitted despite lower reward.
    distinct = _run(0.91, [0.0, 1.0, 0.0, 0.0], "distinct")
    sp._validate_open_ended([seed], [distinct], threshold=0.01)
    assert len(sp._archive) == 2, [m.uuid for m in sp._archive]

    # Length penalty discourages an oversized genotype: a 10x-longer
    # neutral candidate should land but with qd_score reduced.
    bloated = _run(0.85, [0.7, 0.7, 0.0, 0.0], "bloated", code_len=60000)
    sp._validate_open_ended([seed], [bloated], threshold=0.01)
    bloated_member = next((m for m in sp._archive if m.uuid == "bloated"), None)
    assert bloated_member is not None
    assert _length_penalty(60000, 5000) == 1.0

    # previous_n mode: cosine distance against the recent window.
    sp_pn = SelectionPressure(
        strategy="qd", population_size=50, novelty_weight=0.4,
        novelty_comparison="previous_n", previous_n=4,
    )
    sp_pn._validate_open_ended([seed], [seed], threshold=0.01)
    sp_pn._validate_open_ended([seed], [distinct], threshold=0.01)
    assert len(sp_pn._previous_descriptors) == 2
    assert sp_pn._comparison_peers(), "previous_n peers should be populated"

    print("smoke OK: cosine novelty + length penalty + previous_n mode")
