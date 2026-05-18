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


class SelectionStrategy(Enum):
    """Available selection strategies for the evolution loop."""
    GREEDY = "greedy"             # Accept only if strictly better (current behaviour)
    TOURNAMENT = "tournament"     # Probabilistic tournament selection
    NOVELTY = "novelty"           # Novelty search: reward behavioural diversity
    QUALITY_DIVERSITY = "qd"      # MAP-Elites style: novelty + quality combined


@dataclass
class PopulationMember:
    """A single individual in the evolution archive."""
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
        novelty_k_neighbours: int = 5,
        novelty_weight: float = 0.4,
        admit_threshold: float = 0.3,
    ):
        """
        Args:
            min_improvement_threshold: Minimum relative improvement for greedy mode (5% default).
            strategy: Selection strategy (greedy | tournament | novelty | qd).
            population_size: Max individuals kept in the archive (for open-ended modes).
            novelty_k_neighbours: k for k-nearest novelty calculation.
            novelty_weight: Weight of novelty vs quality in QD score (0 = pure quality, 1 = pure novelty).
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
        Args:
            baseline_runs: One or more previous runs (list or single IndividualRun).
            new_runs: One or more candidate runs (list or single IndividualRun).
            threshold: Override for min_improvement_threshold.
        Returns:
            dict with keys: valid, relative_improvement, absolute_improvement,
            baseline_reward, new_reward, confidence, threshold_used, strategy,
            validated_at.
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

    def select_parent(self, runs: list[Any]) -> Any:
        """Select a single parent for mutation.

        In greedy mode: always returns the best-scoring run from `runs`.
        In tournament mode: probabilistic tournament among a random subset of `runs`.
        In novelty/QD mode: samples a `PopulationMember` from `_archive`
            biased toward high QD-score, falling back to greedy over `runs`
            when the archive is empty (cold start).
        Callers driving from archive must rehydrate the chosen member's
        UUID into their domain object (e.g., WorkflowInfo).
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
        # If the caller passed PopulationMember items (multi-parent draw),
        # sample from those so pool-exclusion is respected. Else sample
        # from the global archive.
        if self._archive:
            members = [c for c in (runs or []) if isinstance(c, PopulationMember)]
            if not members:
                members = self._archive
            weights = [max(m.qd_score, 0.01) for m in members]
            return random.choices(members, weights=weights, k=1)[0]

        # Cold start fallback
        return max(runs, key=lambda r: _safe_attr(r, "reward", 0.0)) if runs else None

    def select_parents(
        self,
        candidates: list[Any],
        n_parents: int = 2,
        crossover_rate: float = 0.3,
    ) -> tuple[list[Any], bool]:
        """Select one or more parents from a candidate pool

        The per-strategy selection logic reuses `select_parent` internally
        so that greedy / tournament / novelty / QD biases are respected.
        Args:
            candidates: Pool of objects with a reward (or overall_score) attribute
            n_parents:  Number of parents to pick when crossover fires (≥2).
            crossover_rate: Probability ∈ [0, 1] of choosing crossover over mutation.
        Returns:
            (list[parent], bool) — selected parents and whether to crossover.
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
            parent = self.select_parent(candidates)
            return [parent], False
        selected: list[Any] = []
        pool = list(candidates)  # shallow copy so we can remove picked items

        for _ in range(min(n_parents, len(pool))):
            parent = self.select_parent(pool)
            if parent is None:
                break
            selected.append(parent)
            pool = [c for c in pool if c is not parent]
        # Safety: if we ended up with < 2, fall back to mutation
        if len(selected) < 2:
            return selected or [self.select_parent(candidates)], False
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
        """Classic greedy validation: best-of-new must beat mean-of-baseline."""
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
        to its advantage over a random baseline sample."""
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
        """Novelty / QD validation: admit to archive if the candidate is
        either improving or behaviourally novel.

        QD weighting uses ``reward_uncapped`` (base + info_bonus − cheat)
        so the 0.7 hard-fail cap stops compressing the parent-draw
        gradient. The capped ``reward`` remains the admissibility score
        reported to callers.
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
        """Extract a behaviour descriptor vector from a run.

        Currently uses [reward, cost, iteration_count] as a simple proxy.
        Override or extend this to use richer descriptors (e.g., code
        structure features, tool usage patterns, output characteristics).
        """
        return [
            _safe_attr(run, "reward", 0.0),
            _safe_attr(run, "cost", 0.0),
            float(_safe_attr(run, "iteration_count", 0)),
        ]

    def _compute_novelty(self, descriptor: list[float]) -> float:
        """Compute novelty as mean distance to k-nearest archive members."""
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
        """Estimate the typical novelty scale from the archive."""
        if len(self._archive) < 2:
            return 1.0
        novelties = [m.novelty_score for m in self._archive if m.novelty_score > 0]
        return max(novelties) if novelties else 1.0

    def _is_dominated(self, candidate: PopulationMember) -> bool:
        """Pareto domination on (reward_uncapped, novelty_score).

        ``candidate`` is dominated iff some archive member is ≥ on both
        axes and strictly greater on at least one.
        """
        for m in self._archive:
            ge_reward = m.reward_uncapped >= candidate.reward_uncapped
            ge_novelty = m.novelty_score >= candidate.novelty_score
            strictly = (
                m.reward_uncapped > candidate.reward_uncapped
                or m.novelty_score > candidate.novelty_score
            )
            if ge_reward and ge_novelty and strictly:
                return True
        return False

    def _try_admit(self, member: PopulationMember, is_valid: bool) -> bool:
        """Gate archive admission. Returns True when ``member`` is added.

        Rejects regressions that are neither improving nor novel, and
        rejects anything strictly Pareto-dominated by an existing
        archive member. Rejections increment ``_n_admit_rejected``.
        """
        if not is_valid or self._is_dominated(member):
            self._n_admit_rejected += 1
            self.logger.info(
                f"🚫 ADMIT REJECTED (uncapped={member.reward_uncapped:.3f}, "
                f"qd={member.qd_score:.3f}, novelty={member.novelty_score:.3f}) — "
                f"total rejected={self._n_admit_rejected}"
            )
            return False
        self._add_to_archive(member)
        return True

    def _add_to_archive(self, member: PopulationMember) -> None:
        """Add a member to the archive, evicting the weakest if full."""
        self._archive.append(member)

        if len(self._archive) > self.population_size:
            # Evict the member with the lowest QD score
            weakest = min(self._archive, key=lambda m: m.qd_score)
            self._archive.remove(weakest)
            self.logger.debug(
                f" Evicted archive member (qd={weakest.qd_score:.3f}, "
                f"reward={weakest.reward:.3f}) — archive full"
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
    """Safely get an attribute from an object, returning default if missing."""
    return getattr(obj, attr, default) if obj is not None else default


def _mean_reward(runs: list[Any]) -> float:
    """Mean reward across a list of runs."""
    rewards = [_safe_attr(r, "reward", 0.0) for r in runs if r is not None]
    return sum(rewards) / max(len(rewards), 1)


def _best_reward(runs: list[Any]) -> float:
    """Best (max) reward across a list of runs."""
    rewards = [_safe_attr(r, "reward", 0.0) for r in runs if r is not None]
    return max(rewards) if rewards else 0.0


def _euclidean(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two vectors of equal length."""
    if len(a) != len(b):
        return float("inf")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))