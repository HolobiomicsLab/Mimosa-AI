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
        novelty_k_neighbours: int = 10,
        novelty_weight: float = 0.4,
        admit_threshold: float = 0.3,
        pareto_reward_epsilon: float = 0.02,
        pareto_novelty_epsilon: float = 0.0,
        max_children: int = MAX_CHILDREN_PER_PARENT
    ):
        """
        Args:
            min_improvement_threshold: Minimum relative improvement for greedy mode (5% default).
            strategy: Selection strategy (greedy | tournament | novelty | qd).
            population_size: Max individuals kept in the archive (for open-ended modes).
            novelty_k_neighbours: k for k-nearest novelty calculation.
            novelty_weight: Weight of novelty vs quality in QD score (0 = pure quality, 1 = pure novelty).
            pareto_reward_epsilon: Absolute reward delta below which two members are
                treated as equivalent on the reward axis of the Pareto admit gate.
                Without ε, a 0.001 lead on a near-saturated reward axis is enough
                to dominate every behaviourally-distinct sibling.
            pareto_novelty_epsilon: Same idea on the novelty axis. Default 0 because
                novelty is k-NN distance and already scale-relative.
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
        self.pareto_reward_epsilon = pareto_reward_epsilon
        self.pareto_novelty_epsilon = pareto_novelty_epsilon
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
        """Select one or more parents from a candidate pool
        Args:
            candidates: Pool of objects with a reward (or overall_score) attribute
            n_parents:  Number of parents to pick when crossover fires (≥2).
            crossover_rate: Probability ∈ [0, 1] of choosing crossover over mutation.
            child_counts: Optional ``{uuid: n_children_already}`` map forwarded to
                `select_parent` to apply an inverse-child-count penalty.
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
        """Novelty / QD validation: admit to archive if the candidate is improving or behaviourally novel.
        QD weighting uses ``reward_uncapped`` (base + info_bonus − cheat)
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
           Reads run.code and returns a fixed-length vector of structural features.
        """
        return extract_code_features(_safe_attr(run, "code", None))

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

    def _hypothetical_novelties(
        self, candidate: PopulationMember
    ) -> tuple[float, dict[int, float]]:
        """Return novelty values computed against ``archive ∪ {candidate}``.

        Without this, the first archive member is stored with the bootstrap
        default ``novelty=1.0`` (computed when no peers existed yet) which
        permanently dominates every later arrival on the novelty axis of the
        Pareto admit gate. Computing both the candidate's *and* the existing
        members' novelty inside a hypothetical archive that contains the
        candidate makes the comparison scale-consistent.

        Returns:
            (candidate_novelty, {id(member): member_novelty}) — keyed by
            id() so callers don't conflate distinct members with equal hashes.
        """
        hypothetical = self._archive + [candidate]

        def _knn(target: PopulationMember) -> float:
            distances = [
                _euclidean(target.behaviour_descriptor, o.behaviour_descriptor)
                for o in hypothetical
                if o is not target
            ]
            if not distances:
                return 0.0
            distances.sort()
            k = min(self.novelty_k, len(distances))
            return sum(distances[:k]) / k if k > 0 else 0.0

        cand_nov = _knn(candidate)
        member_novs = {id(m): _knn(m) for m in self._archive}
        return cand_nov, member_novs

    def _is_dominated(self, candidate: PopulationMember) -> bool:
        """Pareto domination on (reward_uncapped, novelty_score) with ε-bands.

        Uses *hypothetical* novelties (see ``_hypothetical_novelties``) so the
        comparison reflects the archive that would exist *after* admission, not
        the stale snapshot from the bootstrap. Applies an absolute ε on each
        axis when deciding "strictly better" — a 0.001 reward lead on a near-
        saturated axis should not be enough to dominate a behaviourally-distinct
        sibling.
        """
        if not self._archive:
            return False
        cand_nov, member_novs = self._hypothetical_novelties(candidate)
        eps_r = self.pareto_reward_epsilon
        eps_n = self.pareto_novelty_epsilon
        for m in self._archive:
            m_nov = member_novs[id(m)]
            ge_reward = m.reward_uncapped >= candidate.reward_uncapped - eps_r
            ge_novelty = m_nov >= cand_nov - eps_n
            strictly = (
                m.reward_uncapped > candidate.reward_uncapped + eps_r
                or m_nov > cand_nov + eps_n
            )
            if ge_reward and ge_novelty and strictly:
                return True
        return False

    def _try_admit(self, member: PopulationMember, is_valid: bool) -> bool:
        """Gate archive admission. Returns True when ``member`` is added.

        Rejects regressions that are neither improving nor novel, and
        rejects anything strictly Pareto-dominated by an existing
        archive member.
        """
        if not is_valid or self._is_dominated(member):
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
        """Add a member to the archive, evicting the weakest if full.

        After membership changes (admission and eviction) we refresh every
        member's stored ``novelty_score`` and ``qd_score``. The bootstrap
        member is admitted with ``novelty=1.0`` (no peers existed yet to
        measure against), but as soon as a second member arrives that value
        is stale — and used by ``select_parent``'s QD weighting and by the
        Pareto admit gate. Recomputing keeps both consistent with the current
        k-NN scale.
        """
        self._archive.append(member)

        if len(self._archive) > self.population_size:
            # Evict the member with the lowest QD score
            weakest = min(self._archive, key=lambda m: m.qd_score)
            self._archive.remove(weakest)
            self.logger.debug(
                f" Evicted archive member (qd={weakest.qd_score:.3f}, "
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