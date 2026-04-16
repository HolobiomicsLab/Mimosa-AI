"""
Evolution Strategy for Darwin Gödel Machine.

Supports two modes:
  1. Greedy (current default): validates that the latest run improved over recent history.
  2. Open-ended (future): maintains a population archive, uses novelty + quality
     to decide which individuals survive — giving low-performers a chance if they
     explore a novel region of workflow-space.

The public API is intentionally simple so the DGM can switch strategies via config
without changing its own code.
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
    created_at: datetime = field(default_factory=datetime.now)


class ImprovementValidator:
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
        min_improvement_threshold: float = 0.05,
        strategy: str | SelectionStrategy = SelectionStrategy.GREEDY,
        population_size: int = 20,
        novelty_k_neighbours: int = 5,
        novelty_weight: float = 0.4,
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

        # Population archive for open-ended modes
        self._archive: list[PopulationMember] = []

    # ------------------------------------------------------------------
    # Public API — called by DGM
    # ------------------------------------------------------------------

    def validate_improvement(
        self,
        baseline_runs: list[Any] | Any,
        new_runs: list[Any] | Any,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Validate whether the new run(s) represent a meaningful step forward.

        Accepts both single ``IndividualRun`` objects **and** lists of runs
        for population-aware evaluation.

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
            # Fallback to greedy
            return self._validate_greedy(baseline_list, new_list, threshold)

    def should_continue_iteration(
        self,
        current_reward: float,
        best_reward: float,
        iterations_without_improvement: int,
        max_iterations_without_improvement: int = 3,
    ) -> bool:
        """Determine whether the evolution loop should keep iterating.

        In open-ended mode this is more permissive — it allows continued
        exploration even without reward improvement, as long as the archive
        is still gaining novel members.
        """
        if current_reward > best_reward:
            return True

        if self.strategy in (SelectionStrategy.NOVELTY, SelectionStrategy.QUALITY_DIVERSITY):
            # Open-ended: allow more patience (2× the greedy budget)
            effective_max = max_iterations_without_improvement * 2
        else:
            effective_max = max_iterations_without_improvement

        if iterations_without_improvement >= effective_max:
            self.logger.warning(
                f"⏹️ Stopping: {iterations_without_improvement} iterations "
                f"without improvement (max: {effective_max}, strategy: {self.strategy.value})"
            )
            return False
        return True

    def get_improvement_type(
        self,
        baseline_run: Any,
        new_run: Any,
    ) -> str:
        """Classify the type of improvement: performance, cost, both, or none."""
        baseline_reward = _safe_attr(baseline_run, "reward", 0.0)
        new_reward = _safe_attr(new_run, "reward", 0.0)
        baseline_cost = _safe_attr(baseline_run, "cost", 0.0)
        new_cost = _safe_attr(new_run, "cost", 0.0)

        reward_improved = new_reward > baseline_reward * 1.01
        cost_improved = new_cost < baseline_cost * 0.99

        if reward_improved and cost_improved:
            return "both"
        elif reward_improved:
            return "performance"
        elif cost_improved:
            return "cost"
        return "none"

    def select_parent(self, runs: list[Any]) -> Any:
        """Select a parent for the next mutation from the run history.

        In greedy mode: always returns the best-scoring run.
        In tournament mode: probabilistic tournament among a random subset.
        In novelty/QD mode: selects from archive biased toward high QD-score.
        """
        if not runs:
            return None

        if self.strategy == SelectionStrategy.GREEDY:
            return max(runs, key=lambda r: _safe_attr(r, "reward", 0.0))

        if self.strategy == SelectionStrategy.TOURNAMENT:
            k = min(3, len(runs))
            candidates = random.sample(runs, k)
            return max(candidates, key=lambda r: _safe_attr(r, "reward", 0.0))

        # Novelty / QD: use archive if populated, else fallback
        if self._archive:
            weights = [max(m.qd_score, 0.01) for m in self._archive]
            chosen = random.choices(self._archive, weights=weights, k=1)[0]
            # Find the matching run (by iteration or uuid)
            for r in runs:
                if _safe_attr(r, "current_uuid", None) == chosen.uuid:
                    return r
            # Fallback: return most novel from archive mapped to runs
            return runs[-1]

        return max(runs, key=lambda r: _safe_attr(r, "reward", 0.0))

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
        """Novelty / QD validation: add new run to archive if it brings
        either performance or behavioural novelty."""
        baseline_reward = _mean_reward(baseline_list)
        new_reward = _best_reward(new_list)
        best_new = max(new_list, key=lambda r: _safe_attr(r, "reward", 0.0))

        # Build behaviour descriptor from available run features
        descriptor = self._extract_behaviour_descriptor(best_new)
        novelty = self._compute_novelty(descriptor)

        # QD score = weighted combination of normalised quality + novelty
        quality_norm = min(new_reward, 1.0)  # rewards are 0-1
        novelty_norm = min(novelty / max(self._novelty_range(), 1e-6), 1.0)
        qd_score = (1 - self.novelty_weight) * quality_norm + self.novelty_weight * novelty_norm

        # Accept if QD score exceeds threshold OR if it improves reward
        absolute_improvement = new_reward - baseline_reward
        relative_improvement = absolute_improvement / max(abs(baseline_reward), 1e-6)
        is_valid = relative_improvement > threshold or qd_score > 0.3

        # Add to archive
        member = PopulationMember(
            iteration=_safe_attr(best_new, "iteration_count", 0),
            reward=new_reward,
            cost=_safe_attr(best_new, "cost", 0.0),
            uuid=_safe_attr(best_new, "current_uuid", None),
            behaviour_descriptor=descriptor,
            novelty_score=novelty,
            qd_score=qd_score,
        )
        self._add_to_archive(member)

        confidence = min(1.0, qd_score / max(0.3, 1e-6))

        result = self._build_result(
            is_valid, relative_improvement, absolute_improvement,
            baseline_reward, new_reward, confidence, threshold,
        )
        result["novelty_score"] = novelty
        result["qd_score"] = qd_score
        result["archive_size"] = len(self._archive)

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


# ------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    from sources.core.schema import IndividualRun

    baseline = IndividualRun(goal="test", prompt="test", reward=0.50)
    improved = IndividualRun(goal="test", prompt="test", reward=0.65)
    marginal = IndividualRun(goal="test", prompt="test", reward=0.51)

    # ── Greedy (backward-compatible) ──────────────────────────────
    print("=== GREEDY strategy ===")
    validator = ImprovementValidator(min_improvement_threshold=0.05, strategy="greedy")

    print("Test 1: Valid improvement (list input)")
    result = validator.validate_improvement([baseline], [improved])
    print(f"  Valid: {result['valid']}, Improvement: {result['relative_improvement']:.1%}")

    print("Test 2: Marginal improvement")
    result = validator.validate_improvement([baseline, baseline], [marginal])
    print(f"  Valid: {result['valid']}, Improvement: {result['relative_improvement']:.1%}")

    print("Test 3: Single-object input (backward compat)")
    result = validator.validate_improvement(baseline, improved)
    print(f"  Valid: {result['valid']}, Improvement: {result['relative_improvement']:.1%}")

    # ── Tournament ────────────────────────────────────────────────
    print("\n=== TOURNAMENT strategy ===")
    validator_t = ImprovementValidator(strategy="tournament")
    result = validator_t.validate_improvement([baseline, baseline], [marginal])
    print(f"  Valid: {result['valid']}, Improvement: {result['relative_improvement']:.1%}")

    # ── Quality-Diversity ─────────────────────────────────────────
    print("\n=== QUALITY-DIVERSITY strategy ===")
    validator_qd = ImprovementValidator(strategy="qd", novelty_weight=0.4)
    for i in range(5):
        run = IndividualRun(goal="test", prompt="test", reward=0.3 + i * 0.1, cost=1.0 - i * 0.1)
        run.iteration_count = i
        result = validator_qd.validate_improvement([baseline], [run])
        print(f"  Iter {i}: valid={result['valid']}, qd={result.get('qd_score', 'N/A'):.3f}, "
              f"archive={result.get('archive_size', 0)}")

    print(f"\n  Archive contents ({len(validator_qd.archive)} members):")
    for m in validator_qd.archive:
        print(f"    reward={m.reward:.2f}, novelty={m.novelty_score:.3f}, qd={m.qd_score:.3f}")

    # ── Improvement type ──────────────────────────────────────────
    print("\n=== Improvement type ===")
    baseline_cost = IndividualRun(goal="test", prompt="test", reward=0.50, cost=1.00)
    improved_both = IndividualRun(goal="test", prompt="test", reward=0.65, cost=0.80)
    print(f"  Type: {validator.get_improvement_type(baseline_cost, improved_both)}")

    # ── Parent selection ──────────────────────────────────────────
    print("\n=== Parent selection ===")
    runs = [baseline, marginal, improved]
    print(f"  Greedy parent: reward={validator.select_parent(runs).reward}")
    print(f"  Tournament parent: reward={validator_t.select_parent(runs).reward}")
