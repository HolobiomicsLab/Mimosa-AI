"""Selection policies for the evolution loop (Layer 2: engine + QD search).

``SelectionPressure`` decides which workflow runs survive a generation and
which parents feed the next one. ``sources.core.evolution_engine`` builds one
per session and calls it after every evaluated run; ``workflow_selection``
reuses it to pick parents among stored workflows. Greedy mode keeps a run only
when it beats the recent baseline; novelty and quality-diversity (QD) modes
also keep behaviourally novel runs. Novelty is ``1 - cosine_similarity``
between unit-norm genotype embeddings from :mod:`sources.core.code_features`,
and a length penalty stops ``qd_score`` from rewarding code bloat.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .code_features import genotype_embedding_descriptor

MAX_CHILDREN_PER_PARENT = 8

# Comparison-set modes for novelty.
NOVELTY_ARCHIVE_KNN = "archive_knn"
NOVELTY_PREVIOUS_N = "previous_n"


class SelectionStrategy(Enum):
    """Selection strategies offered to the evolution loop.

    ``sources.core.evolution_engine`` reads ``selection_strategy`` from config
    and passes the chosen name here. GREEDY keeps only strict improvements;
    TOURNAMENT accepts near-ties and regressions with decaying probability;
    NOVELTY and QUALITY_DIVERSITY also admit behaviourally novel runs into the
    archive.
    """
    GREEDY = "greedy"             # Accept only if strictly better (current behaviour)
    TOURNAMENT = "tournament"     # Probabilistic tournament selection
    NOVELTY = "novelty"           # Novelty search: reward behavioural diversity
    QUALITY_DIVERSITY = "qd"      # Unstructured archive scored by scalarised novelty + quality


@dataclass
class PopulationMember:
    """One individual stored in the quality-diversity archive.

    ``SelectionPressure`` builds a member whenever an open-ended run is
    validated, and eviction keeps the archive within ``population_size``.
    Invariant: ``behaviour_descriptor`` is unit-norm or empty, and ``qd_score``
    is recomposed from ``reward``, ``novelty_score`` and ``genotype_chars``.

    Attributes:
        iteration: int — engine iteration index that produced this run.
        reward: float — capped ``overall_score``; the quality term of QD.
        cost: float — monetary or compute cost of producing this run.
        uuid: str | None — workflow UUID; ``None`` when the run has no
            on-disk artifacts.
        behaviour_descriptor: list[float] — unit-norm genotype embedding for
            the novelty signal; empty when no embedding could be produced.
        genotype_chars: int — source-code length feeding the length penalty.
        novelty_score: float — mean cosine distance to the comparison set.
        qd_score: float — novelty + quality score minus the length penalty.
        created_at: datetime — wall-clock time when the member was built.
    """
    iteration: int
    reward: float
    cost: float
    uuid: str | None = None
    behaviour_descriptor: list[float] = field(default_factory=list)
    genotype_chars: int = 0
    novelty_score: float = 0.0
    qd_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


class SelectionPressure:
    """Keeps or discards each new run and samples parents for the next one.

    ``sources.core.evolution_engine`` constructs one instance per session and
    calls ``validate_survivor`` after every evaluated run. Invariants: the
    archive never holds more than ``population_size`` members, and novelty is
    ``1 - cosine_similarity`` between unit-norm genotype embeddings.
    """

    def __init__(
        self,
        config: dict[str, Any],
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
        """Store thresholds, strategy and archive state for one session.

        ``sources.core.evolution_engine`` calls it once per session with the
        loaded config; tests and the ``__main__`` smoke check pass smaller
        values. It copies config-driven knobs into attributes and creates
        the empty archive.

        Args:
            config: dict[str, Any] — session config from the engine; kept
                for reference only.
            min_improvement_threshold: float — greedy acceptance bar as a
                relative reward gain; from ``min_improvement_threshold`` in
                config (default 0.01).
            strategy: str | SelectionStrategy — strategy name or member;
                from ``selection_strategy`` in config (default
                QUALITY_DIVERSITY).
            population_size: int — archive capacity before eviction
                (default 25).
            novelty_k_neighbours: int — nearest archive members averaged by
                ``archive_knn`` novelty (default 10).
            novelty_weight: float — novelty share of ``qd_score`` in
                [0, 1] (default 0.25).
            admit_threshold: float — minimum ``qd_score`` that admits a run
                when greedy validity fails (default 0.3).
            max_children: int — children allowed per parent before its
                sampling weight decays (default ``MAX_CHILDREN_PER_PARENT``).
            novelty_comparison: str — ``"archive_knn"`` or ``"previous_n"``;
                selects the novelty comparison set. Other values fall back
                to ``"archive_knn"``.
            previous_n: int — window size for ``"previous_n"`` novelty
                (default 5).
            length_penalty_baseline_chars: int — code length where the
                length penalty starts (default 5000).
            length_penalty_lambda: float — length-penalty weight subtracted
                from ``qd_score`` (default 0.05).

        Returns:
            None.

        Raises:
            ValueError: when ``strategy`` is a string that names no member
                of ``SelectionStrategy``.
        """
        self.config = config
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
        """Judge whether the latest run(s) survive into the next generation.

        The evolution engine calls it after each evaluated run with recent
        baseline runs and fresh candidate runs. It dispatches to the
        strategy-specific validator and returns the verdict dict the engine
        logs and acts on.

        Args:
            baseline_runs: list or IndividualRun — previous runs; each needs
                a ``reward`` attribute (engine state).
            new_runs: list or IndividualRun — candidate runs being judged
                (engine state).
            threshold: float | None — override for
                ``min_improvement_threshold``; ``None`` uses the configured
                value.

        Returns:
            dict[str, Any] with ``valid``, ``relative_improvement``,
            ``absolute_improvement``, ``baseline_reward``, ``new_reward``,
            ``confidence``, ``threshold_used``, ``strategy`` and
            ``validated_at``; novelty/QD modes add ``novelty_score``,
            ``qd_score``, ``archive_size``, ``admit_rejected`` and
            ``admit_rejected_total``.
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
        """Sample one parent run for the next mutation or crossover.

        The engine calls it per offspring, and ``workflow_selection`` calls
        it through its score adapter when picking stored workflows. Greedy
        and tournament modes pick by reward from ``runs``; novelty/QD modes
        sample a ``PopulationMember`` from the archive with weight
        proportional to ``qd_score``.

        Args:
            runs: list — candidate pool; run objects with ``reward``, or
                ``PopulationMember`` items for archive-driven sampling.
            child_counts: dict[str, int] | None — ``{uuid: children so
                far}``; caps a parent at ``max_children`` children and
                divides its weight by ``1 + n_children`` in QD/novelty mode.

        Returns:
            The chosen parent — a run object or a ``PopulationMember`` — or
            ``None`` when neither ``runs`` nor the archive holds a
            candidate.
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
        """Pick one or more parents and decide mutation versus crossover.

        The engine calls it before each variation step; a coin flip at
        ``crossover_rate`` picks crossover, which needs at least two
        distinct parents drawn without replacement.

        Args:
            candidates: list — pool of objects with a ``reward`` (or
                ``overall_score``) attribute.
            n_parents: int — parents drawn when crossover fires (default
                2; raised to at least 2).
            crossover_rate: float — probability in [0, 1] of crossover
                over mutation.
            child_counts: dict[str, int] | None — forwarded to
                ``select_parent`` for the inverse-child-count penalty.

        Returns:
            tuple[list, bool] — ``(parents, use_crossover)``;
            ``use_crossover`` is False when fewer than two distinct
            parents could be drawn, and ``parents`` is empty only when
            ``candidates`` is empty.
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
        """Return a defensive copy of the population archive.

        Read-only consumers such as the engine's archive plot call it.
        Returns list[PopulationMember]; empty before the first admitted run.
        """
        return list(self._archive)

    def refresh_rewards(self, rewards: dict[str, float]) -> None:
        """Fold current tournament ratings into archive quality scores.

        The evolution engine calls it after each Bradley-Terry fit so parent
        sampling and eviction use fresh ratings. Members whose UUID is not
        in ``rewards`` keep their old values.

        Args:
            rewards: dict[str, float] — fitted rating per workflow UUID,
                from the policy-BT tournament.

        Returns:
            None. Matching members get ``reward`` and a recomposed
            ``qd_score``; stored novelty and length stay unchanged.
        """
        novelty_range = self._novelty_range()
        for member in self._archive:
            if member.uuid not in rewards:
                continue
            member.reward = rewards[member.uuid]
            member.qd_score = self._compose_qd_score(
                member.reward, member.novelty_score,
                member.genotype_chars, novelty_range,
            )

    # ------------------------------------------------------------------
    # Greedy strategy (backward-compatible)
    # ------------------------------------------------------------------

    def _validate_greedy(
        self,
        baseline_list: list[Any],
        new_list: list[Any],
        threshold: float,
    ) -> dict[str, Any]:
        """Greedy validation: the best new run must beat the mean baseline.

        ``validate_survivor`` dispatches here in greedy mode. Runs expose
        ``reward``; ``threshold`` carries the relative gain required.

        Args:
            baseline_list: list — previous runs forming the mean-reward bar.
            new_list: list — candidate runs; only the best one is judged.
            threshold: float — relative gain required for acceptance.

        Returns:
            dict[str, Any] — result assembled by ``_build_result``.
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
        """Tournament validation with probabilistic acceptance.

        ``validate_survivor`` dispatches here in tournament mode. A gain
        above ``threshold`` always passes; a near-tie (within ``threshold``)
        passes on a 50% coin flip; a larger regression passes with
        probability ``exp(-10 * |delta|)``.

        Args:
            baseline_list: list — previous runs; up to three are sampled as
                the bar.
            new_list: list — candidate runs; only the best one is judged.
            threshold: float — relative gain accepted unconditionally.

        Returns:
            dict[str, Any] — result assembled by ``_build_result``.
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
        """Novelty/QD validation: admit improving or behaviourally novel runs.

        ``validate_survivor`` dispatches here in NOVELTY and QUALITY_DIVERSITY
        modes. Quality uses the capped reward, so ties at the cap are broken
        by novelty and the length penalty. A run passes when its relative
        gain beats ``threshold`` or its ``qd_score`` beats ``admit_threshold``;
        the run is scored, offered to the archive, and recorded in the
        recent-genotype window either way.

        Args:
            baseline_list: list — previous runs forming the mean-reward bar.
            new_list: list — candidate runs; only the best one is judged.
            threshold: float — relative gain required for greedy validity.

        Returns:
            dict[str, Any] — ``_build_result`` keys plus ``novelty_score``,
            ``qd_score``, ``quality_norm``, ``novelty_norm``,
            ``length_penalty``, ``behaviour_descriptor``, ``archive_size``,
            ``admit_rejected``, ``admit_rejected_total`` and ``evicted_uuid``.
        """
        self._last_evicted_uuid = None

        baseline_reward = _mean_reward(baseline_list)
        new_reward = _best_reward(new_list)
        best_new = max(new_list, key=lambda r: _safe_attr(r, "reward", 0.0))
        new_reward = _safe_attr(best_new, "reward", 0.0) or new_reward

        descriptor = self._extract_behaviour_descriptor(best_new)
        genotype_chars = _genotype_chars(best_new)
        novelty = self._compute_novelty(descriptor)

        novelty_range = self._novelty_range()
        quality_norm = min(max(new_reward, 0.0), 1.0)
        novelty_norm = min(novelty / max(novelty_range, 1e-6), 1.0)
        length_penalty = _length_penalty(genotype_chars, self.length_baseline_chars)
        qd_score = self._compose_qd_score(
            new_reward, novelty, genotype_chars, novelty_range,
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
        """Return the unit-norm genotype embedding for one run.

        ``_validate_open_ended`` calls it for every candidate. A precomputed
        ``run.behaviour_descriptor`` (tests inject stub vectors) wins; else
        ``run.code`` is embedded via ``sources.core.code_features``.

        Args:
            run: Any — run object exposing ``code`` and optionally a
                precomputed ``behaviour_descriptor``.

        Returns:
            list[float] | None — unit-norm descriptor, or ``None`` when no
            usable genotype exists. Callers treat ``None`` as neutral, never
            as maximally novel.
        """
        precomputed = _safe_attr(run, "behaviour_descriptor", None)
        if isinstance(precomputed, list) and precomputed:
            return [float(x) for x in precomputed]
        code = _safe_attr(run, "code", None)
        return genotype_embedding_descriptor(code)

    def _compute_novelty(self, descriptor: list[float] | None) -> float:
        """Mean cosine distance from a descriptor to its comparison set.

        ``_validate_open_ended`` calls it once per candidate. ``archive_knn``
        averages the ``novelty_k`` nearest archive members; ``previous_n``
        averages the whole recent window.

        Args:
            descriptor: list[float] | None — candidate embedding; ``None`` or
                empty yields ``0.0`` so a missing embedding adds nothing.

        Returns:
            float — novelty in [0, 2] (cosine-distance range for unit
            vectors); ``0.0`` when the comparison set is empty.
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
        """Return the non-empty descriptors of the active comparison set.

        ``_compute_novelty`` and ``_novelty_range`` call it. ``previous_n``
        mode reads the recent window; other modes read archive members.

        Returns:
            list[list[float]] — comparison descriptors; empty on a cold
            start.
        """
        if self.novelty_comparison == NOVELTY_PREVIOUS_N:
            return [p for p in self._previous_descriptors if p]
        return [m.behaviour_descriptor for m in self._archive if m.behaviour_descriptor]

    def _record_previous(self, descriptor: list[float] | None) -> None:
        """Append a descriptor to the previous-N sliding window.

        ``_validate_open_ended`` calls it after every candidate so later
        runs compare against recent genotypes. Empty descriptors are
        ignored, and the window keeps at most ``previous_n`` entries.

        Args:
            descriptor: list[float] | None — candidate embedding to record.

        Returns:
            None.
        """
        if not descriptor:
            return
        self._previous_descriptors.append(list(descriptor))
        if len(self._previous_descriptors) > self.previous_n:
            self._previous_descriptors.pop(0)

    def _novelty_range(self) -> float:
        """Estimate the novelty scale used to normalise ``qd_score``.

        Novelty scoring and reward refreshes call it. ``previous_n`` mode
        uses the largest pairwise distance inside the window; other modes
        use the largest stored member novelty.

        Returns:
            float — the observed maximum, or ``1.0`` when too few members
            exist to estimate one.
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
        """Admit a validated member to the archive or count the rejection.

        ``_validate_open_ended`` calls it with the validity verdict.
        Rejected members only bump the rejection counter and log a line.

        Args:
            member: PopulationMember — candidate archive entry built from
                the best new run.
            is_valid: bool — verdict computed by the caller.

        Returns:
            bool — True when the member was admitted, False when rejected.
        """
        if not is_valid:
            self._n_admit_rejected += 1
            self.logger.info(
                f"ADMIT REJECTED (reward={member.reward:.3f}, "
                f"qd={member.qd_score:.3f}, novelty={member.novelty_score:.3f}) — "
                f"total rejected={self._n_admit_rejected}"
            )
            return False
        self._add_to_archive(member)
        return True

    def _add_to_archive(self, member: PopulationMember) -> None:
        """Append a member and evict the weakest peer when over capacity.

        ``_try_admit`` calls it for accepted candidates. When the archive
        exceeds ``population_size``, the lowest-``qd_score`` member is
        removed and its UUID recorded for logging.

        Args:
            member: PopulationMember — accepted member to append.

        Returns:
            None.
        """
        self._archive.append(member)

        if len(self._archive) > self.population_size:
            weakest = min(self._archive, key=lambda m: m.qd_score)
            self._archive.remove(weakest)
            self._last_evicted_uuid = getattr(weakest, "uuid", None)
            self.logger.debug(
                f"Evicted archive member (qd={weakest.qd_score:.3f}, "
                f"reward={weakest.reward:.3f}) — archive full"
            )

        self._refresh_member_metrics()

    def _refresh_member_metrics(self) -> None:
        """Recompute stored novelty and ``qd_score`` for every archive member.

        ``_add_to_archive`` calls it after each admission so stored scores
        reflect the changed archive. Skipped while fewer than two members
        exist.

        Returns:
            None. Members are updated in place.
        """
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
                m.reward, m.novelty_score, m.genotype_chars, novelty_range,
            )

    def _knn_novelty_against_peers(self, m: PopulationMember) -> float:
        """k-nearest-neighbour novelty of one member within the archive.

        ``_refresh_member_metrics`` calls it for each member. Distances are
        cosine distances to all other members with non-empty descriptors.

        Args:
            m: PopulationMember — member to score.

        Returns:
            float — mean distance to the ``novelty_k`` nearest peers;
            ``0.0`` when no peer has a descriptor.
        """
        distances = sorted(
            _cosine_distance(m.behaviour_descriptor, o.behaviour_descriptor)
            for o in self._archive
            if o is not m and o.behaviour_descriptor
        )
        k = min(self.novelty_k, len(distances))
        return sum(distances[:k]) / k if k > 0 else 0.0

    def _compose_qd_score(
        self,
        reward: float,
        novelty: float,
        genotype_chars: int,
        novelty_range: float,
    ) -> float:
        """Compose the quality-diversity score for one member.

        Admission and refresh paths call it so the weighting stays in one
        place: ``(1 - w) * quality + w * novelty - lambda * length_penalty``.

        Args:
            reward: float — capped run reward (the quality term).
            novelty: float — cosine-distance novelty of the member.
            genotype_chars: int — source-code length for the length penalty.
            novelty_range: float — scale used to normalise novelty.

        Returns:
            float — composed score; quality and novelty are clipped to
            [0, 1] before weighting.
        """
        quality_norm = min(max(reward, 0.0), 1.0)
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
        """Assemble the standard validation-result dictionary.

        Every strategy validator returns its verdict through it, and the
        engine logs and forwards these keys downstream.

        Args:
            is_valid: bool — whether the candidate passed validation.
            relative_improvement: float — reward delta relative to baseline.
            absolute_improvement: float — raw reward delta.
            baseline_reward: float — reward summary of the baseline runs.
            new_reward: float — reward summary of the new runs.
            confidence: float — confidence score in [0, 1].
            threshold: float — threshold applied to the decision.

        Returns:
            dict[str, Any] — the values above under their result keys plus
            ``validated_at`` (current time) and ``strategy`` (enum value).
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
        """Log one validation verdict as an INFO or WARNING line.

        Strategy validators call it right after building their result dict
        so the engine log records each acceptance or rejection.

        Args:
            is_valid: bool — whether the candidate passed validation.
            relative_improvement: float — reward delta relative to baseline.
            baseline_reward: float — reward summary of the baseline runs.
            new_reward: float — reward summary of the new runs.
            confidence: float — confidence score in [0, 1].
            threshold: float — threshold applied to the decision.

        Returns:
            None. Accepted runs log at INFO; rejected runs log at WARNING.
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
    """Read one attribute from a duck-typed object with a safe default.

    Reward summary helpers call it on run objects whose fields may be
    missing.

    Args:
        obj: Any — source object, possibly ``None``.
        attr: str — attribute name to read.
        default: Any — value returned when the object or attribute is
            missing (callers pass ``0.0`` or ``None``).

    Returns:
        Any — the attribute value, or ``default`` when absent.
    """
    return getattr(obj, attr, default) if obj is not None else default


def _mean_reward(runs: list[Any]) -> float:
    """Mean reward across a list of runs.

    Baseline validators call it to set the bar the new run must beat.

    Args:
        runs: list[Any] — run-like objects with ``reward``; ``None`` entries
            are skipped.

    Returns:
        float — arithmetic mean of the remaining rewards; ``0.0`` when no
        run qualifies.
    """
    rewards = [_safe_attr(r, "reward", 0.0) for r in runs if r is not None]
    return sum(rewards) / max(len(rewards), 1)


def _best_reward(runs: list[Any]) -> float:
    """Best (max) reward across a list of runs.

    Validators call it to score the new runs by their strongest member.

    Args:
        runs: list[Any] — run-like objects with ``reward``; ``None`` entries
            are skipped.

    Returns:
        float — the maximum reward, or ``0.0`` when the list is empty.
    """
    rewards = [_safe_attr(r, "reward", 0.0) for r in runs if r is not None]
    return max(rewards) if rewards else 0.0


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Cosine distance between two vectors of equal length.

    Novelty scoring calls it between unit-norm genotype embeddings.

    Args:
        a: list[float] — first vector.
        b: list[float] — second vector.

    Returns:
        float — ``1 - cosine_similarity`` in [0, 2]; ``1.0`` (neutral) when
        either vector is empty, the lengths differ, or a norm is zero.
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
    """Clipped relative excess of a genotype length over a baseline.

    ``_compose_qd_score`` calls it to subtract a code-growth penalty from
    QD scores.

    Args:
        chars: int — candidate genotype character count.
        baseline_chars: int — configured size where the penalty starts.

    Returns:
        float — ``clip((chars - baseline) / baseline, 0, 1)``; ``0.0`` when
        ``baseline_chars`` is not positive.
    """
    if baseline_chars <= 0:
        return 0.0
    raw = (chars - baseline_chars) / baseline_chars
    return max(0.0, min(1.0, raw))


def _genotype_chars(run: Any) -> int:
    """Character length of a run's workflow source code.

    Open-ended validation calls it to feed the length penalty.

    Args:
        run: Any — run object whose ``code`` may be a string.

    Returns:
        int — length of ``run.code``; ``0`` when the code is missing or not
        a string.
    """
    code = _safe_attr(run, "code", None)
    return len(code) if isinstance(code, str) else 0


if __name__ == "__main__":
    from types import SimpleNamespace

    def _run(reward: float, descriptor: list[float], uuid: str, *, code_len: int = 100) -> SimpleNamespace:
        """Build one stub run for this offline smoke check."""
        return SimpleNamespace(
            reward=reward, current_uuid=uuid,
            iteration_count=1, cost=0.0,
            code="x" * code_len,
            behaviour_descriptor=descriptor,
        )

    sp = SelectionPressure(
        config={}, strategy="qd", population_size=50, novelty_k_neighbours=25,
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
        config={}, strategy="qd", population_size=50, novelty_weight=0.4,
        novelty_comparison="previous_n", previous_n=4,
    )
    sp_pn._validate_open_ended([seed], [seed], threshold=0.01)
    sp_pn._validate_open_ended([seed], [distinct], threshold=0.01)
    assert len(sp_pn._previous_descriptors) == 2
    assert sp_pn._comparison_peers(), "previous_n peers should be populated"

    print("smoke OK: cosine novelty + length penalty + previous_n mode")
