"""Parent selection over persisted workflow generations (Layer 2).

This file owns the evolution engine's view of the ``workflows/``
archive: it discovers valid, scored generation folders, ranks them by
goal similarity (MiniLM embeddings with a lexical Jaccard fallback),
and draws parents under the selection-pressure strategy — steady-state
sampling from the live archive, or a cold-start similarity scan over
disk. It also enforces the oracle leak guard, so genotypes scored by
the benchmark-leaking gold verifier are never reused in a non-gold
run. The evolution engine imports :class:`WorkflowSelector` and
queries it before producing each new generation.
"""

import logging
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from config import Config
from sources.core.selection import PopulationMember, SelectionPressure
from sources.core.workflow_info import WorkflowInfo, is_oracle_generation
from sources.core.lineage import scan_all as _scan_lineage


logger = logging.getLogger(__name__)


class _WorkflowScoreAdapter:
    """Lightweight wrapper so that :class:`SelectionPressure.select_parent(s)`
    can rank :class:`WorkflowInfo` objects via their ``overall_score``.

    Cold-start parent selection builds one adapter per disk candidate.
    The only invariant is the alias: ``reward`` always mirrors
    ``workflow_info.overall_score`` at bind time.

    Attributes:
        workflow_info: The wrapped :class:`WorkflowInfo`.
        reward: Mirror of ``workflow_info.overall_score`` exposed under the
            attribute name expected by selection helpers.
    """

    __slots__ = ("workflow_info", "reward")

    def __init__(self, wf: WorkflowInfo) -> None:
        """Bind a :class:`WorkflowInfo` and expose its score as ``reward``.

        Args:
            wf: Workflow info whose ``overall_score`` becomes the adapter's
                ``reward`` field.
        """
        self.workflow_info = wf
        self.reward = wf.overall_score


class WorkflowSelector:
    """Discover persisted workflows and draw parents under evolutionary pressure.

    The evolution engine constructs one selector per run with the
    application config, and the discovery result is cached on the
    instance. Parent choice runs in two modes: steady-state sampling
    from the live selection archive, or a cold-start similarity scan
    over disk records.
    """

    def __init__(self, config: Config) -> None:
        """Scan persisted workflows and defer the similarity embedder.

        The constructor calls it once with the run config. It fills the
        ``workflows_info`` cache from disk; the MiniLM embedder is not
        loaded here — see :meth:`_get_model` for the lazy strategy.

        Args:
            config: Application configuration providing ``workflow_dir``
                (config file).
        """
        self.config = config
        self.workflows_folder = Path(config.workflow_dir)
        self.workflows_info = self.discover_workflows()
        # Lazy: downloading all-MiniLM-L6-v2 from huggingface.co at startup
        # crashes the whole app where HF is unreachable (e.g. mainland China
        # without VPN). Loaded on first similarity call; falls back to a
        # lexical similarity when the model cannot be loaded.
        self._model = None
        self._model_load_failed = False

    def _get_model(self):
        """Return the MiniLM embedder, or ``None`` when it cannot be loaded.

        Similarity helpers call it before every comparison. The first
        call loads ``all-MiniLM-L6-v2`` (a Hugging Face download; set
        ``HF_ENDPOINT`` to a mirror when the hub is unreachable); later
        calls reuse the cached model. After one failed load it stops
        retrying, and callers take the lexical fallback.

        Returns:
            The loaded sentence-transformer model, or ``None`` when
            loading failed.
        """
        if self._model is not None or self._model_load_failed:
            return self._model
        try:
            from sources.core.genotype_embedding import load_sentence_transformer
            self._model = load_sentence_transformer("all-MiniLM-L6-v2")
        except Exception as exc:
            logger.warning(
                "MiniLM embedder unavailable (%s); falling back to lexical "
                "similarity. Set HF_ENDPOINT=https://hf-mirror.com to "
                "download through the mirror and restore semantic similarity.",
                exc,
            )
            self._model_load_failed = True
        return self._model

    @staticmethod
    def _lexical_similarity(a: str, b: str) -> float:
        """Jaccard similarity over word tokens; offline fallback for MiniLM.

        :meth:`cosine_similarity` calls it when the embedder is
        unavailable, so goal matching keeps working without network
        access.

        Args:
            a: First text; lowercased and tokenized internally.
            b: Second text; lowercased and tokenized internally.

        Returns:
            float: ``|tokens_a ∩ tokens_b| / |tokens_a ∪ tokens_b|``,
            or ``0.0`` when either text has no word tokens.
        """
        tokens_a = set(re.findall(r"\w+", a.lower()))
        tokens_b = set(re.findall(r"\w+", b.lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    def discover_workflows(self) -> dict[str, WorkflowInfo]:
        """Scan the workflow folder and return valid, scored workflows.

        The constructor and tests call it to fill the selector's
        candidate cache. A folder counts only when it validates, holds a
        non-empty state result, and carries loadable genotype code. The
        oracle leak guard applies: in a non-gold run, folders marked by
        :func:`is_oracle_generation` are skipped and counted in a
        warning, so benchmark-leaking genotypes cannot re-enter
        evolution.

        Returns:
            dict[str, WorkflowInfo]: Mapping of UUID to
            :class:`WorkflowInfo`; empty when the workflow directory is
            missing.
        """
        workflows = {}

        if not self.workflows_folder.exists():
            print(f"Workflows directory {self.workflows_folder} does not exist.")
            return workflows

        # Leak guard: genotypes evolved under the benchmark-leaking gold
        # (oracle) verifier are only reusable by another gold run.
        allow_oracle = (
            str(getattr(self.config, "verifier_kind", "")).lower() == "gold"
        )
        skipped_oracle = 0
        for workflow_folder in self.workflows_folder.iterdir():
            if not workflow_folder.is_dir():
                continue

            uuid = workflow_folder.name
            if "single_agent" in uuid:
                continue
            workflow_info = WorkflowInfo(uuid, workflow_folder)

            if not workflow_info.is_valid():
                continue

            # Check if state_result is empty
            state_result = workflow_info.load_state_result()
            if not state_result:
                continue

            if not allow_oracle and is_oracle_generation(workflow_folder, state_result):
                skipped_oracle += 1
                continue

            workflow_info.load_code()
            if not workflow_info.code:
                continue

            workflows[uuid] = workflow_info

        if skipped_oracle:
            logger.warning(
                "[LEAK GUARD] %d generation(s) in %s were evaluated by the "
                "benchmark-leaking gold (oracle) verifier; they are skipped "
                "for this non-gold run. Keep oracle runs in a separate "
                "workflow_dir.",
                skipped_oracle,
                self.workflows_folder,
            )
        return workflows

    def cosine_similarity(self, a: str, b: str) -> float:
        """Calculate cosine similarity between two strings.

        Goal-matching helpers call it to rank stored workflows against a
        new task. MiniLM embeddings are used when the model is
        available; otherwise :meth:`_lexical_similarity` keeps selection
        working offline.

        Args:
            a: First text to embed.
            b: Second text to embed.

        Returns:
            float: Similarity between 0.0 and 1.0.
        """
        model = self._get_model()
        if model is None:
            return self._lexical_similarity(a, b)

        import torch.nn.functional as F

        embeddings_a = model.encode(
            a, convert_to_tensor=True, show_progress_bar=False
        )
        embeddings_b = model.encode(
            b, convert_to_tensor=True, show_progress_bar=False
        )
        return F.cosine_similarity(embeddings_a, embeddings_b, dim=0).item()

    def sort_similar_workflows(
        self, goal: str, threshold: float = 0.8, debug: bool = False
    ) -> list[WorkflowInfo]:
        """Find workflows with similar goals using original unwrapped tasks.

        Cold-start parent selection calls it to rank persisted
        generations against the new task. Only the last 512 characters
        of each text are compared, which keeps long wrapped goals
        comparable.

        Args:
            goal: The task to match against (compared with each
                workflow's ``original_task``).
            threshold: Minimum similarity score (0.0-1.0).
            debug: When ``True``, print each candidate with its
                similarity to stdout.

        Returns:
            list[WorkflowInfo]: Workflows sorted by similarity
            (descending) and filtered to ``>= threshold``; empty when no
            workflows were discovered.
        """
        assert threshold >= 0.0, "Threshold must be non-negative"
        assert threshold <= 1.0, "Threshold must be at most 1.0"
        if not self.workflows_info:
            print("No workflows found.")
            return []

        # Use original_task for comparison to avoid knowledge wrapper interference
        similar_workflows = sorted(
            self.workflows_info.values(),
            key=lambda wf: self.cosine_similarity(wf.original_task[-512:], goal[-512:]),
            reverse=True,
        )

        if debug:
            for wf in similar_workflows:
                sim = self.cosine_similarity(wf.original_task[-512:], goal[-512:])
                print(f"UUID: {wf.uuid}\n"
                      f"Original Task:\n{wf.original_task[:512]}\n"
                      f"Target:\n{goal[:512]}\n"
                      f"Similarity: {sim:.4f}\n---\n")

        return [
            wf
            for wf in similar_workflows
            if self.cosine_similarity(wf.original_task[-512:], goal[-512:]) >= threshold
        ]

    def sort_workflows_by_score(
        self, workflows_info: list[WorkflowInfo], threshold: float
    ) -> list[WorkflowInfo]:
        """Sort workflows by their overall score.

        :meth:`select_best_workflows` calls it after similarity
        filtering to keep only strong candidates.

        Args:
            workflows_info: Candidate workflows to rank (caller state).
            threshold: Minimum ``overall_score`` kept in the output.

        Returns:
            list[WorkflowInfo]: Workflows sorted descending by
            ``overall_score`` and filtered to ``>= threshold``.
        """
        sorted_workflows = sorted(
            workflows_info, key=lambda wf: wf.overall_score, reverse=True
        )
        return [wf for wf in sorted_workflows if wf.overall_score >= threshold]

    def select_best_workflows(
        self, goal: str, threshold_similarity: float = 0.9, threshold_score: float = 0.1
    ) -> list[WorkflowInfo]:
        """Choose workflows that match the goal past both thresholds.

        Cold-start parent selection and the ``__main__`` smoke block
        call it to shortlist candidates for evolutionary pressure.

        Args:
            goal: Task description used for similarity matching.
            threshold_similarity: Minimum cosine similarity to keep
                (0.0-1.0).
            threshold_score: Minimum workflow score to keep.

        Returns:
            list[WorkflowInfo]: Workflows passing both thresholds, best
            match first; empty when nothing qualifies.
        """
        similar_workflows = self.sort_similar_workflows(goal, threshold_similarity)
        best_workflows = self.sort_workflows_by_score(similar_workflows, threshold_score)
        return best_workflows

    def _rehydrate_workflow_info(self, uuid: str) -> WorkflowInfo | None:
        """Materialize a WorkflowInfo from disk by UUID, or return ``None``.

        :meth:`_select_from_archive` calls it to turn sampled archive
        members back into full info objects; candidates whose folder
        vanished or no longer validates are dropped silently.

        Args:
            uuid: Workflow UUID (folder name) to rehydrate (disk scan).

        Returns:
            WorkflowInfo | None: The valid info object, or ``None``
            when `uuid` is empty or the on-disk record fails
            validation.
        """
        if not uuid:
            return None
        wf = WorkflowInfo(uuid, self.workflows_folder / uuid)
        return wf if wf.is_valid() else None

    def _count_children_on_disk(self) -> dict[str, int]:
        """Build ``{parent_uuid: n_children}`` from the on-disk lineage records.

        Archive sampling and cold-start selection pass the result to the
        selection pressure so repeatedly mined parents lose draws —
        without it, a single high-scoring ancestor monopolises the
        offspring stream (observed in early evolution runs).

        Returns:
            dict[str, int]: Mapping of parent UUID to the number of
            recorded direct children; empty when the lineage scan finds
            no records.
        """
        counts: dict[str, int] = {}
        for rec in _scan_lineage(self.workflows_folder).values():
            for parent_uuid in rec.get("parents", []) or []:
                if parent_uuid:
                    counts[parent_uuid] = counts.get(parent_uuid, 0) + 1
        return counts

    def _select_from_archive(
        self,
        selection_pressure: SelectionPressure,
        n_parents: int,
        crossover_rate: float,
    ) -> tuple[list[WorkflowInfo], bool]:
        """Archive-driven parent selection (steady state, current session).

        :meth:`select_parent_workflows` calls it when the selection
        pressure already holds a live archive, so parents come from the
        current session's population instead of a fresh disk scan.

        Args:
            selection_pressure: Pressure instance whose ``_archive`` is
                sampled for parents (caller state).
            n_parents: Maximum parents to draw when crossover fires.
            crossover_rate: Probability of crossover over mutation
                (caller config).

        Returns:
            tuple[list[WorkflowInfo], bool]: ``(workflows, use_crossover)``.
            ``workflows`` is empty when rehydration fails for every
            sampled member; ``use_crossover`` is forced to ``False`` if
            fewer than two parents survive rehydration.
        """
        archive = selection_pressure._archive
        child_counts = self._count_children_on_disk()
        selected_members, use_crossover = selection_pressure.select_parents(
            candidates=archive,
            n_parents=n_parents,
            crossover_rate=crossover_rate,
            child_counts=child_counts,
        )
        # Rehydrate PopulationMember -> WorkflowInfo
        selected_workflows: list[WorkflowInfo] = []
        for m in selected_members:
            uuid = m.uuid if isinstance(m, PopulationMember) else None
            wf = self._rehydrate_workflow_info(uuid)
            if wf is not None:
                selected_workflows.append(wf)
        # If rehydration failed for everyone, treat as cold start (caller falls back)
        if not selected_workflows:
            return [], False
        # Crossover requires ≥ 2 parents post-rehydration
        if use_crossover and len(selected_workflows) < 2:
            use_crossover = False
        return selected_workflows, use_crossover

    def select_parent_workflows(
        self,
        goal: str,
        selection_pressure: SelectionPressure,
        n_parents: int = 2,
        crossover_rate: float = 0.1,
        threshold_similarity: float = 0.8,
        threshold_score: float = 0.1,
    ) -> tuple[list[WorkflowInfo], bool]:
        """Select one or more parent workflows under evolutionary pressure.

        The evolution engine calls it once per generation before
        variation. Steady-state path: when ``selection_pressure._archive``
        is populated, parents are sampled from the live archive
        (current-session population). Cold-start path: fall back to a
        similarity-filtered disk scan for cross-task transfer when the
        archive is empty.

        Args:
            goal: Task description to match against stored workflows
                (current run).
            selection_pressure: The SelectionPressure instance that
                governs strategy and decides crossover vs mutation.
            n_parents: Maximum number of parents when crossover fires
                (≥ 2).
            crossover_rate: Probability ∈ [0, 1] that crossover is
                attempted.
            threshold_similarity: Cosine-similarity floor for
                cold-start candidates.
            threshold_score: Minimum workflow score for cold-start
                candidates.

        Returns:
            tuple[list[WorkflowInfo], bool]: ``(selected_workflows,
            use_crossover)``; the list is empty when no candidate
            qualifies, and ``use_crossover`` is ``False`` in that case.
        """
        # Steady-state: archive-driven selection
        if selection_pressure._archive:
            selected_workflows, use_crossover = self._select_from_archive(
                selection_pressure, n_parents, crossover_rate
            )
            if selected_workflows:
                uuids = [wf.uuid for wf in selected_workflows]
                scores = [f"{wf.overall_score:.2f}" for wf in selected_workflows]
                mode = "CROSSOVER" if use_crossover else "MUTATION"
                logger.info(
                    f"Archive selection ({mode}, strategy={selection_pressure.strategy.value}): "
                    f"{len(selected_workflows)} parent(s) from archive size={len(selection_pressure._archive)} "
                    f"— UUIDs={uuids}, scores={scores}"
                )
                return selected_workflows, use_crossover

        # Cold start: similarity-filtered disk scan
        candidates = self.select_best_workflows(
            goal=goal,
            threshold_similarity=threshold_similarity,
            threshold_score=threshold_score,
        )

        if not candidates:
            logger.info("No candidate workflows found for parent selection.")
            return [], False

        adapters = [_WorkflowScoreAdapter(wf) for wf in candidates]
        child_counts = self._count_children_on_disk()
        selected_adapters, use_crossover = selection_pressure.select_parents(
            candidates=adapters,
            n_parents=n_parents,
            crossover_rate=crossover_rate,
            child_counts=child_counts,
        )
        selected_workflows = [a.workflow_info for a in selected_adapters]
        uuids = [wf.uuid for wf in selected_workflows]
        scores = [f"{wf.overall_score:.2f}" for wf in selected_workflows]
        mode = "CROSSOVER" if use_crossover else "MUTATION"
        logger.info(
            f"Cold-start selection ({mode}, strategy={selection_pressure.strategy.value}): "
            f"{len(selected_workflows)} parent(s) from {len(candidates)} disk candidates "
            f"— UUIDs={uuids}, scores={scores}"
        )

        return selected_workflows, use_crossover


if __name__ == "__main__":
    config = Config()
    config.workflow_dir = "../workflows"
    mcts = WorkflowSelector(config)
    goal = "Given the HP sequence HPHPPHHPHPPHPHHPPHPH, find a conformation with energy ≤ −7"
    matching_workflow = mcts.select_best_workflows(goal)
    print("Best matching workflow:")
    for wf in matching_workflow:
        print(f"UUID: {wf.uuid}, Goal: {wf.goal}, Score: {wf.overall_score:.4f}")
    print("\n=== Evolutionary parent selection ===")
    sp = SelectionPressure(config={}, strategy="tournament")
    selected, crossover = mcts.select_parent_workflows(
        goal=goal,
        selection_pressure=sp,
        n_parents=2,
        crossover_rate=0.1,
    )
    mode = "CROSSOVER" if crossover else "MUTATION"
    print(f"  Mode: {mode}, Parents: {len(selected)}")
    for wf in selected:
        print(f"    UUID: {wf.uuid}, Score: {wf.overall_score:.4f}")
