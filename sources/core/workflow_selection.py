import logging
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parent.parent.parent))

from config import Config
from sources.core.selection import PopulationMember, SelectionPressure
from sources.core.workflow_info import WorkflowInfo
from sources.core.lineage import scan_all as _scan_lineage


logger = logging.getLogger(__name__)


class _WorkflowScoreAdapter:
    """Lightweight wrapper so that :class:`SelectionPressure.select_parent(s)`
    can rank :class:`WorkflowInfo` objects via their ``overall_score``.

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
    """Discover persisted workflows and pick parents under evolutionary pressure."""

    def __init__(self, config: Config) -> None:
        """Load on-disk workflows and prepare the similarity embedder.

        Args:
            config: Application configuration providing ``workflow_dir``.
        """
        self.config = config
        self.workflows_folder = Path(config.workflow_dir)
        self.workflows_info = self.discover_workflows()
        self.model = SentenceTransformer("all-MiniLM-L6-v2", token=False)

    def discover_workflows(self) -> dict[str, WorkflowInfo]:
        """Scan the workflow folder and return valid, scored workflows.

        Returns:
            Mapping of UUID to :class:`WorkflowInfo` for every folder that is
            valid, has a non-empty state result and loadable code. Empty when
            the workflow directory is missing.
        """
        workflows = {}

        if not self.workflows_folder.exists():
            print(f"Workflows directory {self.workflows_folder} does not exist.")
            return workflows

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
            if not workflow_info.load_state_result():
                continue

            workflow_info.load_code()
            if not workflow_info.code:
                continue

            workflows[uuid] = workflow_info

        return workflows

    def cosine_similarity(self, a: str, b: str) -> float:
        """Calculate cosine similarity between two strings.

        Args:
            a: First text to embed.
            b: Second text to embed.

        Returns:
            Cosine similarity between MiniLM embeddings of `a` and `b`.
        """
        import torch.nn.functional as F

        embeddings_a = self.model.encode(
            a, convert_to_tensor=True, show_progress_bar=False
        )
        embeddings_b = self.model.encode(
            b, convert_to_tensor=True, show_progress_bar=False
        )
        return F.cosine_similarity(embeddings_a, embeddings_b, dim=0).item()

    def sort_similar_workflows(
        self, goal: str, threshold: float = 0.8, debug: bool = False
    ) -> list[WorkflowInfo]:
        """Find workflows with similar goals using original unwrapped tasks.

        Args:
            goal: The task to match against (will be compared with original_task of workflows)
            threshold: Minimum similarity score (0.0-1.0)
            debug: Whether to print debug information

        Returns:
            list[WorkflowInfo]: Workflows sorted by similarity, filtered by threshold
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

        Args:
            workflows_info: Workflows to rank.
            threshold: Minimum ``overall_score`` retained in the output.

        Returns:
            Workflows sorted descending by ``overall_score`` and filtered to
            those at or above `threshold`.
        """
        sorted_workflows = sorted(
            workflows_info, key=lambda wf: wf.overall_score, reverse=True
        )
        return [wf for wf in sorted_workflows if wf.overall_score >= threshold]

    def select_best_workflows(
        self, goal: str, threshold_similarity: float = 0.9, threshold_score: float = 0.1
    ) -> list[WorkflowInfo]:
        """Choose a workflow that matches the goal with a minimum threshold.

        Args:
            goal: Task description used for similarity matching.
            threshold_similarity: Minimum cosine similarity to retain.
            threshold_score: Minimum workflow score to retain.

        Returns:
            Workflows that pass both the similarity and score thresholds.
        """
        similar_workflows = self.sort_similar_workflows(goal, threshold_similarity)
        best_workflows = self.sort_workflows_by_score(similar_workflows, threshold_score)
        return best_workflows

    def _rehydrate_workflow_info(self, uuid: str) -> WorkflowInfo | None:
        """Materialize a WorkflowInfo from disk by UUID. Returns None if invalid.

        Args:
            uuid: Workflow UUID (folder name) to rehydrate.

        Returns:
            A valid :class:`WorkflowInfo`, or ``None`` when `uuid` is empty
            or the on-disk record fails validation.
        """
        if not uuid:
            return None
        wf = WorkflowInfo(uuid, self.workflows_folder / uuid)
        return wf if wf.is_valid() else None

    def _count_children_on_disk(self) -> dict[str, int]:
        """Build ``{parent_uuid: n_children}`` from the on-disk lineage records.

        Used to penalise repeatedly-mined parents during parent draw — without
        this, a single high-qd ancestor monopolises the offspring stream
        (observed empirically in early evolution runs).

        Returns:
            Mapping of parent UUID to the number of recorded direct children.
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
        """Archive-driven parent selection (steady-state, current session).

        Args:
            selection_pressure: Pressure instance whose ``_archive`` is
                sampled for parents.
            n_parents: Maximum parents to draw when crossover fires.
            crossover_rate: Probability of crossover over mutation.

        Returns:
            ``(workflows, use_crossover)``. ``workflows`` is empty when
            rehydration fails for every sampled member; ``use_crossover`` is
            forced to ``False`` if fewer than two parents survive rehydration.
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
        crossover_rate: float = 0.3,
        threshold_similarity: float = 0.8,
        threshold_score: float = 0.1,
    ) -> tuple[list[WorkflowInfo], bool]:
        """Select one or more parent workflows under evolutionary pressure.

        Steady-state path: when `selection_pressure._archive` is populated,
        sample parents from the live archive (current-session population).
        Cold-start path: fall back to similarity-filtered disk scan for
        cross-task transfer when the archive is empty.

        Args:
            goal: Task description to match against stored workflows.
            selection_pressure: The SelectionPressure instance that governs strategy and
                decides crossover vs mutation.
            n_parents: Maximum number of parents when crossover fires (≥ 2).
            crossover_rate: Probability ∈ [0, 1] that crossover is attempted.
            threshold_similarity: Cosine-similarity floor for cold-start candidates.
            threshold_score: Minimum workflow score for cold-start candidates.

        Returns:
            (list[WorkflowInfo], use_crossover)
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
                    f"🧬 Archive selection ({mode}, strategy={selection_pressure.strategy.value}): "
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
            f"🧬 Cold-start selection ({mode}, strategy={selection_pressure.strategy.value}): "
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
    sp = SelectionPressure(strategy="tournament")
    selected, crossover = mcts.select_parent_workflows(
        goal=goal,
        selection_pressure=sp,
        n_parents=2,
        crossover_rate=0.5,
    )
    mode = "CROSSOVER" if crossover else "MUTATION"
    print(f"  Mode: {mode}, Parents: {len(selected)}")
    for wf in selected:
        print(f"    UUID: {wf.uuid}, Score: {wf.overall_score:.4f}")
