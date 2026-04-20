import logging
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parent.parent.parent))

from config import Config
from sources.core.selection import SelectionPressure
from sources.core.workflow_info import WorkflowInfo


logger = logging.getLogger(__name__)


class _WorkflowScoreAdapter:
    """Lightweight wrapper so that :class:`SelectionPressure.select_parent(s)`
    can rank :class:`WorkflowInfo` objects via their ``overall_score``.

    ``SelectionPressure`` looks for a ``reward`` attribute; WorkflowInfo
    exposes ``overall_score`` instead.  This adapter bridges the gap without
    touching either class.
    """

    __slots__ = ("workflow_info", "reward")

    def __init__(self, wf: WorkflowInfo):
        self.workflow_info = wf
        self.reward = wf.overall_score


class WorkflowSelector:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.workflows_folder = Path(config.workflow_dir)
        self.workflows_info = self.discover_workflows()
        self.model = SentenceTransformer("all-MiniLM-L6-v2", token=False)

    def discover_workflows(self) -> dict[str, WorkflowInfo]:
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
        """Calculate cosine similarity between two strings."""
        import torch.nn.functional as F

        embeddings_a = self.model.encode(
            a, convert_to_tensor=True, show_progress_bar=False
        )
        embeddings_b = self.model.encode(
            b, convert_to_tensor=True, show_progress_bar=False
        )
        return F.cosine_similarity(embeddings_a, embeddings_b, dim=0).item()

    def sort_similar_workflows(
        self, goal: str, threshold=0.8, debug=False
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
        """Sort workflows by their overall score."""
        sorted_workflows = sorted(
            workflows_info, key=lambda wf: wf.overall_score, reverse=True
        )
        return [wf for wf in sorted_workflows if wf.overall_score >= threshold]

    def select_best_workflows(
        self, goal: str, threshold_similarity=0.9, threshold_score=0.1
    ) -> list[WorkflowInfo]:
        """Choose a workflow that matches the goal with a minimum threshold."""
        similar_workflows = self.sort_similar_workflows(goal, threshold_similarity)
        best_workflows = self.sort_workflows_by_score(similar_workflows, threshold_score)
        return best_workflows

    # ------------------------------------------------------------------
    # Evolutionary multi-parent selection
    # ------------------------------------------------------------------

    def select_parent_workflows(
        self,
        goal: str,
        selection_pressure: SelectionPressure,
        n_parents: int = 2,
        crossover_rate: float = 0.3,
        threshold_similarity: float = 0.9,
        threshold_score: float = 0.1,
    ) -> tuple[list[WorkflowInfo], bool]:
        """Select one or more parent workflows under evolutionary pressure.

        This method bridges :class:`WorkflowSelector` (which discovers and
        ranks existing workflows by similarity/score) with
        :class:`SelectionPressure` (which applies strategy-aware selection —
        greedy, tournament, novelty, QD — and decides whether to crossover
        or mutate).

        Workflow:
          1. Discover candidate workflows matching ``goal`` via
             :meth:`select_best_workflows`.
          2. Wrap them in lightweight adapters so ``SelectionPressure`` can
             rank them by ``reward`` (== ``overall_score``).
          3. Delegate to :meth:`SelectionPressure.select_parents` which
             probabilistically picks one parent (mutation) or multiple
             parents (crossover) according to strategy + ``crossover_rate``.

        Args:
            goal: Task description to match against stored workflows.
            selection_pressure: The :class:`SelectionPressure` instance that
                governs strategy (greedy / tournament / novelty / qd) and
                decides crossover vs mutation.
            n_parents: Maximum number of parents when crossover fires (≥ 2).
            crossover_rate: Probability ∈ [0, 1] that crossover is attempted.
                Actual crossover only happens when there are ≥ 2 distinct
                candidates in the pool.
            threshold_similarity: Cosine-similarity floor for candidate
                discovery (passed through to :meth:`select_best_workflows`).
            threshold_score: Minimum workflow score for candidate discovery.

        Returns:
            ``(list[WorkflowInfo], use_crossover)``
            — One or more parent workflows and a flag indicating whether
            the caller should apply crossover (True) or mutation (False).
            Returns ``([], False)`` when no suitable candidate is found.
        """
        candidates = self.select_best_workflows(
            goal=goal,
            threshold_similarity=threshold_similarity,
            threshold_score=threshold_score,
        )

        if not candidates:
            logger.info("No candidate workflows found for parent selection.")
            return [], False

        # Wrap WorkflowInfo objects so SelectionPressure can use .reward
        adapters = [_WorkflowScoreAdapter(wf) for wf in candidates]

        selected_adapters, use_crossover = selection_pressure.select_parents(
            candidates=adapters,
            n_parents=n_parents,
            crossover_rate=crossover_rate,
        )

        # Unwrap back to WorkflowInfo
        selected_workflows = [a.workflow_info for a in selected_adapters]

        # Log selection outcome
        uuids = [wf.uuid for wf in selected_workflows]
        scores = [f"{wf.overall_score:.2f}" for wf in selected_workflows]
        mode = "CROSSOVER" if use_crossover else "MUTATION"
        logger.info(
            f"🧬 Parent selection ({mode}, strategy={selection_pressure.strategy.value}): "
            f"{len(selected_workflows)} parent(s) from {len(candidates)} candidates "
            f"— UUIDs={uuids}, scores={scores}"
        )

        return selected_workflows, use_crossover


if __name__ == "__main__":
    config = Config()
    config.workflow_dir = "../workflows"
    mcts = WorkflowSelector(config)
    goal = "Compare reproduction results with original paper results and generate comprehensive validation report. Steps: (1) Load all evaluation metrics from results/metrics/ JSON files, (2) Extract original paper results from reproduction_gernermed.md (all tables with F1, precision, recall by dataset and entity type), (3) Create detailed comparison tables matching paper's format: overall performance table, per-entity-type performance table, per-dataset performance table, (4) Calculate absolute and relative differences between reproduced and original results for each metric, (5) Perform statistical significance tests where appropriate if multiple runs were conducted, (6) Generate visualizations: bar charts comparing F1-scores across models and datasets, confusion matrices if possible, entity-type performance heatmaps, (7) Create 'validation_report.html' in results/ directory with: executive summary (reproduction success percentage), side-by-side comparison tables (original vs reproduced), visualization plots embedded, detailed metric breakdowns, assessment of reproduction fidelity (successful/partial/failed), (8) Analyze and document any significant discrepancies (>5% difference in F1) in 'results/discrepancies_analysis.md' including: specific metrics that differ, potential explanations (dataset version differences, random seed variations, implementation details, hardware differences), (9) Document model behavior on different medical entity types (medication vs diagnosis vs symptoms, etc.), (10) Create summary statistics in 'results/summary_statistics.csv' with columns: dataset, entity_type, original_f1, reproduced_f1, absolute_diff, relative_diff, status. Include overall assessment of whether GERNERMED's claimed performance on German medical NER was successfully validated."
    matching_workflow = mcts.select_best_workflows(goal)
    print("Best matching workflow:")
    for wf in matching_workflow:
        print(f"UUID: {wf.uuid}, Goal: {wf.goal}, Score: {wf.overall_score:.4f}")

    # ── Demonstrate evolutionary multi-parent selection ────────────
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
