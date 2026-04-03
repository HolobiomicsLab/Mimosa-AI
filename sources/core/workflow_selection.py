import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parent.parent.parent))

from config import Config
from sources.core.workflow_info import WorkflowInfo

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
            workflow_info = WorkflowInfo(uuid, workflow_folder)

            if not workflow_info.is_valid():
                continue

            # Check if state_result is empty
            if not workflow_info.load_state_result():
                print(f"Skipping workflow {uuid}: empty state_result.json")
                continue

            workflow_info.load_code()
            if not workflow_info.code:
                print(f"Skipping workflow {uuid}: unable to load code")
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
        self, goal: str, threshold_similary=0.7, threshod_score=0.5
    ) -> list[WorkflowInfo]:
        """Choose a workflow that matches the goal with a minimum threshold."""
        similar_workflows = self.sort_similar_workflows(goal, threshold_similary)
        best_workflows = self.sort_workflows_by_score(similar_workflows, threshod_score)
        return best_workflows


if __name__ == "__main__":
    config = Config()
    config.workflow_dir = "../workflows"
    mcts = WorkflowSelector(config)
    goal = "Compare reproduction results with original paper results and generate comprehensive validation report. Steps: (1) Load all evaluation metrics from results/metrics/ JSON files, (2) Extract original paper results from reproduction_gernermed.md (all tables with F1, precision, recall by dataset and entity type), (3) Create detailed comparison tables matching paper's format: overall performance table, per-entity-type performance table, per-dataset performance table, (4) Calculate absolute and relative differences between reproduced and original results for each metric, (5) Perform statistical significance tests where appropriate if multiple runs were conducted, (6) Generate visualizations: bar charts comparing F1-scores across models and datasets, confusion matrices if possible, entity-type performance heatmaps, (7) Create 'validation_report.html' in results/ directory with: executive summary (reproduction success percentage), side-by-side comparison tables (original vs reproduced), visualization plots embedded, detailed metric breakdowns, assessment of reproduction fidelity (successful/partial/failed), (8) Analyze and document any significant discrepancies (>5% difference in F1) in 'results/discrepancies_analysis.md' including: specific metrics that differ, potential explanations (dataset version differences, random seed variations, implementation details, hardware differences), (9) Document model behavior on different medical entity types (medication vs diagnosis vs symptoms, etc.), (10) Create summary statistics in 'results/summary_statistics.csv' with columns: dataset, entity_type, original_f1, reproduced_f1, absolute_diff, relative_diff, status. Include overall assessment of whether GERNERMED's claimed performance on German medical NER was successfully validated."
    matching_workflow = mcts.select_best_workflows(goal)
    print("Best matching workflow:")
    for wf in matching_workflow:
        print(f"UUID: {wf.uuid}, Goal: {wf.goal}, Score: {wf.overall_score:.4f}")
