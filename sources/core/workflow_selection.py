
import json
import sys
from pathlib import Path
from statistics import mean

from sentence_transformers import SentenceTransformer

from config import Config

sys.path.append(str(Path(__file__).parent.parent.parent))

class WorkflowInfo:
    def __init__(self, uuid, state_result, goal, code, overall_score=0.0):
        self.uuid = uuid
        self.goal = goal
        self.state_result = state_result
        self.code = code
        self.overall_score = overall_score

#WF = WorkflowInfo  # Alias for easier reference

class WorkflowSelector:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.workflows_folder = Path(config.workflow_dir)
        self.workflows_info = self.discover_workflows()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def discover_workflows(self) -> dict[str, WorkflowInfo]:
        workflows = {}

        if not self.workflows_folder.exists():
            print(f"Workflows directory {self.workflows_folder} does not exist.")
            return workflows
        for workflow_folder in self.workflows_folder.iterdir():
            goal = ""
            code = None
            state_result = None
            uuid = workflow_folder.name
            overall_score = 0.0
            state_file = workflow_folder / "state_result.json"
            code_file = workflow_folder / f"workflow_code_{uuid}.py"

            if not workflow_folder.is_dir():
                continue
            if not state_file.exists():
                continue
            if not code_file.exists():
                raise ValueError(
                    f"Workflow code file {code_file} does not exist for UUID {uuid}."
                )

            try:
                with open(state_file) as f:
                    content = f.read().strip()
                    if not content:
                        print(f"Skipping workflow {uuid}: empty state_result.json")
                        continue
                    state_result = json.loads(content)
                with open(code_file) as f:
                    code = f.read()
            except Exception as e:
                print(f"can't read workflow files for UUID {uuid}: {e}")
                continue

            if state_result:
                goal = state_result.get("goal", "")
                evaluation = state_result.get("evaluation", {})
                scores = []
                if evaluation:
                    if 'generic' in evaluation:
                        scores.append(evaluation['generic']['overall_score'])
                    elif 'scenario' in evaluation:
                        scores.append(evaluation['scenario']['score'])
                overall_score = mean(scores) if scores else 0.0

            uuid = workflow_folder.name
            workflows[uuid] = WorkflowInfo(uuid,
                                           state_result, 
                                           goal, code,
                                           overall_score
                                          )
        return workflows
    
    def cosine_similarity(self, a: str, b: str) -> float:
        """Calculate cosine similarity between two strings."""
        import torch.nn.functional as F
        embeddings_a = self.model.encode(a, convert_to_tensor=True, show_progress_bar=False)
        embeddings_b = self.model.encode(b, convert_to_tensor=True, show_progress_bar=False)
        return F.cosine_similarity(embeddings_a, embeddings_b, dim=0).item()
    
    def sort_similar_workflows(self, goal: str,
                                     threshold=0.8,
                                     debug=False) -> list[WorkflowInfo]:
        """Find workflows with similar goals."""
        assert threshold >= 0.0, "Threshold must be non-negative"
        assert threshold <= 1.0, "Threshold must be at most 1.0"
        if not self.workflows_info:
            print("No workflows found.")
            return []
        similar_workflows = sorted(
            self.workflows_info.values(),
            key=lambda wf: self.cosine_similarity(wf.goal, goal),
            reverse=True
        )
        if debug:
            for wf in similar_workflows:
                sim = self.cosine_similarity(wf.goal, goal)
                print(f"UUID: {wf.uuid}, Goal: {wf.goal}, Similarity: {sim:.4f}")
        return [wf for wf in similar_workflows
                if self.cosine_similarity(wf.goal, goal) >= threshold]
    
    def sort_workflows_by_score(self, workflows_info: list[WorkflowInfo], threshold:float) -> list[WorkflowInfo]:
        """Sort workflows by their overall score."""
        sorted_workflows = sorted(
            workflows_info,
            key=lambda wf: wf.overall_score,
            reverse=True
        )
        return [wf for wf in sorted_workflows if wf.overall_score >= threshold]
    
    def select_best_workflows(self, goal: str, threshold_similary=0.8, threshod_score = 0.7) -> list[WorkflowInfo]:
        """Choose a workflow that matches the goal with a minimum threshold."""
        similar_workflows = self.sort_similar_workflows(goal, threshold_similary)
        best_workflows = self.sort_workflows_by_score(similar_workflows,threshod_score)
        return best_workflows

if __name__ == "__main__":
    config = Config()
    config.workflow_dir = "../workflows"
    mcts = WorkflowSelector(config)
    goal = "install prima.cpp and run a simple script"
    matching_workflow = mcts.select_best_workflows(goal)
    print("Best matching workflow:")
    for wf in matching_workflow:
        print(f"UUID: {wf.uuid}, Goal: {wf.goal}, Score: {wf.overall_score:.4f}")