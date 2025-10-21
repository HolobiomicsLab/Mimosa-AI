import os
import json
import re
from pathlib import Path
from statistics import mean

class WorkflowInfo:
    def __init__(self, uuid, workflow_folder: Path | str):
        self.uuid = uuid
        self.workflow_folder = workflow_folder if isinstance(workflow_folder, Path) else Path(workflow_folder)
        self._goal = None
        self._state_result = None
        self._code = None
        self._overall_score = None
        self._original_task = None

    @property
    def goal(self) -> str:
        if self._goal is None:
            state_result = self.load_state_result()
            self._goal = state_result.get("goal", "") if state_result else ""
        return self._goal
    
    @property
    def original_task(self) -> str:
        """Load the original unwrapped task for similarity matching.
        
        Returns:
            str: The original task without knowledge wrapper. Falls back to
                 extracting from goal if original_task file doesn't exist.
        """
        if self._original_task is None:
            # Try to load from original_task_{uuid}.txt file first
            task_file = self.workflow_folder / f"original_task_{self.uuid}.txt"
            if task_file.exists():
                try:
                    with open(task_file) as f:
                        self._original_task = f.read().strip()
                except Exception as e:
                    print(f"⚠️ Could not load original_task_{self.uuid}.txt: {e}")
                    self._original_task = self._extract_original_from_wrapped(self.goal)
            else:
                # Fallback: extract from goal if it has the wrapper pattern
                self._original_task = self._extract_original_from_wrapped(self.goal)
        return self._original_task
    
    def _extract_original_from_wrapped(self, text: str) -> str:
        """Extract original task from knowledge-wrapped text.
        
        Args:
            text: Potentially wrapped text
            
        Returns:
            str: Extracted task or original text if not wrapped
        """
        if not text:
            return ""
        
        # Pattern: "...complete the following task:\n<actual_task>"
        match = re.search(r'complete the following task:\s*\n(.*)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # If not wrapped, return as-is
        return text

    @property
    def state_result(self) -> dict:
        if self._state_result is None:
            self._state_result = self.load_state_result()
        return self._state_result

    @property
    def answers(self) -> dict:
        state_result = self.load_state_result()
        return state_result.get('answers', [])

    @property
    def success(self) -> dict:
        state_result = self.load_state_result()
        return state_result.get('success', [])

    @property
    def is_success(self) -> dict:
        state_result = self.load_state_result()
        success_list = state_result.get('success', [False]) if isinstance(state_result, dict) else [False]
        return success_list[-1]

    @property
    def code(self) -> str:
        if self._code is None:
            self._code = self.load_code()
        return self._code

    @property
    def overall_score(self) -> float:
        if self._overall_score is None:
            self._overall_score = self.calculate_overall_score()
        return self._overall_score

    def load_state_result(self) -> dict:
        """Load state_result.json file."""
        state_file = self.workflow_folder / "state_result.json"
        if not state_file.exists():
            return {}
        
        try:
            with open(state_file) as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except Exception as e:
            print(f"❌ Can't read state_result.json for UUID {self.uuid}: {e}")
            return {}

    def load_code(self) -> str:
        """Load workflow code file."""
        code_file = self.workflow_folder / f"workflow_code_{self.uuid}.py"
        if not code_file.exists():
            raise ValueError(
                f"❌ Workflow code file {code_file} does not exist for UUID {self.uuid}."
            )
        
        try:
            with open(code_file) as f:
                return f.read()
        except Exception as e:
            print(f"❌ Can't read workflow code for UUID {self.uuid}: {e}")
            return ""

    def calculate_overall_score(self) -> float:
        """Calculate overall score from evaluation data."""
        state_result = self.load_state_result()
        if not state_result:
            return 0.0
            
        evaluation = state_result.get("evaluation", {})
        scores = []
        if evaluation:
            if "generic" in evaluation:
                scores.append(evaluation["generic"]["overall_score"])
            elif "scenario" in evaluation:
                scores.append(evaluation["scenario"]["score"])
        return mean(scores) if scores else 0.0

    def is_valid(self) -> bool:
        """Check if workflow has all required files."""
        state_file = self.workflow_folder / "state_result.json"
        code_file = self.workflow_folder / f"workflow_code_{self.uuid}.py"
        return state_file.exists() and code_file.exists()

    def __str__(self) -> str:
        """Return a string representation of the WorkflowInfo."""
        goal_preview = self.goal[:100] + "..." if len(self.goal) > 100 else self.goal
        return (
            f"WorkflowInfo(uuid={self.uuid}, "
            f"goal='{goal_preview}', "
            f"score={self.overall_score:.2f}, "
            f"valid={self.is_valid()})"
        )

if __name__ == "__main__":
    # test
    wf = WorkflowInfo("20250926_165556_9e2402c5", "./20250926_165556_9e2402c5")
    print(wf.goal)
    print(wf.state_result)
    print(wf.answers)
