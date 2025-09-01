
import json
from pathlib import Path
from statistics import mean

class WorkflowInfo:
    def __init__(self, uuid, workflow_folder: Path):
        self.uuid = uuid
        self.workflow_folder = workflow_folder
        self._goal = None
        self._state_result = None
        self._code = None
        self._overall_score = None

    @property
    def goal(self) -> str:
        if self._goal is None:
            state_result = self.load_state_result()
            self._goal = state_result.get("goal", "") if state_result else ""
        return self._goal

    @property
    def state_result(self) -> dict:
        if self._state_result is None:
            self._state_result = self.load_state_result()
        return self._state_result

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