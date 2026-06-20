import os
import json
import re
from pathlib import Path
from statistics import mean

class WorkflowInfo:
    """Lazy accessor for the artefacts of one workflow folder.

    Caches per-property reads so repeated access (e.g. during selection or
    visualisation) doesn't repeatedly hit disk. Construction is cheap; I/O
    happens only when properties are first read.
    """

    def __init__(self, uuid: str, workflow_folder: Path | str) -> None:
        """Bind the accessor to a workflow folder on disk.

        Args:
            uuid: Workflow UUID (the folder name and embedded in filenames).
            workflow_folder: Filesystem path to the workflow's folder. Accepts
                a string or a :class:`~pathlib.Path`.
        """
        self.uuid = uuid
        self.workflow_folder = workflow_folder if isinstance(workflow_folder, Path) else Path(workflow_folder)
        self._goal = None
        self._state_result = None
        self._code = None
        self._overall_score = None
        self._original_task = None

    @property
    def goal(self) -> str:
        """Return the workflow goal as recorded in ``state_result.json``."""
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
        """Extract the original task from a knowledge-wrapped text.

        Looks for the marker ``"complete the following task:"`` and returns
        the text that follows it; otherwise returns the input unchanged.

        Args:
            text: Potentially wrapped text.

        Returns:
            The extracted task or the original text if not wrapped. Returns
            an empty string when ``text`` is falsy.
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
        """Cached contents of ``state_result.json``."""
        if self._state_result is None:
            self._state_result = self.load_state_result()
        return self._state_result

    @property
    def answers(self) -> list:
        """List of per-step answers recorded in ``state_result.json``."""
        state_result = self.load_state_result()
        if state_result is None:
            return []
        return state_result.get('answers', [])

    @property
    def success(self) -> list:
        """List of per-step success booleans recorded in ``state_result.json``."""
        state_result = self.load_state_result()
        return state_result.get('success', [])

    @property
    def is_success(self) -> bool:
        """True when the last recorded success flag is True."""
        state_result = self.load_state_result()
        success_list = state_result.get('success', [False]) if isinstance(state_result, dict) else [False]
        return success_list[-1]

    @property
    def code(self) -> str:
        """Cached workflow genotype source code."""
        if self._code is None:
            self._code = self.load_code()
        return self._code

    @property
    def overall_score(self) -> float:
        """Cached overall (post-cap) workflow score."""
        if self._overall_score is None:
            self._overall_score = self.calculate_overall_score()
        return self._overall_score or 0.0

    @property
    def overall_score_uncapped(self) -> float:
        """Reward without the hard-fail cap, still applied.

        Equals ``max(0, base_mean)``. Used for parent-draw weighting so
        distinct refuted-but-improving runs stay rank-ordered. Falls back
        to ``overall_score`` when verifier scores are unavailable.
        """
        verifier = (self.state_result.get("evaluation") or {}).get("verifier") or {}
        if "overall_score_uncapped" not in verifier:
            return self.overall_score
        uncapped = float(verifier.get("overall_score_uncapped", 0.0))
        return max(0.0, uncapped)

    @property
    def judge_evaluation(self) -> dict:
        """Return the contents of ``evaluation.txt`` for this workflow.

        Despite the annotation, this returns the raw string contents of the
        sidecar ``evaluation.txt`` file (or an empty dict / fallback string
        when the file is missing or unreadable).

        Returns:
            Stripped evaluation text, ``{}`` if the file does not exist, or a
            fallback string if it cannot be read.
        """
        eval_file = self.workflow_folder / "evaluation.txt"
        if not eval_file.exists():
            return {}

        try:
            with open(eval_file) as f:
                return f.read().strip()
        except Exception as e:
            print(f"❌ Can't read state_result.json for UUID {self.uuid}: {e}")
            return "No evaluation. execution failed."

    @property
    def abstracted_textual_gradient(self) -> str:
        """Behavioral textual_gradient written by the verifier abstractor (Layer 1)."""
        state = self.load_state_result()
        if isinstance(state, dict):
            verifier = (state.get("evaluation") or {}).get("verifier") or {}
            text = verifier.get("abstracted_textual_gradient")
            if isinstance(text, str) and text.strip():
                return text.strip()
        sidecar = self.workflow_folder / "textual_gradient.txt"
        if sidecar.exists():
            try:
                return sidecar.read_text(encoding="utf-8").strip()
            except OSError:
                return ""
        return ""

    def load_state_result(self) -> dict:
        """Load and parse ``state_result.json`` from disk.

        Returns:
            Parsed JSON dictionary, or an empty dict when the file is missing,
            empty, or cannot be parsed.
        """
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
        """Load the workflow genotype Python file from disk.

        Returns:
            Source code as a string, or an empty string when the file is
            missing or cannot be read.
        """
        genotype_file = self.workflow_folder / f"workflow_genotype_{self.uuid}.py"
        if not genotype_file.exists():
            print(f"❌ Workflow code file {genotype_file} does not exist for UUID {self.uuid}.")
            return ""

        try:
            with open(genotype_file) as f:
                return f.read()
        except Exception as e:
            print(f"❌ Can't read workflow code for UUID {self.uuid}: {e}")
            return ""

    def calculate_overall_score(self) -> float:
        """Compute the overall score from the workflow's evaluation block.

        Uses the first matching evaluator in priority order (``generic`` →
        ``verifier`` → ``scenario``) and returns the mean of the collected
        scores. Returns ``0.0`` when no evaluation data is available.

        Returns:
            The mean of the collected score(s), or ``0.0`` when none exist.
        """
        state_result = self.load_state_result()
        if not state_result:
            return 0.0

        evaluation = state_result.get("evaluation", {})
        scores = []
        if evaluation:
            try:
                if "generic" in evaluation:
                        scores.append(evaluation["generic"]["overall_score"])
                elif "verifier" in evaluation:
                        scores.append(evaluation["verifier"]["overall_score"])
                elif "scenario" in evaluation:
                    scores.append(evaluation["scenario"]["score"])
            except Exception as _:
                scores.append(0)
        return mean(scores) if scores else 0.0

    def is_valid(self) -> bool:
        """Return True when both ``state_result.json`` and the genotype file exist."""
        state_file = self.workflow_folder / "state_result.json"
        genotype_file = self.workflow_folder / f"workflow_genotype_{self.uuid}.py"
        return state_file.exists() and genotype_file.exists()

    def __str__(self) -> str:
        """Return a human-readable summary of the WorkflowInfo."""
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
