import os
import json
import re
from pathlib import Path
from statistics import mean

#: ``evaluation.verifier.reward_fallback`` values whose reward is NOT on the
#: comparable scale and must stay out of the capsule argmax. ``mean_claim``
#: is the hybrid verifier's first-generation fallback; ``oracle_censored``
#: marks a full-oracle gold-feedback generation whose benchmark grade was
#: censored (it only carries the hybrid win-rate). ``short_circuit`` is a
#: real measured 0.0 and is deliberately absent.
FALLBACK_REWARD_KINDS = frozenset({"mean_claim", "oracle_censored"})

#: Sidecar written by the gold-feedback (oracle) verifier in every
#: generation folder it evaluates — the cheap on-disk leak marker.
ORACLE_SIDECAR = "gold_feedback.json"


def is_oracle_generation(workflow_folder: Path | str, state_result: dict | None = None) -> bool:
    """True when a generation was evaluated by the benchmark-leaking gold verifier.

    Args:
        workflow_folder: The generation folder.
        state_result: Its parsed ``state_result.json`` when already loaded.

    Returns:
        ``True`` if the gold-feedback sidecar exists or the persisted
        ``evaluation.verifier.oracle`` flag is set.
    """
    folder = Path(workflow_folder)
    if (folder / ORACLE_SIDECAR).exists():
        return True
    if isinstance(state_result, dict):
        evaluation = state_result.get("evaluation") or {}
        verifier = evaluation.get("verifier") if isinstance(evaluation, dict) else None
        return isinstance(verifier, dict) and verifier.get("oracle") is True
    return False


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

    def _selected_evaluation(self) -> dict:
        """The evaluation block that backs ``overall_score`` (or ``{}``).

        ``calculate_overall_score`` reads the first evaluator block in
        priority order (``generic`` → ``verifier`` → ``scenario``); the
        fallback flag MUST come from that same block, otherwise a state
        result written by two evaluators can pair one block's score with
        another block's ``reward_fallback`` (N1: the pains_brenk gen-13
        mean-claim flag was read from a different block than the 0.745
        score that won the capsule argmax).
        """
        state_result = self.load_state_result()
        if not state_result:
            return {}
        evaluation = state_result.get("evaluation", {})
        if not isinstance(evaluation, dict):
            return {}
        for key in ("generic", "verifier", "scenario"):
            block = evaluation.get(key)
            if isinstance(block, dict):
                return block
        return {}

    @property
    def reward_is_fallback(self) -> bool:
        """True when the selected evaluation's reward is a fallback, not a pairwise win.

        The hybrid verifier persists ``reward_fallback`` alongside
        ``overall_score``: ``"mean_claim"`` marks a first-generation reward
        computed as the mean claim score (no rivals to compare against yet)
        — a different scale from the pairwise win-rate every later
        generation earns. Comparing the two in one argmax ships the wrong
        generation (E37 R1: bulk_modulus shipped gen 0's 0.89 fallback over
        gen 11's winning 0.70). ``"short_circuit"`` (execution failed,
        reward 0.0) is a real measured outcome, not a fallback.

        Read from :meth:`_selected_evaluation` so the flag always comes
        from the SAME block that produced ``overall_score``.
        """
        return self._selected_evaluation().get("reward_fallback") in FALLBACK_REWARD_KINDS

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
        ``verifier`` → ``scenario`` — the same block
        :meth:`_selected_evaluation` feeds to ``reward_is_fallback``) and
        returns its score. Returns ``0.0`` when no evaluation data is
        available.

        Returns:
            The evaluation's overall score, or ``0.0`` when none exists.
        """
        block = self._selected_evaluation()
        if not block:
            return 0.0
        try:
            # generic/verifier persist "overall_score"; scenario uses "score".
            value = block.get("overall_score", block.get("score"))
            return mean([float(value)]) if value is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

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
