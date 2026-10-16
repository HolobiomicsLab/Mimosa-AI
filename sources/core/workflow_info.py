"""Workflow-folder metadata accessors for the evolution engine (Layer 2).

This file owns lazy, cached reads of one generation's artefacts:
``state_result.json``, the genotype source file, the evaluation sidecar,
and the goal/task texts. :class:`WorkflowInfo` backs similarity-based
parent selection, lineage scans, and audit reports, while
:func:`is_oracle_generation` and ``ORACLE_SIDECAR`` mark generations
scored by the benchmark-leaking gold verifier so selection can skip
them in non-gold runs. :mod:`sources.core.workflow_selection` and
audit scripts import it; it runs whenever the engine scans the
``workflows/`` directory on disk.
"""

import json
import re
from pathlib import Path
from statistics import mean

FALLBACK_REWARD_KINDS = frozenset(
    {"mean_claim", "oracle_censored", "unmeasured_prior"}
)

#: Sidecar written by the gold-feedback (oracle) verifier in every
#: generation folder it evaluates — the cheap on-disk leak marker.
ORACLE_SIDECAR = "gold_feedback.json"


def is_oracle_generation(workflow_folder: Path | str, state_result: dict | None = None) -> bool:
    """Report whether a generation was scored by the benchmark-leaking gold verifier.

    Workflow selection calls this during discovery to enforce the leak
    guard: genotypes evolved under gold feedback may only be reused by
    another gold run. The caller passes the generation folder path and,
    when already loaded, its parsed ``state_result.json``.

    Args:
        workflow_folder: Path to the generation folder on disk.
        state_result: Parsed ``state_result.json`` from the caller's
            cache, or ``None`` when not loaded yet.

    Returns:
        bool: ``True`` when the sidecar exists or the oracle flag is
        set; ``False`` otherwise, including when ``state_result`` is
        not a mapping and no sidecar is present.
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
    """Lazy, cached accessor for one workflow generation folder.

    Workflow selection, lineage scans, and audit scripts construct it
    with a UUID plus the folder path under ``workflows/``. Construction
    performs no I/O; each property reads its file on first access and
    caches the value for the object's lifetime.
    """

    def __init__(self, uuid: str, workflow_folder: Path | str) -> None:
        """Bind the accessor to one workflow folder.

        Discovery scans, rehydration, and tests call it with values read
        from the workflows directory. It stores the identifiers and
        resets every cache; no file is read here.

        Args:
            uuid: Workflow UUID; it is the folder name and is embedded
                in the stored file names.
            workflow_folder: Path to the generation folder on disk;
                a string is converted to :class:`~pathlib.Path`.
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
        """Return the workflow goal recorded in ``state_result.json``.

        Selection and reporting read it to match and display what the
        generation tried to do. The text comes from the ``goal`` key the
        runner wrote into the state result at execution time.

        Returns:
            str: The goal text, or ``""`` when the state result is
            missing or has no ``goal`` key.
        """
        if self._goal is None:
            state_result = self.load_state_result()
            self._goal = state_result.get("goal", "") if state_result else ""
        return self._goal

    @property
    def original_task(self) -> str:
        """Return the original unwrapped task text for similarity matching.

        Parent selection compares it with a new goal so the knowledge
        wrapper around stored goals cannot skew the match. It reads
        ``original_task_<uuid>.txt`` from the folder first; when that
        file is missing or unreadable, it recovers the task from the
        wrapped goal text instead.

        Returns:
            str: The unwrapped task, recovered by
            :meth:`_extract_original_from_wrapped` when the sidecar
            file is unavailable.
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

        :attr:`original_task` calls this when the sidecar file is
        absent or unreadable. It looks for the marker ``"complete the
        following task:"`` and returns the text that follows it;
        otherwise it returns the input unchanged.

        Args:
            text: Potentially wrapped goal text from ``state_result.json``.

        Returns:
            str: The text after the wrapper marker, the input unchanged
            when no marker is present, or ``""`` when ``text`` is empty.
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
        """Return the cached contents of ``state_result.json``.

        The scoring properties of this class read it, so the JSON file
        is parsed at most once per instance.

        Returns:
            dict: Parsed state result, or ``{}`` when the file is
            missing or unreadable.
        """
        if self._state_result is None:
            self._state_result = self.load_state_result()
        return self._state_result

    @property
    def answers(self) -> list:
        """Return the per-step answers recorded in ``state_result.json``.

        Trajectory reports and audits read it to see what the workflow's
        agents produced at each step of the run.

        Returns:
            list: The ``answers`` entries, or an empty list when the
            state result is missing or has no ``answers`` key.
        """
        state_result = self.load_state_result()
        if state_result is None:
            return []
        return state_result.get('answers', [])

    @property
    def success(self) -> list:
        """Return the per-step success flags recorded in ``state_result.json``.

        Evolution bookkeeping reads it to see which steps of a run
        reported success.

        Returns:
            list: The ``success`` entries, or an empty list when the
            state result is missing or has no ``success`` key.
        """
        state_result = self.load_state_result()
        return state_result.get('success', [])

    @property
    def is_success(self) -> bool:
        """Report whether the last recorded step succeeded.

        Callers use it as a quick pass/fail signal for the whole run. It
        reads the ``success`` list from the state result and returns its
        final entry.

        Returns:
            bool: The last success flag, or ``False`` when the state
            result is missing or holds no entries.
        """
        state_result = self.load_state_result()
        success_list = state_result.get('success', [False]) if isinstance(state_result, dict) else [False]
        return success_list[-1]

    @property
    def code(self) -> str:
        """Return the cached workflow genotype source code.

        Selection inspects parent candidates with it, and reports quote
        it. The text is read once from ``workflow_genotype_<uuid>.py``
        and cached for the object's lifetime.

        Returns:
            str: The genotype source, or ``""`` when the file is
            missing or unreadable.
        """
        if self._code is None:
            self._code = self.load_code()
        return self._code

    @property
    def overall_score(self) -> float:
        """Return the cached overall score of the workflow.

        Parent selection and ranking sort candidates by it. The value is
        computed once by :meth:`calculate_overall_score` and cached.

        Returns:
            float: The capped overall score, or ``0.0`` when no
            evaluation data is available.
        """
        if self._overall_score is None:
            self._overall_score = self.calculate_overall_score()
        return self._overall_score or 0.0

    @property
    def overall_score_uncapped(self) -> float:
        """Return the reward without the hard-fail cap.

        Parent-draw weighting reads it so refuted-but-improving runs
        stay rank-ordered. The value is ``max(0, base_mean)`` taken from
        the verifier block of the state result.

        Returns:
            float: The uncapped reward, or :attr:`overall_score` when
            verifier scores are unavailable.
        """
        verifier = (self.state_result.get("evaluation") or {}).get("verifier") or {}
        if "overall_score_uncapped" not in verifier:
            return self.overall_score
        uncapped = float(verifier.get("overall_score_uncapped", 0.0))
        return max(0.0, uncapped)

    def _selected_evaluation(self) -> dict:
        """Return the evaluation block that backs ``overall_score`` (or ``{}``).

        :meth:`calculate_overall_score` and :attr:`reward_is_fallback`
        call it so the score and the fallback flag always come from the
        same block. Blocks are tried in priority order (``generic`` →
        ``verifier`` → ``scenario``) and the first mapping found wins.

        Returns:
            dict: The first evaluation block found, or ``{}`` when the
            state result is missing or holds no mapping.
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
        """Report whether the selected evaluation's reward is a fallback.

        Capsule selection and parent ranking call this before an argmax:
        a fallback reward is not comparable with measured rewards and
        must stay out of the comparison. The hybrid verifier persists
        ``reward_fallback`` alongside
        ``overall_score``: ``"mean_claim"`` marks a first-generation fallback
        computed as the mean claim score (no rivals to compare against yet)
        — a different scale from the pairwise win-rate every later
        generation earns. Comparing the two in one argmax ships the wrong generation

        The flag counts as a fallback when it is ``mean_claim``,
        ``oracle_censored``, or ``unmeasured_prior`` (see
        ``FALLBACK_REWARD_KINDS``); a missing flag is a measured reward.

        Read from :meth:`_selected_evaluation` so the flag always comes
        from the SAME block that produced ``overall_score``.

        Returns:
            bool: ``True`` when the fallback flag is a non-comparable
            kind, ``False`` otherwise.
        """
        return self._selected_evaluation().get("reward_fallback") in FALLBACK_REWARD_KINDS

    @property
    def judge_evaluation(self) -> dict:
        """Return the human-readable evaluation text for this workflow.

        Reporting and audit scripts read it to show the judge's written
        verdict. Despite the annotation, this returns the raw string
        contents of the sidecar ``evaluation.txt`` file, not a mapping.

        Returns:
            str | dict: Stripped evaluation text; ``{}`` when the file
            does not exist, or a fallback string when it cannot be read.
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
        """Return the verifier's behavioral gradient for this generation.

        The evolution engine feeds it to the variation prompt so the
        next genotype changes behavior, not only the score. It reads
        ``abstracted_textual_gradient`` from the verifier block and
        falls back to the ``textual_gradient.txt`` sidecar written
        during evaluation.

        Returns:
            str: The stripped gradient text, or ``""`` when neither
            source holds a non-empty value.
        """
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

    @property
    def run_stdout(self) -> str:
        """Return the raw execution output saved for this generation.

        The variation engine shows it to the directive LLM as an
        unverified raw log when this workflow is a parent without a
        verifier gradient. It reads ``run_stdout.log``, which the
        evolution engine writes once, right after execution. The text is
        read on each access and not cached.

        Returns:
            str: The log text (UTF-8, verbatim), or ``""`` when the file
            is missing or unreadable.
        """
        log_file = self.workflow_folder / "run_stdout.log"
        try:
            return log_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""

    def load_state_result(self) -> dict:
        """Load and parse ``state_result.json`` from disk.

        Every property in this class starts from it, and audit scripts
        call it directly for one-off reads. A missing, empty, or invalid
        file is reported on stdout and degrades to an empty result
        instead of raising.

        Returns:
            dict: The parsed JSON object, or ``{}`` when the file is
            missing, empty, or cannot be parsed.
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

        Discovery calls it to keep only workflows whose source survived,
        and parent reuse reads it through the :attr:`code` cache. A
        missing or unreadable file is reported on stdout and degrades to
        an empty string instead of raising.

        Returns:
            str: The genotype source code, or ``""`` when the file is
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
        """Compute the workflow's overall score from its evaluation block.

        :attr:`overall_score` calls it once and caches the result. The
        block is chosen by :meth:`_selected_evaluation` in priority
        order (``generic`` → ``verifier`` → ``scenario`` — the same
        block that feeds :attr:`reward_is_fallback`). Malformed values
        degrade to ``0.0`` instead of raising.

        Returns:
            float: The block's ``overall_score`` (or ``score`` for
            scenario blocks), or ``0.0`` when no evaluation data exists.
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
        """Report whether this folder is a usable workflow record.

        Discovery and rehydration call it before spending I/O on a
        candidate folder. Both required files must exist on disk: the
        state result and the genotype source.

        Returns:
            bool: ``True`` when ``state_result.json`` and
            ``workflow_genotype_<uuid>.py`` both exist, else ``False``.
        """
        state_file = self.workflow_folder / "state_result.json"
        genotype_file = self.workflow_folder / f"workflow_genotype_{self.uuid}.py"
        return state_file.exists() and genotype_file.exists()

    def __str__(self) -> str:
        """Return a human-readable summary for logs and debugging.

        Returns:
            str: UUID, goal preview, score, and validity flag.
        """
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
