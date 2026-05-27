
"""
VariationEngine: search-schedule and prompt assembly for LLM-guided workflow evolution.
"""

import math
from .workflow_info import WorkflowInfo

from sources.cli.pretty_print import (
    print_info,
    CYAN, GREEN, YELLOW, RED, DIM, RESET, BOLD,
)

import numpy as np

class VariationEngine:
    """
    Orchestrates iterative LLM-driven workflow search via structured prompt mutation.

    Each call to mutation_prompt() or crossover_prompt() produces a prompt that:
      - Anchors the LLM on concrete execution feedback (agent answers, judge eval).
      - Injects a freshly-sampled multi-dimensional perturbation (via Mutagen)
      - Applies a step-aware annealing schedule that governs exploration breadth
        and permitted topology complexity as iterations stagnation.
    """

    def __init__(self):
        self.diagnosis_history = []
        self.agent_count_history = []
        self.max_possible_agents = 7

    def _sample_agent_count(self, stagnation: float, lo: int, hi: int, concentration: float = 4.0) -> int:
        """
        Sample agent random count within [lo, hi], biased upward by stagnation.
        """
        if lo == hi:
            return lo
        target_mean = lo + stagnation * (hi - lo)
        p = np.clip((target_mean - lo) / (hi - lo), 0.05, 0.95)
        alpha = p * concentration
        beta = (1 - p) * concentration
        prob = np.random.beta(alpha, beta)
        return lo + int(np.random.binomial(hi - lo, prob))

    def _diagnosis_similarity(self, a: str, b: str) -> float:
        """Token-level Jaccard. Cheap stand-in for embedding cosine; no model load."""
        if not a or not b:
            return 0.0
        a = ' '.join([a_c for a_c in a.split() if len(a_c) > 5 or a_c.isupper()])
        b = ' '.join([b_c for b_c in b.split() if len(b_c) > 5 or b_c.isupper()])
        print(f"Comparing diagnoses:\nA: {a}\nB: {b}")
        tokens_a = set(a.lower().split(' '))
        tokens_b = set(b.lower().split(' '))
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    def _compute_stagnation(self, window: int = 3) -> float:
        """Mean pairwise Jaccard over the last `window` diagnoses, ∈ [0, 1].

        High value ⇒ the LLM-mutator is cycling on similar failure modes
        (mode collapse). Used to drive stochastic step regression.
        """
        recent = self.diagnosis_history[-window:]
        if len(recent) < 2:
            return 0.0
        sims = [
            self._diagnosis_similarity(recent[i], recent[j])
            for i in range(len(recent))
            for j in range(i + 1, len(recent))
        ]
        raw = sum(sims) / len(sims) if sims else 0.0
        stagnation = np.clip(raw / 0.5, 0, 1) ** 0.5
        return stagnation

    def _get_prompt_step_size(self):
        """
        prompt gradient 'step size' to adapt mutation boldness based on stagnation level. High stagnation → bolder mutations.
        """
        stagnation = self._compute_stagnation()
        curr_agent_count = self.agent_count_history[-1] if self.agent_count_history else 1
        max_agent = int((curr_agent_count + stagnation * self.max_possible_agents))  # more stagnation → allow more agents
        n_agents = self._sample_agent_count(stagnation, 1, max_agent)
        self.agent_count_history.append(n_agents)

        if stagnation < 0.25:
            return (
                f"Priority mutations: small change, prompt only. Max Agent count: {n_agents}.\n"
            )
        elif stagnation < 0.50:
            return (
                f"Permitted mutations: small change, prompt (primary), tools. Max Agent count: {n_agents}.\n"
            )
        elif stagnation < 0.65:
            return (
                f"Permitted mutations: medium change, topology, prompt, handoff format. Max Agent count: {n_agents}.\n"
            )
        elif stagnation < 0.85:
            return (
                f"Permitted mutations: big change, prompt, handoff format, Max Agent count {n_agents}.\n"
            )
        else:
            return (
                f"Permitted mutations: complete rethink. Max Agent count: {n_agents}.\n"
            )

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_agent_answers(wf_state: dict | None) -> str:
        """Flatten per-agent answers from workflow state into a readable string."""
        if not wf_state or "answers" not in wf_state:
            return "No agent answers captured."
        answers = wf_state["answers"]
        if isinstance(answers, list):
            return "\n".join(
                f"agent {name}: {str(answer)[:2048]}..."
                for name, answer in zip(wf_state["step_name"], answers, strict=True)
            )
        return str(answers)

    # ── Prompt builders ───────────────────────────────────────────────────────

    def seed_genome_prompt(self, goal: str) -> str:
        """
        Prompt for very first workflow generation (generation 0).
        """
        return (
            "## First workflow generation\n"
            f"Goal to assemble a workflow for:\n{goal}\n"
            "Build the minimal workflow for the task with maximum 2 agents.\n"
        )

    def mutation_prompt(
        self,
        goal: str,
        wf_info: WorkflowInfo | None,
        genotype: str | None,
        run_stderr: str,
        iteration_count: int,
        max_iterations: int = 10,
    ) -> str:
        """
        Build a prompt for one mutation step in the evolutionary search.

        The prompt has three layers:
          1. Voice framing   — sets the LLM's reasoning tone for this iteration.
          2. Execution grounding — previous code, agent answers, and judge eval.
        """
        score      = wf_info.overall_score        if wf_info else 0.0
        diagnosis  = wf_info.abstracted_diagnosis if wf_info else ""
        wf_state   = wf_info.state_result         if wf_info else None

        # ── Execution evidence ───────────────────────────────────────────────
        agent_answers = self._extract_agent_answers(wf_state)
        diagnosis_block = (
            diagnosis.strip()
            if diagnosis and diagnosis.strip()
            else (run_stderr or "FAILURE:unknown failure").strip()
            or "NO_DIAGNOSIS:No diagnosis captured."
        ).replace('_', ' ')[:2048]
        self.diagnosis_history.append(diagnosis_block)
        step_block   = self._get_prompt_step_size()

        if genotype is None:
            body = "Previous attempt failed. Fix syntax errors."
        else:
            body = "\n".join([
                "## WORKFLOW EVOLUTION STEP",
                "",
                "Your previous workflow attempt did not reach the success threshold.",
                "",
                "## Previous workflow code:",
                "<python>",
                genotype,
                "</python>",
                "",
                "## EXECUTION RESULTS:",
                "<agents_answers>",
                agent_answers,
                "</agents_answers>",
                "<diagnosis>",
                "",
                diagnosis_block,
                "</diagnosis>",
                "<boldness>",
                step_block,
                "</boldness>",
                "",
                "## Task: apply a single mutation to the workflow code.",
            ])

        return "\n".join([
            f"Attempt {iteration_count + 1} of workflow generation.",
            body,
            "\nTarget goal:",
            goal,
        ])

    def crossover_prompt(
        self,
        goal: str,
        wf_infos: list[WorkflowInfo],
        genotypes: list[str],
        run_stderrs: list[str],
        iteration_count: int,
        max_iterations: int = 10,
    ) -> str:
        """
        Build a prompt for a crossover step: recombine N parent workflows into one offspring.
        Parents are sorted best → worst so the LLM sees the strongest candidates first.
        """
        # ── Assemble parent records ──────────────────────────────────────────
        parents = []
        for i, (wf_info, genotype, stderr) in enumerate(
            zip(wf_infos, genotypes, run_stderrs)
        ):
            score     = wf_info.overall_score        if wf_info else 0.0
            diagnosis = wf_info.abstracted_diagnosis if wf_info else ""
            # Layer 1: rubric-blind diagnosis instead of raw judge log.
            # Fall back to stderr tail only when no diagnosis exists.
            diagnosis = diagnosis.strip() or (stderr or "").strip()[-1024:]
            answers   = self._extract_agent_answers(wf_info.state_result if wf_info else None)
            parents.append({
                "index":     i + 1,
                "score":     score,
                "code":      genotype,
                "diagnosis": diagnosis,
                "answers":   answers,
            })

        # Best parents first — LLM primacy bias helps inherit strong traits
        parents.sort(key=lambda p: p["score"], reverse=True)

        parents_block = []
        for p in parents:
            parents_block.append("\n".join([
                f"### Parent {p['index']} (score={p['score']:.2f})",
                "<python>",
                p["code"] or "(construction failed)",
                "</python>",
                "<agents_answers>",
                p["answers"],
                "</agents_answers>"
            ]))

        return "\n".join([
            f"Attempt {iteration_count + 1}: CROSSOVER — synthesize from {len(parents)} parent workflows.",
            f"Goal: {goal}",
            "",
            "## PARENT WORKFLOWS:",
            "\n\n".join(parents_block),
            "",
            "## YOUR TASK:",
            "Do not pick one parent and patch it. Genuinely recombine.",
            "1. Identify which structural decisions worked in each parent (look at agent answers, not just score).",
            "2. Identify which decisions failed and why.",
            "3. Build a new workflow that inherits the strongest sub-structures across parents",
            "   and discards the weakest, even if that means a topology none of the parents used.",
        ])


if __name__ == "__main__":
    ve = VariationEngine()
    ve.diagnosis_history = ["verification failed: claim X not supported"] * 3
    s_high = ve._compute_stagnation()

    np.random.seed(0)
    ve.diagnosis_history = []
    simulate_diagnosis = [
        "AMBIGUOUS_PROBABILITIES: the workflow largely built the intended multitask molecular predictor and produced complete-looking probabilities, but it was weakened by ambiguous/reproducibility issues in how probabilities were generated, missing held-out classification evaluation, leftover competing scripts, and a minor implementation/tooling inconsistency that made parts of the solution hard to verify.",
        "FALLBACK_ECFP: the workflow produced a correctly shaped prediction table, but it appears to rely on fallback or constant baseline probabilities rather than a trained ECFP-based multitask ClinTox classifier, so the main fix is to remove bypass logic and ensure the script actually featurizes structures, trains on the labeled training split, predicts with the fitted two-task model, and saves non-placeholder positive-class probabilities.",
        "TRAIN_TEST: the workflow produced a plausible prediction table, but its script was not sufficiently transparent or complete: it did not clearly demonstrate proper train/test split usage, two-task positive-class probability extraction, or required validation AUC reporting, making the scientific results hard to trust despite the output file looking valid.",
        "TRAIN_TEST_ISSUE: the workflow produced a well-formed prediction file, but its implementation did not convincingly use the intended train/test separation, clearly model both required binary targets, or reliably extract positive-class probabilities, making the results structurally valid but scientifically unreliable.",
        "BROKEN_CLASSIFIER: the workflow produced a plausible prediction CSV, but the underlying script is broken and does not reliably demonstrate a real trained two-task molecular classifier with proper featurization, train/test use, probability extraction, and alignment-safe molecule handling, suggesting the deliverable may have come from fallback or non-reproducible output generation rather than the intended workflow.",
        "FALLBACK_ECFP: probabilities are constant baselines rather than outputs of a trained ECFP multitask ClinTox model; remove bypass logic and ensure real training and prediction.",
        "FALLBACK_ECFP: the script writes baseline probabilities instead of trained ECFP predictions; the fix is to train the multitask model and persist real positive-class scores.",
        "FALLBACK_FEATURIZER: featurization silently degraded to a placeholder when RDKit failed, so downstream predictions are not based on real molecular structure; harden the featurizer path.",
        "FEATURIZER_DEGRADED: the molecular featurizer fell back to a degenerate representation under partial RDKit failure, yielding predictions that do not reflect structure-aware learning.",
        "RDKIT_PARTIAL: RDKit loaded but several molecules failed to parse and were silently dropped, biasing the trained model toward an unrepresentative subset of the data.",
        "TIMEOUT_AGENT_3: agent 3 hit the wall-clock limit while iterating over the full dataset; reduce per-agent scope, batch inputs, or split the responsibility across two agents instead of one monolithic loop.",
        "JUDGE_REJECTED_FORMAT: the final CSV had the right columns but used semicolons as separators, causing the judge's pandas read to misparse; enforce comma-delimited output in the writer agent.",
        "EMPTY_HANDOFF: agent 2 returned an empty string to agent 3, breaking the chain; add a validation gate that retries or escalates when an upstream answer is empty or below a length threshold.",
        "INFINITE_LOOP_PLANNER: the planner agent kept re-emitting the same plan because the critic's feedback was not threaded back into its context; route critic output explicitly into the planner's next prompt."
    ]

    for diag in simulate_diagnosis:
        print("Adding diagnosis to history:", diag)
        ve.diagnosis_history.append(diag)  # only keep code for stagnation sim
        s_low = ve._compute_stagnation()
        print(f"Stagnation scores: {s_low:.3f}")
        block = ve._get_prompt_step_size()
        print("Block:", block)