"""
Claim extraction (six independent sources) and importance rating.
"""

from __future__ import annotations

import os
import sys
from typing import Any

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sources.cli.pretty_print import (
    print_info,
    print_ok,
    print_warn,
)
from sources.core.evaluators.base import LLMEvaluationError


# ----- Claim extraction: rules shared across all source prompts ---------------
# Importance is NOT assigned here. Each source emits raw claims; a separate
# post-merge pass (``_declare_claim_importance``) rates each surviving claim
# against the goal on a 1–10 scale and drops near-duplicates.
_CLAIM_RULES_BLOCK = """For each claim, list `likely_relevant_files`: relative
paths whose contents the verifier would need to read in order to check the
claim.
- ONLY use paths that appear verbatim in the WORKSPACE FILES listing above.
  Do not invent or guess paths the workflow's answer mentions but that are
  not in the listing.
- Use `[]` if the claim is purely about the workflow's output text and has
  no on-disk artefact to consult.

POLARITY (mandatory). Every claim is a POSITIVE SUCCESS ASSERTION about what
the workflow ACHIEVED scientifically. A claim is well-formed only if
"verified TRUE" is equivalent to "the workflow succeeded at this aspect".
Never extract a claim that a FAILURE MODE would satisfy. Extract the
success condition the workflow failed: a workflow that produced no usable
answer should FAIL the claim "produced <the deliverable, meeting <the
bar>>", not pass the claim "the final answer is empty".

ARTIFACT CLAIMS — STRICT. Bare file-existence or file-size claims are weak
and easy to game. Extract an artifact claim only when chained to a
functional property — not "predictions.csv exists" but "predictions.csv
contains a valid probability in [0,1] for every row of the test set".

Return STRICT JSON only, no prose, in this exact form:
{
  "claims": [
    {
      "id": "<short_slug>",
      "description": "<concise restatement of the claim>",
      "likely_relevant_files": ["<relative/path>", ...]
    },
    ...
  ]
}
"""


# Importance anchors shown to the rater LLM so it doesn't collapse to
# the middle of the scale. Kept short on purpose — long anchors waste
# tokens and tend to confuse small judges.
_IMPORTANCE_ANCHOR_BLOCK = """Importance scale (1–10), anchored:
- 10: literal deliverable named in the goal (the exact file, the headline metric).
-  8: required methodology step without which the result is invalid.
-  6: non-negotiable sanity property (probabilities in [0,1], no NaN, train/test disjoint).
-  4: literature-recommended best practice (seeded RNG, pinned dependencies).
-  2: minor / advisory (entrypoint name, workspace clutter).
Use the FULL scale; do not collapse to 5–7 by default. Goal-alignment dominates."""


class _VerifierClaimExtractionMixin:
    """Claim-extraction and importance-rating methods.

    Sentinel claim emitted by ``_extract_claims`` when the workflow produced
    nothing actionable; rated importance 10 because failure to execute is
    failure of the literal deliverable.
    """

    # ------------------------------------------------------------------
    # Stage 1 — claim extraction
    # ------------------------------------------------------------------

    def _extract_claims(
        self,
        uuid: str,
        goal: str,
        execution_text: str,
        workspace_listing: str,
        is_truly_empty: bool,
        grounding: str = "",
    ) -> list[dict[str, Any]]:
        """Extract atomic claims by polling six independent source prompts.

        Source A asks an LLM for what the LITERATURE demands of a correct
        solution; Source B asks what the USER explicitly required in the goal
        text; Source C asks for mathematical sanity properties of the
        produced artefacts; Source D asks for the non-negotiable
        computational-reproducibility requirements; Source E asks the
        statistical-fingerprint / non-triviality questions.

        After merging across sources, ``_declare_claim_importance`` runs a
        single rater pass that drops near-duplicates and assigns each
        surviving claim an integer importance (1-10) anchored on the goal.

        Args:
            uuid: Workflow identifier (used for logging and judge calls).
            goal: Original workflow goal text.
            execution_text: Agent narration / produced output text.
            workspace_listing: Rendered listing of workspace files.
            is_truly_empty: When True, returns a single sentinel "execution
                succeeded" claim and skips extraction.
            grounding: Optional peer-reviewed literature grounding block.

        Returns:
            Filtered claim list with stable ids, source labels, importance
            (1-10) and a one-sentence importance rationale on every entry.
        """
        if is_truly_empty:
            return [{
                "id": "c0_execution_succeeded",
                "description": "The workflow executed to completion and produced a non-empty answer.",
                "importance": 10,
                "importance_rationale": "literal deliverable; absent here",
                "likely_relevant_files": [],
            }]
        per_source_min, per_source_max = self._per_source_targets(n_sources=6)
        sources = (
            ("a", self._build_source_a_prompt(goal, grounding, workspace_listing, per_source_min, per_source_max)),
            ("b", self._build_source_b_prompt(goal, workspace_listing, per_source_min, per_source_max)),
            ("c", self._build_source_c_prompt(goal, workspace_listing, per_source_min, per_source_max)),
            ("d", self._build_source_d_prompt(goal, workspace_listing, per_source_min, per_source_max)),
            ("e", self._build_source_e_prompt(goal, execution_text, workspace_listing, per_source_min, per_source_max)),
        )

        merged: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for label, prompt in sources:
            data, err = self._call_judge_for_json(
                uuid, f"verifier_extract_claims_{label}", prompt
            )
            if err is not None:
                self.logger.warning(
                    f"Claim extraction source {label} failed for {uuid}: {err}"
                )
                continue
            for claim in self._parse_and_filter_claims(uuid, data):
                print_info(f"Extracted claim {claim['id']} from source {label} for {uuid}")
                claim_id = claim["id"]
                if claim_id in seen_ids:
                    claim_id = f"{claim_id}_{label}"
                claim["id"] = claim_id
                claim["source"] = f"source_{label}"
                seen_ids.add(claim_id)
                merged.append(claim)

        print_ok(f"Extracted claims for workflow {uuid} from {len(sources)} sources...")
        if len(merged) < self.min_claims:
            print_warn("Claim extraction yielded fewer than the minimum required claims ")
            self.logger.warning(
                f"Claim extraction yielded only {len(merged)} claims "
                f"(min_claims={self.min_claims}); proceeding with what we got"
            )
        ranked = self._declare_claim_importance(uuid, goal, merged, grounding)
        return ranked

    def _per_source_targets(self, n_sources: int = 3) -> tuple[int, int]:
        """Per-source min/max claim targets derived from the global bounds.

        Scaled by the number of extraction sources to ensure the overall target is met.

        Args:
            n_sources: Number of extraction sources to share the global budget across.

        Returns:
            Tuple ``(per_source_min, per_source_max)`` of target claim counts.
        """
        n_sources = max(1, n_sources)
        per_min = max(2, self.min_claims // n_sources)
        per_max = max(per_min, max(2, self.max_claims // n_sources))
        return per_min, per_max

    # ------------------------------------------------------------------
    # Source-specific extraction prompts
    # ------------------------------------------------------------------

    def _build_source_a_prompt(
        self,
        goal: str,
        grounding: str,
        workspace_listing: str,
        target_min: int,
        target_max: int,
    ) -> str:
        """Source A — what the LITERATURE demands of a correct solution.

        Args:
            goal: Workflow goal text.
            grounding: Literature grounding block; may be empty.
            workspace_listing: Rendered listing of workspace files.
            target_min: Lower bound on the number of claims to elicit.
            target_max: Upper bound on the number of claims to elicit.

        Returns:
            Fully formatted prompt string for the judge.
        """
        grounding_block = grounding.strip() if grounding else "(no literature grounding available)"
        return f"""You are extracting  claims for a verification rubric: requirements the peer-reviewed literature places on any correct solution to this task, independent of what the agents actually did.

LITERATURE GROUNDING:
{grounding_block}

WORKSPACE FILES (relative to workspace root):
{workspace_listing}

TASK:
Extract claims that represent the things the LITERATURE demands of a
correct solution:
- Required methodology steps (e.g. "data was normalised before PCA",
  "cross-validation was performed with k≥5", "the energy minimisation
  converged to a stationary point").
- Required outputs / quality bars the field treats as load-bearing
  (e.g. "the predicted structure has RMSD ≤ X to the reference",
  "the regression model reports an R² on a held-out test set").
- Required constraints / sanity properties standard in the field
  (e.g. "probabilities sum to 1", "the contact matrix is symmetric",
  "the conformation is a valid self-avoiding walk").

MANDATORY GOAL CLAIM. The first claim MUST assert that the workflow
produced the specific scientific deliverable the task requested AND that
it meets the literature-standard success criterion. If the task names a
quantitative bar (accuracy ≥ x, energy ≤ y, AUC ≥ z, p < α), this claim
must encode that bar — not merely "a result exists". Phrase it so a
workflow that skipped, faked, or left the deliverable empty FAILS it.

{_CLAIM_RULES_BLOCK}

Aim for {target_min}–{target_max} claims.
"""

    def _build_source_b_prompt(
        self,
        goal: str,
        workspace_listing: str,
        target_min: int,
        target_max: int,
    ) -> str:
        """Source B — what the USER explicitly required in the goal text.

        Args:
            goal: Workflow goal text.
            workspace_listing: Rendered listing of workspace files.
            target_min: Lower bound on the number of claims to elicit.
            target_max: Upper bound on the number of claims to elicit.

        Returns:
            Fully formatted prompt string for the judge.
        """
        return f"""You are extracting claims for a verification rubric: requirements the user explicitly stated in the workflow goal.

WORKFLOW GOAL:
{goal}

WORKSPACE FILES (relative to workspace root):
{workspace_listing}

TASK:
Read ONLY the goal text above. Extract claims that capture instructions
and deliverables the user spelled out. Claim FAILS if the
agents skipped, weakened, or substituted what the user asked for.

Look for, in the goal:
- Explicit deliverables ("produce a CSV with columns A,B,C", "save the
  trained model to disk", "render a phylogenetic tree as SVG").
- Explicit method / tool choices ("use random forest with 100 trees",
  "run BLAST against the nr database", "fit with sklearn's PCA").
- Explicit numeric or qualitative success bars ("accuracy ≥ 90%",
  "energy ≤ −20 kJ/mol", "p < 0.05", "all residues classified").
- Explicit comparisons or controls ("compare against a random baseline",
  "include a negative control", "report both train and test metrics").
- Explicit scope constraints ("over the 2020–2024 window", "for the
  test split only", "use the 20-mer sequence HPHPPHHPHPPHPHHPPHPH").
- Explicit output format constraints ("as JSON", "one row per sample",
  "rounded to 3 decimal places").
- Input dataset's exact column names, order, and data types in your output.
- The claims Ensure no suffixes (e.g., _prob, _score) or renamed columns unless the task explicitly specifies a different output schema.
- Any hint, advice, recommandations, treat them as explicit user requirements that must be followed

If the goal is short and contains few explicit requirements, return a
short list — DO NOT pad with claims the user did not write. It is fine
to return fewer than {target_min} claims when the goal is terse; do not
invent constraints.

{_CLAIM_RULES_BLOCK}

Aim for up to {target_max} Source-B claims, but only as many as the goal
text actually warrants.
"""

    def _build_source_c_prompt(
        self,
        goal: str,
        workspace_listing: str,
        target_min: int,
        target_max: int,
    ) -> str:
        """Source C — mathematical sanity properties of the produced artefacts.

        Args:
            goal: Workflow goal text.
            workspace_listing: Rendered listing of workspace files.
            target_min: Lower bound on the number of claims to elicit.
            target_max: Upper bound on the number of claims to elicit.

        Returns:
            Fully formatted prompt string for the judge.
        """
        return f"""You are extracting claims for a verification rubric: closed-form mathematical sanity properties.

WORKFLOW GOAL:
{goal}

WORKSPACE FILES (relative to workspace root):
{workspace_listing}

TASK:
Extract  claims that are mathematical invariants and structural
properties that follow from the type of object produced and that a small
numerical check can confirm directly against the on-disk artefact. A
 claim FAILS if the artefact violates a property any correct
solution would have respected.

Look for properties such as:
- Probability constraints (values in [0,1]; rows of a probability matrix
  sum to 1; class probabilities non-negative).
- Matrix / tensor properties (symmetry of distance or covariance matrices;
  positive semi-definiteness of covariance; zero diagonal of distance
  matrices; triangle inequality; correct shapes / dimensions).
- Numerical sanity (no NaN, no infinity, no negative variances, no
  negative counts, no out-of-domain values for log/sqrt).
- Conservation, monotonicity, dimensional consistency (an energy below a
  physical upper bound; cumulative distributions monotonic; unit
  consistency between inputs and outputs).
- Structural validity (a self-avoiding walk has no repeated coordinates;
  a tree on n nodes has n-1 edges; an alignment has matching sequence
  lengths; a graph's adjacency matrix matches its edge list).
- Cardinality / shape consistency (output row count matches input row
  count on a per-row task; predictions equal the test set size; feature
  counts agree across train and test).

Prefer claims that can be checked with a tiny script reading the relevant
artefact. Violating a mathematical invariant means the result is not just
suboptimal — it is incorrect.

{_CLAIM_RULES_BLOCK}

Aim for {target_min}–{target_max} claims, but only ones grounded
in the actual artefacts visible in the workspace listing. Do not invent
properties for objects the task does not produce.
"""

    def _build_source_d_prompt(
        self,
        goal: str,
        workspace_listing: str,
        target_min: int,
        target_max: int,
    ) -> str:
        """Source D — non-negotiable computational reproducibility / CS practice.

        Args:
            goal: Workflow goal text.
            workspace_listing: Rendered listing of workspace files.
            target_min: Lower bound on the number of claims to elicit.
            target_max: Upper bound on the number of claims to elicit.

        Returns:
            Fully formatted prompt string for the judge.
        """
        return f"""You are extracting claims for a verification rubric: NON-NEGOTIABLE computational reproducibility requirementsto re-run this work on a fresh machine.

WORKFLOW GOAL:
{goal}

WORKSPACE FILES (relative to workspace root):
{workspace_listing}

TASK:
Extract claims that capture essential computational-reproducibility
requirements. The scope is intentionally narrow: only things without
which a second party CANNOT re-run this work on a fresh machine. The
bar is "can it be re-run", NOT "is it nicely engineered".

ALLOWED claim shapes:
- The workspace declares its dependencies in a standard manifest
  (`requirements.txt`, `pyproject.toml`, or `environment.yml`) AND the
  declared packages cover the third-party imports actually used by the
  produced code — i.e. the manifest is non-empty and is not missing a
  library that the workspace's `.py` files import.
- The dependencies are pinned to specific versions (e.g. `numpy==1.25.3` rather than `numpy>=1.20` or `numpy`).
- If the produced code uses stochastic operations (random sampling,
  shuffling, model training, weight init, train/test split), a random
  seed is fixed in code (`numpy.random.seed`, `random.seed`,
  `torch.manual_seed`, `random_state=...`) so the run is reproducible.
- A clearly identifiable runnable entrypoint exists (a single top-level
  `.py` such as `main.py`, `run.py`, `pipeline.py`, or unambiguous from
  the layout) so a re-runner knows what to launch.
- The workspace is not pathologically cluttered with junk (no thousands
  of unrelated files; no obvious accumulation of failed intermediate
  dumps that would confuse a re-runner).

EXPLICITLY FORBIDDEN — DO NOT extract claims about any of these:
- README files, documentation, markdown, or doc presence of any kind.
- Docstrings, comments, or in-code documentation.
- Tests, test coverage, or test presence.
- Code style (PEP8, line length, naming conventions, formatting).
- Type hints / type annotations.
- Logging structure, log file presence, or log verbosity.
This source verifies non-negotiable computer-science PRACTICE — not
engineering aesthetics.

Eachclaim MUST chain a file / structural property to a
functional reproducibility consequence — never bare existence. Example
WELL-FORMED claim: "the workspace declares its dependencies in a standard
manifest covering the packages actually imported by the produced code".
Example MALFORMED claim: "a requirements.txt file exists in the workspace".

The deps-manifest, absolute-paths, and seed-on-stochastic claims genuinely
block re-execution and matter most; entrypoint, clutter, and output-location
claims are nice-to-have. The post-extraction importance pass will weight them
accordingly.

{_CLAIM_RULES_BLOCK}

Aim for up to {target_max} Source-D claims, but only as many as the
workspace actually warrants — fewer is fine. Do not pad.
"""

    def _build_source_e_prompt(
        self,
        goal: str,
        execution_text: str,
        workspace_listing: str,
        target_min: int,
        target_max: int,
    ) -> str:
        """Source E — statistical fingerprint / non-triviality of the result.

        Args:
            goal: Workflow goal text.
            execution_text: Agent narration / produced output text.
            workspace_listing: Rendered listing of workspace files.
            target_min: Lower bound on the number of claims to elicit.
            target_max: Upper bound on the number of claims to elicit.

        Returns:
            Fully formatted prompt string for the judge.
        """
        return f"""You are creating a list of verification claims for a verification rubric: statistical-fingerprint and non-triviality checks.

WORKFLOW GOAL:
{goal}

WORKFLOW OUTPUT (agents narration — names the headline metrics they report):
{execution_text}

WORKSPACE FILES (relative to workspace root):
{workspace_listing}

TASK:
Create a list of check that the produced result is
NON-TRIVIAL and STATISTICALLY REAL — i.e. that it could not have been
achieved by a degenerate, leaking, or hard-coded "solution".
The claim FAILS if the on-disk artefact bears the fingerprint of a vacuous
success.

Look for properties such as:
- The headline metric beats a trivial baseline by a non-trivial margin
  (random / majority-class / mean predictor / shuffled-label baseline);
  on a balanced binary task, accuracy is above 0.55; on a regression
  task, the model beats the mean predictor in R² or RMSE.
- The prediction distribution is not degenerate: not constant, not all
  one class, not a single value repeated, not uniformly 0.5, with non-zero
  variance across rows in continuous outputs.
- No data-leakage signatures: train and test sets are disjoint (no
  overlapping IDs or rows); the test set is not a subset of training data;
  perfect or near-perfect scores on a known-hard task are flagged as
  suspect unless the artefact explicitly justifies them.
- No suspicious hard-coded or fallback patterns in outputs (predictions
  all identical, all integers when probabilities were expected, exact
  reproduction of an input column as the "prediction").
- Sample sizes are adequate for the test (n above a sensible floor for
  the statistic being claimed; enough samples per class for stratified
  metrics).
- Where probabilities are produced, they show inter-class separation
  rather than collapsing to a single point.

A result statistically indistinguishable from a baseline is not a
scientific success — claims that target the headline result deserve
extraction. Skip baseline claims for tasks with no obvious null to
compare against — do not invent one.

{_CLAIM_RULES_BLOCK}

Aim for {target_min}–{target_max} claims, only as many as the
on-disk artefacts can actually support.
"""

    # ------------------------------------------------------------------
    # Per-source parsing + path validation
    # ------------------------------------------------------------------

    def _parse_and_filter_claims(
        self,
        uuid: str,
        data: Any,
    ) -> list[dict[str, Any]]:
        """Validate the LLM JSON, normalise each claim, drop confabulated paths.

        Importance is intentionally NOT assigned here — it is rated by a
        separate post-merge pass (``_declare_claim_importance``) so the
        per-source extractors only need to enumerate candidates.

        Args:
            uuid: Workflow identifier (used in error context).
            data: Parsed JSON payload from the judge; expected to contain a
                ``claims`` list or to be a list itself.

        Returns:
            List of normalised claim dicts with stable ids and workspace-validated
            relevant file lists.

        Raises:
            LLMEvaluationError: When ``data`` does not carry a usable claims list.
        """
        claims = data.get("claims", []) if isinstance(data, dict) else data
        if not isinstance(claims, list):
            raise LLMEvaluationError(f"Claim extractor JSON has no 'claims' list for {uuid}")

        cleaned: list[dict[str, Any]] = []
        for idx, c in enumerate(claims):
            if not isinstance(c, dict) or "description" not in c:
                continue
            raw_files = c.get("likely_relevant_files") or []
            relevant = self._validate_workspace_paths(
                raw_files,
                allowed=self._workspace_files or None,
                label=str(c.get("id") or f"c{idx}"),
            )
            cleaned.append({
                "id": str(c.get("id") or f"c{idx}"),
                "description": str(c["description"]).strip(),
                "likely_relevant_files": relevant,
            })
        return cleaned

    # Note: ``_validate_workspace_paths`` lives on the workspace mixin since
    # its state (``self._workspace_files``) is owned there. Both the claim
    # parser above and the file selector in the per-claim mixin call into it
    # via MRO.

    # ------------------------------------------------------------------
    # Stage 1b — rate claim importance (1-10) against the goal, dedupe
    # ------------------------------------------------------------------

    def _declare_claim_importance(
        self,
        uuid: str,
        goal: str,
        claims: list[dict[str, Any]],
        grounding: str = "",
    ) -> list[dict[str, Any]]:
        """Rate each merged claim on a 1-10 importance scale and drop duplicates.

        Replaces the previous hard/soft step-function tier. A single judge call
        sees the goal and the full merged claim list; it returns (a) ids to drop
        as near-duplicates and (b) an importance + one-sentence rationale for
        each surviving claim. Goal-alignment is the dominant axis — the prompt
        anchors the scale on what the user explicitly asked for.

        On parse failure or judge error every input claim is returned with
        ``importance = _DEFAULT_CLAIM_IMPORTANCE``: the run still scores rather
        than crashing, but the gradient is intentionally muted so the next
        iteration is not steered by a noisy rating.

        Args:
            uuid: Workflow identifier (used for the judge call).
            goal: Workflow goal text — primary anchor for importance.
            claims: Merged, source-tagged claims from the per-source extractors.
            grounding: Optional peer-reviewed literature grounding block.

        Returns:
            Filtered claim list with ``importance`` (int 1-10) and
            ``importance_rationale`` (one-sentence str) populated on every entry.
        """
        if not claims:
            return claims

        prompt = self._build_importance_prompt(goal, claims, grounding)
        data, err = self._call_judge_for_json(
            uuid, "verifier_declare_importance", prompt
        )
        if err is not None or not isinstance(data, dict):
            self.logger.warning(
                f"importance rater failed for {uuid} ({err or 'non-dict JSON'}); "
                f"falling back to uniform importance={self._DEFAULT_CLAIM_IMPORTANCE}"
            )
            return [self._with_default_importance(c) for c in claims]

        drop_ids = self._extract_drop_ids(data)
        importance_by_id = self._extract_importance_map(data)

        kept: list[dict[str, Any]] = []
        for c in claims:
            cid = c.get("id")
            if cid in drop_ids:
                self.logger.debug(f"importance rater dropped duplicate claim {cid}")
                continue
            imp, rationale = importance_by_id.get(
                cid, (self._DEFAULT_CLAIM_IMPORTANCE, "")
            )
            kept.append({
                **c,
                "importance": imp,
                "importance_rationale": rationale,
            })

        print_ok(
            f"Rated {len(kept)} claims for {uuid} "
            f"(dropped {len(drop_ids)} duplicate(s))"
        )
        return kept

    def _build_importance_prompt(
        self,
        goal: str,
        claims: list[dict[str, Any]],
        grounding: str,
    ) -> str:
        """Render the rater prompt: scale anchors + goal + claim list.

        Args:
            goal: Workflow goal text.
            claims: Merged claims to rate.
            grounding: Optional grounding block; may be empty.

        Returns:
            Fully formatted prompt string for the rater judge.
        """
        grounding_block = (
            grounding.strip() if grounding else "(no literature grounding available)"
        )
        claim_lines = "\n".join(
            f"- id={c.get('id')!r}  source={c.get('source', 'unknown')}  "
            f"desc={str(c.get('description', '')).strip()[:300]}"
            for c in claims
        )
        return f"""You are rating verification claims by how much they matter for the user's goal.

WORKFLOW GOAL:
{goal}

LITERATURE GROUNDING:
{grounding_block}

{_IMPORTANCE_ANCHOR_BLOCK}

CLAIMS TO RATE:
{claim_lines}

TASKS:
1. Identify near-duplicate claims (same checked property, different wording or
   source) and list the redundant ids to drop. Keep the clearest version of
   each cluster. Do NOT drop claims that check different facets — only true
   duplicates.
2. For every surviving claim, return an integer importance 1–10 anchored on
   the scale above (goal-alignment dominates), plus a one-sentence rationale
   stating what makes the claim that important.

Return STRICT JSON only, in this exact shape:
{{
  "drop_ids": ["<id>", ...],
  "importance": [
    {{"id": "<id>", "importance": <int 1-10>, "rationale": "<one sentence>"}},
    ...
  ]
}}
"""

    @staticmethod
    def _extract_drop_ids(data: dict[str, Any]) -> set[str]:
        """Best-effort parse of the ``drop_ids`` list from the rater JSON."""
        raw = data.get("drop_ids") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return set()
        return {str(x) for x in raw if isinstance(x, str) and x}

    def _extract_importance_map(
        self,
        data: dict[str, Any],
    ) -> dict[str, tuple[int, str]]:
        """Project the ``importance`` array into ``{id: (importance, rationale)}``.

        Importance values are clamped to ``[1, 10]`` and rationales coerced to
        strings; malformed entries are skipped silently — the caller falls back
        to the default importance for any claim missing from the map.

        Args:
            data: Parsed rater JSON.

        Returns:
            Map from claim id to ``(importance, rationale)``.
        """
        out: dict[str, tuple[int, str]] = {}
        raw = data.get("importance") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return out
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            cid = entry.get("id")
            if not isinstance(cid, str) or not cid:
                continue
            try:
                imp = int(entry.get("importance", self._DEFAULT_CLAIM_IMPORTANCE))
            except (TypeError, ValueError):
                imp = self._DEFAULT_CLAIM_IMPORTANCE
            imp = max(1, min(10, imp))
            rationale = str(entry.get("rationale") or "").strip()
            out[cid] = (imp, rationale)
        return out

    def _with_default_importance(self, claim: dict[str, Any]) -> dict[str, Any]:
        """Stamp a claim with the default importance + an empty rationale."""
        return {
            **claim,
            "importance": self._DEFAULT_CLAIM_IMPORTANCE,
            "importance_rationale": "",
        }


if __name__ == "__main__":
    expected = {
        "_extract_claims",
        "_per_source_targets",
        "_build_source_a_prompt",
        "_build_source_b_prompt",
        "_build_source_c_prompt",
        "_build_source_d_prompt",
        "_build_source_e_prompt",
        "_parse_and_filter_claims",
        "_declare_claim_importance",
        "_build_importance_prompt",
        "_extract_drop_ids",
        "_extract_importance_map",
        "_with_default_importance",
    }
    actual = {n for n in dir(_VerifierClaimExtractionMixin) if not n.startswith("__")}
    missing = expected - actual
    assert not missing, f"claims mixin missing methods: {missing}"
    print("verifier_claims: smoke ok")
