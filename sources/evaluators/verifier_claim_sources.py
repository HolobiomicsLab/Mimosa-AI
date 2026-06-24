"""Claim-extraction source registry: the five prompts and their dispatch table.

Each source = a labelled prompt that elicits a different *kind* of claim
(literature requirements, user goal, math sanity, reproducibility, statistical
fingerprint). Adding a sixth source = define one builder and append one entry
to ``SOURCES`` — no edits in the extraction loop.

The shared rules block (``_CLAIM_RULES_BLOCK``) is defined here so all five
builders pull from one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


# Rules every source prompt appends verbatim. Kept here so the five builders
# share one definition rather than five copies that can drift.
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


@dataclass(frozen=True)
class ClaimContext:
    """Inputs available to every source-prompt builder.

    Attributes:
        goal: Original workflow goal text.
        workspace_listing: Rendered listing of workspace files.
        target_min: Lower bound on the number of claims to elicit.
        target_max: Upper bound on the number of claims to elicit.
        grounding: Peer-reviewed literature grounding block; consumed by
            source A only. Empty string when unavailable.
        execution_text: Agent narration / produced output; consumed by
            source E only.
        prior_claims: Rendered text of claims produced for this source on
            an earlier workflow for the same task. The builder appends it
            so the LLM is steered to reproduce the same ids and descriptions
            where they still apply. Empty string when no prior cache exists.
    """

    goal: str
    workspace_listing: str
    target_min: int
    target_max: int
    grounding: str = ""
    execution_text: str = ""
    prior_claims: str = ""


# Rendered when prior claims for this source exist. The instruction keeps the
# claim_id set stable across iterations of the same task so verifier scores
# are comparable, while still permitting adaptation when the current workspace
# legitimately diverges from the seed (e.g. a referenced file no longer
# exists, or the workflow now produces a different artefact type).
_PRIOR_CLAIMS_INSTRUCTION = """
PRIOR CLAIMS — REPRODUCE THE SAME LIST WHERE STILL APPLICABLE
The list below was extracted from a prior workflow attempt for THIS SAME
task. Reproduce this list as closely as possible:
- KEEP the same `id` verbatim for any prior claim that still applies to the
  current workspace; reusing the id is how downstream scoring stays
  comparable across iterations.
- KEEP the description if it still describes a true success condition; adapt
  it only when the current artefact's shape or naming makes the old wording
  wrong (e.g. a referenced file is gone, the workflow now produces a PNG
  where it previously produced a CSV).
- DROP a prior claim only when no artefact in the current workspace could
  plausibly let it be checked (the file is gone and nothing replaces it).
- ADD new claims only when the current workflow exposes a success condition
  not covered by any prior claim. Do not duplicate intents already covered.

PRIOR CLAIMS (id — description; likely_relevant_files):
{prior_claims}
"""


def _render_prior_claims_block(prior_claims: str) -> str:
    """Return the prior-claims instruction block, or empty when no cache exists."""
    text = (prior_claims or "").strip()
    if not text:
        return ""
    return _PRIOR_CLAIMS_INSTRUCTION.format(prior_claims=text)


@dataclass(frozen=True)
class ClaimSource:
    """One claim-extraction source: a label and its prompt builder.

    Attributes:
        label: Single-letter source tag (used in claim ids and logs).
        build: Function rendering the full prompt from a ``ClaimContext``.
    """

    label: str
    build: Callable[[ClaimContext], str]


def _build_source_a(ctx: ClaimContext) -> str:
    """Source A — what the LITERATURE demands of a correct solution."""
    grounding_block = (
        ctx.grounding.strip() if ctx.grounding else "(no literature grounding available)"
    )
    return f"""You are extracting  claims for a verification rubric: requirements the peer-reviewed literature places on any correct solution to this task, independent of what the agents actually did.
GOAL:
{ctx.goal}

LITERATURE GROUNDING:
{grounding_block}

WORKSPACE FILES (relative to workspace root):
{ctx.workspace_listing}

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
- Do not extract claims that conflict with the goal requirements or aren't possible per goal requirements
  (e.g. if the goal explicitly relaxes a standard or dataset preview imply that a method cann't be used, don't extract claims that would require it.)
- When the literature grounding lists multiple co-equal preprocessing steps in one bullet (e.g. "normalization, handling missing values, dimensionality reduction"), emit one claim per named step rather than a single composite claim.
- Do NOT emit hyperparameter-value claims (`n_estimators`, `max_depth`, `learning_rate`, `batch_size`, `n_layers`, `dropout`, `epochs`, `kernel_size`, `n_folds`) unless the goal text literally pins that value.
- When the goal cites a package or repo, prefer claims that the workflow uses that package's documented API rather than re-implementing the method by hand.
- On visual deliverables (output is `.png` / `.pdf` / `.svg`): do not prescribe panel count, named plotting methods, or specific overlay types unless the goal literally names them.

MANDATORY GOAL CLAIM. The first claim MUST assert that the workflow
produced the specific scientific deliverable the task requested AND that
it meets the literature-standard success criterion. If the task names a
quantitative bar (accuracy ≥ x, energy ≤ y, AUC ≥ z, p < α), this claim
must encode that bar — not merely "a result exists". Phrase it so a
workflow that skipped, faked, or left the deliverable empty FAILS it.

{_render_prior_claims_block(ctx.prior_claims)}
{_CLAIM_RULES_BLOCK}

Aim for {ctx.target_min}–{ctx.target_max} claims.
"""


def _build_source_b(ctx: ClaimContext) -> str:
    """Source B — what the USER explicitly required in the goal text."""
    return f"""You are extracting claims for a verification rubric: requirements the user explicitly stated in the workflow goal.

WORKFLOW GOAL:
{ctx.goal}

WORKSPACE FILES (relative to workspace root):
{ctx.workspace_listing}

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
- Any hint, advice, recommendation: treat as an explicit user requirement.

LITERAL IDENTIFIERS — MINE THEM AGGRESSIVELY. The goal may contain
several kinds of literal identifiers the user is implicitly or explicitly
asking the workflow to reproduce. Find every such identifier and emit
ONE CLAIM PER IDENTIFIER, quoted verbatim in backticks. The kinds you
must scan for, regardless of how the goal is labelled or sectioned:

  * Example data (sample rows, header excerpts, JSON or YAML snippets,
    schema previews): the column names, key names, or field names shown
    are the EXACT identifiers the user expects in the output. Emit a
    claim of the form:
    "The output `<path>` has columns/keys exactly equal to
    `[<id1>, <id2>, ...]` in that order (same names, same order, no
    added suffixes like `_prob`/`_score`, no renames)."
    Use the identifiers as-is from the example; do NOT paraphrase.
  * Explicit deliverable specifications (output paths, filenames,
    file formats stated literally): every path, filename, and "EXACT"
    phrase becomes a claim of the form
    "The file `<exact/path/from/goal>` exists at the workspace location
    the goal specifies." Quote the path verbatim.
  * Named library, class, function, model, or dataset identifier —
    any backticked or capitalised code-identifier-shaped token in the
    goal body, plus anything introduced by "use X" / "implement with X"
    / "as defined by X": emit a claim
    "The workflow uses `<Identifier>` as named in the goal."
    One claim per distinct identifier.

When DATASET PREVIEW or an explicit output path is present, emit the exact-name and exact-path claims FIRST, before any concept paraphrases.
Numeric thresholds or cut-offs in the goal (e.g. `>= 0.6`, `top-10`, `5.5km`, `>280K`) are also literal identifiers — emit one claim per value, quoted verbatim, before any paraphrase.
"The output includes a primary-key column" (concept) and "The output header equals `['record_id','value_a','value_b']`" (exact) are DIFFERENT claims and you should emit BOTH when both are extractable from the goal.

If an example is truncated (ends with `...`, contains `[truncated]`, or
shows only a partial row), emit a weaker positional claim instead:
"The first N columns of `<path>` are exactly `[<visible names>]`" —
do NOT invent the omitted names.

If the goal contains no example data, no deliverable specifications,
and no named identifiers, skip this section silently. Do not fabricate
identifiers to fill the quota.

If the goal is short and contains few explicit requirements, return a
short list — DO NOT pad with claims the user did not write. It is fine
to return fewer than {ctx.target_min} claims when the goal is terse; do not
invent constraints.

{_render_prior_claims_block(ctx.prior_claims)}
{_CLAIM_RULES_BLOCK}

Aim for up to {ctx.target_max} Source-B claims, but only as many as the goal
text actually warrants.
"""


def _build_source_c(ctx: ClaimContext) -> str:
    """Source C — mathematical sanity properties of the produced artefacts."""
    return f"""You are extracting claims for a verification rubric: closed-form mathematical sanity properties.

WORKFLOW GOAL:
{ctx.goal}

WORKSPACE FILES (relative to workspace root):
{ctx.workspace_listing}

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
- Image-artefact properties (when the goal asks for a figure, plot,
  or other rendered image — PNG/PDF/SVG): the file opens with PIL,
  has non-zero width and height, has pixel variance above a trivial
  threshold (i.e. not a blank canvas), and — if the goal names a
  specific plot type or panel layout — has an aspect ratio and panel
  count consistent with that type. Express each threshold explicitly
  in the claim ("standard deviation of grayscale pixel values > 5",
  "image width >= 400 px", "at least 2 horizontally-tiled subregions
  detected by column-variance scan").
  Default to single-panel claims; emit multi-panel claims ONLY when the goal literally names a multi-panel helper (e.g. `bio_ecg_plot`, `subplots(N,M)`) or a panel count.

Prefer claims that can be checked with a tiny script reading the relevant
artefact. Violating a mathematical invariant means the result is not just
suboptimal — it is incorrect.

{_render_prior_claims_block(ctx.prior_claims)}
{_CLAIM_RULES_BLOCK}

Aim for {ctx.target_min}–{ctx.target_max} claims, but only ones grounded
in the actual artefacts visible in the workspace listing. Do not invent
properties for objects the task does not produce.
"""


def _build_source_d(ctx: ClaimContext) -> str:
    """Source D — non-negotiable computational reproducibility / CS practice."""
    return f"""You are extracting claims for a verification rubric: NON-NEGOTIABLE computational reproducibility requirementsto re-run this work on a fresh machine.

WORKFLOW GOAL:
{ctx.goal}

WORKSPACE FILES (relative to workspace root):
{ctx.workspace_listing}

TASK:
If the goal's deliverable is an image (`.png` / `.pdf` / `.svg`), emit no claims — the visual grader cannot see manifests or pinning.
Extract claims that capture essential computational-reproducibility
requirements. The scope is intentionally narrow: only things without
which a second party CANNOT re-run this work on a fresh machine. The
bar is "can it be re-run", NOT "is it nicely engineered".

ALLOWED claim shapes (only emit when the workspace listing already contains a manifest file — never emit a manifest-presence claim that would deterministically fail a workspace that has none):
- The workspace declares its dependencies in a standard manifest
  (`requirements.txt`, `pyproject.toml`, or `environment.yml`) AND the
  declared packages cover the third-party imports actually used by the
  produced code — i.e. the manifest is non-empty and is not missing a
  library that the workspace's `.py` files import.
- The dependencies are pinned to specific versions (e.g. `numpy==1.25.3` rather than `numpy>=1.20` or `numpy`).
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

{_render_prior_claims_block(ctx.prior_claims)}
{_CLAIM_RULES_BLOCK}

Aim for up to {ctx.target_max} Source-D claims, but only as many as the
workspace actually warrants — fewer is fine. Do not pad.
"""


def _build_source_e(ctx: ClaimContext) -> str:
    """Source E — statistical fingerprint / non-triviality of the result."""
    return f"""You are creating a list of verification claims for a verification rubric: statistical-fingerprint and non-triviality checks.

WORKFLOW GOAL:
{ctx.goal}

WORKFLOW OUTPUT (agents narration — names the headline metrics they report):
{ctx.execution_text}

WORKSPACE FILES (relative to workspace root):
{ctx.workspace_listing}

TASK:
If the goal's deliverable is an image (`.png` / `.pdf` / `.svg`), emit no claims — statistical fingerprints do not contribute to a visual resemblance judge.
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
  If any class proportion is ≥ 0.80, do NOT emit a majority-baseline accuracy claim — emit `AUC-ROC ≥ 0.6 on a separated test split` instead.
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
- No dataset-sentinel leakage into the workflow's outputs. If any
  column the workflow consumes contains values clearly outside the
  expected scientific range for that measurement (e.g. -999, -9999,
  -1 in a non-negative column, NaN, inf, or string markers like
  "missing", "?", "NA", ""), the claim must verify that rows
  carrying those sentinels were filtered BEFORE they entered
  training, slicing (top-k / bottom-k / quantile selection),
  thresholding, aggregation, or visualisation. A workflow that
  feeds sentinel-bearing rows into a min / max / sort, a histogram
  bin, a model fit, or a plot has produced a polluted result even
  if every per-row arithmetic step "succeeded". Sentinel leakage
  invalidates extreme-value selection in particular: the "bottom
  10" of a column containing -999 is not the bottom 10 of the real
  measurements. Sentinel-leakage claims target methodology
  validity and rate importance 8-9 whenever the polluted column
  drives a headline selection, ranking, fit, or figure.
- Sample sizes are adequate for the test (n above a sensible floor for
  the statistic being claimed; enough samples per class for stratified
  metrics).
- Where probabilities are produced, they show inter-class separation
  rather than collapsing to a single point.

A result statistically indistinguishable from a baseline is not a
scientific success — claims that target the headline result deserve
extraction. Skip baseline claims for tasks with no obvious null to
compare against — do not invent one.

{_render_prior_claims_block(ctx.prior_claims)}
{_CLAIM_RULES_BLOCK}

Aim for {ctx.target_min}–{ctx.target_max} claims, only as many as the
on-disk artefacts can actually support.
"""


SOURCES: tuple[ClaimSource, ...] = (
    ClaimSource("a", _build_source_a),
    ClaimSource("b", _build_source_b),
    ClaimSource("c", _build_source_c),
    ClaimSource("d", _build_source_d),
    ClaimSource("e", _build_source_e),
)


if __name__ == "__main__":
    ctx = ClaimContext(
        goal="dummy goal",
        workspace_listing="(no files)",
        target_min=2,
        target_max=5,
        grounding="(dummy)",
        execution_text="(dummy run)",
    )
    for src in SOURCES:
        rendered = src.build(ctx)
        assert "dummy goal" in rendered, f"source {src.label} did not interpolate goal"
        assert "<short_slug>" in rendered, f"source {src.label} missing claim-rules block"
    print(f"verifier_claim_sources: smoke ok ({len(SOURCES)} sources)")
