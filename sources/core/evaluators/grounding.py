
"""
Perspicacite grounding for litterature-retrieve of science goals success indicators.
"""

from sources.utils.perspicacite_client import query_perspicacite


def get_perspicacite_grounding(goal: str) -> str:
    """Retrieve a literature grounding block for a workflow goal.

    Delegates to the local Perspicacite agentic retriever. Prompts it to
    PLAN multiple parallel sub-searches across every dimension that applies
    to the goal (dataset provenance, standard hyperparameters, preprocessing
    conventions, model families, evaluation metrics, pitfalls) so that
    dataset-canonical and method-canonical papers surface alongside
    generic background.
    """
    prompt = f"""
TASK: Provide scientific literature evidence about how the following science goal is typically and successfully achieved, so that an agent's eventual answer can be evaluated against established practice.

SCIENCE GOAL:
{goal}

INSTRUCTIONS:
1. INFER SUCCESS INDICATORS: Identify 15-35 concrete scientific claims, findings, methods, or measurable outcomes whose presence in an answer would credibly indicate the goal has been achieved. They must be specific enough to be checked against literature (not generic virtues like "rigorous methodology").

2. PLAN MANY SUB-SEARCHES — do not collapse to a single query. Run separate, targeted searches across every dimension that applies to this goal. At minimum, consider:
   - DATASET / SOURCE: if the goal names a dataset, references named columns, or shows an example-data preview, look up the canonical paper that introduced the data; surface its activity cutoffs, train/test split conventions, recommended featurisers, known label-noise caveats, and sentinel/masked-value codes.
   - STANDARD HYPERPARAMETERS: for each model, optimiser, or algorithm implied by the goal, return the parameter ranges and defaults peer-reviewed work has converged on (number of estimators, learning rates, regularisation strengths, fingerprint radii, k-fold counts, batch sizes, early-stopping criteria, etc.).
   - PREPROCESSING: standard data cleaning, normalisation / scaling, deduplication, train/test splitting conventions, scaffolding, sentinel-value handling, leakage controls relevant to this kind of input.
   - MODEL FAMILY: the model families and architectures the literature treats as competitive baselines for this task; their reported strengths, weaknesses, and failure modes.
   - EVALUATION: metrics the field uses, reported empirical bars, statistical-significance conventions, and the trivial-baseline numbers an answer must beat to be credible.
   - PITFALLS: documented failure modes specific to this kind of task (data leakage signatures, common over-claiming patterns, brittle preprocessing assumptions).
   Each dimension that applies to the goal should drive at least one sub-search; skip a dimension only when the goal makes it irrelevant.

3. PROVIDE EVIDENCE: For each indicator, cite specific papers establishing the expected approach, result, or constraint.

4. SUMMARIZE: Give a concise overview of where the literature converges or disagrees on how this goal should be achieved, including common failure modes and methodological caveats relevant to judging an answer.

5. IF GOAL IS VAGUE: If the goal is too underspecified to extract concrete success indicators, provide foundational literature on the broader topic that would help evaluate any reasonable answer, and flag the underspecification.

RULES:
- Do NOT explain what grounding means or discuss AI/XAI concepts
- Do NOT give generic methodology advice outside what the literature supports
- ONLY output concrete expected claims/methods and their supporting evidence
- If no relevant literature exists for an indicator, state "No peer-reviewed evidence found for this specific aspect"
- Prioritize empirical findings and established methods over theoretical discussions
- If the literature is silent on a specific threshold or hyperparameter, say so — do not invent a number
"""
    try:
        response = query_perspicacite(prompt)
        return response
    except Exception:
        return "Perspicacite query failed, unable to provide grounded expectations. Proceeding without external grounding."
