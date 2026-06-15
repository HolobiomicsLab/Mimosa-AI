
from sources.utils.perspicacite_client import query_perspicacite

def get_perspicacite_grounding(goal: str) -> str:
    """Query Perspicacite for citation-grounded expectations of how a science goal should be achieved.

    Args:
        goal: The science goal given to the agents.

    Returns:
        The Perspicacite response text, or a fallback message indicating that
        the query failed (so the caller can proceed without external grounding).
    """
    prompt = f"""
TASK: Provide scientific literature evidence about how the following science goal is typically and successfully achieved, so that an agent's eventual answer can be evaluated against established practice.

SCIENCE GOAL:
{goal}

INSTRUCTIONS:
1. INFER SUCCESS INDICATORS: Identify 15-35 concrete scientific claims, findings, methods, or measurable outcomes whose presence in an answer would credibly indicate the goal has been achieved. They must be specific enough to be checked against literature (not generic virtues like "rigorous methodology").
2. SEARCH: For each indicator, find relevant peer-reviewed literature on how the goal is approached in practice — state-of-the-art methods, reported empirical results, established quantitative ranges, and known pitfalls. Prioritize empirical studies, meta-analyses, and methods papers.
3. PROVIDE EVIDENCE: For each indicator, cite specific papers establishing the expected approach, result, or constraint.
4. SUMMARIZE: Give a concise overview of where the literature converges or disagrees on how this goal should be achieved, including common failure modes and methodological caveats relevant to judging an answer.
5. IF GOAL IS VAGUE: If the goal is too underspecified to extract concrete success indicators, provide foundational literature on the broader topic that would help evaluate any reasonable answer, and flag the underspecification.

RULES:
- Do NOT explain what grounding means or discuss AI/XAI concepts
- Do NOT give generic methodology advice outside what the literature supports
- ONLY output concrete expected claims/methods and their supporting evidence
- If no relevant literature exists for an indicator, state "No peer-reviewed evidence found for this specific aspect"
- Prioritize empirical findings and established methods over theoretical discussions
"""
    try:
        response = query_perspicacite(prompt)
        return response
    except Exception as e:
        return "Perspicacite query failed, unable to provide grounded expectations. Proceeding without external grounding."