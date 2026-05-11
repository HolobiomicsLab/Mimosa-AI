
from sources.utils.perspicacite_client import query_perspicacite

def get_perspicacite_grounding(execution_text: str) -> str:
    """Query Perspicacite for citation grounded verification of workflow execution.
    Args:
        execution_text: The execution text to send to Perspicacite
    """
    prompt = f"""
TASK: Provide scientific literature evidence to verify specific claims from an AI agent's answer.

AGENT OUTPUT TO VERIFY:
{execution_text}

INSTRUCTIONS:
1. EXTRACT: Identify 2-5 specific factual/scientific claims made in the agent's answer that aim to address the workflow's goal.
2. SEARCH: For each claim, find relevant peer-reviewed literature that supports or contradicts it, prioritizing empirical studies and meta-analyses
3. PROVIDE EVIDENCE: For each claim, cite specific papers that support or contradict it
4. SUMMARIZE: For all claims, summarize the evidence and state whether it supports, contradicts, or is inconclusive regarding the claim
5. IF NO CLAIMS: If the agent's answer is vague or non-specific, focus on the goal to provide litterature grounding useful for the evaluation of the answer rather than specific claims.

RULES:
- Do NOT explain what verification means or discuss AI/XAI concepts
- Do NOT provide general methodology advice
- ONLY output concrete claims and their supporting/contradicting evidence
- If no relevant literature exists for a claim, state "No peer-reviewed evidence found for this specific claim"
- Prioritize empirical findings over theoretical discussions
"""
    try:
        response = query_perspicacite(prompt)
        return response
    except Exception as e:
        return "Perspicacite query failed, unable to provide grounded verification. Estimating plausibility based on available evidence without external grounding."