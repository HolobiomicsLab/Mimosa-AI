# ================================================
# LangGraph ★ SmolAgent  – Transformers Experiment
# ================================================

# ---------  IMPORTS (MANDATORY) -----------------
from langgraph.graph import StateGraph, START, END

# The SmolAgentFactory and WorkflowNodeFactory ARE
# pre-defined in the execution environment
# Tools lists are already available in scope:
# SHELL_TOOLS, BROWSER_TOOLS, CSV_TOOLS
# ------------------------------------------------

# ---------------- STATE TYPE --------------------
# (Provided by system description – no re-declaration needed)
# class WorkflowState(TypedDict): ...

# ------------- AGENT INSTRUCTIONS ---------------
instruct_search_primary = """
You are a specialised scholarly web-search agent.

TASK
- Use academic search engines & arXiv to find the LATEST (≤12 months)
  research papers on Transformer architecture improvements.
- Return a BULLETED list with title, authors, date, one-sentence summary
  and link for at least 5 papers.

SUCCESS CRITERIA
- If you gathered ≥5 recent papers say SEARCH_COMPLETE
- If search failed or you found <5 suitable papers say SEARCH_FAILURE
- If tool error beyond your control say GIVE_UP
"""

instruct_search_secondary = """
You are a fallback broad web search agent.

TASK
- Perform a broader internet search (blog posts, conference talks, GitHub)
  for new Transformer techniques or variants not captured in scholarly search.
- Produce a list (min 3) with title, source, brief summary, and link.

SUCCESS CRITERIA
- If you compiled ≥3 distinct techniques say SEARCH_COMPLETE
- If unable to compile list say SEARCH_FAILURE
- On tool errors say GIVE_UP
"""

instruct_extract_primary = """
You are an information-extraction agent.

INPUT
- You receive the raw search result list in previous observation.

TASK
- For EACH item, extract: technique name, key idea, reported gains, and any
  code link.
- Output a JSON list with keys: 'technique','idea','gains','code_link'.

SUCCESS CRITERIA
- If JSON is well-formed and covers all items say EXTRACT_COMPLETE
- Otherwise say EXTRACT_FAILURE
- If tool error say GIVE_UP
"""

instruct_select = """
You are a critical evaluation agent.

INPUT
- A JSON list of techniques with ideas and reported gains.

TASK
- Score each technique from 1-10 for novelty & practicality.
- Select the TOP technique and justify in ≤150 chars.
- Output exactly:
  CHOSEN_TECHNIQUE: <name>
  JUSTIFICATION: <text>
  SELECTION_COMPLETE

FAILURE/ERROR
- If data insufficient say SELECTION_FAILURE
- On unexpected error say GIVE_UP
"""

instruct_implement = """
You are an implementation agent with shell access.

TASK
- Create a minimal PyTorch prototype of the CHOSEN_TECHNIQUE.
- Save code to 'model_variant.py'.
- Train on a tiny synthetic dataset to ensure code runs (epochs<=2).
- Dump training metrics to 'results.csv'.

SUCCESS CRITERIA
- If script executed without exception and 'results.csv' exists say IMPLEMENT_COMPLETE
- If training failed say IMPLEMENT_FAILURE
- On tool error say GIVE_UP
"""

instruct_evaluate = """
You are an evaluation agent.

TASK
- Read 'results.csv'.
- Compute final accuracy or loss.
- Compare against a vanilla Transformer baseline of your knowledge; estimate improvement.
- Output succinct report and say EVALUATE_COMPLETE

FAILURE
- If 'results.csv' missing or unreadable say EVALUATE_FAILURE
- On tool error say GIVE_UP
"""

# ------------- AGENT CREATION -------------------
search_primary_agent   = SmolAgentFactory(instruct_search_primary,   BROWSER_TOOLS)
search_secondary_agent = SmolAgentFactory(instruct_search_secondary, BROWSER_TOOLS)
extract_primary_agent  = SmolAgentFactory(instruct_extract_primary,  BROWSER_TOOLS)
select_agent           = SmolAgentFactory(instruct_select,           [])
implement_agent        = SmolAgentFactory(instruct_implement,        SHELL_TOOLS)
evaluate_agent         = SmolAgentFactory(instruct_evaluate,         CSV_TOOLS)

# ------------- NODE WRAPPERS --------------------
node_search_primary   = WorkflowNodeFactory.create_agent_node(search_primary_agent)
node_search_secondary = WorkflowNodeFactory.create_agent_node(search_secondary_agent)
node_extract_primary  = WorkflowNodeFactory.create_agent_node(extract_primary_agent)
node_select           = WorkflowNodeFactory.create_agent_node(select_agent)
node_implement        = WorkflowNodeFactory.create_agent_node(implement_agent)
node_evaluate         = WorkflowNodeFactory.create_agent_node(evaluate_agent)

# ------------- ROUTING FUNCTIONS ----------------
def route_search(state: "WorkflowState") -> str:
    """
    Decide next step after a search agent.
    Fallback hierarchy:
        1) retry primary search (≤2 retries)
        2) switch to secondary search
        3) give up (END)
    """
    try:
        answers   = state.get("answers", [])
        steps     = state.get("step_name", [])
        last_ans  = answers[-1] if answers else ""
        retry_cnt = steps.count("search_primary")

        if "SEARCH_COMPLETE" in last_ans:
            return "extract_primary"
        if "GIVE_UP" in last_ans:
            return "search_secondary"
        # search failure
        if retry_cnt < 2:
            return "search_primary"          # retry same agent
        return "search_secondary"            # escalate to secondary
    except Exception as e:
        print(f"💥 route_search error: {e}")
        return END

def route_secondary_search(state: "WorkflowState") -> str:
    try:
        last_ans = state.get("answers", [])[-1]
        if "SEARCH_COMPLETE" in last_ans:
            return "extract_primary"
        return END
    except Exception as e:
        print(f"💥 route_secondary_search error: {e}")
        return END

def route_extract(state: "WorkflowState") -> str:
    try:
        answers   = state.get("answers", [])
        steps     = state.get("step_name", [])
        last_ans  = answers[-1] if answers else ""
        retry_cnt = steps.count("extract_primary")

        if "EXTRACT_COMPLETE" in last_ans:
            return "select"
        if "GIVE_UP" in last_ans:
            return "search_secondary"
        if retry_cnt < 2:
            return "extract_primary"         # retry extraction
        return "search_secondary"            # go back & broaden search
    except Exception as e:
        print(f"💥 route_extract error: {e}")
        return END

def route_select(state: "WorkflowState") -> str:
    try:
        last_ans = state.get("answers", [])[-1]
        if "SELECTION_COMPLETE" in last_ans:
            return "implement"
        return END                           # cannot proceed without selection
    except Exception as e:
        print(f"💥 route_select error: {e}")
        return END

def route_implement(state: "WorkflowState") -> str:
    try:
        answers   = state.get("answers", [])
        steps     = state.get("step_name", [])
        last_ans  = answers[-1] if answers else ""
        retry_cnt = steps.count("implement")

        if "IMPLEMENT_COMPLETE" in last_ans:
            return "evaluate"
        if "GIVE_UP" in last_ans:
            return END
        if retry_cnt < 1:
            return "implement"               # single retry allowed (heavy step)
        return END
    except Exception as e:
        print(f"💥 route_implement error: {e}")
        return END

def route_evaluate(state: "WorkflowState") -> str:
    try:
        last_ans = state.get("answers", [])[-1]
        if "EVALUATE_COMPLETE" in last_ans:
            return END
        return END                           # graceful degradation: end anyway
    except Exception as e:
        print(f"💥 route_evaluate error: {e}")
        return END

# ------------- WORKFLOW DEFINITION --------------
workflow = StateGraph(WorkflowState)

# ---- Nodes ----
workflow.add_node("search_primary",   node_search_primary)
workflow.add_node("search_secondary", node_search_secondary)
workflow.add_node("extract_primary",  node_extract_primary)
workflow.add_node("select",           node_select)
workflow.add_node("implement",        node_implement)
workflow.add_node("evaluate",         node_evaluate)

# ---- Edges & Conditional Routing ----
workflow.add_edge(START, "search_primary")

workflow.add_conditional_edges(
    "search_primary",
    route_search,
    {
        "extract_primary":  "extract_primary",
        "search_primary":   "search_primary",   # retry
        "search_secondary": "search_secondary",
        END:                END,
    }
)

workflow.add_conditional_edges(
    "search_secondary",
    route_secondary_search,
    {
        "extract_primary": "extract_primary",
        END:               END,
    }
)

workflow.add_conditional_edges(
    "extract_primary",
    route_extract,
    {
        "select":           "select",
        "extract_primary":  "extract_primary",
        "search_secondary": "search_secondary",
        END:                END,
    }
)

workflow.add_conditional_edges(
    "select",
    route_select,
    {
        "implement": "implement",
        END:         END,
    }
)

workflow.add_conditional_edges(
    "implement",
    route_implement,
    {
        "evaluate":  "evaluate",
        "implement": "implement",
        END:         END,
    }
)

workflow.add_conditional_edges(
    "evaluate",
    route_evaluate,
    {
        END: END,
    }
)

# ---- Compile ----
app = workflow.compile()

# The compiled `app` can now be invoked with an initial empty WorkflowState.
# Example:
# initial_state = {
#     "step_name": [], "actions": [], "observations": [],
#     "rewards": [], "answers": [], "success": []
# }
# result = app.invoke(initial_state)