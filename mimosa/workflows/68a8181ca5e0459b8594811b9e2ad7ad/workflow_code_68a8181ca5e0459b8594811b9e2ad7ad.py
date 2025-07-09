# ============================================
# LangGraph ‑ SmolAgent workflow
# Task:  “Search who is working at HolobiomicsLab and make a report as a CSV (name, role)”
# ============================================

# ---- PRE-DECLARED OBJECTS (already in interpreter context) --------------
#   • WorkflowState TypedDict (step_name, actions, observations, rewards, answers, success)
#   • SmolAgentFactory
#   • WorkflowNodeFactory
#   • Tool packages:  CSV_TOOL_TOOLS , BROWSER_TOOL_TOOLS , SHELL_TOOL_TOOLS
# -------------------------------------------------------------------------

from langgraph.graph import StateGraph, START, END

# ============ 1)  AGENT PROMPTS  =========================================
# ── A1 :  WEB RESEARCHER  ────────────────────────────────────────────────
instruct_web_research = """
You are a focused WEB RESEARCH agent.
GOAL
- Discover *authoritative* public web pages that list **people working at HolobiomicsLab**.

YOUR TASK
1. Use browser tools to search the Internet.
2. Collect up to 5 DIRECT URLs that clearly list staff / members / researchers at HolobiomicsLab.
3. Provide a concise bullet list of those URLs for downstream agents (no parsing, no names yet).

COMPLETION PROTOCOL
SUCCESS → reply ONLY with:  
SEARCH_COMPLETE: <url_1> , <url_2> , …

FAILURE → if no suitable pages found, reply ONLY with:  
SEARCH_FAILURE: [short explanation of what was tried]

ERROR  → tool problems / technical error, reply ONLY with:  
GIVE_UP: [error description]
"""

# ── A2 :  DATA EXTRACTOR  ───────────────────────────────────────────────
instruct_extractor = """
You are a DATA-EXTRACTION agent.

INPUT
The previous agent’s answer supplies URLs containing HolobiomicsLab personnel info.

YOUR TASK
1. Visit each provided URL.
2. Extract EVERY individual’s full **Name** and **Role/Title** mentioned on those pages.
3. Output ONE line per person in the exact format:  <Name> | <Role>

COMPLETION PROTOCOL
SUCCESS → reply ONLY with:  
EXTRACT_COMPLETE:
<Name1> | <Role1>
<Name2> | <Role2>
…

INSUFFICIENT_INFORMATION → if pages lack clear names/roles, reply ONLY with:  
INSUFFICIENT_INFORMATION: [what was missing]

FAILURE → scraping obstacles, reply ONLY with:  
EXTRACT_FAILURE: [reason]

ERROR → technical error, reply ONLY with:  
GIVE_UP: [error]
"""

# ── A3 :  VALIDATOR  ────────────────────────────────────────────────────
instruct_validator = """
You are a VALIDATION agent ensuring data quality.

INPUT
A raw list of lines: <Name> | <Role>

YOUR TASK
1. Check each line contains both a Name and a Role.
2. Remove exact duplicates.
3. Confirm at least 3 distinct people exist.

COMPLETION PROTOCOL
VALIDATION_OK → if data passes all checks, reply ONLY with:  
VALIDATION_OK

INSUFFICIENT_INFORMATION → if any issue (missing role, <3 people, etc.) reply ONLY with:  
INSUFFICIENT_INFORMATION: [explanation & what is missing]

ERROR → unexpected technical issue, reply ONLY with:  
GIVE_UP: [error]
"""

# ── A4 :  CSV FORMATTER  ────────────────────────────────────────────────
instruct_csv = """
You are a CSV-FORMATTING agent.

INPUT
Clean list of <Name> | <Role> pairs already validated.

YOUR TASK
1. Create a CSV file (header: Name,Role) with each pair on its own row.
2. Save / output via the CSV tools.

COMPLETION PROTOCOL
SUCCESS → reply ONLY with:  
CSV_COMPLETE: [path or textual CSV content]

FAILURE → tool or write error, reply ONLY with:  
CSV_FAILURE: [reason]

ERROR → unexpected technical error, reply ONLY with:  
GIVE_UP: [error]
"""

# ============ 2)  AGENT INSTANTIATION  ==================================
agent_web       = SmolAgentFactory(instruct_web_research, BROWSER_TOOL_TOOLS)
agent_extract   = SmolAgentFactory(instruct_extractor,      BROWSER_TOOL_TOOLS)
agent_validate  = SmolAgentFactory(instruct_validator,      SHELL_TOOL_TOOLS)
agent_csv       = SmolAgentFactory(instruct_csv,            CSV_TOOL_TOOLS)

# ============ 3)  WORKFLOW CONSTRUCTION  ================================
workflow = StateGraph(WorkflowState)

# ---- Nodes -------------------------------------------------------------
workflow.add_node("web_researcher", WorkflowNodeFactory.create_agent_node(agent_web))
workflow.add_node("extractor",      WorkflowNodeFactory.create_agent_node(agent_extract))
workflow.add_node("validator",      WorkflowNodeFactory.create_agent_node(agent_validate))
workflow.add_node("csv_maker",      WorkflowNodeFactory.create_agent_node(agent_csv))

# ============ 4)  ROUTING / ERROR-HANDLING FUNCTIONS ====================
MAX_RETRIES = 3   # global retry ceiling per agent


def _count_attempts(state: WorkflowState, step_tag: str) -> int:
    """Count how many times a given step (or its retry variants) appears."""
    names = state.get("step_name", [])
    return sum(1 for n in names if n.startswith(step_tag))


# ---------- Router after WEB RESEARCHER ---------------------------------
def router_after_web(state: WorkflowState) -> str:
    try:
        last_answer = state.get("answers", [""])[-1]
        attempts    = _count_attempts(state, "web_researcher")

        if last_answer.startswith("SEARCH_COMPLETE"):
            return "extractor"

        if last_answer.startswith("SEARCH_FAILURE") and attempts < MAX_RETRIES:
            return "web_researcher"           # retry same agent
        if last_answer.startswith("GIVE_UP"):
            return END

        # Unexpected output – treat as failure but retry once
        if attempts < MAX_RETRIES:
            return "web_researcher"
        return END
    except Exception as e:
        print(f"Routing error after web: {e}")
        return END


# ---------- Router after EXTRACTOR --------------------------------------
def router_after_extract(state: WorkflowState) -> str:
    try:
        last_answer = state.get("answers", [""])[-1]
        attempts    = _count_attempts(state, "extractor")

        if last_answer.startswith("EXTRACT_COMPLETE"):
            return "validator"

        if last_answer.startswith("INSUFFICIENT_INFORMATION"):
            return "web_researcher"            # go back, need more sources

        if last_answer.startswith("EXTRACT_FAILURE") and attempts < MAX_RETRIES:
            return "extractor"                 # retry extraction

        if last_answer.startswith("GIVE_UP"):
            return END

        # Unexpected output
        if attempts < MAX_RETRIES:
            return "extractor"
        return END
    except Exception as e:
        print(f"Routing error after extract: {e}")
        return END


# ---------- Router after VALIDATOR --------------------------------------
def router_after_validate(state: WorkflowState) -> str:
    try:
        last_answer = state.get("answers", [""])[-1]
        attempts    = _count_attempts(state, "validator")

        if last_answer.startswith("VALIDATION_OK"):
            return "csv_maker"

        if last_answer.startswith("INSUFFICIENT_INFORMATION"):
            return "extractor"                 # back for better data

        if last_answer.startswith("GIVE_UP"):
            return END

        # Unexpected text
        if attempts < MAX_RETRIES:
            return "validator"
        return END
    except Exception as e:
        print(f"Routing error after validate: {e}")
        return END


# ---------- Router after CSV MAKER --------------------------------------
def router_after_csv(state: WorkflowState) -> str:
    try:
        last_answer = state.get("answers", [""])[-1]
        attempts    = _count_attempts(state, "csv_maker")

        if last_answer.startswith("CSV_COMPLETE"):
            return END

        if last_answer.startswith("CSV_FAILURE") and attempts < MAX_RETRIES:
            return "csv_maker"

        if last_answer.startswith("GIVE_UP"):
            return END

        # Unexpected
        if attempts < MAX_RETRIES:
            return "csv_maker"
        return END
    except Exception as e:
        print(f"Routing error after csv: {e}")
        return END


# ============ 5)  EDGE DEFINITIONS  =====================================
workflow.add_edge(START, "web_researcher")

workflow.add_conditional_edges(
    "web_researcher",
    router_after_web,
    {
        "extractor":      "extractor",
        "web_researcher": "web_researcher",
        END:              END,
    }
)

workflow.add_conditional_edges(
    "extractor",
    router_after_extract,
    {
        "validator":  "validator",
        "web_researcher": "web_researcher",
        "extractor":  "extractor",
        END:          END,
    }
)

workflow.add_conditional_edges(
    "validator",
    router_after_validate,
    {
        "csv_maker":  "csv_maker",
        "extractor":  "extractor",
        "validator":  "validator",
        END:          END,
    }
)

workflow.add_conditional_edges(
    "csv_maker",
    router_after_csv,
    {
        END:         END,
        "csv_maker": "csv_maker",
    }
)

# ============ 6)  COMPILE WORKFLOW  =====================================
app = workflow.compile()

# ------------------------------------------------------------------------
# The compiled `app` can now be invoked with an initial (possibly empty)
# WorkflowState dictionary, e.g.  app.invoke(initial_state)
# ------------------------------------------------------------------------