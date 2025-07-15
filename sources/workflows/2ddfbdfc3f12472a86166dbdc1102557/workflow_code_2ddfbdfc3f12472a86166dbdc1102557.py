# ================================================================
#  LANGGRAPH – SMOLAGENT WORKFLOW
#  Objective (placeholder):  "YOUR OBJECTIVE HERE"
#  Demonstrates rigorous task-decomposition, multi–agent hand-off,
#  robust routing, retry loops (≤3) and emergency fallbacks.
# ================================================================

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

# ----------------------------------------------------------------
#  STATE SCHEMA (ALREADY DEFINED IN RUNTIME – **DO NOT** REDECLARE)
# ----------------------------------------------------------------
# class Action(TypedDict): ...
# class Observation(TypedDict): ...
# class WorkflowState(TypedDict): ...

# ----------------------------------------------------------------
#  1.  AGENT PROMPTS (one atomic responsibility each)
# ----------------------------------------------------------------
# --- 1a. Web Researcher (find raw information) -------------------
instruct_web_researcher = """
You are a specialised WEB RESEARCH agent.

TOPIC TO RESEARCH:  "YOUR OBJECTIVE HERE"

YOUR TASK (atomic):
1. Search the web and collect the most relevant, credible sources.
2. Extract key facts, figures, links & citations.

COMPLETION PROTOCOL
SUCCESS  -> End with:   RESEARCH_COMPLETE:
FAILURE  -> End with:   RESEARCH_FAILURE:
ERROR    -> End with:   GIVE_UP:

Always give a concise summary (bullet points + sources) before the
trigger phrase.
"""

# --- 1b. Content Validator (quick quality & link check) ----------
instruct_validator = """
You are a CONTENT VALIDATION agent.

INPUT: You receive raw research notes & links from previous step.

YOUR TASK (atomic):
1. Verify that at least 3 valid links exist and are accessible.
2. Flag broken or suspicious links.

COMPLETION PROTOCOL
SUCCESS  -> End with:   VALIDATION_COMPLETE:
INSUFFICIENT_INFORMATION -> End with: VALIDATION_INSUFFICIENT:
FAILURE  -> End with:   VALIDATION_FAILURE:
ERROR    -> End with:   GIVE_UP:
"""

# --- 1c. CSV Creator (persist data) ------------------------------
instruct_csv_creator = """
You are a CSV CREATOR agent.

INPUT: Validated research facts in free-text.

YOUR TASK (atomic):
1. Create a new CSV dataset 'research_data.csv'
   Columns: source, headline, summary, url
2. Insert one row per validated source.

COMPLETION PROTOCOL
SUCCESS  -> End with:   CSV_COMPLETE:
FAILURE  -> End with:   CSV_FAILURE:
ERROR    -> End with:   GIVE_UP:
"""

# --- 1d. R Analyzer (run simple descriptive stats) ---------------
instruct_r_analyzer = """
You are an R ANALYSIS agent.

INPUT: The dataset 'research_data.csv'.

YOUR TASK (atomic):
1. Using R, load the csv.
2. Produce counts, wordcloud of headlines or any simple summary stats.
3. Save any script as 'analysis.R'.

COMPLETION PROTOCOL
SUCCESS  -> End with:   ANALYSIS_COMPLETE:
FAILURE  -> End with:   ANALYSIS_FAILURE:
ERROR    -> End with:   GIVE_UP:
"""

# --- 1e. Report Generator (create human friendly report) ---------
instruct_reporter = """
You are a REPORT GENERATOR agent.

INPUT: Analytical results + original csv.

YOUR TASK (atomic):
1. Write a concise executive summary of findings.
2. Highlight interesting statistics.
3. Provide recommendations or insights.

COMPLETION PROTOCOL
SUCCESS  -> End with:   REPORT_COMPLETE:
FAILURE  -> End with:   REPORT_FAILURE:
ERROR    -> End with:   GIVE_UP:
"""

# --- 1f. Bash Emergency Executor (last-resort tool) --------------
instruct_bash_emergency = """
You are an EMERGENCY RECOVERY agent.

YOUR TASK (atomic):
Attempt any bash command that may fix previous step's fatal error
(e.g., install missing package, fetch file again, etc.).

COMPLETION PROTOCOL
SUCCESS  -> End with:   BASH_FIX_COMPLETE:
FAILURE  -> End with:   BASH_FIX_FAILURE:
ERROR    -> End with:   GIVE_UP:
"""

# ----------------------------------------------------------------
#  2.  AGENT INSTANTIATION (SmolAgentFactory is already available)
# ----------------------------------------------------------------
web_researcher    = SmolAgentFactory("web_researcher",    instruct_web_researcher, WEB_BROWSER_MCP_TOOLS)
content_validator  = SmolAgentFactory("content_validator", instruct_validator,       WEB_BROWSER_MCP_TOOLS)
csv_creator        = SmolAgentFactory("csv_creator",       instruct_csv_creator,     CSV_MANAGEMENT_TOOLS)
r_analyzer         = SmolAgentFactory("r_analyzer",        instruct_r_analyzer,      R_COMMAND_MCP_TOOLS)
report_generator   = SmolAgentFactory("report_generator",  instruct_reporter,        CSV_MANAGEMENT_TOOLS)
bash_emergency     = SmolAgentFactory("bash_emergency",    instruct_bash_emergency,  BASH_COMMAND_MCP_TOOLS)

# ----------------------------------------------------------------
#  3.  WORKFLOW INITIALISATION
# ----------------------------------------------------------------
workflow = StateGraph(WorkflowState)

# ----------------------------------------------------------------
#  4.  HELPER – attempt counter
# ----------------------------------------------------------------
def count_attempts(state: WorkflowState, node_name: str) -> int:
    try:
        return [n for n in state.get("step_name", []) if node_name in n].__len__()
    except Exception:
        return 0

# ----------------------------------------------------------------
#  5.  ROUTING FUNCTIONS (one per stage, each with retries & fallback)
# ----------------------------------------------------------------
MAX_RETRIES = 3

def router_after_research(state: WorkflowState) -> str:
    print("🔀 RouterAfterResearch")
    try:
        last_answer = state["answers"][-1] if state.get("answers") else ""
        retries     = count_attempts(state, "web_researcher")
        if "RESEARCH_COMPLETE" in last_answer:
            return "content_validator"
        if retries < MAX_RETRIES:
            return "web_researcher"            # retry
        else:
            return "bash_emergency"            # fallback
    except Exception as e:
        print(f"Router error: {e}")
        return "bash_emergency"

def router_after_validation(state: WorkflowState) -> str:
    print("🔀 RouterAfterValidation")
    try:
        last_answer = state["answers"][-1] if state.get("answers") else ""
        retries     = count_attempts(state, "content_validator")
        if "VALIDATION_COMPLETE" in last_answer:
            return "csv_creator"
        if "VALIDATION_INSUFFICIENT" in last_answer and retries < MAX_RETRIES:
            return "web_researcher"            # go back gather more
        if retries < MAX_RETRIES:
            return "content_validator"         # retry validation
        return "bash_emergency"                # fallback
    except Exception as e:
        print(f"Router error: {e}")
        return "bash_emergency"

def router_after_csv(state: WorkflowState) -> str:
    print("🔀 RouterAfterCSV")
    try:
        last_answer = state["answers"][-1] if state.get("answers") else ""
        retries     = count_attempts(state, "csv_creator")
        if "CSV_COMPLETE" in last_answer:
            return "r_analyzer"
        if retries < MAX_RETRIES:
            return "csv_creator"               # retry
        return "bash_emergency"
    except Exception as e:
        print(f"Router error: {e}")
        return "bash_emergency"

def router_after_analysis(state: WorkflowState) -> str:
    print("🔀 RouterAfterAnalysis")
    try:
        last_answer = state["answers"][-1] if state.get("answers") else ""
        retries     = count_attempts(state, "r_analyzer")
        if "ANALYSIS_COMPLETE" in last_answer:
            return "report_generator"
        if retries < MAX_RETRIES:
            return "r_analyzer"                # retry
        return "bash_emergency"
    except Exception as e:
        print(f"Router error: {e}")
        return "bash_emergency"

def router_after_report(state: WorkflowState) -> str:
    print("🔀 RouterAfterReport")
    try:
        last_answer = state["answers"][-1] if state.get("answers") else ""
        if "REPORT_COMPLETE" in last_answer:
            return END
        else:
            return "bash_emergency"
    except Exception as e:
        print(f"Router error: {e}")
        return "bash_emergency"

def router_after_bash(state: WorkflowState) -> str:
    print("🔀 RouterAfterBash (EMERGENCY PATH)")
    # After emergency attempt, stop workflow regardless of result
    return END

# ----------------------------------------------------------------
#  6.  NODE REGISTRATION
# ----------------------------------------------------------------
workflow.add_node("web_researcher",   WorkflowNodeFactory.create_agent_node(web_researcher))
workflow.add_node("content_validator",WorkflowNodeFactory.create_agent_node(content_validator))
workflow.add_node("csv_creator",      WorkflowNodeFactory.create_agent_node(csv_creator))
workflow.add_node("r_analyzer",       WorkflowNodeFactory.create_agent_node(r_analyzer))
workflow.add_node("report_generator", WorkflowNodeFactory.create_agent_node(report_generator))
workflow.add_node("bash_emergency",   WorkflowNodeFactory.create_agent_node(bash_emergency))

# ----------------------------------------------------------------
#  7.  EDGES & CONDITIONAL ROUTING
# ----------------------------------------------------------------
workflow.add_edge(START, "web_researcher")

workflow.add_conditional_edges(
    "web_researcher",
    router_after_research,
    {
        "content_validator": "content_validator",
        "web_researcher":    "web_researcher",   # retry
        "bash_emergency":    "bash_emergency"
    }
)

workflow.add_conditional_edges(
    "content_validator",
    router_after_validation,
    {
        "csv_creator":       "csv_creator",
        "web_researcher":    "web_researcher",   # back to research
        "content_validator": "content_validator",# retry
        "bash_emergency":    "bash_emergency"
    }
)

workflow.add_conditional_edges(
    "csv_creator",
    router_after_csv,
    {
        "r_analyzer":     "r_analyzer",
        "csv_creator":    "csv_creator",         # retry
        "bash_emergency": "bash_emergency"
    }
)

workflow.add_conditional_edges(
    "r_analyzer",
    router_after_analysis,
    {
        "report_generator": "report_generator",
        "r_analyzer":       "r_analyzer",
        "bash_emergency":   "bash_emergency"
    }
)

workflow.add_conditional_edges(
    "report_generator",
    router_after_report,
    {
        END:             END,
        "bash_emergency": "bash_emergency"
    }
)

workflow.add_conditional_edges(
    "bash_emergency",
    router_after_bash,
    { END: END }
)

# ----------------------------------------------------------------
#  8.  COMPILE WORKFLOW
# ----------------------------------------------------------------
app = workflow.compile()