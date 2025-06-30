# ======================  LangGraph ‑ SmolAgent WORKFLOW  ======================
#
#  OVERALL GOAL
#  ---------------------------------------------------------------------------
#  1) Verify that the requested task is actually possible
#  2) Perform a deep web-search on how to use *mzmind* in batch / CLI mode
#  3) Extract the CLI information into a structured CSV
#  4) Build an installation plan for macOS Apple-Silicon (arm64)
#  5) Execute the installation plan via shell commands
#  ---------------------------------------------------------------------------
#  The graph contains:
#     • 6 SmolAgents  (feasibility, primary search, fallback search,
#                      extractor, planner, shell-executor)
#     • 2 specialised routing functions with multi-path fallbacks
#     • 2 separate fallback mechanisms (alternate researcher & manual review)
#  ---------------------------------------------------------------------------

# --------  Mandatory imports (pre-installed in execution environment)  -------
from langgraph.graph import StateGraph, START, END

# ---------- Agent instruction definitions  -----------------------------------
# 1) Feasibility-check agent ---------------------------------------------------
instruct_feasibility = """
You are a feasibility assessment agent.

GOAL
Determine if the following objective is realistically achievable with
currently available public resources:

    "Perform a deep search on how to use mzmind in batch/command-line mode
     and then install the software on macOS Apple-Silicon (arm64)."

TASK
1. Use web search to confirm ALL of the following:
   • The software 'mzmind' exists.
   • It supports batch or CLI usage.
   • Downloadable source or binaries exist (or can be built) for macOS arm64.

2. OUTPUT PROTOCOL (start your answer with ONE keyword only):
   • FEASIBLE        – if all conditions are met
   • NOT_FEASIBLE    – if the goal cannot be achieved (give short reason)
   • GIVE_UP         – on technical errors (explain the error)
"""

# 2) Primary deep-search agent -------------------------------------------------
instruct_search = """
You are an advanced web-research agent.

OBJECTIVE
   Gather authoritative information on using 'mzmind' via batch / CLI.

ON SUCCESS – end your answer with exactly: SEARCH_COMPLETE
   • Provide concrete command examples.
   • Document required flags, config files, or scripts.
   • List and link ALL sources (official docs, blog posts, issues, etc.).

ON FAILURE – end with: SEARCH_FAIL
   • Detail what search terms + strategies you tried.
   • Explain which information could not be found.
   • Suggest next avenues of search.

ON ERROR   – end with: GIVE_UP
"""

# 3) Fallback deep-search agent (alternate approach) --------------------------
instruct_alt_search = """
You are a rescue research agent called after earlier failure.

OBJECTIVE
   Try alternative approaches to find CLI/batch documentation for 'mzmind'.

ALLOWED STRATEGIES
   • Search non-English sources.
   • Query code-hosting sites (GitHub, GitLab).
   • Look for archived copies or forks.

OUTPUT PROTOCOL identical to the primary agent:
   SEARCH_COMPLETE / SEARCH_FAIL / GIVE_UP
"""

# 4) Extraction agent ---------------------------------------------------------
instruct_extract = """
You are an information extraction agent.

INPUT – previous answer(s) that contain raw notes & URLs about mzmind CLI.

TASK
1. Pull out each distinct CLI command or flag.
2. Produce a CSV with columns: command, description, source_url.

SUCCESS  – Start with: EXTRACTION_DONE
FAILURE  – Start with: EXTRACTION_FAIL   (explain what was missing)
ERROR    – Start with: GIVE_UP           (explain error)
"""

# 5) Installation plan agent --------------------------------------------------
instruct_plan = """
You are an installation planning agent.

INPUT – CSV of commands + sources for mzmind CLI.

OUTPUT PLAN (macOS 13+ Apple-Silicon):
1. Pre-requisites (e.g. brew install …)
2. Exact shell commands to download/build/install mzmind.
3. Verification command to confirm CLI works.

SUCCESS – Begin with: PLAN_READY
FAILURE – Begin with: PLAN_FAIL  (state information gaps)
ERROR   – Begin with: GIVE_UP
"""

# 6) Shell execution agent ----------------------------------------------------
instruct_shell = """
You are a shell-execution agent.

TASK
Execute the installation plan step-by-step on the local machine.

PROTOCOL
• For every command: run → capture stdout/stderr.
• Retry a failing command once; if still non-zero, stop.

SUCCESS – Start with: INSTALL_SUCCESS
FAILURE – Start with: INSTALL_FAIL  (include failing command & output)
ERROR   – Start with: GIVE_UP
"""

# 7) Manual review fallback ----------------------------------------------------
instruct_manual = """
You are a human-support agent.

If automatic installation failed, provide detailed manual instructions,
troubleshooting suggestions, and next steps for a human operator.

End with: REVIEW_COMPLETE
"""

# ----------  SmolAgent creation ----------------------------------------------
feasibility_agent   = SmolAgentFactory(instruct_feasibility, BROWSER_TOOLS)
search_agent        = SmolAgentFactory(instruct_search, BROWSER_TOOLS)
alt_search_agent    = SmolAgentFactory(instruct_alt_search, BROWSER_TOOLS)
extract_agent       = SmolAgentFactory(instruct_extract, CSV_TOOLS)
plan_agent          = SmolAgentFactory(instruct_plan, BROWSER_TOOLS)
shell_agent         = SmolAgentFactory(instruct_shell, SHELL_TOOLS)
manual_agent        = SmolAgentFactory(instruct_manual, BROWSER_TOOLS)   # tools unused

# ----------  Workflow initialisation  ----------------------------------------
workflow = StateGraph(WorkflowState)

# ----------  Add agent nodes -------------------------------------------------
workflow.add_node("feasibility_checker", WorkflowNodeFactory.create_agent_node(feasibility_agent))
workflow.add_node("deep_search",         WorkflowNodeFactory.create_agent_node(search_agent))
workflow.add_node("alt_deep_search",     WorkflowNodeFactory.create_agent_node(alt_search_agent))
workflow.add_node("extract_info",        WorkflowNodeFactory.create_agent_node(extract_agent))
workflow.add_node("install_planner",     WorkflowNodeFactory.create_agent_node(plan_agent))
workflow.add_node("shell_installer",     WorkflowNodeFactory.create_agent_node(shell_agent))
workflow.add_node("manual_review",       WorkflowNodeFactory.create_agent_node(manual_agent))

# ----------  Routing / decision functions ------------------------------------
def feasibility_router(state: WorkflowState) -> str:
    print("=== Routing: Feasibility Check ===")
    try:
        if not state.get("answers"):
            print("No answer present – emergency END.")
            return "END"
        answer = state["answers"][-1]
        if answer.startswith("FEASIBLE"):
            return "deep_search"
        elif answer.startswith("NOT_FEASIBLE"):
            return "END"
        else:
            # Any other keyword treated as error
            return "END"
    except Exception as e:
        print(f"Feasibility router error: {e}")
        return "END"

def search_router(state: WorkflowState) -> str:
    print("=== Routing: Primary Search ===")
    try:
        answer = state["answers"][-1] if state.get("answers") else ""
        attempts = state["step_name"].count("deep_search")
        if "SEARCH_COMPLETE" in answer:
            return "extract_info"
        if "SEARCH_FAIL" in answer and attempts < 3:
            print(f"Retrying deep_search (attempt {attempts+1}/3)")
            return "deep_search"
        # either exhausted retries or GIVE_UP / repeated failure
        return "alt_deep_search"
    except Exception as e:
        print(f"Search router error: {e}")
        return "alt_deep_search"

def alt_search_router(state: WorkflowState) -> str:
    print("=== Routing: Alternate Search ===")
    try:
        answer = state["answers"][-1] if state.get("answers") else ""
        if "SEARCH_COMPLETE" in answer:
            return "extract_info"
        return "END"   # give up after alternate search
    except Exception as e:
        print(f"Alt search router error: {e}")
        return "END"

def extract_router(state: WorkflowState) -> str:
    print("=== Routing: Extraction ===")
    try:
        answer = state["answers"][-1]
        if "EXTRACTION_DONE" in answer:
            return "install_planner"
        if "EXTRACTION_FAIL" in answer:
            return "deep_search"   # back to search for more data
        return "END"
    except Exception as e:
        print(f"Extraction router error: {e}")
        return "END"

def plan_router(state: WorkflowState) -> str:
    print("=== Routing: Plan ===")
    try:
        answer = state["answers"][-1]
        if "PLAN_READY" in answer:
            return "shell_installer"
        if "PLAN_FAIL" in answer:
            return "deep_search"   # gather more info
        return "END"
    except Exception as e:
        print(f"Plan router error: {e}")
        return "END"

def shell_router(state: WorkflowState) -> str:
    print("=== Routing: Shell Execution ===")
    try:
        answer = state["answers"][-1]
        attempts = state["step_name"].count("shell_installer")
        if "INSTALL_SUCCESS" in answer:
            return "END"
        if "INSTALL_FAIL" in answer and attempts < 2:
            print(f"Retrying shell installer (attempt {attempts+1}/2)")
            return "shell_installer"
        # ultimate failure or GIVE_UP
        return "manual_review"
    except Exception as e:
        print(f"Shell router error: {e}")
        return "manual_review"

def manual_router(state: WorkflowState) -> str:
    # Manual review ends the workflow
    return "END"

# ----------  Edges & conditional paths ---------------------------------------
workflow.add_edge(START, "feasibility_checker")

workflow.add_conditional_edges(
    "feasibility_checker",
    feasibility_router,
    {
        "deep_search": "deep_search",
        "END": END
    }
)

workflow.add_conditional_edges(
    "deep_search",
    search_router,
    {
        "extract_info": "extract_info",
        "deep_search":  "deep_search",       # retry
        "alt_deep_search": "alt_deep_search"
    }
)

workflow.add_conditional_edges(
    "alt_deep_search",
    alt_search_router,
    {
        "extract_info": "extract_info",
        "END": END
    }
)

workflow.add_conditional_edges(
    "extract_info",
    extract_router,
    {
        "install_planner": "install_planner",
        "deep_search": "deep_search",
        "END": END
    }
)

workflow.add_conditional_edges(
    "install_planner",
    plan_router,
    {
        "shell_installer": "shell_installer",
        "deep_search": "deep_search",
        "END": END
    }
)

workflow.add_conditional_edges(
    "shell_installer",
    shell_router,
    {
        "shell_installer": "shell_installer",   # retry
        "manual_review": "manual_review",
        "END": END
    }
)

workflow.add_conditional_edges(
    "manual_review",
    manual_router,
    {
        "END": END
    }
)

# ----------  Compile workflow -------------------------------------------------
app = workflow.compile()

# -----------  (Execution happens outside this snippet)  -----------------------