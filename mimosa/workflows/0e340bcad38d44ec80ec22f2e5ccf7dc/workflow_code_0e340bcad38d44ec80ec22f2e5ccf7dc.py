# -------------------------------------------
#  LangGraph – SmolAgent Workflow Definition
# -------------------------------------------

from langgraph.graph import StateGraph, START, END

# === 1.  Agents’ TOOLBOXES (already available) ==========
# SHELL_TOOLS
# BROWSER_TOOLS
# CSV_TOOLS
# =========================================================


# === 2.  AGENT INSTRUCTIONS  ==============================

instruct_search = """
You are a focused WEB SEARCH agent.

TASK
1. Search the web for “MetaboT Holobiomics CNRS GitHub”.
2. Identify the OFFICIAL GitHub repository URL of the MetaboT project.

OUTPUT RULES
- On success, respond only with: SEARCH_COMPLETE <repo_url>
- If search fails, say: SEARCH_FAILURE
- If an unrecoverable error occurs, say: GIVE_UP
"""

instruct_scrape = """
You are a CONTRIBUTOR SCRAPER agent.

INPUT
- A GitHub repository URL of MetaboT (from previous step).

TASK
1. Open the GitHub repository contributors page.
2. Extract a flat, comma-separated list of contributor names (no duplicates).

OUTPUT RULES
- On success, respond only with: SCRAPE_COMPLETE <name1,name2,…>
- If extraction fails, say: SCRAPE_FAILURE
- If tools throw unknown errors, say: GIVE_UP
"""

instruct_csv = """
You are a CSV WRITER agent.

INPUT
- List of contributor names.

TASK
1. Create / overwrite file  contributors_metabot.csv  with header “name”.
2. Write each contributor on its own row.

OUTPUT RULES
- On success, respond only with: CSV_COMPLETE
- If writing fails, say: CSV_FAILURE
- On unknown error, say: GIVE_UP
"""

instruct_install = """
You are an INSTALLATION agent working in shell.

INPUT
- GitHub repository URL for MetaboT.

TASK
1. Run: git clone <url> metabot
2. cd metabot
3. Run installation instructions (prefer `pip install -e .` or `pip install .`).

OUTPUT RULES
- On success, respond only with: INSTALL_COMPLETE
- If installation fails, say: INSTALL_FAILURE
- For unknown tool errors, say: GIVE_UP
"""

instruct_verify = """
You are a VERIFICATION agent.

TASK
1. In shell, run: metabot --help  (or equivalent) to ensure executable works.

OUTPUT RULES
- If command works, say: VERIFY_SUCCESS
- If command not found or fails, say: VERIFY_FAILURE
- For unexpected errors, say: GIVE_UP
"""

# === 3.  AGENT CREATION  =================================

smol_search   = SmolAgentFactory(instruct_search,  BROWSER_TOOLS)
smol_scrape   = SmolAgentFactory(instruct_scrape,  BROWSER_TOOLS)
smol_csv      = SmolAgentFactory(instruct_csv,     CSV_TOOLS)
smol_install  = SmolAgentFactory(instruct_install, SHELL_TOOLS)
smol_verify   = SmolAgentFactory(instruct_verify,  SHELL_TOOLS)

# === 4.  ROUTING UTILITIES  ===============================

def make_router(step_name:str, success_token:str, next_step:str):
    """
    Creates a router function for a particular step.
    Provides:
        - retry (max 3)
        - fallback (skip to next_step with degraded data)
        - emergency_fallback (END)
    """
    def router(state):
        try:
            answers   = state.get("answers", [])
            steps     = state.get("step_name", [])
            last_ans  = answers[-1] if answers else ""
            attempts  = sum(1 for s in steps if s == step_name)

            # --- SUCCESS ------------------------
            if success_token in last_ans:
                return next_step

            # --- GIVE-UP shortcut --------------
            if "GIVE_UP" in last_ans:
                print(f"🚨 {step_name} gave up -> emergency END")
                return "emergency_end"

            # --- RETRY (<3) --------------------
            if attempts < 3:
                print(f"🔄 {step_name} retry #{attempts+1}")
                return "retry_path"

            # --- FALLBACK (>3) -----------------
            print(f"⚠️ {step_name} fallback after {attempts} attempts")
            return "fallback_path"

        except Exception as e:
            print(f"💥 Router error at {step_name}: {e}")
            return "emergency_end"
    return router

# === 5.  CUSTOM FALLBACK NODES  ===========================

def empty_csv_fallback(state):
    """Graceful degradation: create placeholder CSV content in state."""
    state["observations"].append({"data": "Placeholder CSV generated – contributors unknown"})
    state["answers"].append("CSV_PLACEHOLDER_COMPLETE")
    state["success"].append(False)
    state["step_name"].append("empty_csv_fallback")
    return state

def noop_success(state):
    """Allows pipeline to continue to END when previous step already failed gracefully."""
    state["observations"].append({"data": "Workflow ended with partial success"})
    state["answers"].append("NOOP_END")
    state["success"].append(True)
    state["step_name"].append("noop_success")
    return state

# === 6.  WORKFLOW GRAPH  =================================

workflow = StateGraph(WorkflowState)

# --- Nodes ------------------------------------------------
workflow.add_node("search_repo",        WorkflowNodeFactory.create_agent_node(smol_search))
workflow.add_node("scrape_contributors",WorkflowNodeFactory.create_agent_node(smol_scrape))
workflow.add_node("write_csv",          WorkflowNodeFactory.create_agent_node(smol_csv))
workflow.add_node("install_project",    WorkflowNodeFactory.create_agent_node(smol_install))
workflow.add_node("verify_install",     WorkflowNodeFactory.create_agent_node(smol_verify))

# Fallback custom nodes
workflow.add_node("empty_csv_fallback", empty_csv_fallback)
workflow.add_node("noop_success",       noop_success)

# --- Edges -----------------------------------------------
workflow.add_edge(START, "search_repo")

# search_repo routing
workflow.add_conditional_edges(
    "search_repo",
    make_router("search_repo", "SEARCH_COMPLETE", "scrape_contributors"),
    {
        "scrape_contributors": "scrape_contributors",   # success
        "retry_path":          "search_repo",           # retry
        "fallback_path":       "install_project",       # skip scraping if cannot find repo
        "emergency_end":       END
    }
)

# scrape_contributors routing
workflow.add_conditional_edges(
    "scrape_contributors",
    make_router("scrape_contributors", "SCRAPE_COMPLETE", "write_csv"),
    {
        "write_csv":           "write_csv",
        "retry_path":          "scrape_contributors",
        "fallback_path":       "empty_csv_fallback",     # use placeholder CSV
        "emergency_end":       END
    }
)

# write_csv routing
workflow.add_conditional_edges(
    "write_csv",
    make_router("write_csv", "CSV_COMPLETE", "install_project"),
    {
        "install_project":     "install_project",
        "retry_path":          "write_csv",
        "fallback_path":       "install_project",        # proceed without CSV guarantee
        "emergency_end":       END
    }
)

# empty_csv_fallback -> continue to install
workflow.add_edge("empty_csv_fallback", "install_project")

# install_project routing
workflow.add_conditional_edges(
    "install_project",
    make_router("install_project", "INSTALL_COMPLETE", "verify_install"),
    {
        "verify_install":      "verify_install",
        "retry_path":          "install_project",
        "fallback_path":       "verify_install",         # attempt verification anyway
        "emergency_end":       END
    }
)

# verify_install routing
workflow.add_conditional_edges(
    "verify_install",
    make_router("verify_install", "VERIFY_SUCCESS", "noop_success"),
    {
        "noop_success":        "noop_success",           # graceful end even on failure
        "retry_path":          "verify_install",
        "fallback_path":       "noop_success",
        "emergency_end":       END
    }
)

# Final edge
workflow.add_edge("noop_success", END)

# === 7.  COMPILE WORKFLOW  ===============================
app = workflow.compile()