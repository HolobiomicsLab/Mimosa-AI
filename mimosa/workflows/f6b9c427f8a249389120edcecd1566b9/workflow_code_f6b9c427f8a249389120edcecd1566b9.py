# ==============================================================
# LANGGRAPH – SMOLAGENT WORKFLOW : METABOT PROJECT PIPELINE
# ==============================================================

# MANDATORY IMPORTS
from langgraph.graph import StateGraph, START, END

# ----  TOOL PACKAGES (already available in the runtime) -------
# SHELL_TOOLS      : for cloning / installing the project
# BROWSER_TOOLS    : for every web-search / scraping step
# CSV_TOOLS        : for writing the contributors.csv file
# --------------------------------------------------------------


# ==============================================================
# 1.  AGENT INSTRUCTION PROMPTS  (ONE ATOMIC RESPONSIBILITY EACH)
# ==============================================================

# --- 1) PRIMARY SEARCH AGENT ----------------------------------
instruct_search_primary = """
You are a WEB SEARCH agent.

TASK
1. Search the web for “MetaboT Holobiomics CNRS GitHub”.
2. Find the canonical GitHub repository URL for the MetaboT project.
3. Return ONLY the final GitHub URL. Do not visit the repo yet.

OUTPUT FORMAT
Return one line: FOUND_URL=<url>

UPON COMPLETION
If successful say SEARCH_COMPLETE
If no result say SEARCH_FAILED
If tool crash say GIVE_UP
"""

# --- 2) FALLBACK SEARCH AGENT ---------------------------------
instruct_search_fallback = """
You are a SECOND-PASS WEB SEARCH agent.

TASK
1. Repeat the search for MetaboT Holobiomics CNRS repository.
2. Broaden keywords if needed (e.g. “MetaboT CNRS”, “Holobiomics MetaboT”).
3. Return the best candidate GitHub URL.

OUTPUT FORMAT
Return one line: FOUND_URL=<url>

UPON COMPLETION
If successful say SEARCH_COMPLETE
If still no result say SEARCH_FAILED
If tool crash say GIVE_UP
"""

# --- 3) CONTRIBUTOR SCRAPER – PRIMARY -------------------------
instruct_scrape_primary = """
You are a GITHUB SCRAPER agent.

INPUT
You will receive the GitHub repository URL in the current observation.
Navigate to the repo’s contributors page (or use GitHub API via browser).

TASK
- Collect the visible contributor names (usernames) – minimum 5 expected.
- Return them as a JSON list of strings, e.g. ["alice","bob"].

UPON COMPLETION
If list length >0 say SCRAPE_COMPLETE
If empty list say SCRAPE_FAILED
If tool crash say GIVE_UP
"""

# --- 4) CONTRIBUTOR SCRAPER – FALLBACK ------------------------
instruct_scrape_fallback = """
You are a SECOND-PASS CONTRIBUTOR SCRAPER agent.

STRATEGY
If normal page scrape failed, try:
- Append “/graphs/contributors” to the repo URL OR
- Use raw GitHub API: https://api.github.com/repos/<owner>/<repo>/contributors

Return contributors as JSON list of strings.

UPON COMPLETION
If list length >0 say SCRAPE_COMPLETE
If empty list say SCRAPE_FAILED
If tool crash say GIVE_UP
"""

# --- 5) CSV WRITER AGENT --------------------------------------
instruct_csv_writer = """
You are a CSV WRITER agent.

INPUT
The last observation contains a JSON list of contributor names.

TASK
1. Create / overwrite a local file named contributors.csv
2. First column header: contributor
3. Write one contributor per row.

UPON COMPLETION
If file saved successfully say CSV_COMPLETE
If failure say CSV_FAILED
If unknown error say GIVE_UP
"""

# --- 6) PROJECT INSTALLER – PRIMARY ---------------------------
instruct_install_primary = """
You are a SHELL INSTALLER agent.

INPUT
You have the GitHub URL in the observation.

TASK STEPS
1. git clone <url> metabot_repo
2. cd metabot_repo
3. python -m pip install -e .

UPON COMPLETION
If install finishes with exit status 0 say INSTALL_COMPLETE
If any command fails say INSTALL_FAILED
"""

# --- 7) PROJECT INSTALLER – FALLBACK --------------------------
instruct_install_fallback = """
You are a FALLBACK SHELL INSTALLER agent.

STRATEGY
1. Retry installation with --no-cache-dir and upgrade pip.
   Commands:
   python -m pip install --upgrade pip
   git clone <url> metabot_repo || true
   cd metabot_repo
   python -m pip install --no-cache-dir -e .

UPON COMPLETION
If install finishes with exit status 0 say INSTALL_COMPLETE
If still failing say INSTALL_FAILED
"""


# ==============================================================
# 2.  CREATE SMOLAGENT INSTANCES
# ==============================================================

# NOTE: SmolAgentFactory is available in the environment
search_agent_primary      = SmolAgentFactory(instruct_search_primary,   BROWSER_TOOLS)
search_agent_fallback     = SmolAgentFactory(instruct_search_fallback,  BROWSER_TOOLS)

scrape_agent_primary      = SmolAgentFactory(instruct_scrape_primary,   BROWSER_TOOLS)
scrape_agent_fallback     = SmolAgentFactory(instruct_scrape_fallback,  BROWSER_TOOLS)

csv_writer_agent          = SmolAgentFactory(instruct_csv_writer,       CSV_TOOLS)

install_agent_primary     = SmolAgentFactory(instruct_install_primary,  SHELL_TOOLS)
install_agent_fallback    = SmolAgentFactory(instruct_install_fallback, SHELL_TOOLS)


# ==============================================================
# 3.  WORKFLOW GRAPH INITIALISATION
# ==============================================================

workflow = StateGraph(WorkflowState)

# ---------- NODE REGISTRATION ---------------------------------
workflow.add_node("search_primary",   WorkflowNodeFactory.create_agent_node(search_agent_primary))
workflow.add_node("search_fallback",  WorkflowNodeFactory.create_agent_node(search_agent_fallback))

workflow.add_node("scrape_primary",   WorkflowNodeFactory.create_agent_node(scrape_agent_primary))
workflow.add_node("scrape_fallback",  WorkflowNodeFactory.create_agent_node(scrape_agent_fallback))

# Validator node – pure function (no tools)
def validate_contributors(state: WorkflowState) -> WorkflowState:
    try:
        last_obs = state.get("observations", [])[-1]["data"]
        valid = False
        try:
            import json, ast
            contributors = json.loads(last_obs) if last_obs.strip().startswith("[") else ast.literal_eval(last_obs)
            valid = isinstance(contributors, list) and len(contributors) > 0
        except Exception:
            valid = False
        state["success"].append(valid)
        state["step_name"].append("validate_contributors")
        return state
    except Exception as err:
        state["success"].append(False)
        state["step_name"].append("validate_contributors_error")
        state["observations"].append({"data": f"Validator error: {err}"})
        return state

workflow.add_node("validate_contributors", validate_contributors)

workflow.add_node("csv_writer",       WorkflowNodeFactory.create_agent_node(csv_writer_agent))

workflow.add_node("install_primary",  WorkflowNodeFactory.create_agent_node(install_agent_primary))
workflow.add_node("install_fallback", WorkflowNodeFactory.create_agent_node(install_agent_fallback))


# ==============================================================
# 4.  ROUTING FUNCTIONS WITH ROBUST ERROR HANDLING
# ==============================================================

# ---------- A) ROUTER AFTER SEARCH ----------------------------
def route_after_search(state: WorkflowState) -> str:
    try:
        answers = state.get("answers", [])
        last_answer = answers[-1] if answers else ""
        step = state.get("step_name", [""])[-1]
        retry_count = state.get("step_name", []).count(step)

        if "SEARCH_COMPLETE" in last_answer:
            return "scrape_primary"
        if "GIVE_UP" in last_answer:
            return END
        # SEARCH_FAILED path
        if retry_count < 2 and step == "search_primary":
            return "search_fallback"
        else:
            return END
    except Exception as e:
        state["observations"].append({"data": f"Routing error after search: {e}"})
        return END

# ---------- B) ROUTER AFTER SEARCH_FALLBACK -------------------
def route_after_search_fallback(state: WorkflowState) -> str:
    try:
        answers = state.get("answers", [])
        last_answer = answers[-1] if answers else ""
        if "SEARCH_COMPLETE" in last_answer:
            return "scrape_primary"
        return END  # give up if still not found
    except Exception:
        return END

# ---------- C) ROUTER AFTER SCRAPE ----------------------------
def route_after_scrape(state: WorkflowState) -> str:
    try:
        answers = state.get("answers", [])
        last_answer = answers[-1] if answers else ""
        step = state.get("step_name", [""])[-1]
        retry_count = state.get("step_name", []).count(step)

        if "SCRAPE_COMPLETE" in last_answer:
            return "validate_contributors"
        if "GIVE_UP" in last_answer:
            return END
        # SCRAPE_FAILED path
        if retry_count < 2 and step == "scrape_primary":
            return "scrape_fallback"
        else:
            return END
    except Exception:
        return END

# ---------- D) ROUTER AFTER SCRAPE_FALLBACK -------------------
def route_after_scrape_fallback(state: WorkflowState) -> str:
    try:
        answers = state.get("answers", [])
        if "SCRAPE_COMPLETE" in (answers[-1] if answers else ""):
            return "validate_contributors"
        return END
    except Exception:
        return END

# ---------- E) ROUTER AFTER VALIDATION ------------------------
def route_after_validation(state: WorkflowState) -> str:
    try:
        if state.get("success", [])[-1]:
            return "csv_writer"
        # validation failed – retry scraping fallback directly
        return "scrape_fallback"
    except Exception:
        return END

# ---------- F) ROUTER AFTER CSV WRITER ------------------------
def route_after_csv(state: WorkflowState) -> str:
    try:
        answers = state.get("answers", [])
        if "CSV_COMPLETE" in (answers[-1] if answers else ""):
            return "install_primary"
        # If CSV writing failed once – try again
        if "CSV_FAILED" in (answers[-1] if answers else ""):
            retry = state.get("step_name", []).count("csv_writer")
            if retry < 2:
                return "csv_writer"
        return END
    except Exception:
        return END

# ---------- G) ROUTER AFTER INSTALL ---------------------------
def route_after_install(state: WorkflowState) -> str:
    try:
        answers = state.get("answers", [])
        last = answers[-1] if answers else ""
        step = state.get("step_name", [""])[-1]
        retry_count = state.get("step_name", []).count(step)

        if "INSTALL_COMPLETE" in last:
            return END
        if retry_count < 2 and step == "install_primary":
            return "install_fallback"
        return END
    except Exception:
        return END

# ---------- H) ROUTER AFTER INSTALL_FALLBACK ------------------
def route_after_install_fallback(state: WorkflowState) -> str:
    try:
        answers = state.get("answers", [])
        if "INSTALL_COMPLETE" in (answers[-1] if answers else ""):
            return END
        return END
    except Exception:
        return END


# ==============================================================
# 5.  EDGE DEFINITIONS
# ==============================================================

workflow.add_edge(START, "search_primary")

workflow.add_conditional_edges(
    "search_primary",
    route_after_search,
    {
        "scrape_primary": "scrape_primary",
        "search_fallback": "search_fallback",
        END: END
    }
)

workflow.add_conditional_edges(
    "search_fallback",
    route_after_search_fallback,
    {
        "scrape_primary": "scrape_primary",
        END: END
    }
)

workflow.add_conditional_edges(
    "scrape_primary",
    route_after_scrape,
    {
        "validate_contributors": "validate_contributors",
        "scrape_fallback": "scrape_fallback",
        END: END
    }
)

workflow.add_conditional_edges(
    "scrape_fallback",
    route_after_scrape_fallback,
    {
        "validate_contributors": "validate_contributors",
        END: END
    }
)

workflow.add_conditional_edges(
    "validate_contributors",
    route_after_validation,
    {
        "csv_writer": "csv_writer",
        "scrape_fallback": "scrape_fallback",
        END: END
    }
)

workflow.add_conditional_edges(
    "csv_writer",
    route_after_csv,
    {
        "install_primary": "install_primary",
        "csv_writer": "csv_writer",   # retry same node
        END: END
    }
)

workflow.add_conditional_edges(
    "install_primary",
    route_after_install,
    {
        END: END,
        "install_fallback": "install_fallback"
    }
)

workflow.add_conditional_edges(
    "install_fallback",
    route_after_install_fallback,
    {
        END: END
    }
)


# ==============================================================
# 6.  COMPILE THE WORKFLOW
# ==============================================================

app = workflow.compile()