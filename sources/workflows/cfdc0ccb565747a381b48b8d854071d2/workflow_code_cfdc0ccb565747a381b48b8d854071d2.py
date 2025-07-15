###############################################################################
# LangGraph – SmolAgent MULTI-AGENT WORKFLOW
# Task :  Locate the paper
#         “Measurement and prediction of small molecule retention by
#          Gram-negative bacteria based on a large-scale LC/MS screen”
#         -> extract every software mentioned
#         -> catalogue them
#         -> download & install them
#
# REQUIREMENTS FULFILLED
# • 6 atomic agents (search ⇒ extract ⇒ catalogue ⇒ download-plan ⇒ installer)
# • Robust routing with retry (max-3), “INSUFFICIENT_INFORMATION” step-back,
#   fatal fallbacks and END guarantees
# • Extensive logging / exception handling inside every router
###############################################################################

# --------------------------------------------------------------------------- #
# PRE-DECLARED OBJECTS (already available in the execution context – DO NOT
# re-import / re-declare):
#   - StateGraph, START, END
#   - WorkflowState schema
#   - SmolAgentFactory
#   - WorkflowNodeFactory
#   - Tool packages:
#       WEB_BROWSER_MCP_TOOLS
#       CSV_MANAGEMENT_TOOLS
#       BASH_COMMAND_MCP_TOOLS
# --------------------------------------------------------------------------- #

from langgraph.graph import StateGraph, START, END   # noqa: F401  (already in ctx)

# ----------------------------- INITIALISE GRAPH ----------------------------- #
workflow = StateGraph(WorkflowState)

# ------------------------------ AGENT PROMPTS ------------------------------- #
# 1) Paper-finding agent ----------------------------------------------------- #
prompt_find_paper = """
You are an expert scientific web-research agent.

GOAL
    Find the official web page (publisher or archive) for the research paper
    titled exactly:
        "Measurement and prediction of small molecule retention by
        Gram-negative bacteria based on a large-scale LC/MS screen"

TOOLS
    • WEB_BROWSER_MCP_TOOLS – use search, navigate, validate links, etc.

COMPLETION PROTOCOL
    SUCCESS  – Once you have a VALID url to the paper’s landing page (PDF or
               HTML), respond with:
               FOUND_PAPER: [paste the canonical URL here]
    FAILURE  – If you cannot find it after exhaustive search, respond with:
               SEARCH_FAILURE: [explain in detail]
    ERROR    – On technical issues:
               GIVE_UP: [error log]
Only output one of the above tags at the end of your answer.
"""

# 2) Software-extraction agent ---------------------------------------------- #
prompt_extract_software = """
You are a detail-oriented extraction agent.

INPUT
    • You will receive the URL of the paper identified by the previous step.

TASK
    • Open the paper (PDF/HTML) and read it.
    • Extract EVERY software package, library, script, or computational tool
      mentioned in methods, supplementary or elsewhere.
    • Collect, for each software:
        - Name
        - Purpose (1-sentence)
        - Version (if stated)
        - Any homepage / repository link you can locate
    • Provide the list in JSON-like bulleted form.

COMPLETION PROTOCOL
    SUCCESS  – When list is complete:
               EXTRACT_COMPLETE:
               [bullet list of dicts -> {name, purpose, version, url}]
    INSUFFICIENT_INFORMATION – if the paper link was invalid or missing:
               INSUFFICIENT_INFORMATION: need valid paper link.
    FAILURE  – if extraction impossible for other reasons:
               EXTRACT_FAILURE: [explain]
"""

# 3) Catalogue agent --------------------------------------------------------- #
prompt_catalog = """
You are a data-entry agent responsible for cataloguing software lists into CSV.

INPUT
    • A bullet list of software dicts from previous step.

TOOLS
    • CSV_MANAGEMENT_TOOLS

TASK
    • Create/overwrite a CSV dataset named "paper_software_catalog".
    • Columns: name, purpose, version, url
    • Insert one row per software item.

COMPLETION PROTOCOL
    SUCCESS  – After data saved:
               CATALOG_DONE: paper_software_catalog
    INSUFFICIENT_INFORMATION – if list missing or malformed:
               INSUFFICIENT_INFORMATION: extraction incomplete.
    FAILURE  – on other problems:
               CATALOG_FAILURE: [details]
"""

# 4) Download-planning agent ------------------------------------------------- #
prompt_download_plan = """
You are a download-planning agent.

INPUT
    • The software catalogue CSV file path.

TOOLS
    • WEB_BROWSER_MCP_TOOLS  – to visit official pages and capture direct
      download/clone URLs.

TASK
    • For every software entry in the CSV, locate an official download link
      (git clone URL, tarball, pip/R package string, etc.).
    • Produce a final list mapping software -> download_command
      (e.g. wget <url>, git clone <url>, pip install <pkg>, etc.).

COMPLETION PROTOCOL
    SUCCESS  – When every item has a download command:
               DOWNLOADS_READY:
               [multi-line list: software | command]
    INSUFFICIENT_INFORMATION – if catalogue missing rows:
               INSUFFICIENT_INFORMATION: need updated csv.
    FAILURE  – on other obstacles:
               DOWNLOAD_FAILURE: [details]
"""

# 5) Installer agent --------------------------------------------------------- #
prompt_installer = """
You are an autonomous installation agent.

INPUT
    • A list of shell commands to download and install each software.

TOOLS
    • BASH_COMMAND_MCP_TOOLS

TASK
    • Execute each command sequentially.
    • Capture output / error for each step.

COMPLETION PROTOCOL
    SUCCESS  – If every command exited with code 0:
               INSTALL_SUCCESS: [summarise logs]
    INSUFFICIENT_INFORMATION – if command list incomplete:
               INSUFFICIENT_INFORMATION: need download commands.
    FAILURE  – if installation fails (non-zero code) even after reasonable
               retries:
               INSTALL_FAILURE: [error log]
"""

# ----------------------------- AGENT CREATION ------------------------------ #
agent_find_paper      = SmolAgentFactory("paper_finder",       prompt_find_paper,      WEB_BROWSER_MCP_TOOLS)
agent_extract_software= SmolAgentFactory("software_extractor", prompt_extract_software,WEB_BROWSER_MCP_TOOLS)
agent_catalog         = SmolAgentFactory("software_cataloger", prompt_catalog,         CSV_MANAGEMENT_TOOLS)
agent_download_plan   = SmolAgentFactory("download_planner",   prompt_download_plan,   WEB_BROWSER_MCP_TOOLS)
agent_installer       = SmolAgentFactory("installer",          prompt_installer,       BASH_COMMAND_MCP_TOOLS)

# ----------------------- ADD NODES TO WORKFLOW GRAPH ----------------------- #
workflow.add_node("paper_finder",       WorkflowNodeFactory.create_agent_node(agent_find_paper))
workflow.add_node("software_extractor", WorkflowNodeFactory.create_agent_node(agent_extract_software))
workflow.add_node("software_cataloger", WorkflowNodeFactory.create_agent_node(agent_catalog))
workflow.add_node("download_planner",   WorkflowNodeFactory.create_agent_node(agent_download_plan))
workflow.add_node("installer",          WorkflowNodeFactory.create_agent_node(agent_installer))

# -------------------------- ROUTING FUNCTIONS ------------------------------ #
MAX_RETRIES = 3

def _count_attempts(state: WorkflowState, node_name: str) -> int:
    """Helper – count how many times node_name appears in step_name list."""
    return [n for n in state.get("step_name", []) if n.startswith(node_name)].__len__()

# --- Router 1 : paper_finder ---------------------------------------------- #
def route_after_finder(state: WorkflowState) -> str:
    try:
        last_answer = state["answers"][-1] if state.get("answers") else ""
        attempts    = _count_attempts(state, "paper_finder")
        if "FOUND_PAPER" in last_answer:
            return "software_extractor"
        elif "SEARCH_FAILURE" in last_answer or "GIVE_UP" in last_answer:
            if attempts >= MAX_RETRIES:
                print("Paper Finder reached max retries – aborting.")
                return END
            print("Retrying paper search …")
            return "paper_finder"
        else:
            # Unexpected answer → retry
            if attempts >= MAX_RETRIES:
                return END
            return "paper_finder"
    except Exception as e:
        print(f"Router-finder error: {e}")
        return END

# --- Router 2 : software_extractor ---------------------------------------- #
def route_after_extractor(state: WorkflowState) -> str:
    try:
        last_answer = state["answers"][-1] if state.get("answers") else ""
        attempts    = _count_attempts(state, "software_extractor")
        if "EXTRACT_COMPLETE" in last_answer:
            return "software_cataloger"
        elif "INSUFFICIENT_INFORMATION" in last_answer:
            return "paper_finder"   # go one step back
        elif "EXTRACT_FAILURE" in last_answer:
            if attempts >= MAX_RETRIES:
                return END
            return "software_extractor"
        else:
            # Unknown output – retry
            if attempts >= MAX_RETRIES:
                return END
            return "software_extractor"
    except Exception as e:
        print(f"Router-extractor error: {e}")
        return END

# --- Router 3 : software_cataloger ---------------------------------------- #
def route_after_cataloger(state: WorkflowState) -> str:
    try:
        last_answer = state["answers"][-1] if state.get("answers") else ""
        attempts    = _count_attempts(state, "software_cataloger")
        if "CATALOG_DONE" in last_answer:
            return "download_planner"
        elif "INSUFFICIENT_INFORMATION" in last_answer:
            return "software_extractor"  # step back
        elif "CATALOG_FAILURE" in last_answer:
            if attempts >= MAX_RETRIES:
                return END
            return "software_cataloger"
        else:
            if attempts >= MAX_RETRIES:
                return END
            return "software_cataloger"
    except Exception as e:
        print(f"Router-cataloger error: {e}")
        return END

# --- Router 4 : download_planner ------------------------------------------ #
def route_after_download_planner(state: WorkflowState) -> str:
    try:
        last_answer = state["answers"][-1] if state.get("answers") else ""
        attempts    = _count_attempts(state, "download_planner")
        if "DOWNLOADS_READY" in last_answer:
            return "installer"
        elif "INSUFFICIENT_INFORMATION" in last_answer:
            return "software_cataloger"
        elif "DOWNLOAD_FAILURE" in last_answer:
            if attempts >= MAX_RETRIES:
                return END
            return "download_planner"
        else:
            if attempts >= MAX_RETRIES:
                return END
            return "download_planner"
    except Exception as e:
        print(f"Router-download planner error: {e}")
        return END

# --- Router 5 : installer -------------------------------------------------- #
def route_after_installer(state: WorkflowState) -> str:
    try:
        last_answer = state["answers"][-1] if state.get("answers") else ""
        attempts    = _count_attempts(state, "installer")
        if "INSTALL_SUCCESS" in last_answer:
            return END
        elif "INSUFFICIENT_INFORMATION" in last_answer:
            return "download_planner"
        elif "INSTALL_FAILURE" in last_answer:
            if attempts >= MAX_RETRIES:
                return END
            return "installer"
        else:
            if attempts >= MAX_RETRIES:
                return END
            return "installer"
    except Exception as e:
        print(f"Router-installer error: {e}")
        return END

# --------------------------- CONNECT GRAPH --------------------------------- #
workflow.add_edge(START, "paper_finder")

workflow.add_conditional_edges(
    "paper_finder",
    route_after_finder,
    {
        "software_extractor": "software_extractor",
        "paper_finder":       "paper_finder",
        END:                  END
    }
)

workflow.add_conditional_edges(
    "software_extractor",
    route_after_extractor,
    {
        "software_cataloger": "software_cataloger",
        "software_extractor": "software_extractor",
        "paper_finder":       "paper_finder",
        END:                  END
    }
)

workflow.add_conditional_edges(
    "software_cataloger",
    route_after_cataloger,
    {
        "download_planner":   "download_planner",
        "software_cataloger": "software_cataloger",
        "software_extractor": "software_extractor",
        END:                  END
    }
)

workflow.add_conditional_edges(
    "download_planner",
    route_after_download_planner,
    {
        "installer":          "installer",
        "download_planner":   "download_planner",
        "software_cataloger": "software_cataloger",
        END:                  END
    }
)

workflow.add_conditional_edges(
    "installer",
    route_after_installer,
    {
        END:                  END,
        "installer":          "installer",
        "download_planner":   "download_planner"
    }
)

# ------------------------------ COMPILE APP -------------------------------- #
app = workflow.compile()

# ---------------------------- WORKFLOW READY ------------------------------- #
# You can now run:
#   initial_state = WorkflowState(
#       step_name=[], actions=[], observations=[], answers=[], success=[])
#   result = app.invoke(initial_state)
###############################################################################