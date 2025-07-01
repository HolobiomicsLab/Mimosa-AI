# -----------------------------  LANGGRAPH – SMOLAGENT WORKFLOW  -----------------------------
# PURPOSE:  
# 1. Deep-search for “mzmind” batch/CLI usage instructions  
# 2. Gather & validate installation steps for macOS  
# 3. Execute installation through shell commands  
# 4. Produce human-readable summary  
# -------------------------------------------------------------------------------------------

from langgraph.graph import StateGraph, START, END

# -------------------------------------------------------------------------------------------
# 1)  AGENT PROMPTS  (all single-purpose, termination keywords in ALL-CAPS)
# -------------------------------------------------------------------------------------------

# --- Web Search Agent -----------------------------------------------------------
search_prompt = """
You are a focused WEB SEARCH agent.

OBJECTIVE  
- Find ALL authoritative resources (docs, blog posts, forums, GitHub, etc.) explaining:  
  a) How to run "mzmind" in batch / command-line mode  
  b) How to install "mzmind" on macOS (Intel & Apple Silicon)

OUTPUT PROTOCOL  
Use the browser tool to visit, click and read pages.  
When finished:

SUCCESS → reply with:  
SEARCH_COMPLETE:
<bulleted list of URLs + 1-sentence description each>

FAILURE (nothing useful) → reply with:  
SEARCH_FAILURE:
<what you tried, why not useful, ideas for next search>

ERROR (tool crash) → reply with:  
GIVE_UP:
<error details>
"""
# --- Extraction Agent -----------------------------------------------------------
extract_prompt = """
You are a DATA EXTRACTION agent.

INPUT  
- You receive raw URLs & short notes provided by previous agent.

TASK  
- Visit each URL, read content and EXTRACT two separate artifacts:  
  1. "CLI_USAGE" → every concrete example of running mzmind via batch or command-line  
  2. "MAC_INSTALL" → step-by-step macOS installation instructions with prerequisites  
  
- Provide clean Markdown blocks for both artifacts.

OUTPUT PROTOCOL
If both artifacts are complete & detailed:  
EXTRACT_COMPLETE:
CLI_USAGE:
<markdown block>
MAC_INSTALL:
<markdown block>

If information missing/inadequate:  
INSUFFICIENT_INFORMATION:
<what is missing – be explicit>

On unexpected errors:  
EXTRACT_FAILURE:
<error explanation>
"""
# --- Validation Agent -----------------------------------------------------------
validate_prompt = """
You are a VALIDATION agent.

TASK  
- Review CLI_USAGE and MAC_INSTALL sections from previous answer.  
- Check for completeness (dependencies, path variables, version numbers, Apple Silicon notes).

OUTPUT PROTOCOL
If info is complete & accurate:  
VALIDATE_PASSED

If missing pieces detected:  
VALIDATE_INSUFFICIENT:
<list missing items>

Critical error (formatting, corruption):  
VALIDATE_FAILURE:
<error details>
"""
# --- Install Agent --------------------------------------------------------------
install_prompt = """
You are an INSTALLATION EXECUTOR agent for macOS.

TASK  
- Use the shell tool to perform the installation steps exactly as provided.  
- Capture command outputs via observations.  
- Only run SAFE commands (brew install, git clone, etc.).  
- NO sudo rm ‑rf or destructive actions.

OUTPUT PROTOCOL  
If installation completed without errors & `mzmind --help` works:  
INSTALL_SUCCESS:
<last 30 lines of `mzmind --help`>

If some steps failed but recoverable:  
INSTALL_PARTIAL:
<which commands failed, stderr, suggestions>

If completely failed / dangerous:  
INSTALL_FAILURE:
<error logs, advice>
"""
# --- Summary Agent --------------------------------------------------------------
summary_prompt = """
You are a FINAL REPORT agent.

TASK  
- Combine validated instructions and actual shell output.  
- Produce concise, reader-friendly guide for using mzmind in batch mode on macOS.

OUTPUT PROTOCOL  
Always finish with the exact tag: WORKFLOW_DONE
"""

# -------------------------------------------------------------------------------------------
# 2)  AGENT DECLARATION (SmolAgentFactory is pre-loaded)
# -------------------------------------------------------------------------------------------

search_agent    = SmolAgentFactory(search_prompt,   BROWSER_TOOLS)
extract_agent   = SmolAgentFactory(extract_prompt,  BROWSER_TOOLS)
validate_agent  = SmolAgentFactory(validate_prompt, CSV_TOOLS)
install_agent   = SmolAgentFactory(install_prompt,  SHELL_TOOLS)
summary_agent   = SmolAgentFactory(summary_prompt,  CSV_TOOLS)

# -------------------------------------------------------------------------------------------
# 3)  WORKFLOW  GRAPH
# -------------------------------------------------------------------------------------------

workflow = StateGraph(WorkflowState)

# ---- 3.1  Nodes ---------------------------------------------------------------------------
workflow.add_node("search_web",   WorkflowNodeFactory.create_agent_node(search_agent))
workflow.add_node("extract_info", WorkflowNodeFactory.create_agent_node(extract_agent))
workflow.add_node("validate",     WorkflowNodeFactory.create_agent_node(validate_agent))
workflow.add_node("install",      WorkflowNodeFactory.create_agent_node(install_agent))
workflow.add_node("summarize",    WorkflowNodeFactory.create_agent_node(summary_agent))

# ---- 3.2  Helper: attempt counter ---------------------------------------------------------
def _count_attempts(state: WorkflowState, step: str) -> int:
    try:
        return [s for s in state.get("step_name", []) if step in s].__len__()
    except Exception:
        return 0

# ---- 3.3  Routers with robust fallbacks ---------------------------------------------------

# AFTER SEARCH -----------------------------------------------------------------
def router_after_search(state: WorkflowState) -> str:
    try:
        answer = (state.get("answers") or [""])[-1]
        attempts = _count_attempts(state, "search_web")
        
        if "SEARCH_COMPLETE" in answer:
            return "extract_info"
        elif "GIVE_UP" in answer:
            return "summarize"          # emergency end
        else:  # FAILURE or unexpected
            if attempts < 3:
                return "search_web"     # retry
            else:
                return "summarize"      # give up after 3 tries
    except Exception as e:
        print(f"Router search error: {e}")
        return "summarize"

# AFTER EXTRACTION -------------------------------------------------------------
def router_after_extract(state: WorkflowState) -> str:
    try:
        answer = (state.get("answers") or [""])[-1]
        attempts = _count_attempts(state, "extract_info")
        
        if "EXTRACT_COMPLETE" in answer:
            return "validate"
        elif "INSUFFICIENT_INFORMATION" in answer and attempts < 2:
            return "search_web"         # go get more sources
        elif "EXTRACT_FAILURE" in answer:
            return "search_web" if attempts < 2 else "summarize"
        else:
            # unknown format
            return "search_web"
    except Exception as e:
        print(f"Router extract error: {e}")
        return "summarize"

# AFTER VALIDATION -------------------------------------------------------------
def router_after_validation(state: WorkflowState) -> str:
    try:
        answer = (state.get("answers") or [""])[-1]
        if "VALIDATE_PASSED" in answer:
            return "install"
        elif "VALIDATE_INSUFFICIENT" in answer:
            return "search_web"   # missing info → back to web
        else:  # failure or corruption
            return "summarize"
    except Exception as e:
        print(f"Router validate error: {e}")
        return "summarize"

# AFTER INSTALL ---------------------------------------------------------------
def router_after_install(state: WorkflowState) -> str:
    try:
        answer = (state.get("answers") or [""])[-1]
        attempts = _count_attempts(state, "install")
        
        if "INSTALL_SUCCESS" in answer:
            return "summarize"
        elif "INSTALL_PARTIAL" in answer and attempts < 2:
            return "install"      # retry once with suggestions
        else:
            return "summarize"    # failure branch
    except Exception as e:
        print(f"Router install error: {e}")
        return "summarize"

# ---- 3.4  Edges ---------------------------------------------------------------------------
workflow.add_edge(START, "search_web")

workflow.add_conditional_edges(
    "search_web",
    router_after_search,
    {
        "extract_info": "extract_info",
        "search_web":   "search_web",
        "summarize":    "summarize"
    }
)

workflow.add_conditional_edges(
    "extract_info",
    router_after_extract,
    {
        "validate":     "validate",
        "search_web":   "search_web",
        "summarize":    "summarize"
    }
)

workflow.add_conditional_edges(
    "validate",
    router_after_validation,
    {
        "install":      "install",
        "search_web":   "search_web",
        "summarize":    "summarize"
    }
)

workflow.add_conditional_edges(
    "install",
    router_after_install,
    {
        "install":      "install",
        "summarize":    "summarize"
    }
)

workflow.add_edge("summarize", END)

# -------------------------------------------------------------------------------------------
# 4)  COMPILE APP
# -------------------------------------------------------------------------------------------
app = workflow.compile()
# -------------------------------------------------------------------------------------------
#  READY TO RUN  –  `app.invoke(initial_state)`  with a WorkflowState-shaped dict
# -------------------------------------------------------------------------------------------