# =========================  MANDATORY IMPORTS  =========================
from langgraph.graph import StateGraph, START, END
import re

# ======================  WORKFLOW STATE ALREADY DEFINED  ===============
# (Action, Observation & WorkflowState classes are assumed pre-defined)

# =========================  AGENT INSTRUCTIONS  ========================

search_instr = """
You are a focused web-search agent.

GOAL
- Use your browser tools to find the most reliable, up-to-date instructions
  for installing the GPAW software on a macOS system with an Apple M1 chip.

OUTPUT RULES
- Summarize the best installation resource URL(s) and the key steps discovered.
- End with either:
  • SEARCH_SUCCESS  – if you are confident you found valid instructions
  • SEARCH_FAILURE  – if you could not find suitable information
"""

alt_search_instr = """
You are an alternative search agent called in after repeated failures.

GOAL
- Perform deeper web searches (forums, GitHub issues, Homebrew docs, etc.)
  to locate ANY viable installation procedure for GPAW on Apple-silicon macOS.

OUTPUT RULES
- Provide at least one working procedure or resource link.
- Finish with SEARCH_SUCCESS or SEARCH_FAILURE.
"""

extract_instr = """
You are an extraction agent.

INPUT
- You receive the web summary from previous search steps in the most recent
  observation.

TASK
- Parse and list, in correct sequence, all terminal commands required to
  install GPAW on an Apple-silicon Mac (M1).

OUTPUT RULES
- Return ONLY the command list, one per line, ready to be executed by shell.
- End with:
  • EXTRACT_SUCCESS  – commands extracted and look valid
  • EXTRACT_FAILURE  – unable to extract useful commands
"""

validate_instr = """
You are an environment-validation agent.

TASK
- Run shell checks to confirm that the current machine:
   1. Is running macOS on arm64 (Apple M-series)
   2. Has Homebrew installed
   3. Has Python ≥3.9 available
- Report any missing prerequisites.

OUTPUT RULES
- If all checks pass, end with VALIDATION_SUCCESS.
- Otherwise, clearly list missing items and end with VALIDATION_FAILURE.
"""

install_instr = """
You are an installation agent.

INPUT
- You receive a list of shell commands (from extraction agent) in the last
  observation. Execute them sequentially.

TASK
- Run each command with shell tools.
- Capture and report any errors.

OUTPUT RULES
- When GPAW appears successfully installed (importable in Python), finish with
  INSTALL_SUCCESS.
- If installation fails, finish with INSTALL_FAILURE.
"""

brew_install_instr = """
You are a last-resort installer that attempts a Homebrew-only approach.

TASK
- Try:  brew install gpaw  (and any dependencies: libxc, fftw, hdf5, etc.)
- Verify installation by launching python -c "import gpaw; print(gpaw.__version__)"

OUTPUT RULES
- End with INSTALL_SUCCESS if import succeeds, otherwise INSTALL_FAILURE.
"""

# =========================  AGENT CREATION  ============================

# Tool bundles (lists are provided in program scope)
AGENT_TOOLS_SEARCH = BROWSER_TOOLS
AGENT_TOOLS_ALTSEARCH = BROWSER_TOOLS
AGENT_TOOLS_EXTRACT = BROWSER_TOOLS          # parsing of page text
AGENT_TOOLS_VALIDATE = SHELL_TOOLS
AGENT_TOOLS_INSTALL = SHELL_TOOLS
AGENT_TOOLS_BREWINSTALL = SHELL_TOOLS

smol_search         = SmolAgentFactory(search_instr,        AGENT_TOOLS_SEARCH)
smol_alt_search     = SmolAgentFactory(alt_search_instr,    AGENT_TOOLS_ALTSEARCH)
smol_extract        = SmolAgentFactory(extract_instr,       AGENT_TOOLS_EXTRACT)
smol_validate       = SmolAgentFactory(validate_instr,      AGENT_TOOLS_VALIDATE)
smol_install        = SmolAgentFactory(install_instr,       AGENT_TOOLS_INSTALL)
smol_brew_install   = SmolAgentFactory(brew_install_instr,  AGENT_TOOLS_BREWINSTALL)

# =========================  WORKFLOW CREATION  =========================

workflow = StateGraph(WorkflowState)

# -----------------------  ADD AGENT NODES  -----------------------------
workflow.add_node("search_agent",       WorkflowNodeFactory.create_agent_node(smol_search))
workflow.add_node("alt_search_agent",   WorkflowNodeFactory.create_agent_node(smol_alt_search))
workflow.add_node("extract_agent",      WorkflowNodeFactory.create_agent_node(smol_extract))
workflow.add_node("validate_agent",     WorkflowNodeFactory.create_agent_node(smol_validate))
workflow.add_node("install_agent",      WorkflowNodeFactory.create_agent_node(smol_install))
workflow.add_node("brew_install_agent", WorkflowNodeFactory.create_agent_node(smol_brew_install))

# ===================  ROUTING / ERROR-HANDLING NODES  ==================

def search_router(state: WorkflowState) -> str:
    print("🔀 [search_router] deciding next step")
    try:
        last_ans   = state.get("answers", [])[-1] if state.get("answers") else ""
        step_names = state.get("step_name", [])
        retries    = step_names.count("search_agent")
        if "SEARCH_SUCCESS" in last_ans:
            return "extract_agent"
        if retries < 3:
            print(f"🔄 search retry {retries+1}/3")
            state["step_name"].append("search_agent_retry")
            return "search_agent"
        print("⚠️ switching to alt_search_agent")
        return "alt_search_agent"
    except Exception as e:
        print(f"🚨 search_router error: {e}")
        return "alt_search_agent"

def alt_search_router(state: WorkflowState) -> str:
    print("🔀 [alt_search_router]")
    try:
        last_ans = state.get("answers", [])[-1] if state.get("answers") else ""
        if "SEARCH_SUCCESS" in last_ans:
            return "extract_agent"
        return END
    except Exception as e:
        print(f"🚨 alt_search_router error: {e}")
        return END

def extract_router(state: WorkflowState) -> str:
    print("🔀 [extract_router]")
    try:
        last_ans   = state.get("answers", [])[-1] if state.get("answers") else ""
        step_names = state.get("step_name", [])
        retries    = step_names.count("extract_agent")
        if "EXTRACT_SUCCESS" in last_ans:
            return "validate_agent"
        if retries < 3:
            state["step_name"].append("extract_agent_retry")
            return "extract_agent"
        # fallback to alt_search to gather new info
        return "alt_search_agent"
    except Exception as e:
        print(f"🚨 extract_router error: {e}")
        return "alt_search_agent"

def validate_router(state: WorkflowState) -> str:
    print("🔀 [validate_router]")
    try:
        last_ans   = state.get("answers", [])[-1] if state.get("answers") else ""
        step_names = state.get("step_name", [])
        retries    = step_names.count("validate_agent")
        if "VALIDATION_SUCCESS" in last_ans:
            return "install_agent"
        if retries < 2:
            state["step_name"].append("validate_agent_retry")
            return "validate_agent"
        # graceful degradation: proceed to install anyway
        print("⚠️ Skipping validation – proceeding to install")
        return "install_agent"
    except Exception as e:
        print(f"🚨 validate_router error: {e}")
        return "install_agent"

def install_router(state: WorkflowState) -> str:
    print("🔀 [install_router]")
    try:
        last_ans   = state.get("answers", [])[-1] if state.get("answers") else ""
        step_names = state.get("step_name", [])
        retries    = step_names.count("install_agent")
        if "INSTALL_SUCCESS" in last_ans:
            return END
        if retries < 2:
            state["step_name"].append("install_agent_retry")
            return "install_agent"
        print("⚠️ Switching to brew_install_agent fallback")
        return "brew_install_agent"
    except Exception as e:
        print(f"🚨 install_router error: {e}")
        return "brew_install_agent"

def brew_install_router(state: WorkflowState) -> str:
    print("🔀 [brew_install_router]")
    try:
        last_ans = state.get("answers", [])[-1] if state.get("answers") else ""
        if "INSTALL_SUCCESS" in last_ans:
            return END
        return END  # final fail
    except Exception as e:
        print(f"🚨 brew_install_router error: {e}")
        return END

# -----------------------  EDGE DEFINITIONS  ----------------------------

workflow.add_edge(START, "search_agent")

workflow.add_conditional_edges(
    "search_agent",
    search_router,
    {
        "extract_agent":      "extract_agent",
        "search_agent":       "search_agent",
        "alt_search_agent":   "alt_search_agent",
        END:                  END
    }
)

workflow.add_conditional_edges(
    "alt_search_agent",
    alt_search_router,
    {
        "extract_agent": "extract_agent",
        END:             END
    }
)

workflow.add_conditional_edges(
    "extract_agent",
    extract_router,
    {
        "validate_agent":    "validate_agent",
        "extract_agent":     "extract_agent",
        "alt_search_agent":  "alt_search_agent",
        END:                 END
    }
)

workflow.add_conditional_edges(
    "validate_agent",
    validate_router,
    {
        "install_agent": "install_agent",
        "validate_agent": "validate_agent",
        END: END
    }
)

workflow.add_conditional_edges(
    "install_agent",
    install_router,
    {
        END: END,
        "install_agent": "install_agent",
        "brew_install_agent": "brew_install_agent"
    }
)

workflow.add_conditional_edges(
    "brew_install_agent",
    brew_install_router,
    {
        END: END
    }
)

# --------------------------  COMPILE APP  ------------------------------
app = workflow.compile()