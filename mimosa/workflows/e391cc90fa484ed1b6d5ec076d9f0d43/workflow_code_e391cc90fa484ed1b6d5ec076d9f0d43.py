# ================================================================
#  LANGGRAPH × SMOLAGENT WORKFLOW : GPAW INSTALLATION FOR MAC-ARM
# ================================================================
# This code is runnable as-is (all factories/utilities presumed pre-defined
# in the execution environment as described in the system instructions).
# ------------------------------------------------

from langgraph.graph import StateGraph, START, END

# ============= 1. AGENT INSTRUCTIONS (VERY SPECIFIC) =============

instruct_search = """
You are SEARCH_AGENT.

OBJECTIVE
- Use browser tools to look for official or community instructions to
  install the GPAW quantum-chemistry package on macOS running Apple Silicon.
- Compile a concise list of promising URLs with one-line descriptions.

SUCCESS CONDITIONS
- If you found ≥1 URL clearly describing Apple-Silicon installation steps
  say exactly: SEARCH_COMPLETE
- Otherwise say: SEARCH_FAILURE
- On unexpected tool error say: GIVE_UP
"""

instruct_alt_search = """
You are ALT_SEARCH_AGENT.

OBJECTIVE
- Perform an alternative broader search (include "conda", "brew", "pip",
  "arch=arm64", "mac m1", "mac m2", etc.) for GPAW installation info.
- Provide URL list just like SEARCH_AGENT.

SUCCESS CONDITIONS
- If suitable URLs discovered say: ALT_SEARCH_COMPLETE
- Otherwise say: ALT_SEARCH_FAILURE
- On unexpected error say: GIVE_UP
"""

instruct_extract = """
You are EXTRACT_AGENT.

OBJECTIVE
- Open the URLs provided by previous agent (they are in observations).
- Extract concrete shell commands or guidelines needed to install GPAW
  on macOS Apple-Silicon (e.g. brew install, conda install, pip wheels, etc.).
- Output clean command snippets only (no commentary).

SUCCESS CONDITIONS
- If ≥1 command extracted say: EXTRACTION_COMPLETE
- Otherwise say: EXTRACTION_FAILURE
- On unknown error say: GIVE_UP
"""

instruct_validate = """
You are VALIDATE_AGENT.

OBJECTIVE
- Check each installation command (from observations) for feasibility:
  • Syntax looks correct
  • Package/channel exists (use shell commands like `brew info gpaw`,
    `conda search gpaw`, or `pip index versions gpaw`).
- Mark each command as VALID or INVALID.

SUCCESS CONDITIONS
- If at least one command marked VALID say: VALIDATION_PASS
- Else say: VALIDATION_FAIL
- If shell tools raise unexpected error say: GIVE_UP
"""

instruct_install = """
You are INSTALL_AGENT.

OBJECTIVE
- Execute the validated installation command(s) in shell.
- Confirm GPAW is available afterwards with a check like `python -c "import gpaw, sys; print(gpaw.__version__)"`.

SUCCESS CONDITIONS
- If installation succeeds and check prints a version, say: INSTALL_SUCCESS
- If command fails, retry with alternatives then say: INSTALL_FAILURE
- On unrecoverable tool error say: GIVE_UP
"""

instruct_manual = """
You are MANUAL_FALLBACK_AGENT.

OBJECTIVE
- Provide the user with a clear, step-by-step manual guide to install GPAW
  on macOS Apple-Silicon including prerequisites (Homebrew, Conda, MPICH),
  and at least two alternative methods (Conda & Brew).
- End with: MANUAL_COMPLETE
"""


# ============= 2. AGENT FACTORIES & NODES ========================

smol_search      = SmolAgentFactory(instruct_search,      BROWSER_TOOLS)
smol_alt_search  = SmolAgentFactory(instruct_alt_search,  BROWSER_TOOLS)
smol_extract     = SmolAgentFactory(instruct_extract,     BROWSER_TOOLS)
smol_validate    = SmolAgentFactory(instruct_validate,    SHELL_TOOLS)
smol_install     = SmolAgentFactory(instruct_install,     SHELL_TOOLS)
smol_manual      = SmolAgentFactory(instruct_manual,      [])               # no tools needed

# Wrap into LangGraph nodes
node_search      = WorkflowNodeFactory.create_agent_node(smol_search)
node_alt_search  = WorkflowNodeFactory.create_agent_node(smol_alt_search)
node_extract     = WorkflowNodeFactory.create_agent_node(smol_extract)
node_validate    = WorkflowNodeFactory.create_agent_node(smol_validate)
node_install     = WorkflowNodeFactory.create_agent_node(smol_install)
node_manual      = WorkflowNodeFactory.create_agent_node(smol_manual)

# ============= 3. ROUTING / ERROR-HANDLING FUNCTIONS ============

MAX_RETRIES = 3  # global cap

def _count_occurrences(state: "WorkflowState", step_keyword: str) -> int:
    """Utility to count how many times a given keyword appears in step_name."""
    return sum(1 for name in state.get("step_name", []) if step_keyword in name)

# ---------- ROUTER AFTER SEARCH_AGENT ----------
def router_search(state: "WorkflowState") -> str:
    try:
        answers = state.get("answers", [])
        last = answers[-1] if answers else ""
        retries = _count_occurrences(state, "search")
        if "SEARCH_COMPLETE" in last:
            return "extract"
        if "GIVE_UP" in last:
            return "alt_search"
        # failure path
        if retries < MAX_RETRIES:
            return "search_retry"
        else:
            return "alt_search"
    except Exception as e:
        print(f"🚨 router_search error: {e}")
        return "alt_search"

# ---------- ROUTER AFTER ALT_SEARCH_AGENT ----------
def router_alt_search(state: "WorkflowState") -> str:
    try:
        last = state.get("answers", [])[-1] if state.get("answers") else ""
        retries = _count_occurrences(state, "alt_search")
        if "ALT_SEARCH_COMPLETE" in last:
            return "extract"
        if "GIVE_UP" in last:
            return END
        if retries < MAX_RETRIES:
            return "alt_search_retry"
        else:
            return END
    except Exception as e:
        print(f"🚨 router_alt_search error: {e}")
        return END

# ---------- ROUTER AFTER EXTRACT_AGENT ----------
def router_extract(state: "WorkflowState") -> str:
    try:
        last = state.get("answers", [])[-1] if state.get("answers") else ""
        retries = _count_occurrences(state, "extract")
        if "EXTRACTION_COMPLETE" in last:
            return "validate"
        if "GIVE_UP" in last:
            return "alt_search"
        if retries < MAX_RETRIES:
            return "extract_retry"
        else:
            return "alt_search"
    except Exception as e:
        print(f"🚨 router_extract error: {e}")
        return "alt_search"

# ---------- ROUTER AFTER VALIDATE_AGENT ----------
def router_validate(state: "WorkflowState") -> str:
    try:
        last = state.get("answers", [])[-1] if state.get("answers") else ""
        retries = _count_occurrences(state, "validate")
        if "VALIDATION_PASS" in last:
            return "install"
        if "GIVE_UP" in last:
            return "manual"
        if retries < MAX_RETRIES:
            return "validate_retry"
        else:
            return "manual"
    except Exception as e:
        print(f"🚨 router_validate error: {e}")
        return "manual"

# ---------- ROUTER AFTER INSTALL_AGENT ----------
def router_install(state: "WorkflowState") -> str:
    try:
        last = state.get("answers", [])[-1] if state.get("answers") else ""
        retries = _count_occurrences(state, "install")
        if "INSTALL_SUCCESS" in last:
            return END
        if "GIVE_UP" in last:
            return "manual"
        if retries < MAX_RETRIES:
            return "install_retry"
        else:
            return "manual"
    except Exception as e:
        print(f"🚨 router_install error: {e}")
        return "manual"


# ============= 4. BUILD THE STATE GRAPH ==========================
workflow = StateGraph(WorkflowState)

# ---- 4.1  ADD NODES ----
workflow.add_node("search",          node_search)
workflow.add_node("search_retry",    node_search)
workflow.add_node("alt_search",      node_alt_search)
workflow.add_node("alt_search_retry",node_alt_search)

workflow.add_node("extract",         node_extract)
workflow.add_node("extract_retry",   node_extract)

workflow.add_node("validate",        node_validate)
workflow.add_node("validate_retry",  node_validate)

workflow.add_node("install",         node_install)
workflow.add_node("install_retry",   node_install)

workflow.add_node("manual",          node_manual)

# ---- 4.2  START EDGE ----
workflow.add_edge(START, "search")

# ---- 4.3  CONDITIONAL EDGES WITH FALLBACKS ----
workflow.add_conditional_edges(
    "search",
    router_search,
    {
        "extract":        "extract",
        "search_retry":   "search_retry",
        "alt_search":     "alt_search",
    }
)
workflow.add_conditional_edges(
    "search_retry",
    router_search,
    {
        "extract":        "extract",
        "search_retry":   "search_retry",
        "alt_search":     "alt_search",
    }
)

workflow.add_conditional_edges(
    "alt_search",
    router_alt_search,
    {
        "extract":            "extract",
        "alt_search_retry":   "alt_search_retry",
        END:                  END
    }
)
workflow.add_conditional_edges(
    "alt_search_retry",
    router_alt_search,
    {
        "extract":            "extract",
        "alt_search_retry":   "alt_search_retry",
        END:                  END
    }
)

workflow.add_conditional_edges(
    "extract",
    router_extract,
    {
        "validate":       "validate",
        "extract_retry":  "extract_retry",
        "alt_search":     "alt_search"
    }
)
workflow.add_conditional_edges(
    "extract_retry",
    router_extract,
    {
        "validate":       "validate",
        "extract_retry":  "extract_retry",
        "alt_search":     "alt_search"
    }
)

workflow.add_conditional_edges(
    "validate",
    router_validate,
    {
        "install":        "install",
        "validate_retry": "validate_retry",
        "manual":         "manual"
    }
)
workflow.add_conditional_edges(
    "validate_retry",
    router_validate,
    {
        "install":        "install",
        "validate_retry": "validate_retry",
        "manual":         "manual"
    }
)

workflow.add_conditional_edges(
    "install",
    router_install,
    {
        END:              END,
        "install_retry":  "install_retry",
        "manual":         "manual"
    }
)
workflow.add_conditional_edges(
    "install_retry",
    router_install,
    {
        END:              END,
        "install_retry":  "install_retry",
        "manual":         "manual"
    }
)

# Manual agent always finishes
workflow.add_edge("manual", END)

# ============= 5. COMPILE APP ====================================
app = workflow.compile()