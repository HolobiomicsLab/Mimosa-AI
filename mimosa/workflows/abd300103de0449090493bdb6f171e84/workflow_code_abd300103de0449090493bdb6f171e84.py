# ==============================================================
# EXECUTABLE LangGraph – SmolAgent WORKFLOW
# GOAL :  “Search and install the GPAW software for macOS Apple-Silicon”
# ==============================================================

from langgraph.graph import StateGraph, START, END
from typing import List, TypedDict, Callable
import re
import time

# ----------------------------------------------------------------
# 1)  STATE  (already provided in the environment description)
# ----------------------------------------------------------------
class Action(TypedDict):
    tool: str
    inputs: dict

class Observation(TypedDict):
    data: str

class WorkflowState(TypedDict):
    step_name: List[str]
    actions: List[Action]
    observations: List[Observation]
    rewards: List[float]
    answers: List[str]
    success: List[bool]

# ----------------------------------------------------------------
# 2)  AGENT INSTRUCTIONS (one atomic responsibility each)
# ----------------------------------------------------------------
instruct_search_web = """
You are a focused WEB SEARCH agent.

TASK
1. Use browser tools to query the web for “install GPAW macos apple silicon”.
2. Capture urls and two-sentence summaries of the best 3 sources.

SUCCESS CRITERIA
- If at least one relevant source is found, finish with the exact word: SEARCH_OK
- If nothing useful is found, finish with: SEARCH_FAIL
- On unexpected error write: GIVE_UP
"""

instruct_docs_search = """
You are a DOCUMENTATION SEARCH agent specialised in official manuals.

TASK
1. Look specifically for GPAW official documentation / GitHub / Conda-Forge pages mentioning Apple-Silicon or arm64.
2. Return url(s) and short note.

SUCCESS WORDS
- DOCS_OK   (found)
- DOCS_FAIL (not found)
- GIVE_UP   (error)
"""

instruct_extract_content = """
You are a CONTENT EXTRACTION agent.

TASK
1. From the last browser page visited (in observations) extract ONLY:
   - concrete installation commands (brew, conda, pip, etc.)
   - mentioned prerequisites
2. Output them as a bullet list.

FINISH WORDS
- EXTRACT_OK  (commands extracted)
- EXTRACT_FAIL (no commands)
- GIVE_UP
"""

instruct_validate_compat = """
You are a COMPATIBILITY VALIDATOR agent.

INPUT
- Installation commands bullet list from previous agent.

TASK
- Ensure commands are valid for macOS 13+ Apple-Silicon arm64.
- Detect intel-only or deprecated commands.
- Mark each command as VALID or INVALID and explain briefly.

FINISH WORDS
- VALID_OK  (all commands valid)
- VALID_FAIL (any invalid)
- GIVE_UP
"""

instruct_adapt_commands = """
You are an ADAPTATION agent.

INPUT
- A list of INVALID GPAW installation commands.

TASK
- Rewrite each invalid command so it works on Apple-Silicon.
- Prefer Homebrew or Conda-Forge with arm64 builds.
- Return adapted commands.

FINISH WORDS
- ADAPT_OK   (adapted set ready)
- ADAPT_FAIL (cannot adapt)
- GIVE_UP
"""

instruct_installer = """
You are an INSTALLATION EXECUTOR agent.

TASK
1. Execute each installation command in order using shell tool.
2. Capture output / errors in observations.

FINISH WORDS
- INSTALL_SUCCESS (everything installed)
- INSTALL_FAILURE (any command failed)
- GIVE_UP
"""

# ----------------------------------------------------------------
# 3)  CREATE SMOL AGENTS
# ----------------------------------------------------------------
# Tool sets already available in global scope:
# SHELL_TOOLS, BROWSER_TOOLS, CSV_TOOLS

smol_search_web     = SmolAgentFactory(instruct_search_web,  BROWSER_TOOLS)
smol_docs_search    = SmolAgentFactory(instruct_docs_search, BROWSER_TOOLS)
smol_extract        = SmolAgentFactory(instruct_extract_content, BROWSER_TOOLS)
smol_validate       = SmolAgentFactory(instruct_validate_compat, [])
smol_adapt          = SmolAgentFactory(instruct_adapt_commands, [])
smol_installer      = SmolAgentFactory(instruct_installer, SHELL_TOOLS)

# ----------------------------------------------------------------
# 4)  WORKFLOW INITIALISATION
# ----------------------------------------------------------------
workflow = StateGraph(WorkflowState)

# ----------------------------------------------------------------
# 5)  ADD NODES
# ----------------------------------------------------------------
workflow.add_node("search_web",  WorkflowNodeFactory.create_agent_node(smol_search_web))
workflow.add_node("docs_search", WorkflowNodeFactory.create_agent_node(smol_docs_search))
workflow.add_node("extract",     WorkflowNodeFactory.create_agent_node(smol_extract))
workflow.add_node("validate",    WorkflowNodeFactory.create_agent_node(smol_validate))
workflow.add_node("adapt",       WorkflowNodeFactory.create_agent_node(smol_adapt))
workflow.add_node("install",     WorkflowNodeFactory.create_agent_node(smol_installer))

# ----------------------------------------------------------------
# 6)  ROUTING / ERROR-HANDLING FUNCTIONS
# ----------------------------------------------------------------
def _retry_counter(state: WorkflowState, step: str) -> int:
    return sum(1 for s in state.get("step_name", []) if s.startswith(step))

def router_search(state: WorkflowState) -> str:
    print("🔀 Router SEARCH")
    try:
        answer = state.get("answers", [""])[-1]
        if "SEARCH_OK" in answer:
            return "extract"
        if "GIVE_UP" in answer:
            return "docs_search"
        # retry limit
        retries = _retry_counter(state, "search_web")
        return "search_web" if retries < 2 else "docs_search"
    except Exception as e:
        print("⚠️ router_search error:", e)
        return "docs_search"

def router_docs(state: WorkflowState) -> str:
    print("🔀 Router DOCS_SEARCH")
    try:
        answer = state.get("answers", [""])[-1]
        if "DOCS_OK" in answer:
            return "extract"
        retries = _retry_counter(state, "docs_search")
        return "docs_search" if retries < 2 else END
    except Exception as e:
        print("⚠️ router_docs error:", e)
        return END

def router_extract(state: WorkflowState) -> str:
    print("🔀 Router EXTRACT")
    try:
        answer = state.get("answers", [""])[-1]
        if "EXTRACT_OK" in answer:
            return "validate"
        retries = _retry_counter(state, "extract")
        return "extract" if retries < 2 else END
    except Exception as e:
        print("⚠️ router_extract error:", e)
        return END

def router_validate(state: WorkflowState) -> str:
    print("🔀 Router VALIDATE")
    try:
        answer = state.get("answers", [""])[-1]
        if "VALID_OK" in answer:
            return "install"
        if "VALID_FAIL" in answer:
            return "adapt"
        return END
    except Exception as e:
        print("⚠️ router_validate error:", e)
        return END

def router_adapt(state: WorkflowState) -> str:
    print("🔀 Router ADAPT")
    try:
        answer = state.get("answers", [""])[-1]
        if "ADAPT_OK" in answer:
            return "install"
        retries = _retry_counter(state, "adapt")
        return "adapt" if retries < 1 else END
    except Exception as e:
        print("⚠️ router_adapt error:", e)
        return END

def router_install(state: WorkflowState) -> str:
    print("🔀 Router INSTALL")
    try:
        answer = state.get("answers", [""])[-1]
        if "INSTALL_SUCCESS" in answer:
            return END
        retries = _retry_counter(state, "install")
        return "install" if retries < 2 else END
    except Exception as e:
        print("⚠️ router_install error:", e)
        return END

# ----------------------------------------------------------------
# 7)  EDGE DEFINITIONS WITH MULTIPLE FALLBACKS
# ----------------------------------------------------------------
workflow.add_edge(START, "search_web")

workflow.add_conditional_edges(
    "search_web",
    router_search,
    {
        "extract":      "extract",
        "search_web":   "search_web",   # retry
        "docs_search":  "docs_search"   # fallback
    }
)

workflow.add_conditional_edges(
    "docs_search",
    router_docs,
    {
        "extract":     "extract",
        "docs_search": "docs_search",   # retry
        END:           END
    }
)

workflow.add_conditional_edges(
    "extract",
    router_extract,
    {
        "validate": "validate",
        "extract":  "extract",          # retry
        END:        END
    }
)

workflow.add_conditional_edges(
    "validate",
    router_validate,
    {
        "install": "install",
        "adapt":   "adapt",
        END:       END
    }
)

workflow.add_conditional_edges(
    "adapt",
    router_adapt,
    {
        "install": "install",
        "adapt":   "adapt",             # one retry
        END:       END
    }
)

workflow.add_conditional_edges(
    "install",
    router_install,
    {
        END:     END,
        "install": "install"            # retry
    }
)

# ----------------------------------------------------------------
# 8)  COMPILE WORKFLOW
# ----------------------------------------------------------------
app = workflow.compile()