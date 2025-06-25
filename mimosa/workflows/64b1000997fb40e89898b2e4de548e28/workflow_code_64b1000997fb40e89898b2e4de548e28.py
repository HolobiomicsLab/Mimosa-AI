# --------------------------------------------
# GPAW (macOS ‑ Apple Silicon) Installation
# LangGraph  +  SmolAgent  Robust Workflow
# --------------------------------------------

from langgraph.graph import StateGraph, START, END
from typing import Callable, List
# ↓↓↓  PRE-DECLARED COMPONENTS (already in working env)  ↓↓↓
# - WorkflowState / Action / Observation  (schema)
# - SmolAgentFactory
# - WorkflowNodeFactory
# - SHELL_TOOLS, BROWSER_TOOLS, CSV_TOOLS
# --------------------------------------------

# 1️⃣  AGENT INSTRUCTIONS  (one atomic task each) ------------------------------

instruct_search = """
You are an Internet SEARCH agent.

GOAL: find official & reliable information about installing GPAW on macOS with Apple-Silicon (arm64).

STEPS
1. Use browser search queries such as:
   - "GPAW install macOS Apple silicon"
   - "GPAW brew formula apple silicon"
2. Collect 3-5 promising links (docs, GitHub, Homebrew, Conda).
3. Summarise key installation approaches in ≤100 words.

UPON COMPLETION
- If summary & at least one viable approach found → reply with: SEARCH_SUCCESS
- If no useful info → reply with: SEARCH_FAIL
- If tooling error → reply with: GIVE_UP
"""

instruct_extract = """
You are a COMMAND EXTRACTION agent.

INPUT
- You receive web search summary & links in the latest observations list.

TASK
- Extract concrete shell commands required to install GPAW on Apple-Silicon macOS
  (e.g. brew install gpaw, pip install gpaw, conda ...).
- Output the commands clearly labelled.

UPON COMPLETION
- If you produced at least one full command set → reply with: EXTRACT_SUCCESS
- If extraction impossible → reply with: EXTRACT_FAIL
- On tool error → GIVE_UP
"""

instruct_prereq = """
You are a PREREQUISITE CHECK agent using shell tools.

TASK
1. Verify presence of Homebrew & Python3 (≥3.9) with:
   - which brew
   - brew --version
   - python3 --version
2. If missing, install Homebrew OR prompt user to install Xcode CLT.
3. Output status report.

UPON COMPLETION
- If all prerequisites satisfied → PREREQ_OK
- If still missing after attempts → PREREQ_FAIL
- Tool error → GIVE_UP
"""

instruct_install = """
You are an INSTALLATION agent.

INPUT
- You receive extracted installation commands in observations.

TASK
- Execute the commands step by step using shell tools.
- Handle sudo prompt gracefully (Homebrew shouldn't need it).
- After install, cache stdout/stderr lines (first 20 lines only).

UPON COMPLETION
- If commands finished without fatal error → INSTALL_OK
- If any command failed → INSTALL_FAIL
- If shell tool error beyond control → GIVE_UP
"""

instruct_validate = """
You are a VALIDATION agent.

TASK
- Run: python3 -c "import gpaw, sys, json, platform, importlib; print(gpaw.__version__)"
- Verify exit code is 0.
- Report detected GPAW version & Python version.

UPON COMPLETION
- If import succeeded → VALIDATION_SUCCESS
- If import failed → VALIDATION_FAIL
- If shell tool error → GIVE_UP
"""

# 2️⃣  CREATE SMOL AGENTS -------------------------------------------------------

search_agent      = SmolAgentFactory(instruct_search,   BROWSER_TOOLS)
extract_agent     = SmolAgentFactory(instruct_extract,  CSV_TOOLS)
prereq_agent      = SmolAgentFactory(instruct_prereq,   SHELL_TOOLS)
install_agent     = SmolAgentFactory(instruct_install,  SHELL_TOOLS)
validate_agent    = SmolAgentFactory(instruct_validate, SHELL_TOOLS)

# 3️⃣  INITIALISE WORKFLOW GRAPH ----------------------------------------------

workflow = StateGraph(WorkflowState)

# --- Add agent nodes (atomic tasks)
workflow.add_node("search",   WorkflowNodeFactory.create_agent_node(search_agent))
workflow.add_node("extract",  WorkflowNodeFactory.create_agent_node(extract_agent))
workflow.add_node("prereq",   WorkflowNodeFactory.create_agent_node(prereq_agent))
workflow.add_node("install",  WorkflowNodeFactory.create_agent_node(install_agent))
workflow.add_node("validate", WorkflowNodeFactory.create_agent_node(validate_agent))

# 4️⃣  ROUTING HELPERS  --------------------------------------------------------

def make_router(success_kw: str, failure_kw: str, node_name: str,
                max_retry: int, fallback_node: str = None) -> Callable[[WorkflowState], str]:
    """
    Generic router factory with:
    - success detection
    - retry counter
    - optional fallback to another node
    - GIVE_UP handling
    """
    def router(state: WorkflowState) -> str:
        try:
            answers: List[str] = state.get("answers", [])
            steps:   List[str] = state.get("step_name", [])
            if not answers:
                print(f"⚠️  {node_name}: No answer captured, retrying …")
                return "retry"

            last = answers[-1]

            # Immediate GIVE_UP from agent
            if "GIVE_UP" in last:
                print(f"❌  {node_name} indicated GIVE_UP")
                return "give_up"

            # Success branch
            if success_kw in last:
                print(f"✅  {node_name} succeeded.")
                return "success"

            # Explicit failure branch
            if failure_kw in last:
                retry_cnt = steps.count(node_name)
                if retry_cnt < max_retry:
                    print(f"🔄  {node_name} failed, retry {retry_cnt+1}/{max_retry}")
                    return "retry"
                else:
                    print(f"🚨  {node_name} max retries exceeded.")
                    return "fallback" if fallback_node else "give_up"

            # Unknown pattern -> treat as retry
            print(f"❓  {node_name} unrecognised answer, retrying …")
            return "retry"
        except Exception as e:
            print(f"💥 Router error in {node_name}: {e}")
            return "give_up"
    return router


# 5️⃣  DEFINE ROUTERS WITH ROBUST FALLBACKS ------------------------------------

search_router   = make_router("SEARCH_SUCCESS",   "SEARCH_FAIL",   "search",   max_retry=3)
extract_router  = make_router("EXTRACT_SUCCESS",  "EXTRACT_FAIL",  "extract",  max_retry=2, fallback_node="search")
prereq_router   = make_router("PREREQ_OK",        "PREREQ_FAIL",   "prereq",   max_retry=2, fallback_node="search")
install_router  = make_router("INSTALL_OK",       "INSTALL_FAIL",  "install",  max_retry=2, fallback_node="prereq")
validate_router = make_router("VALIDATION_SUCCESS","VALIDATION_FAIL","validate",max_retry=1, fallback_node="install")

# 6️⃣  GRAPH EDGES + CONDITIONAL PATHS -----------------------------------------

# ---- START
workflow.add_edge(START, "search")

# ---- search node
workflow.add_conditional_edges(
    "search",
    search_router,
    {
        "success":  "extract",
        "retry":    "search",
        "give_up":  END
    }
)

# ---- extract node
workflow.add_conditional_edges(
    "extract",
    extract_router,
    {
        "success":   "prereq",
        "retry":     "extract",
        "fallback":  "search",   # go back to fresh search with new keywords
        "give_up":   END
    }
)

# ---- prereq node
workflow.add_conditional_edges(
    "prereq",
    prereq_router,
    {
        "success":   "install",
        "retry":     "prereq",
        "fallback":  "search",   # attempt brand-new information if prereqs impossible
        "give_up":   END
    }
)

# ---- install node
workflow.add_conditional_edges(
    "install",
    install_router,
    {
        "success":   "validate",
        "retry":     "install",
        "fallback":  "prereq",   # re-check prerequisites
        "give_up":   END
    }
)

# ---- validate node
workflow.add_conditional_edges(
    "validate",
    validate_router,
    {
        "success":   END,
        "retry":     "validate",
        "fallback":  "install",  # attempt reinstall if validation fails
        "give_up":   END
    }
)

# 7️⃣  COMPILE WORKFLOW --------------------------------------------------------

app = workflow.compile()