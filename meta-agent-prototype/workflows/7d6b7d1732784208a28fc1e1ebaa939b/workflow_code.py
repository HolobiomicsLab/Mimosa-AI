# ---------------------------------------------
# LANGGRAPH – SMOLAGENT WORKFLOW (GPAW install)
# ---------------------------------------------
#
# Prerequisites (already available in runtime):
# - SHELL_TOOLS
# - BROWSER_TOOLS
# - CSV_TOOLS
# - WorkflowState / Action / Observation  (schema)
# - SmolAgentFactory
# - WorkflowNodeFactory
# - langgraph.graph : StateGraph, START, END
#

from langgraph.graph import StateGraph, START, END

# ------------------------------------------------------------------
# 1. AGENT INSTRUCTIONS  (one atomic responsibility – no overlap)
# ------------------------------------------------------------------

instruct_search = """
You are a focused WEB SEARCH agent.

YOUR TASK
- Use browser-type tools to search the web for instructions on
  installing GPAW on macOS with an M1 (Apple-Silicon) chip.
- Save links, titles, and short snippets that look relevant.

UPON COMPLETION
- If you found at least one promising source say SEARCH_COMPLETE
- If you did not find anything useful say SEARCH_FAILURE
- If a tool error blocks you say GIVE_UP
"""

instruct_extract = """
You are an INFORMATION EXTRACTION agent.

YOUR TASK
- Read the latest browser observations (web pages) and extract ONLY
  the concrete step-by-step shell commands required to install GPAW
  on macOS (M1).  Ignore intel-only or linux-only commands.
- Output the ordered list of commands.

UPON COMPLETION
- If you have a clear list of commands say EXTRACT_COMPLETE
- If extraction failed say EXTRACT_FAILURE
- On unknown errors say GIVE_UP
"""

instruct_validate = """
You are a VALIDATION agent.

YOUR TASK
- Inspect the extracted installation commands.
- Check they explicitly target macOS on Apple Silicon (arm64/m1) and
  do not include incompatible binaries.
- If anything is missing, explain what and mark failure.

UPON COMPLETION
- If commands look correct say VALIDATION_PASS
- If they are wrong/incomplete say VALIDATION_FAIL
- On unhandled error say GIVE_UP
"""

instruct_install = """
You are an INSTALLATION EXECUTION agent.

YOUR TASK
- Run the validated shell commands *exactly* in order using the
  provided shell tool set.
- Capture stdout/stderr for each command.

UPON COMPLETION
- If every command succeeded say INSTALL_SUCCESS
- If a command failed say INSTALL_FAIL
- For unexpected tool errors say GIVE_UP
"""

instruct_verify = """
You are a VERIFICATION agent.

YOUR TASK
- Run `gpaw --version` (or an equivalent command) using the shell
  tool to confirm the installation.
  
UPON COMPLETION
- If GPAW returns a version string say VERIFY_SUCCESS
- If the command errors or GPAW missing say VERIFY_FAIL
- On unknown errors say GIVE_UP
"""

# ------------------------------------------------------------------
# 2. CREATE SMOLAGENTS (tool selection strictly matches responsibility)
# ------------------------------------------------------------------

search_agent_factory   = SmolAgentFactory(instruct_search,  BROWSER_TOOLS)
extract_agent_factory  = SmolAgentFactory(instruct_extract, BROWSER_TOOLS)
validate_agent_factory = SmolAgentFactory(instruct_validate, CSV_TOOLS)
install_agent_factory  = SmolAgentFactory(instruct_install, SHELL_TOOLS)
verify_agent_factory   = SmolAgentFactory(instruct_verify,  SHELL_TOOLS)

# ------------------------------------------------------------------
# 3. WORKFLOW GRAPH
# ------------------------------------------------------------------

workflow = StateGraph(WorkflowState)

# ---- 3.1  Agent Nodes ----
workflow.add_node("search_web",      WorkflowNodeFactory.create_agent_node(search_agent_factory))
workflow.add_node("extract_steps",   WorkflowNodeFactory.create_agent_node(extract_agent_factory))
workflow.add_node("validate_steps",  WorkflowNodeFactory.create_agent_node(validate_agent_factory))
workflow.add_node("install_software",WorkflowNodeFactory.create_agent_node(install_agent_factory))
workflow.add_node("verify_install",  WorkflowNodeFactory.create_agent_node(verify_agent_factory))

# ---- 3.2  Manual fallback node (function) ----
def manual_review(state: WorkflowState) -> dict:
    """
    Emergency graceful-degradation node.
    Simply marks the workflow as unsuccessful but returns all
    gathered information for human review.
    """
    state.setdefault("step_name", []).append("manual_review")
    state.setdefault("answers",   []).append("MANUAL_INTERVENTION_REQUIRED")
    state.setdefault("success",   []).append(False)
    return state

workflow.add_node("manual_review", manual_review)

# ------------------------------------------------------------------
# 4. ROUTING FUNCTIONS  (robust, multi-path, with retry counters)
# ------------------------------------------------------------------

MAX_RETRIES = 3        # universal retry cap per node

def _attempts(state: WorkflowState, node_name: str) -> int:
    """Helper: count how many times a node has already executed."""
    return sum(1 for n in state.get("step_name", []) if n.startswith(node_name))

# ---- search_router ----
def search_router(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        if "SEARCH_COMPLETE" in answer:
            return "extract_steps"
        elif _attempts(state, "search_web") < MAX_RETRIES:
            return "search_web"              # retry
        else:
            return "manual_review"           # hard fallback
    except Exception as e:
        print(f"Routing error (search): {e}")
        return "manual_review"

# ---- extract_router ----
def extract_router(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        if "EXTRACT_COMPLETE" in answer:
            return "validate_steps"
        elif _attempts(state, "extract_steps") < MAX_RETRIES:
            return "extract_steps"           # retry extraction
        else:
            return "search_web"              # re-search different sources
    except Exception as e:
        print(f"Routing error (extract): {e}")
        return "manual_review"

# ---- validate_router ----
def validate_router(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        if "VALIDATION_PASS" in answer:
            return "install_software"
        elif "VALIDATION_FAIL" in answer and _attempts(state, "search_web") < MAX_RETRIES:
            return "search_web"              # gather better resources
        elif _attempts(state, "validate_steps") < MAX_RETRIES:
            return "validate_steps"          # retry validation
        else:
            return "manual_review"
    except Exception as e:
        print(f"Routing error (validate): {e}")
        return "manual_review"

# ---- install_router ----
def install_router(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        if "INSTALL_SUCCESS" in answer:
            return "verify_install"
        elif _attempts(state, "install_software") < MAX_RETRIES:
            return "install_software"        # retry installation
        else:
            return "manual_review"
    except Exception as e:
        print(f"Routing error (install): {e}")
        return "manual_review"

# ---- verify_router ----
def verify_router(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        if "VERIFY_SUCCESS" in answer:
            return END
        elif _attempts(state, "install_software") < MAX_RETRIES:
            return "install_software"        # attempt re-install
        else:
            return "manual_review"
    except Exception as e:
        print(f"Routing error (verify): {e}")
        return "manual_review"

# ------------------------------------------------------------------
# 5. GRAPH EDGES & CONDITIONAL ROUTING
# ------------------------------------------------------------------

# start
workflow.add_edge(START, "search_web")

# search node edges
workflow.add_conditional_edges(
    "search_web",
    search_router,
    {
        "extract_steps":    "extract_steps",
        "search_web":       "search_web",      # retry
        "manual_review":    "manual_review"
    }
)

# extract node edges
workflow.add_conditional_edges(
    "extract_steps",
    extract_router,
    {
        "validate_steps":   "validate_steps",
        "extract_steps":    "extract_steps",   # retry
        "search_web":       "search_web",
        "manual_review":    "manual_review"
    }
)

# validate node edges
workflow.add_conditional_edges(
    "validate_steps",
    validate_router,
    {
        "install_software": "install_software",
        "validate_steps":   "validate_steps",  # retry
        "search_web":       "search_web",
        "manual_review":    "manual_review"
    }
)

# install node edges
workflow.add_conditional_edges(
    "install_software",
    install_router,
    {
        "verify_install":   "verify_install",
        "install_software": "install_software",# retry
        "manual_review":    "manual_review"
    }
)

# verify node edges
workflow.add_conditional_edges(
    "verify_install",
    verify_router,
    {
        END:                END,
        "install_software": "install_software",
        "manual_review":    "manual_review"
    }
)

# manual_review edges – always end
workflow.add_edge("manual_review", END)

# ------------------------------------------------------------------
# 6. COMPILE APP
# ------------------------------------------------------------------

app = workflow.compile()