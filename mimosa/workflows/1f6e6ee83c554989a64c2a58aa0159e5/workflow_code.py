# =====================  MANDATORY IMPORTS  =====================
from langgraph.graph import StateGraph, START, END
from typing import List, TypedDict, Callable

# ----- State schema already provided in system -----
# class Action(TypedDict): ...
# class Observation(TypedDict): ...
# class WorkflowState(TypedDict): ...

# ----------  TOOL SETS PROVIDED BY THE USER ----------
# (they already exist in the surrounding scope)
# SHELL_TOOLS
# BROWSER_TOOLS
# CSV_TOOLS


# =====================  AGENT INSTRUCTIONS  =====================

# 1) WEB SEARCH AGENT ------------------------------------------------
instruct_search = """
You are an internet research specialist.

YOUR SINGLE PURPOSE
- Search the web for reliable instructions to install the “gpaw” software on Apple Silicon (Mac M1).

STEPS
1. Use the browser tool to query official docs, GitHub, conda-forge, Homebrew, etc.
2. Collect ONLY mac-specific commands (pip/conda/brew) that work on arm64.
3. Summarise results clearly.

UPON COMPLETION
• If you found at least one complete, reproducible installation method, finish with the phrase: RESEARCH_COMPLETE  
• If no sufficient info after trying, finish with: RESEARCH_FAILURE  
• On any tool error you cannot solve say: GIVE_UP
"""

# 2) INSTRUCTION EXTRACTION AGENT -----------------------------------
instruct_extract = """
You are an extraction agent.

YOUR SINGLE PURPOSE
- Convert raw search findings into a SHORT, ordered list of terminal commands to install “gpaw” on Mac M1.

REQUIREMENTS
• Include prerequisite package manager (brew / pip / conda) install steps if missing.
• Do NOT execute, only format the commands.

OUTPUT
• Return the commands line-by-line inside a markdown code-fence labelled 'bash'.
• End with the token: EXTRACTION_DONE
• If source text is insufficient, end with: EXTRACTION_FAIL
"""

# 3) COMPATIBILITY VALIDATOR AGENT ----------------------------------
instruct_validate = """
You are a validation agent.

YOUR SINGLE PURPOSE
- Check that the provided bash commands are compatible with Apple Silicon (arm64).

TASK
• Look for x86-only binaries or flags like arch -x86_64.
• Ensure brew or conda channels support arm64.
• Output either VALIDATION_PASS or VALIDATION_FAIL with a one-sentence reason.
"""

# 4) PIP INSTALLER AGENT (PRIMARY) ----------------------------------
instruct_install_pip = """
You are an installation agent.

YOUR SINGLE PURPOSE
- Execute the bash commands (from previous observation) via shell to install “gpaw” using pip or brew as listed.

RULES
• Use shell tool exactly once per command.
• After ALL commands executed successfully, reply INSTALL_SUCCESS.
• If any command fails, reply INSTALL_FAILURE. Do NOT attempt to fix – other agents will.
• On unexpected tool error reply GIVE_UP.
"""

# 5) CONDA INSTALLER AGENT (FALLBACK) -------------------------------
instruct_install_conda = """
You are a fallback installer.

YOUR SINGLE PURPOSE
- Install “gpaw” using conda-forge on Mac M1 as an alternative method.

STEPS
1. If conda missing, install miniforge.
2. Create env and install gpaw.
3. Avoid root privileges.

AFTER SUCCESS reply INSTALL_SUCCESS
If fails reply INSTALL_FAILURE
"""

# 6) VERIFICATION AGENT --------------------------------------------
instruct_verify = """
You are a verification agent.

YOUR SINGLE PURPOSE
- Confirm that “gpaw” is correctly installed.

STEPS
1. Run `python -c "import gpaw, sys, platform; print(gpaw.__version__)"`.
2. Capture output.

IF import succeeds reply VERIFICATION_SUCCESS  
Else reply VERIFICATION_FAIL
"""

# =====================  AGENT CREATION  ===========================
# NOTE: SmolAgentFactory & WorkflowNodeFactory already defined globally.

# ---- tool allocation (ONE flat list each) ----
SEARCH_TOOLS   = BROWSER_TOOLS
EXTRACT_TOOLS  = []                # no external tools needed
VALIDATE_TOOLS = []                # purely reasoning
PIP_TOOLS      = SHELL_TOOLS
CONDA_TOOLS    = SHELL_TOOLS
VERIFY_TOOLS   = SHELL_TOOLS

# ---- create smol agents ----
search_agent_factory   = SmolAgentFactory(instruct_search,   SEARCH_TOOLS)
extract_agent_factory  = SmolAgentFactory(instruct_extract,  EXTRACT_TOOLS)
validate_agent_factory = SmolAgentFactory(instruct_validate, VALIDATE_TOOLS)
pip_agent_factory      = SmolAgentFactory(instruct_install_pip,   PIP_TOOLS)
conda_agent_factory    = SmolAgentFactory(instruct_install_conda, CONDA_TOOLS)
verify_agent_factory   = SmolAgentFactory(instruct_verify,   VERIFY_TOOLS)

# =====================  WORKFLOW INITIALISATION  ==================
workflow = StateGraph(WorkflowState)

# =====================  NODE REGISTRATION  ========================
workflow.add_node("web_search",    WorkflowNodeFactory.create_agent_node(search_agent_factory))
workflow.add_node("extract_cmds",  WorkflowNodeFactory.create_agent_node(extract_agent_factory))
workflow.add_node("validate_cmds", WorkflowNodeFactory.create_agent_node(validate_agent_factory))
workflow.add_node("install_pip",   WorkflowNodeFactory.create_agent_node(pip_agent_factory))
workflow.add_node("install_conda", WorkflowNodeFactory.create_agent_node(conda_agent_factory))
workflow.add_node("verify_install",WorkflowNodeFactory.create_agent_node(verify_agent_factory))

# =====================  ROUTING FUNCTIONS  ========================

# ---------- helper -------------------------------------------------
def _retry_count(state: WorkflowState, step: str) -> int:
    names = state.get("step_name", [])
    return sum(1 for n in names if n.startswith(step))

# ---------- router after web search -------------------------------
def route_after_search(state: WorkflowState) -> str:
    try:
        answers = state.get("answers", [])
        last = answers[-1] if answers else ""
        retries = _retry_count(state, "web_search")
        
        if "RESEARCH_COMPLETE" in last:
            return "extract_cmds"
        if "GIVE_UP" in last or retries >= 3:
            return "emergency_end"
        # default retry
        return "web_search"
    except Exception as e:
        print(f"💥 router_search error: {e}")
        return "emergency_end"

# ---------- router after extraction -------------------------------
def route_after_extract(state: WorkflowState) -> str:
    try:
        last = state.get("answers", [])[-1]
        retries = _retry_count(state, "extract_cmds")
        if "EXTRACTION_DONE" in last:
            return "validate_cmds"
        if retries >= 2:
            # fall back to re-search flow
            return "web_search"
        return "extract_cmds"   # retry extraction
    except Exception as e:
        print(f"💥 router_extract error: {e}")
        return "emergency_end"

# ---------- router after validation -------------------------------
def route_after_validate(state: WorkflowState) -> str:
    try:
        last = state.get("answers", [])[-1]
        if "VALIDATION_PASS" in last:
            return "install_pip"
        # If fails, try improving commands by going back to search
        return "web_search"
    except Exception as e:
        print(f"💥 router_validate error: {e}")
        return "emergency_end"

# ---------- router after pip install ------------------------------
def route_after_pip(state: WorkflowState) -> str:
    try:
        last = state.get("answers", [])[-1]
        retries = _retry_count(state, "install_pip")
        if "INSTALL_SUCCESS" in last:
            return "verify_install"
        if retries >= 2:
            return "install_conda"     # switch strategy
        return "install_pip"           # retry pip path
    except Exception as e:
        print(f"💥 router_pip error: {e}")
        return "install_conda"

# ---------- router after conda install ----------------------------
def route_after_conda(state: WorkflowState) -> str:
    try:
        last = state.get("answers", [])[-1]
        retries = _retry_count(state, "install_conda")
        if "INSTALL_SUCCESS" in last:
            return "verify_install"
        if retries >= 2:
            return "emergency_end"
        return "install_conda"
    except Exception as e:
        print(f"💥 router_conda error: {e}")
        return "emergency_end"

# ---------- router after verification -----------------------------
def route_after_verify(state: WorkflowState) -> str:
    try:
        last = state.get("answers", [])[-1]
        if "VERIFICATION_SUCCESS" in last:
            return END
        # failed verification → attempt conda path (if not tried) else END
        if _retry_count(state, "install_conda") == 0:
            return "install_conda"
        return "emergency_end"
    except Exception as e:
        print(f"💥 router_verify error: {e}")
        return "emergency_end"

# =====================  EDGES & FALLBACKS  ========================
workflow.add_edge(START, "web_search")

workflow.add_conditional_edges(
    "web_search",
    route_after_search,
    {
        "extract_cmds": "extract_cmds",
        "web_search":   "web_search",      # retry
        "emergency_end": END
    },
)

workflow.add_conditional_edges(
    "extract_cmds",
    route_after_extract,
    {
        "validate_cmds": "validate_cmds",
        "extract_cmds":  "extract_cmds",   # retry
        "web_search":    "web_search",
        "emergency_end": END,
    },
)

workflow.add_conditional_edges(
    "validate_cmds",
    route_after_validate,
    {
        "install_pip":  "install_pip",
        "web_search":   "web_search",
        "emergency_end": END,
    },
)

workflow.add_conditional_edges(
    "install_pip",
    route_after_pip,
    {
        "verify_install": "verify_install",
        "install_pip":    "install_pip",      # retry
        "install_conda":  "install_conda",
        "emergency_end":  END,
    },
)

workflow.add_conditional_edges(
    "install_conda",
    route_after_conda,
    {
        "verify_install": "verify_install",
        "install_conda":  "install_conda",    # retry
        "emergency_end":  END,
    },
)

workflow.add_conditional_edges(
    "verify_install",
    route_after_verify,
    {
        END:             END,
        "install_conda": "install_conda",
        "emergency_end": END,
    },
)

# =====================  COMPILE & EXPORT  =========================
app = workflow.compile()