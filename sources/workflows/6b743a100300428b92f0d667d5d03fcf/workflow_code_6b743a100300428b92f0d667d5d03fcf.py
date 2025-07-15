# ================================================================
# LangGraph ‑ SmolAgent WORKFLOW :  “Search and install llama.cpp”
# ================================================================

# ---  Context objects that ALREADY EXIST in the interpreter -----
#  - WorkflowState   (TypedDict)            ✅
#  - StateGraph, START, END                 ✅
#  - WorkflowNodeFactory                    ✅
#  - SmolAgentFactory                       ✅
#  - Tool packages:
#       • WEB_BROWSER_MCP_TOOLS             ✅
#       • BASH_COMMAND_MCP_TOOLS            ✅
#       • CSV_MANAGEMENT_TOOLS              ✅
# ----------------------------------------------------------------

from langgraph.graph import StateGraph, START, END


# 1️⃣ ==========  AGENT INSTRUCTIONS  =================================

# A.  ARCHITECTURE DETECTOR  -----------------------------------------
arch_prompt = """
You are a system-inspection agent.

GOAL
Determine the current operating-system name and CPU architecture so that other
agents can choose the correct llama.cpp build procedure.

TOOLS
You can execute shell commands through the available tools.

COMPLETION PROTOCOL
SUCCESS  →  ARCH_DETECT_SUCCESS: [OS=<os>, ARCH=<arch>, full uname output…]
FAILURE  →  ARCH_DETECT_FAILURE: [what you tried, what went wrong]
ERROR    →  GIVE_UP: [error details]

Only reply with a short explanation followed by the completion tag.
"""

# B.  WEB RESEARCHER  -------------------------------------------------
research_prompt = """
You are a web-research specialist.

GOAL
Find the exact installation / build instructions for the open-source project
"llama.cpp" that match the provided OS and architecture.

INPUT CONTEXT
`user_arch_info` will be passed to you as the previous agent’s answer.

DELIVERABLE
A numbered list of shell commands ready to be executed, plus any dependencies
that must be installed first (cmake, clang, etc.).

COMPLETION PROTOCOL
SUCCESS  →  RESEARCH_COMPLETE: [step-by-step commands with short comments]
FAILURE  →  RESEARCH_FAILURE: [what was missing, new ideas to try]
ERROR    →  GIVE_UP: [technical error]
"""

# C.  GENERIC-RESEARCH FALLBACK  -------------------------------------
generic_research_prompt = """
You are a fallback research agent.

GOAL
Produce GENERIC build instructions for llama.cpp that should work on most
Linux x86_64 systems when specific instructions could not be located.

COMPLETION PROTOCOL
SUCCESS  →  GENERIC_RESEARCH_COMPLETE: [generic steps]
FAILURE  →  GENERIC_RESEARCH_FAILURE: [why even generic steps failed]
ERROR    →  GIVE_UP: [error info]
"""

# D.  INSTALLER  ------------------------------------------------------
install_prompt = """
You are an installation agent.

GOAL
Execute the provided llama.cpp build commands exactly as given. Capture all
stdout/stderr and report success or failure.

PRE-CONDITION
Previous answer contains the commands to run (labelled RESEARCH_COMPLETE).

COMPLETION PROTOCOL
SUCCESS  →  INSTALL_SUCCESS: [binary path, compile log summary]
FAILURE  →  INSTALL_FAILURE: [last 50 lines of log, hypothesis]
ERROR    →  GIVE_UP: [bash error]
"""

# E.  ALT-INSTALLER (pre-built binaries)  -----------------------------
alt_install_prompt = """
You are an alternative installer.

GOAL
If compilation failed, attempt to download and unpack a PRE-BUILT binary of
llama.cpp that matches the architecture, or use a package manager.

COMPLETION PROTOCOL
SUCCESS  →  ALT_INSTALL_SUCCESS: [where binary located]
FAILURE  →  ALT_INSTALL_FAILURE: [explain]
ERROR    →  GIVE_UP: [error]
"""

# F.  VERIFIER  -------------------------------------------------------
verify_prompt = """
You are an installation verification agent.

GOAL
Run the compiled (or downloaded) llama.cpp binary with the -h flag to confirm
it starts correctly.

COMPLETION PROTOCOL
SUCCESS  →  VERIFY_SUCCESS: [stdout excerpt proving success]
FAILURE  →  VERIFY_FAILURE: [stdout/stderr, analysis]
ERROR    →  GIVE_UP: [error]
"""

# G.  LOGGER  ---------------------------------------------------------
logger_prompt = """
You are a logging agent.

GOAL
Append a record of the whole installation attempt to a CSV file called
"llama_cpp_install_log".

REQUIRED COLUMNS
timestamp, os, arch, result, notes

COMPLETION PROTOCOL
SUCCESS  →  LOG_COMPLETE
FAILURE  →  LOG_FAILURE: [explain]
ERROR    →  GIVE_UP: [error]
"""


# 2️⃣ ==========  AGENT DECLARATIONS  ================================

arch_detector       = SmolAgentFactory("arch_detector",       arch_prompt,           BASH_COMMAND_MCP_TOOLS)
web_researcher      = SmolAgentFactory("web_researcher",      research_prompt,       WEB_BROWSER_MCP_TOOLS)
generic_researcher  = SmolAgentFactory("generic_researcher",  generic_research_prompt, WEB_BROWSER_MCP_TOOLS)
installer           = SmolAgentFactory("installer",           install_prompt,        BASH_COMMAND_MCP_TOOLS)
alt_installer       = SmolAgentFactory("alt_installer",       alt_install_prompt,    BASH_COMMAND_MCP_TOOLS)
verifier            = SmolAgentFactory("verifier",            verify_prompt,         BASH_COMMAND_MCP_TOOLS)
logger_agent        = SmolAgentFactory("logger_agent",        logger_prompt,         CSV_MANAGEMENT_TOOLS)


# 3️⃣ ==========  WORKFLOW BUILD  =====================================

workflow = StateGraph(WorkflowState)

# -----  NODE REGISTRATION -------------------------------------------
workflow.add_node("arch_detector",       WorkflowNodeFactory.create_agent_node(arch_detector))
workflow.add_node("web_researcher",      WorkflowNodeFactory.create_agent_node(web_researcher))
workflow.add_node("generic_researcher",  WorkflowNodeFactory.create_agent_node(generic_researcher))
workflow.add_node("installer",           WorkflowNodeFactory.create_agent_node(installer))
workflow.add_node("alt_installer",       WorkflowNodeFactory.create_agent_node(alt_installer))
workflow.add_node("verifier",            WorkflowNodeFactory.create_agent_node(verifier))
workflow.add_node("logger_agent",        WorkflowNodeFactory.create_agent_node(logger_agent))


# 4️⃣ ==========  ROUTING FUNCTIONS WITH ROBUST FALLBACKS ==============

# Helper to count how many times a step appears in the history
def _count_occurrences(state: WorkflowState, step: str) -> int:
    return sum(1 for s in state.get("step_name", []) if s.startswith(step))


# A. ARCH ROUTER ------------------------------------------------------
def arch_router(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        if "ARCH_DETECT_SUCCESS" in answer:
            return "web_researcher"
        elif "ARCH_DETECT_FAILURE" in answer and _count_occurrences(state, "arch_detector") < 3:
            return "arch_detector"          # retry
        else:
            print("⚠️  Falling back to generic architecture (x86_64)")
            return "generic_researcher"
    except Exception as e:
        print(f"💥 Arch router error: {e}")
        return END


# B. RESEARCH ROUTER --------------------------------------------------
def research_router(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]

        # SUCCESS paths
        if "RESEARCH_COMPLETE" in answer or "GENERIC_RESEARCH_COMPLETE" in answer:
            return "installer"

        # FAILURE handling
        if any(tag in answer for tag in ["RESEARCH_FAILURE", "GENERIC_RESEARCH_FAILURE"]):
            retries = _count_occurrences(state, state["step_name"][-1])
            if retries < 3:
                return state["step_name"][-1]   # retry same researcher
            elif state["step_name"][-1] == "web_researcher":
                return "generic_researcher"     # switch to generic fallback
            else:
                return END                      # give up after generic fails

        # Unknown → retry primary researcher
        return "web_researcher"
    except Exception as e:
        print(f"💥 Research router error: {e}")
        return END


# C. INSTALL ROUTER ---------------------------------------------------
def install_router(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]

        if "INSTALL_SUCCESS" in answer:
            return "verifier"

        if "ALT_INSTALL_SUCCESS" in answer:
            return "verifier"

        # Failure handling
        if any(tag in answer for tag in ["INSTALL_FAILURE", "ALT_INSTALL_FAILURE"]):
            step = state["step_name"][-1]
            retries = _count_occurrences(state, step)
            if retries < 3:
                return step                     # retry same installer
            elif step == "installer":
                return "alt_installer"          # fallback to alt installer
            else:
                return END

        return "installer"
    except Exception as e:
        print(f"💥 Install router error: {e}")
        return END


# D. VERIFY ROUTER ----------------------------------------------------
def verify_router(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]

        if "VERIFY_SUCCESS" in answer:
            return "logger_agent"

        if "VERIFY_FAILURE" in answer:
            retries = _count_occurrences(state, "verifier")
            if retries < 3:
                return "verifier"               # retry verification
            else:
                print("❌ Verification failed after 3 attempts.")
                return END

        return "verifier"
    except Exception as e:
        print(f"💥 Verify router error: {e}")
        return END


# E. LOGGER ROUTER (trivial) -----------------------------------------
def logger_router(state: WorkflowState) -> str:
    try:
        return END
    except Exception as e:
        print(f"💥 Logger router error: {e}")
        return END


# 5️⃣ ==========  EDGE DEFINITIONS  ====================================

workflow.add_edge(START, "arch_detector")

workflow.add_conditional_edges(
    "arch_detector",
    arch_router,
    {
        "web_researcher":      "web_researcher",
        "generic_researcher":  "generic_researcher",
        END:                   END,
        "arch_detector":       "arch_detector",   # implicit retry
    }
)

workflow.add_conditional_edges(
    "web_researcher",
    research_router,
    {
        "installer":           "installer",
        "web_researcher":      "web_researcher",
        "generic_researcher":  "generic_researcher",
        END:                   END,
    }
)

workflow.add_conditional_edges(
    "generic_researcher",
    research_router,
    {
        "installer":           "installer",
        "generic_researcher":  "generic_researcher",
        END:                   END,
    }
)

workflow.add_conditional_edges(
    "installer",
    install_router,
    {
        "verifier":            "verifier",
        "installer":           "installer",
        "alt_installer":       "alt_installer",
        END:                   END,
    }
)

workflow.add_conditional_edges(
    "alt_installer",
    install_router,
    {
        "verifier":            "verifier",
        "alt_installer":       "alt_installer",
        END:                   END,
    }
)

workflow.add_conditional_edges(
    "verifier",
    verify_router,
    {
        "logger_agent":        "logger_agent",
        "verifier":            "verifier",
        END:                   END,
    }
)

workflow.add_conditional_edges(
    "logger_agent",
    logger_router,
    {
        END: END
    }
)

# 6️⃣ ==========  COMPILE WORKFLOW  ====================================
app = workflow.compile()

# The workflow is now ready to be executed:
# `app.invoke(input_state_dict)` where input_state_dict respects WorkflowState.