# LangGraph-SmolAgent workflow
# Goal: search and install llama.cpp for the current OS / architecture

# ------------------------------------------------------------------
# Context objects (already available – DO NOT REDECLARE):
#   ‑ WorkflowState  (state schema)
#   ‑ SmolAgentFactory
#   ‑ WorkflowNodeFactory
#   ‑ WEB_BROWSER_MCP_TOOLS
#   ‑ BASH_COMMAND_MCP_TOOLS
# ------------------------------------------------------------------

from langgraph.graph import StateGraph, START, END

# 1 ───────────────────────── PROMPTS ───────────────────────────────
instruct_os_detect = """
You are a system analysis agent with bash access.

TASK
1. Run commands (uname ‑a / ‑m / ‑s, cat /etc/os-release …) to identify:
   • OS name + version
   • CPU architecture
   • Distribution (if Linux)
2. Summarise the findings in ONE short sentence.

COMPLETION PROTOCOL
SUCCESS  → OS_DETECTED: <summary>
FAILURE  → OS_DETECT_FAILURE: <reason>
ERROR    → GIVE_UP: <technical error>
"""

instruct_web_research = """
You are a web-research agent.

TASK
Locate reliable, up-to-date instructions to build and install *llama.cpp* that match
any Linux/Unix architecture (include architecture-specific flags when needed).

Return:
• Dependencies to install
• Clone / download commands
• Build commands (make …)
• Post-build verification commands
• Links / sources

COMPLETION PROTOCOL
SUCCESS  → RESEARCH_COMPLETE: <detailed, ordered instructions + links>
FAILURE  → RESEARCH_FAILURE: <reason & alternative queries>
ERROR    → GIVE_UP: <technical error>
"""

instruct_install = """
You are an installation agent with bash privileges.

INPUT
You will receive step-by-step instructions from the research phase.

TASK
Execute the commands in order, handle dependencies, compile llama.cpp, and place the
binary in a location on PATH.

COMPLETION PROTOCOL
SUCCESS  → INSTALL_SUCCESS: <summary, binary location>
FAILURE  → INSTALL_FAILURE: <reason, key error excerpts>
ERROR    → GIVE_UP: <technical error>
"""

instruct_verify = """
You are a verification agent.

TASK
Run the built binary (e.g. ./main --help) or llama-cli --help.
If help text is displayed and exit-code is 0, deem it successful.

COMPLETION PROTOCOL
SUCCESS  → VERIFY_SUCCESS: <snippet of help output>
FAILURE  → VERIFY_FAILURE: <reason>
ERROR    → GIVE_UP: <technical error>
"""

# 2 ────────────────────── AGENT DEFINITIONS ───────────────────────
smolagent_os      = SmolAgentFactory("os_detector",   instruct_os_detect,  BASH_COMMAND_MCP_TOOLS)
smolagent_web     = SmolAgentFactory("web_researcher",instruct_web_research,WEB_BROWSER_MCP_TOOLS)
smolagent_install = SmolAgentFactory("installer",     instruct_install,    BASH_COMMAND_MCP_TOOLS)
smolagent_verify  = SmolAgentFactory("verifier",      instruct_verify,     BASH_COMMAND_MCP_TOOLS)

# 3 ───────────────────────── WORKFLOW ─────────────────────────────
workflow = StateGraph(WorkflowState)

# ── Nodes
workflow.add_node("os_detect",   WorkflowNodeFactory.create_agent_node(smolagent_os))
workflow.add_node("web_research",WorkflowNodeFactory.create_agent_node(smolagent_web))
workflow.add_node("install",     WorkflowNodeFactory.create_agent_node(smolagent_install))
workflow.add_node("verify",      WorkflowNodeFactory.create_agent_node(smolagent_verify))

# ── Routing helpers
def route_after_os_detect(state: WorkflowState) -> str:
    try:
        last   = state.get("answers", [""])[-1]
        tries  = state["step_name"].count("os_detect")
        print(f"[Route-OS] try {tries} → {last[:60]}")
        if "OS_DETECTED" in last:
            return "web_research"
        if "GIVE_UP" in last or tries >= 2:
            return "end"
        return "os_detect"
    except Exception as e:
        print(f"[Route-OS] exception: {e}")
        return "end"

def route_after_web_research(state: WorkflowState) -> str:
    try:
        last   = state.get("answers", [""])[-1]
        tries  = state["step_name"].count("web_research")
        print(f"[Route-Research] try {tries} → {last[:60]}")
        if "RESEARCH_COMPLETE" in last:
            return "install"
        if "GIVE_UP" in last or tries >= 3:
            return "end"
        return "web_research"      # retry
    except Exception as e:
        print(f"[Route-Research] exception: {e}")
        return "end"

def route_after_install(state: WorkflowState) -> str:
    try:
        last   = state.get("answers", [""])[-1]
        tries  = state["step_name"].count("install")
        print(f"[Route-Install] try {tries} → {last[:60]}")
        if "INSTALL_SUCCESS" in last:
            return "verify"
        if "GIVE_UP" in last or tries >= 2:
            return "end"
        return "web_research"      # go back – maybe need better instructions
    except Exception as e:
        print(f"[Route-Install] exception: {e}")
        return "end"

def route_after_verify(state: WorkflowState) -> str:
    try:
        last   = state.get("answers", [""])[-1]
        tries  = state["step_name"].count("verify")
        print(f"[Route-Verify] try {tries} → {last[:60]}")
        if "VERIFY_SUCCESS" in last:
            return "end"
        if "GIVE_UP" in last or tries >= 2:
            return "end"
        return "install"           # re-install / fix build
    except Exception as e:
        print(f"[Route-Verify] exception: {e}")
        return "end"

# ── Edges
workflow.add_edge(START, "os_detect")

workflow.add_conditional_edges(
    "os_detect",
    route_after_os_detect,
    {
        "web_research": "web_research",
        "os_detect":    "os_detect",
        "end":          END
    }
)

workflow.add_conditional_edges(
    "web_research",
    route_after_web_research,
    {
        "install":      "install",
        "web_research": "web_research",
        "end":          END
    }
)

workflow.add_conditional_edges(
    "install",
    route_after_install,
    {
        "verify":       "verify",
        "web_research": "web_research",
        "end":          END
    }
)

workflow.add_conditional_edges(
    "verify",
    route_after_verify,
    {
        "install":      "install",
        "end":          END
    }
)

# 4 ───────────────────────── COMPILE ──────────────────────────────
app = workflow.compile()