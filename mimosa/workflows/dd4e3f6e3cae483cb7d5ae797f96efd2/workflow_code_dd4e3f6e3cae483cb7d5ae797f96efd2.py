# ================================================================
# LangGraph – SmolAgent workflow: “Find and install MZmine on macOS”
# ================================================================
#
# PURPOSE
# -------
# 1. Locate official MZmine download page (macOS build).
# 2. Extract concrete download URL + installation instructions.
# 3. Download & install on macOS via shell ( curl / brew / dmg ).
# 4. Verify installation succeeded.
# 5. Produce final human-readable report.
#
# ARCHITECTURE
# ------------
#     ┌────────────┐     ┌──────────────┐     ┌──────────┐
#     │ 1. SEARCH  │ --> │ 2. EXTRACT   │ --> │ 3. INSTALL│
#     └────────────┘     └──────────────┘     └──────────┘
#                                                │
#                                                ▼
#                                         ┌────────────┐
#                                         │4. VERIFY   │
#                                         └────────────┘
#                                                │
#                                                ▼
#                                         ┌────────────┐
#                                         │5. REPORT   │
#                                         └────────────┘
#
# Each numbered box is a SmolAgent node with ONE atomic task.
# Robust routers guard every hop (max-3 retries, emergency END).
#
# ================================================================

from langgraph.graph import StateGraph, START, END

# --------- 1. SEARCH AGENT ----------------------------------------------------
search_prompt = """
You are a focused web-search agent.

GOAL
• Find the official website or authoritative hosting page for the “MZmine” project.
• Specifically identify the macOS download section (DMG, ZIP, JAR, or Homebrew).

DELIVERABLE PROTOCOL
If you successfully locate at least one valid macOS download link:
return final_answer("SEARCH_COMPLETE: [Paste the page URL(s) plus a concise description of where the macOS download link/button is located.]")

If you could not find such a page after thorough searching:
return final_answer("SEARCH_FAILURE: [Detail all queries tried, pages visited, and why none satisfied the requirement.]")

If you hit a technical problem (tool error, captcha, etc.):
return final_answer("GIVE_UP: [Describe the technical issue blocking you.]")
"""
search_agent   = SmolAgentFactory(search_prompt, BROWSER_TOOL_TOOLS)

# --------- 2. EXTRACT AGENT ---------------------------------------------------
extract_prompt = """
You are an information extraction agent.

INPUT
• You will receive the web page URL(s) where MZmine for macOS is available.

TASK
• Navigate to the page(s) and pull out:
  1. Direct macOS download URL (DMG/ZIP/JAR or brew formula).
  2. Exact version number.
  3. Any textual installation instructions aimed at macOS.

DELIVERABLE PROTOCOL
If extraction succeeded:
return final_answer("EXTRACT_COMPLETE: [Provide the direct download URL, version, and quoted install instructions.]")

If page lacks clear macOS assets / information insufficient:
return final_answer("EXTRACT_FAILURE: [Explain what content was missing and suggest alternative extraction ideas.]")

If you face technical issues:
return final_answer("GIVE_UP: [State the error encountered.]")
"""
extract_agent  = SmolAgentFactory(extract_prompt, BROWSER_TOOL_TOOLS)

# --------- 3. INSTALL AGENT ---------------------------------------------------
install_prompt = """
You are a macOS installation agent with shell access (non-interactive).

INPUT
• A direct download link or Homebrew formula for MZmine, plus instructions.

TASK
1. Download the installer (curl or brew).
2. Install to /Applications or appropriate location.
3. Handle any dependencies (e.g., Java).
4. Do *not* prompt for passwords (assume user can sudo if required).

DELIVERABLE PROTOCOL
If installation completes without errors:
return final_answer("INSTALL_COMPLETE: [Summarise commands executed and install path]")

If installation fails (missing dependencies, checksum errors, etc.):
return final_answer("INSTALL_FAILURE: [Provide error logs and suggest fixes]")

If shell tool itself errors (no permission, environment issue):
return final_answer("GIVE_UP: [Describe technical blocker]")
"""
install_agent  = SmolAgentFactory(install_prompt, SHELL_TOOL_TOOLS)

# --------- 4. VERIFY AGENT ----------------------------------------------------
verify_prompt = """
You are a verification agent.

TASK
• Run `mzmine --help` or open the installed app headless to confirm installation.
• Check exit code == 0 or GUI bundle exists in /Applications.

DELIVERABLE PROTOCOL
If verification succeeds:
return final_answer("VERIFY_COMPLETE: [Provide command output or file existence confirmation]")

If verification fails (command not found, crash):
return final_answer("VERIFY_FAILURE: [Provide error output and troubleshooting hints]")

If shell tool fails to execute:
return final_answer("GIVE_UP: [Explain technical issue]")
"""
verify_agent   = SmolAgentFactory(verify_prompt, SHELL_TOOL_TOOLS)

# --------- 5. REPORT AGENT ----------------------------------------------------
# Pure formatting agent – no external tools
report_prompt = """
You are a summarisation/reporting agent.

INPUT
• All previous agent answers including verification outcome.

TASK
• Compile a concise plain-language report containing:
  - Download source and version
  - Commands executed
  - Verification result
  - Any caveats

DELIVERABLE PROTOCOL
If sufficient data to make final report:
return final_answer("REPORT_COMPLETE: [Insert full installation report]")

If upstream steps failed:
return final_answer("REPORT_FAILURE: [Explain which step failed and recommend next steps]")
"""
report_agent   = SmolAgentFactory(report_prompt, [])  # no tools needed

# -------------- GENERIC ROUTER ----------------------------------------------
def make_router(success_token: str, failure_token: str, next_node: str):
    """
    Factory returning a router function tailored for each hop.
    Checks latest answer for SUCCESS / FAILURE / GIVE_UP markers.
    Provides retry (≤3) and emergency END paths.
    """
    def router(state: WorkflowState) -> str:
        try:
            answers     = state.get("answers", [])
            steps       = state.get("step_name", [])
            last_answer = answers[-1] if answers else ""
            current     = steps[-1]  if steps else "UNKNOWN"
            retry_count = steps.count(current)

            # ------------- SUCCESS PATH -----------------
            if success_token in last_answer:
                return next_node

            # ------------- FAILURE  ---------------------
            if failure_token in last_answer:
                if retry_count < 3:
                    # Retry same node
                    return current
                else:
                    # escalate to END (irrecoverable)
                    return "emergency_end"

            # ------------- TECH ERROR -------------------
            if "GIVE_UP" in last_answer:
                return "emergency_end"

            # ------------- UNEXPECTED OUTPUT ------------
            # Treat as soft-failure with retry
            if retry_count < 3:
                return current
            else:
                return "emergency_end"

        except Exception as e:
            print(f"🚨 Router exception ({current}): {e}")
            return "emergency_end"
    return router

# -------------------- WORKFLOW BUILD -----------------------------------------
workflow = StateGraph(WorkflowState)

# Add agent nodes
workflow.add_node("search",  WorkflowNodeFactory.create_agent_node(search_agent))
workflow.add_node("extract", WorkflowNodeFactory.create_agent_node(extract_agent))
workflow.add_node("install", WorkflowNodeFactory.create_agent_node(install_agent))
workflow.add_node("verify",  WorkflowNodeFactory.create_agent_node(verify_agent))
workflow.add_node("report",  WorkflowNodeFactory.create_agent_node(report_agent))

# --------- EDGES & CONDITIONALS ----------
# START -> SEARCH
workflow.add_edge(START, "search")

# SEARCH routing
workflow.add_conditional_edges(
    "search",
    make_router("SEARCH_COMPLETE", "SEARCH_FAILURE", "extract"),
    {
        "search":  "search",   # retry
        "extract": "extract",
        "emergency_end": END
    }
)

# EXTRACT routing
workflow.add_conditional_edges(
    "extract",
    make_router("EXTRACT_COMPLETE", "EXTRACT_FAILURE", "install"),
    {
        "extract": "extract",  # retry
        "install": "install",
        "emergency_end": END
    }
)

# INSTALL routing
workflow.add_conditional_edges(
    "install",
    make_router("INSTALL_COMPLETE", "INSTALL_FAILURE", "verify"),
    {
        "install": "install",  # retry
        "verify":  "verify",
        "emergency_end": END
    }
)

# VERIFY routing
workflow.add_conditional_edges(
    "verify",
    make_router("VERIFY_COMPLETE", "VERIFY_FAILURE", "report"),
    {
        "verify":  "verify",   # retry
        "report":  "report",
        "emergency_end": END
    }
)

# REPORT routing (final)
def final_router(state: WorkflowState) -> str:
    """
    Ends if report returned REPORT_COMPLETE, otherwise give up.
    """
    try:
        last = state.get("answers", [])[-1]
        if "REPORT_COMPLETE" in last:
            return END
        # No retry for report; nothing more we can do
        return END
    except Exception as e:
        print(f"💥 Final router error: {e}")
        return END

workflow.add_conditional_edges(
    "report",
    final_router,
    {END: END}
)

# --------------- COMPILE WORKFLOW -----------------
app = workflow.compile()