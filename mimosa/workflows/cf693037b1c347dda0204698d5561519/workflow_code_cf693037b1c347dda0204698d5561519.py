# ================================================================
# LangGraph – SmolAgent workflow : “Install MZmine on macOS”
# ================================================================

# NOTE
# ----
# – State schema, tool packages, SmolAgentFactory and
#   WorkflowNodeFactory are **already** available in the interpreter
#   context – we just use them.
# – This code follows all decomposition / error-handling guidelines:
#     • 5 atomic agents
#     • 4 specialised routing blocks with ≥ 2 fall-back paths each
#     • attempt counters, emergency END, extensive logging
# ================================================================

from langgraph.graph import StateGraph, START, END

# ------------------------------------------------
# 1️⃣  Agent-specific prompts
# ------------------------------------------------
# -- Each prompt ends with mandatory SUCCESS / FAILURE / GIVE_UP tags
# -- Agents have single, atomic responsibility
# ------------------------------------------------

# --- 1. WEB RESEARCH AGENT --------------------------------------
instruct_research = """
You are a focused web-research agent.

GOAL
• Locate the OFFICIAL download link for the latest “MZmine” release for macOS.
• Capture SHA-256 / SHA-512 checksums if provided by the publisher.
• Output ONLY factual data you verified directly on authoritative pages
  (GitHub releases, official website, etc.).

COMPLETION PROTOCOL
SUCCESS  -> End with the tag  RESEARCH_COMPLETE
           Provide JSON with keys: {"url": "...", "checksum": "...", "version": "..."}
FAILURE  -> End with tag       RESEARCH_FAILURE
           Explain what sources you checked and why data is missing.
ERROR    -> End with tag       GIVE_UP
           Describe exact technical problem encountered.
"""
agent_research = SmolAgentFactory(instruct_research, BROWSER_TOOLS)

# --- 2. DOWNLOADER AGENT ----------------------------------------
instruct_download = """
You are a download automation agent.

INPUT
• Expect a macOS download URL for the latest MZmine release.

TASK (atomic)
• Download the file to /tmp/mzmine_installer (use curl or wget).
• Report the FULL local file path and original file name.

COMPLETION PROTOCOL
SUCCESS  -> DOWNLOAD_COMPLETE: /tmp/....
FAILURE  -> DOWNLOAD_FAILURE:  [explanation]
ERROR    -> GIVE_UP:           [technical-error]
"""
agent_download = SmolAgentFactory(instruct_download, SHELL_TOOLS + BROWSER_TOOLS)

# --- 3. VERIFIER AGENT ------------------------------------------
instruct_verify = """
You are a checksum verification agent.

INPUT
• A file path on the local filesystem and its expected SHA hash.

TASK
• Compute SHA-256 and/or SHA-512 of the file.
• Compare with expected value.
• Output VERDICT: MATCH or MISMATCH.

COMPLETION PROTOCOL
SUCCESS  -> VERIFY_COMPLETE: MATCH
FAILURE  -> VERIFY_FAILURE:  MISMATCH
ERROR    -> GIVE_UP:         [technical-error]
"""
agent_verify = SmolAgentFactory(instruct_verify, SHELL_TOOLS)

# --- 4. INSTALLER AGENT -----------------------------------------
instruct_install = """
You are an installation automation agent for macOS.

INPUT
• Path to a downloaded MZmine dmg/zip/jar.

TASK
• If dmg:   hdiutil attach → copy *.app to /Applications → hdiutil detach
• If zip:   unzip → move *.app to /Applications
• If jar:   ensure java, then move to /Applications/MZmine.jar

COMPLETION PROTOCOL
SUCCESS  -> INSTALLATION_COMPLETE
FAILURE  -> INSTALLATION_FAILURE: [reason]
ERROR    -> GIVE_UP:              [technical-error]
"""
agent_install = SmolAgentFactory(instruct_install, SHELL_TOOLS)

# --- 5. SUMMARY AGENT -------------------------------------------
instruct_summary = """
You are a summarisation agent.

TASK
• Produce a concise installation report that includes:
  – Download URL and file name
  – Verified checksum
  – Installation destination
  – Command sample to launch MZmine
• Output as CSV with headers: step,result,detail

COMPLETION PROTOCOL
SUCCESS -> SUMMARY_COMPLETE
ERROR   -> GIVE_UP
"""
agent_summary = SmolAgentFactory(instruct_summary, CSV_TOOLS)

# ------------------------------------------------
# 2️⃣  Workflow graph & helper utilities
# ------------------------------------------------
workflow = StateGraph(WorkflowState)

# Helper: how many times a step (or its retry variants) occurred
def _attempts(state: WorkflowState, key: str) -> int:
    names = state.get("step_name", [])
    return sum(1 for n in names if n.startswith(key))

# ------------------------------------------------
# 3️⃣  Routing functions with robust fallbacks
# ------------------------------------------------
def route_after_research(state: WorkflowState) -> str:
    print("🔀 Router-Research")
    try:
        answer = state["answers"][-1] if state.get("answers") else ""
        attempts = _attempts(state, "research")
        if "RESEARCH_COMPLETE" in answer:
            return "downloader"
        if "GIVE_UP" in answer:
            return END
        # FAILURE or unknown → retry / give-up escalation
        if attempts < 3:
            return "research_retry"
        return END
    except Exception as e:
        print(f"⚠️ Router-Research error: {e}")
        return END

def route_after_download(state: WorkflowState) -> str:
    print("🔀 Router-Download")
    try:
        answer = state["answers"][-1] if state.get("answers") else ""
        attempts = _attempts(state, "downloader")
        if "DOWNLOAD_COMPLETE" in answer:
            return "verifier"
        if "GIVE_UP" in answer:
            return END
        if "DOWNLOAD_FAILURE" in answer and attempts < 3:
            # Maybe link wrong → bounce to research again
            return "research_retry"
        if attempts < 3:
            return "downloader_retry"
        return END
    except Exception as e:
        print(f"⚠️ Router-Download error: {e}")
        return END

def route_after_verify(state: WorkflowState) -> str:
    print("🔀 Router-Verify")
    try:
        answer = state["answers"][-1] if state.get("answers") else ""
        attempts = _attempts(state, "verifier")
        if "VERIFY_COMPLETE" in answer and "MATCH" in answer:
            return "installer"
        if "GIVE_UP" in answer:
            return END
        # Hash mismatch → re-download (fallback) or abort
        if attempts < 2:
            return "downloader_retry"
        return END
    except Exception as e:
        print(f"⚠️ Router-Verify error: {e}")
        return END

def route_after_install(state: WorkflowState) -> str:
    print("🔀 Router-Install")
    try:
        answer = state["answers"][-1] if state.get("answers") else ""
        attempts = _attempts(state, "installer")
        if "INSTALLATION_COMPLETE" in answer:
            return "summary"
        if "GIVE_UP" in answer:
            return END
        if attempts < 2:
            return "installer_retry"
        return END
    except Exception as e:
        print(f"⚠️ Router-Install error: {e}")
        return END

def route_after_summary(state: WorkflowState) -> str:
    print("🔀 Router-Summary")
    try:
        answer = state["answers"][-1] if state.get("answers") else ""
        if "SUMMARY_COMPLETE" in answer:
            return END
        return END  # Any other outcome → END
    except Exception as e:
        print(f"⚠️ Router-Summary error: {e}")
        return END

# ------------------------------------------------
# 4️⃣  Add nodes
# ------------------------------------------------
workflow.add_node("research",        WorkflowNodeFactory.create_agent_node(agent_research))
workflow.add_node("research_retry",  WorkflowNodeFactory.create_agent_node(agent_research))
workflow.add_node("downloader",      WorkflowNodeFactory.create_agent_node(agent_download))
workflow.add_node("downloader_retry",WorkflowNodeFactory.create_agent_node(agent_download))
workflow.add_node("verifier",        WorkflowNodeFactory.create_agent_node(agent_verify))
workflow.add_node("verifier_retry",  WorkflowNodeFactory.create_agent_node(agent_verify))
workflow.add_node("installer",       WorkflowNodeFactory.create_agent_node(agent_install))
workflow.add_node("installer_retry", WorkflowNodeFactory.create_agent_node(agent_install))
workflow.add_node("summary",         WorkflowNodeFactory.create_agent_node(agent_summary))

# ------------------------------------------------
# 5️⃣  Edges with conditional routing & fallbacks
# ------------------------------------------------
workflow.add_edge(START, "research")

# Research routing
workflow.add_conditional_edges(
    "research",
    route_after_research,
    {
        "downloader": "downloader",
        "research_retry": "research_retry",
        END: END
    }
)
workflow.add_conditional_edges(
    "research_retry",
    route_after_research,
    {
        "downloader": "downloader",
        "research_retry": "research_retry",
        END: END
    }
)

# Download routing
workflow.add_conditional_edges(
    "downloader",
    route_after_download,
    {
        "verifier": "verifier",
        "research_retry": "research_retry",
        "downloader_retry": "downloader_retry",
        END: END
    }
)
workflow.add_conditional_edges(
    "downloader_retry",
    route_after_download,
    {
        "verifier": "verifier",
        "research_retry": "research_retry",
        "downloader_retry": "downloader_retry",
        END: END
    }
)

# Verify routing
workflow.add_conditional_edges(
    "verifier",
    route_after_verify,
    {
        "installer": "installer",
        "downloader_retry": "downloader_retry",
        END: END
    }
)
workflow.add_conditional_edges(
    "verifier_retry",
    route_after_verify,
    {
        "installer": "installer",
        "downloader_retry": "downloader_retry",
        END: END
    }
)

# Install routing
workflow.add_conditional_edges(
    "installer",
    route_after_install,
    {
        "summary": "summary",
        "installer_retry": "installer_retry",
        END: END
    }
)
workflow.add_conditional_edges(
    "installer_retry",
    route_after_install,
    {
        "summary": "summary",
        "installer_retry": "installer_retry",
        END: END
    }
)

# Summary routing
workflow.add_conditional_edges(
    "summary",
    route_after_summary,
    {
        END: END
    }
)

# ------------------------------------------------
# 6️⃣  Compile graph
# ------------------------------------------------
app = workflow.compile()