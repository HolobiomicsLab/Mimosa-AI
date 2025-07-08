# =========================================================
# LangGraph Workflow :  “Install MZmine on macOS”
# =========================================================
#
# GOAL
# ----
# Generate a bullet-proof, multi-agent LangGraph workflow that
# researches, prepares, installs and verifies the MZmine software
# on macOS while providing robust fall-backs and detailed logs.
#
# AGENT DECOMPOSITION (minimum 5 atomic agents)
# --------------------------------------------
# 1. info_gather      – find official download + requirements      (BROWSER_TOOL_TOOLS)
# 2. download_planner – craft exact shell commands for download    (SHELL_TOOL_TOOLS)
# 3. installer        – craft / run install & set-up commands      (SHELL_TOOL_TOOLS)
# 4. verifier         – check installation actually works          (SHELL_TOOL_TOOLS)
# 5. summarizer       – produce final consolidated report          (CSV_TOOL_TOOLS)  (any toolbox is fine, CSV chosen)
#
# Each agent finishes with one of three KEYWORDS so routers can
# decide next steps:
#    • <STEP>_COMPLETE      – success
#    • <STEP>_FAILURE       – task failed but recoverable
#    • GIVE_UP              – unrecoverable / technical error
#
# =========================================================

from langgraph.graph import StateGraph, START, END

# ------------ 1. PROMPTS -------------------------------------------------

info_prompt = """
You are an expert web-researcher.

TASK
- Using browser tools, locate the MOST RECENT stable release of MZmine for macOS.
- Collect: version number, dmg/zip download URL, SHA checksum if available,
  Java / system requirements, and any official install notes.

UPON COMPLETION
SUCCESS  ➜ final_answer("INFO_COMPLETE: {<json with version, url, checksum, requirements, notes>}")
FAILURE  ➜ final_answer("INFO_FAILURE: <explain exactly what you tried / why no info>")
ERROR    ➜ final_answer("GIVE_UP: <technical error description>")
STRICTLY end with only one of the three keywords above.
"""

download_prompt = """
You are a shell-planning agent.

CONTEXT
- Previous step already identified the official macOS download URL for MZmine.

TASK
- Generate ONE shell command sequence (bash) that:
  1. Creates ~/Downloads/mzmine
  2. Downloads the dmg/zip to that folder (use curl -L or wget)
  3. Verifies checksum if provided (shasum -a 256) else skip
Return commands only.

UPON COMPLETION
SUCCESS  ➜ final_answer("DOWNLOAD_COMPLETE: <bash commands shown line by line>")
FAILURE  ➜ final_answer("DOWNLOAD_FAILURE: <what went wrong / alternate ideas>")
ERROR    ➜ final_answer("GIVE_UP: <technical error>")
"""

install_prompt = """
You are an installation agent.

CONTEXT
- The downloaded file is in ~/Downloads/mzmine.

TASK
- Write exact shell commands to install MZmine for the CURRENT user:
  • If dmg: hdiutil attach → copy .app to /Applications → hdiutil detach
  • If zip: unzip then move .app to /Applications
  • Ensure correct permissions
  • If Java required, check 'java -version' and guide user to install brew java if absent

UPON COMPLETION
SUCCESS  ➜ final_answer("INSTALL_COMPLETE: <bash install commands>")
FAILURE  ➜ final_answer("INSTALL_FAILURE: <issue and proposal>")
ERROR    ➜ final_answer("GIVE_UP: <technical error>")
"""

verify_prompt = """
You are a verification agent.

TASK
- Run 'open -a MZmine' OR execute the mzmine executable via shell.
- Capture exit code and any stderr/stdout.
- Decide:
    * Works and UI launches? success.
    * Fails to open? failure.

UPON COMPLETION
SUCCESS  ➜ final_answer("VERIFY_SUCCESS: <output log summary>")
FAILURE  ➜ final_answer("VERIFY_FAILURE: <output log + hypothesis>")
ERROR    ➜ final_answer("GIVE_UP: <technical error>")
"""

summary_prompt = """
You are a summarization agent.

TASK
- Produce a FINAL report for human user including:
    • Version installed
    • Exact download location
    • Checksums verified?
    • Installation steps run
    • Verification outcome
- If ANY step earlier ended in *_FAILURE, give clear troubleshooting advice.
- If a step ended in GIVE_UP, apologise and explain next manual actions.

UPON COMPLETION (always end)
final_answer("SUMMARY_COMPLETE")
"""

# ------------ 2. AGENT DECLARATIONS --------------------------------------

info_agent       = SmolAgentFactory(info_prompt,      BROWSER_TOOL_TOOLS)
download_agent   = SmolAgentFactory(download_prompt,  SHELL_TOOL_TOOLS)
install_agent    = SmolAgentFactory(install_prompt,   SHELL_TOOL_TOOLS)
verify_agent     = SmolAgentFactory(verify_prompt,    SHELL_TOOL_TOOLS)
summary_agent    = SmolAgentFactory(summary_prompt,   CSV_TOOL_TOOLS)

# ------------ 3. ROUTING HELPERS -----------------------------------------

MAX_RETRIES = 3    # hard limit per step


def _attempts(state: WorkflowState, step_name: str) -> int:
    """
    Utility: count how many times <step_name> was already executed.
    """
    try:
        return [n for n in state.get("step_name", []) if n == step_name].__len__()
    except Exception:
        return 0


# ---------- Router after INFO_GATHER ----------------
def route_info(state: WorkflowState) -> str:
    print("📡 ROUTER (info_gather)")
    try:
        answers = state.get("answers", [])
        if not answers:
            print("⚠️  No answer yet → retry info_gather")
            return "info_gather"
        last = answers[-1]

        if "INFO_COMPLETE" in last:
            return "download_planner"

        # keyword GIVE_UP leads directly to summary (fatal)
        if "GIVE_UP" in last:
            return "summarizer"

        # failure but recoverable?
        if "INFO_FAILURE" in last:
            if _attempts(state, "info_gather") < MAX_RETRIES:
                print("🔄 Re-attempting info gathering")
                return "info_gather"
            else:
                print("❌ Max info attempts hit – giving up")
                return "summarizer"

        # Unexpected message – safe fallback
        print("❓ Unknown response – fallback to retry")
        return "info_gather"

    except Exception as e:
        print(f"💥 Routing exception {e}")
        return "summarizer"


# ---------- Router after DOWNLOAD -------------------
def route_download(state: WorkflowState) -> str:
    print("💾 ROUTER (download_planner)")
    try:
        answers = state.get("answers", [])
        last = answers[-1] if answers else ""

        if "DOWNLOAD_COMPLETE" in last:
            return "installer"

        if "GIVE_UP" in last:
            return "summarizer"

        if "DOWNLOAD_FAILURE" in last:
            if _attempts(state, "download_planner") < MAX_RETRIES:
                return "download_planner"
            else:
                return "summarizer"

        # if info insufficient, step back
        if "INSUFFICIENT_INFORMATION" in last:
            return "info_gather"

        return "download_planner"  # default retry

    except Exception as e:
        print(f"💥 Routing exception {e}")
        return "summarizer"


# ---------- Router after INSTALL --------------------
def route_install(state: WorkflowState) -> str:
    print("📦 ROUTER (installer)")
    try:
        answers = state.get("answers", [])
        last = answers[-1] if answers else ""

        if "INSTALL_COMPLETE" in last:
            return "verifier"

        if "GIVE_UP" in last:
            return "summarizer"

        if "INSTALL_FAILURE" in last:
            if _attempts(state, "installer") < MAX_RETRIES:
                return "installer"
            else:
                return "summarizer"

        return "installer"

    except Exception as e:
        print(f"💥 Routing exception {e}")
        return "summarizer"


# ---------- Router after VERIFY ---------------------
def route_verify(state: WorkflowState) -> str:
    print("🔍 ROUTER (verifier)")
    try:
        answers = state.get("answers", [])
        last = answers[-1] if answers else ""

        if "VERIFY_SUCCESS" in last:
            # happy path → summarizer
            return "summarizer"

        if "GIVE_UP" in last:
            return "summarizer"

        if "VERIFY_FAILURE" in last:
            # allow one reinstall attempt
            if _attempts(state, "verifier") < MAX_RETRIES:
                # maybe try reinstall then verify again
                return "installer"
            else:
                return "summarizer"

        return "verifier"

    except Exception as e:
        print(f"💥 Routing exception {e}")
        return "summarizer"


# summarizer always ends → END
def route_summary(state: WorkflowState) -> str:
    return END

# ------------ 4. WORKFLOW GRAPH ------------------------------------------

workflow = StateGraph(WorkflowState)

# ---- Nodes
workflow.add_node("info_gather",      WorkflowNodeFactory.create_agent_node(info_agent))
workflow.add_node("download_planner", WorkflowNodeFactory.create_agent_node(download_agent))
workflow.add_node("installer",        WorkflowNodeFactory.create_agent_node(install_agent))
workflow.add_node("verifier",         WorkflowNodeFactory.create_agent_node(verify_agent))
workflow.add_node("summarizer",       WorkflowNodeFactory.create_agent_node(summary_agent))

# ---- Edges & Conditional routing
workflow.add_edge(START, "info_gather")

workflow.add_conditional_edges("info_gather",      route_info,
                               {"download_planner": "download_planner",
                                "info_gather": "info_gather",
                                "summarizer": "summarizer"})

workflow.add_conditional_edges("download_planner", route_download,
                               {"installer": "installer",
                                "download_planner": "download_planner",
                                "info_gather": "info_gather",
                                "summarizer": "summarizer"})

workflow.add_conditional_edges("installer",        route_install,
                               {"verifier": "verifier",
                                "installer": "installer",
                                "summarizer": "summarizer"})

workflow.add_conditional_edges("verifier",         route_verify,
                               {"installer": "installer",
                                "verifier": "verifier",
                                "summarizer": "summarizer"})

workflow.add_conditional_edges("summarizer",       route_summary,
                               {END: END})

# ---- Compile
app = workflow.compile()

# =========================================================
# The compiled `app` object is now an executable LangGraph
# workflow that uses five SmolAgents with robust retry and
# fallback logic to install MZmine on macOS.
# =========================================================