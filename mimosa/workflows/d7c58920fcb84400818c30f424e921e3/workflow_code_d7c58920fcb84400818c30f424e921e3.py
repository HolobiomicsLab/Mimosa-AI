# =====================================================================
# LANGGRAPH ‑ SMOLAGENT WORKFLOW : “mzmind batch-mode deep-dive”
# =====================================================================
#
#  Goal :
#     1. Perform an exhaustive web investigation on “mzmind” batch /
#        command-line usage
#     2. Extract concrete CLI + config-file examples
#     3. (Optionally) validate syntax with a shell probe
#     4. Aggregate results in CSV
#     5. Deliver a final human-readable report
#
#  Architectural highlights
#     • 5 atomic agents  (+2 dedicated fall-backs)  = 7 total nodes
#     • Multiple retry & alternative paths (≥2 fall-backs)
#     • Robust routing with attempt counters & emergency END gate
# =====================================================================

from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------
# 1.  AGENT INSTRUCTIONS
# ---------------------------------------------------------------------

# --- Web searcher ----------------------------------------------------
instruct_web_search = """
You are WEB_SEARCHER, an internet research agent.

TASK
- Search for authoritative information about using "mzmind" in batch / command-line mode
- Collect URLs, page titles and short excerpts that mention:
  * command syntax (mzmind …)
  * batch configuration files (yaml / json / txt etc.)
  * usage examples, flags, options
- Store findings in your final answer as a numbered list

UPON COMPLETION
  If you found ≥3 distinct credible sources say exactly: SEARCH_COMPLETE
  If you failed or found <3 useful sources say: SEARCH_FAILURE
  On unexpected errors say: GIVE_UP
"""

# --- Deep/fallback searcher -----------------------------------------
instruct_alt_search = """
You are ALT_WEB_SEARCHER, a deep-dive research agent specialising in forums,
GitHub issues and niche blogs.

TASK
- Perform secondary searches for "mzmind batch", "mzmind --batch", "mzmind cli",
  "mzmind config file", "mzmind.exe /batch"
- Look into sources such as GitHub, StackOverflow, Gists, archived docs, etc.
- Output additional links and snippets NOT returned by the primary search.

UPON COMPLETION
  If you added ≥2 NEW sources say: ALT_SEARCH_COMPLETE
  If still insufficient info say: ALT_SEARCH_FAILURE
  On technical problems say: GIVE_UP
"""

# --- Content extractor ----------------------------------------------
instruct_extractor = """
You are CONTENT_EXTRACTOR, an agent that visits provided URLs and extracts
specific artefacts.

INPUT
- The previous agent's answer contains URLs and snippets.

TASK
- For each URL: visit the page, extract any CLI commands (mzmind …),
  flags, and configuration file blocks.
- Consolidate extracted artefacts into a clean markdown list.

UPON COMPLETION
  If at least one command and one config snippet extracted say: EXTRACTION_COMPLETE
  Else say: EXTRACTION_FAILURE
  For technical issues say: GIVE_UP
"""

# --- Shell validator -------------------------------------------------
instruct_validator = """
You are SHELL_VALIDATOR, an agent with terminal access (mock / limited).

TASK
- For each extracted command create a dry-run check:
     * If 'mzmind' exists, run 'mzmind --help' to verify availability.
     * Otherwise simulate with 'echo "<command>"'.
- Report whether syntax appears valid (exit code 0 etc.)
- Do NOT modify commands.

UPON COMPLETION
  If all commands checked produce no critical errors say: VALIDATION_COMPLETE
  If some commands problematic say: VALIDATION_FAILURE
  On tool errors say: GIVE_UP
"""

# --- CSV compiler ----------------------------------------------------
instruct_csv = """
You are CSV_COMPILER.

TASK
- Convert validated (or partially validated) examples into CSV with columns:
    use_case, cli_command, config_file_name, notes
- Use CSV tools to generate and store the file.

UPON COMPLETION
  If CSV created successfully say: CSV_COMPLETE
  Otherwise say: CSV_FAILURE
  On errors say: GIVE_UP
"""

# --- Final report formatter -----------------------------------------
instruct_report = """
You are REPORT_FORMATTER.

TASK
- Produce a final human-readable brief that includes:
   * Overview of what mzmind batch-mode is
   * The CLI examples (formatted)
   * The corresponding config files (formatted)
   * Validation results
   * Short CSV preview with link / path
- Be thorough but concise.

UPON COMPLETION
  If satisfied say: REPORT_COMPLETE
  If critical data missing say: REPORT_FAILURE
  On technical problems say: GIVE_UP
"""

# --- Optional “give-up” emitter -------------------------------------
instruct_give_up = """
You are GIVE_UP_AGENT.

TASK
- Explain clearly what went wrong in the workflow and suggest human
  interventions.

Always end with: WORKFLOW_ABORTED
"""

# ---------------------------------------------------------------------
# 2.  AGENT DECLARATION
# ---------------------------------------------------------------------

search_agent          = SmolAgentFactory(instruct_web_search,  BROWSER_TOOLS)
alt_search_agent      = SmolAgentFactory(instruct_alt_search, BROWSER_TOOLS)
extractor_agent       = SmolAgentFactory(instruct_extractor,  BROWSER_TOOLS)
validator_agent       = SmolAgentFactory(instruct_validator,  SHELL_TOOLS)
csv_agent             = SmolAgentFactory(instruct_csv,        CSV_TOOLS)
report_agent          = SmolAgentFactory(instruct_report,     [])              # no tools needed
giveup_agent          = SmolAgentFactory(instruct_give_up,    [])              # explanatory only

# ---------------------------------------------------------------------
# 3.  WORKFLOW GRAPH
# ---------------------------------------------------------------------

workflow = StateGraph(WorkflowState)

# ---- Add nodes ------------------------------------------------------
workflow.add_node("web_searcher",         WorkflowNodeFactory.create_agent_node(search_agent))
workflow.add_node("alt_web_searcher",     WorkflowNodeFactory.create_agent_node(alt_search_agent))
workflow.add_node("content_extractor",    WorkflowNodeFactory.create_agent_node(extractor_agent))
workflow.add_node("shell_validator",      WorkflowNodeFactory.create_agent_node(validator_agent))
workflow.add_node("csv_compiler",         WorkflowNodeFactory.create_agent_node(csv_agent))
workflow.add_node("report_formatter",     WorkflowNodeFactory.create_agent_node(report_agent))
workflow.add_node("give_up",              WorkflowNodeFactory.create_agent_node(giveup_agent))

# ---------------------------------------------------------------------
# 4.  ROUTING / EDGE LOGIC
# ---------------------------------------------------------------------

def safe_get(lst, idx, default=None):
    try:
        return lst[idx]
    except Exception:
        return default


# ---------- Router : generic with step-specific triggers -------------
def router(state: WorkflowState,
           success_phrases: list[str],
           failure_phrases: list[str],
           current_node: str,
           success_target: str,
           retry_target: str,
           alt_target: str):
    """
    Generic decision routine used by individual wrappers below.
    """
    try:
        ans_list     = state.get("answers", [])
        steps        = state.get("step_name", [])
        last_answer  = safe_get(ans_list, -1, "")
        retry_count  = steps.count(current_node)

        # --- success check ------------------------------------------
        if any(phrase in last_answer for phrase in success_phrases):
            return success_target

        # --- failure handling ---------------------------------------
        if any(phrase in last_answer for phrase in failure_phrases):
            if retry_count < 3:
                # simple retry (same node)
                return retry_target
            else:
                # after 3 attempts, escalate
                return alt_target

        # --- explicit GIVE_UP word ----------------------------------
        if "GIVE_UP" in last_answer:
            return "give_up"

        # --- defensive default --------------------------------------
        # Treat unknown response as failure and escalate
        if retry_count < 3:
            return retry_target
        else:
            return alt_target

    except Exception as e:
        print(f"💥 Router error at {current_node}: {e}")
        return "give_up"


# ---- Individual tiny wrappers (for readability) ---------------------

def search_router(state: WorkflowState) -> str:
    return router(state,
                  success_phrases=["SEARCH_COMPLETE"],
                  failure_phrases=["SEARCH_FAILURE"],
                  current_node="web_searcher",
                  success_target="content_extractor",
                  retry_target="web_searcher",
                  alt_target="alt_web_searcher")

def alt_search_router(state: WorkflowState) -> str:
    return router(state,
                  success_phrases=["ALT_SEARCH_COMPLETE"],
                  failure_phrases=["ALT_SEARCH_FAILURE"],
                  current_node="alt_web_searcher",
                  success_target="content_extractor",
                  retry_target="alt_web_searcher",
                  alt_target="give_up")

def extract_router(state: WorkflowState) -> str:
    return router(state,
                  success_phrases=["EXTRACTION_COMPLETE"],
                  failure_phrases=["EXTRACTION_FAILURE"],
                  current_node="content_extractor",
                  success_target="shell_validator",
                  retry_target="content_extractor",
                  alt_target="give_up")

def validate_router(state: WorkflowState) -> str:
    # If validation fails we will still allow progress to CSV after 3 tries
    route = router(state,
                   success_phrases=["VALIDATION_COMPLETE"],
                   failure_phrases=["VALIDATION_FAILURE"],
                   current_node="shell_validator",
                   success_target="csv_compiler",
                   retry_target="shell_validator",
                   alt_target="csv_compiler")
    return route

def csv_router(state: WorkflowState) -> str:
    return router(state,
                  success_phrases=["CSV_COMPLETE"],
                  failure_phrases=["CSV_FAILURE"],
                  current_node="csv_compiler",
                  success_target="report_formatter",
                  retry_target="csv_compiler",
                  alt_target="give_up")

def report_router(state: WorkflowState) -> str:
    route = router(state,
                   success_phrases=["REPORT_COMPLETE"],
                   failure_phrases=["REPORT_FAILURE"],
                   current_node="report_formatter",
                   success_target=END,
                   retry_target="report_formatter",
                   alt_target="give_up")
    # If report passes END else continues / escalates
    return route

# ---------------------------------------------------------------------
# 5.  EDGE DEFINITIONS
# ---------------------------------------------------------------------

workflow.add_edge(START, "web_searcher")

workflow.add_conditional_edges("web_searcher",
                               search_router,
                               {
                                   "content_extractor":  "content_extractor",
                                   "web_searcher":       "web_searcher",       # retry
                                   "alt_web_searcher":   "alt_web_searcher",
                                   "give_up":            "give_up"
                               })

workflow.add_conditional_edges("alt_web_searcher",
                               alt_search_router,
                               {
                                   "content_extractor":  "content_extractor",
                                   "alt_web_searcher":   "alt_web_searcher",   # retry
                                   "give_up":            "give_up"
                               })

workflow.add_conditional_edges("content_extractor",
                               extract_router,
                               {
                                   "shell_validator":    "shell_validator",
                                   "content_extractor":  "content_extractor", # retry
                                   "give_up":            "give_up"
                               })

workflow.add_conditional_edges("shell_validator",
                               validate_router,
                               {
                                   "csv_compiler":       "csv_compiler",
                                   "shell_validator":    "shell_validator",   # retry
                                   "give_up":            "give_up"
                               })

workflow.add_conditional_edges("csv_compiler",
                               csv_router,
                               {
                                   "report_formatter":   "report_formatter",
                                   "csv_compiler":       "csv_compiler",      # retry
                                   "give_up":            "give_up"
                               })

workflow.add_conditional_edges("report_formatter",
                               report_router,
                               {
                                   END:                  END,
                                   "report_formatter":   "report_formatter",  # retry
                                   "give_up":            "give_up"
                               })

# Terminal fallback
workflow.add_edge("give_up", END)

# ---------------------------------------------------------------------
# 6.  COMPILE APP
# ---------------------------------------------------------------------
app = workflow.compile()

# Workflow ready to run:
#   result = app.invoke(initial_state_dict)