# ============================================================
# LangGraph ‑ SmolAgent workflow for:
#  “Deep-search how to use mzmind with batch on the command line,
#   then install the software on macOS (Apple-Silicon)”
# ============================================================

# ---- Pre-existing objects (already loaded in the runtime) ---------------
#  • WorkflowState  (TypedDict)
#  • SHELL_TOOLS, BROWSER_TOOLS, CSV_TOOLS
#  • SmolAgentFactory
#  • WorkflowNodeFactory
# -------------------------------------------------------------------------

from langgraph.graph import StateGraph, START, END

# -------------------------------------------------------------------------
# 1. AGENT PROMPTS  (one atomic responsibility each)
# -------------------------------------------------------------------------

# --- 1) Task-feasibility checker ----------------------------------------
instruct_scope = """
You are a TASK-FEASIBILITY analyst.

MISSION
- Evaluate whether the user’s goal is realistically attainable with the
  resources available (tools: browser/shell) and within normal constraints.

UPON COMPLETION
SUCCESS CASE → provide a short justification followed by TASK_FEASIBLE
INFEASIBLE   → give reasons, blockers, external requirements then TASK_INFEASIBLE
ERROR        → explain problem then GIVE_UP

CRITICAL
- Only say TASK_FEASIBLE if you are confident all steps (research, download,
  Apple-silicon install) are accomplishable with our tools.
"""

# --- 2) Primary web researcher ------------------------------------------
instruct_web_primary = """
You are a WEB-RESEARCH agent.

GOAL
- Perform an in-depth search for documentation on:
  • “mzmind” batch / CLI usage
  • Installation on macOS Apple-silicon

OUTPUT RULES
SUCCESS  → long detailed summary with URLs, commands, version notes, ending with RESEARCH_COMPLETE
FAILURE  → detailed search log, terms tried, issues, suggestions, ending with RESEARCH_FAILURE
ERROR    → explanation, ending with GIVE_UP

MUST
- Provide at least 3 distinct reputable sources on success.
"""

# --- 3) Secondary (fallback) web researcher -----------------------------
instruct_web_fallback = """
You are a SECOND-PASS researcher specialising in niche or hard-to-find
information.

TASK
- Use alternative strategies: deep sub-pages, GitHub issues, forums, cache
  snapshots, etc. for the same “mzmind” CLI + Apple-silicon install info.

Same SUCCESS / FAILURE / ERROR keywords as primary researcher.
"""

# --- 4) Extraction agent -------------------------------------------------
instruct_extract = """
You are an INFORMATION-EXTRACTOR.

INPUT
- Previous answer(s) contain raw web findings.

TASK
- Produce ordered, shell-ready step-by-step instructions:
  a) Using mzmind in batch / CLI
  b) Installing on macOS Apple-silicon (incl. dependencies)

OUTPUT
SUCCESS → numbered list of commands + explanations, end with EXTRACTION_SUCCESS
FAILURE → state exactly what’s missing, end with EXTRACTION_FAILURE
ERROR   → GIVE_UP
"""

# --- 5) Validation agent -------------------------------------------------
instruct_validate = """
You are a VALIDATOR for shell commands.

TASK
- Review the extracted instructions for logic, Apple-silicon compatibility,
  missing sudo / brew steps, version mismatches.

OUTPUT
PASS  → list any minor notes then VALIDATION_PASSED
FAIL  → explain issues precisely then VALIDATION_FAILED
ERROR → GIVE_UP
"""

# --- 6) Installer / executor --------------------------------------------
instruct_install = """
You are an INSTALLER agent executing commands on a macOS Apple-silicon system.

TASK
- Run the validated commands using the shell tool.
- Capture output / errors.

OUTPUT
SUCCESS → summary of execution + INSTALL_SUCCESS
FAILURE → error logs, troubleshooting tips + INSTALL_FAILURE
ERROR   → GIVE_UP
"""

# -------------------------------------------------------------------------
# 2. AGENT CREATION  ------------------------------------------------------
scope_checker       = SmolAgentFactory(instruct_scope,        BROWSER_TOOLS)
web_researcher      = SmolAgentFactory(instruct_web_primary,  BROWSER_TOOLS)
web_researcher_alt  = SmolAgentFactory(instruct_web_fallback, BROWSER_TOOLS)
extractor_agent     = SmolAgentFactory(instruct_extract,      CSV_TOOLS)
validator_agent     = SmolAgentFactory(instruct_validate,     SHELL_TOOLS)
installer_agent     = SmolAgentFactory(instruct_install,      SHELL_TOOLS)

# -------------------------------------------------------------------------
# 3. WORKFLOW INIT  -------------------------------------------------------
workflow = StateGraph(WorkflowState)

# -------------------------------------------------------------------------
# 4. NODE ADDITION  -------------------------------------------------------
workflow.add_node("scope_checker",      WorkflowNodeFactory.create_agent_node(scope_checker))
workflow.add_node("web_researcher",     WorkflowNodeFactory.create_agent_node(web_researcher))
workflow.add_node("alt_researcher",     WorkflowNodeFactory.create_agent_node(web_researcher_alt))
workflow.add_node("extractor",          WorkflowNodeFactory.create_agent_node(extractor_agent))
workflow.add_node("validator",          WorkflowNodeFactory.create_agent_node(validator_agent))
workflow.add_node("installer",          WorkflowNodeFactory.create_agent_node(installer_agent))

# -------------------------------------------------------------------------
# 5. ROUTING / ERROR-HANDLING FUNCTIONS  ----------------------------------
def safe_get(lst, idx, default=None):
    try:
        return lst[idx]
    except Exception:
        return default

# ---- 5.a  Scope-checker router -----------------------------------------
def scope_router(state: WorkflowState) -> str:
    try:
        answer = safe_get(state.get("answers", []), -1, "")
        if "TASK_FEASIBLE" in answer:
            return "web_researcher"
        elif "TASK_INFEASIBLE" in answer:
            return END
        else:
            # unexpected output -> give one retry then end
            retries = state.get("step_name", []).count("scope_checker")
            return "scope_checker" if retries < 2 else END
    except Exception as e:
        print(f"Scope router error: {e}")
        return END

# ---- 5.b  Research router ----------------------------------------------
def research_router(state: WorkflowState) -> str:
    try:
        answer = safe_get(state.get("answers", []), -1, "")
        retries_primary = state.get("step_name", []).count("web_researcher")
        if "RESEARCH_COMPLETE" in answer:
            return "extractor"
        if "RESEARCH_FAILURE" in answer:
            if retries_primary < 2:
                return "web_researcher"        # retry same agent
            else:
                return "alt_researcher"        # fallback agent
        # Any other unexpected message
        return "web_researcher" if retries_primary < 3 else "alt_researcher"
    except Exception as e:
        print(f"Research router error: {e}")
        return END

# ---- 5.c  Alt-research router ------------------------------------------
def alt_research_router(state: WorkflowState) -> str:
    try:
        answer = safe_get(state.get("answers", []), -1, "")
        if "RESEARCH_COMPLETE" in answer:
            return "extractor"
        # Failed even in alt path -> end workflow
        return END
    except Exception as e:
        print(f"Alt research router error: {e}")
        return END

# ---- 5.d  Extraction router --------------------------------------------
def extraction_router(state: WorkflowState) -> str:
    try:
        answer = safe_get(state.get("answers", []), -1, "")
        retries = state.get("step_name", []).count("extractor")
        if "EXTRACTION_SUCCESS" in answer:
            return "validator"
        if "EXTRACTION_FAILURE" in answer:
            # Go back to researcher for more info after 2 extractor retries
            return "extractor" if retries < 2 else "web_researcher"
        return "extractor" if retries < 3 else "web_researcher"
    except Exception as e:
        print(f"Extraction router error: {e}")
        return END

# ---- 5.e  Validation router --------------------------------------------
def validation_router(state: WorkflowState) -> str:
    try:
        answer = safe_get(state.get("answers", []), -1, "")
        retries = state.get("step_name", []).count("validator")
        if "VALIDATION_PASSED" in answer:
            return "installer"
        if "VALIDATION_FAILED" in answer:
            return "extractor" if retries < 2 else END
        return "validator" if retries < 3 else END
    except Exception as e:
        print(f"Validation router error: {e}")
        return END

# ---- 5.f  Installation router ------------------------------------------
def install_router(state: WorkflowState) -> str:
    try:
        answer = safe_get(state.get("answers", []), -1, "")
        retries = state.get("step_name", []).count("installer")
        if "INSTALL_SUCCESS" in answer:
            return END
        if "INSTALL_FAILURE" in answer:
            return "validator" if retries < 2 else END
        # Unexpected output
        return "installer" if retries < 2 else END
    except Exception as e:
        print(f"Install router error: {e}")
        return END

# -------------------------------------------------------------------------
# 6. EDGE DEFINITIONS  ----------------------------------------------------
workflow.add_edge(START, "scope_checker")

workflow.add_conditional_edges(
    "scope_checker",
    scope_router,
    {
        "web_researcher": "web_researcher",
        END: END
    }
)

workflow.add_conditional_edges(
    "web_researcher",
    research_router,
    {
        "extractor": "extractor",
        "web_researcher": "web_researcher",
        "alt_researcher": "alt_researcher"
    }
)

workflow.add_conditional_edges(
    "alt_researcher",
    alt_research_router,
    {
        "extractor": "extractor",
        END: END
    }
)

workflow.add_conditional_edges(
    "extractor",
    extraction_router,
    {
        "validator": "validator",
        "extractor": "extractor",
        "web_researcher": "web_researcher"
    }
)

workflow.add_conditional_edges(
    "validator",
    validation_router,
    {
        "installer": "installer",
        "extractor": "extractor",
        END: END
    }
)

workflow.add_conditional_edges(
    "installer",
    install_router,
    {
        END: END,
        "validator": "validator",
        "installer": "installer"
    }
)

# -------------------------------------------------------------------------
# 7. COMPILE WORKFLOW  ----------------------------------------------------
app = workflow.compile()