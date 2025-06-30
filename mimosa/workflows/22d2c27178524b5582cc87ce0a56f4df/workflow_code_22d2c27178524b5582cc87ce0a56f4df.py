# ============================================================
# PRE-EXISTING CONTEXT (already provided by runtime)
# - WorkflowState TypedDict
# - SHELL_TOOLS, BROWSER_TOOLS, CSV_TOOLS
# - SmolAgentFactory
# - WorkflowNodeFactory
# ============================================================

from langgraph.graph import StateGraph, START, END

# --------------- 1.  WORKFLOW INITIALISATION ----------------
workflow = StateGraph(WorkflowState)


# --------------- 2.  AGENT PROMPTS & CREATION ---------------

# A) Feasibility-check agent
feas_prompt = """You are a feasibility analyst.

GOAL
Determine whether it is realistically possible to:
1. Find reliable information describing how to run "mzmind" in batch/command-line mode.
2. Obtain trustworthy macOS installation steps for "mzmind".

TASK
• Search the web (use your tools) for any sign of official docs, community posts, GitHub issues, etc.
• Evaluate credibility and freshness of sources.

OUTPUT PROTOCOL
If you conclude the goal IS achievable, respond only with:
FEASIBILITY_YES: <one concise sentence justifying why>

If goal is NOT achievable, respond only with:
FEASIBILITY_NO: <brief reason>

For technical/tool issues you cannot overcome, respond only with:
GIVE_UP: <error description>
"""
feas_agent = SmolAgentFactory(feas_prompt, BROWSER_TOOLS)


# B) Deep web-search agent
search_prompt = """You are a deep web-search specialist.

SUB-GOAL
Gather DETAILED, well-sourced information on:
• Using the mzmind application in batch / command-line mode.
• Required flags, parameters, or example command strings.

INSTRUCTIONS
1. Exhaustively search official docs, forums, blogs, GitHub, etc.
2. Collect at least THREE distinct, credible sources.
3. Copy complete CLI examples where available.
4. Store URLs for every claim.

SUCCESS CRITERIA
Return exactly:
SEARCH_COMPLETE: <rich multi-paragraph summary with links & CLI examples>

FAILURE CRITERIA (insufficient info):
SEARCH_FAILURE: <what was tried, what’s missing, suggestions>

ERROR CRITERIA (tool problems):
GIVE_UP: <error>
"""
search_agent = SmolAgentFactory(search_prompt, BROWSER_TOOLS)


# C) Extraction & Mac-install-instruction agent
extract_prompt = """You are an extraction agent focused on macOS installation steps.

INPUT
You will receive previous search results in state.observations / state.answers.

TASK
• Derive a clean, step-by-step macOS installation guide for mzmind.
• Include system requirements, dependencies, path variables, and verification steps.

OUTPUT
EXTRACTION_COMPLETE: <ordered list of steps w/ shell commands & notes>

If info is insufficient:
EXTRACTION_FAILURE: <missing items, request new search angles>

Technical/tool errors:
GIVE_UP: <error>
"""
extract_agent = SmolAgentFactory(extract_prompt, BROWSER_TOOLS)


# D) Validation / sanity-check agent
validate_prompt = """You are a cross-checker.

TASK
• Verify the macOS installation steps and CLI usage instructions for mzmind just produced.
• Cross-reference at least two independent sources.
• Flag contradictions or risky commands.

OUTPUT
VALIDATION_PASS: <concise approval & any minor notes>

If major problems detected:
VALIDATION_FAIL: <why invalid, what needs fixing>

Tool errors:
GIVE_UP: <error>
"""
validate_agent = SmolAgentFactory(validate_prompt, CSV_TOOLS)   # CSV tools reused for structured comparison


# E) Installation executor agent (shell)
install_prompt = """You are a macOS installation executor.

INPUT
You will be given validated installation instructions.

TASK
• Execute each shell command using your tools.
• Capture output & confirm success of each step.

OUTPUT
INSTALL_SUCCESS: <summary of all executed commands & confirmations>

If any command fails (non-zero exit):
INSTALL_FAILURE: <failed command, stderr, suggestion>

Tool issues:
GIVE_UP: <error>
"""
install_agent = SmolAgentFactory(install_prompt, SHELL_TOOLS)


# ---- Convert all agents into LangGraph nodes ----------------
workflow.add_node("feasibility_checker", WorkflowNodeFactory.create_agent_node(feas_agent))
workflow.add_node("deep_searcher",        WorkflowNodeFactory.create_agent_node(search_agent))
workflow.add_node("extractor",            WorkflowNodeFactory.create_agent_node(extract_agent))
workflow.add_node("validator",            WorkflowNodeFactory.create_agent_node(validate_agent))
workflow.add_node("installer",            WorkflowNodeFactory.create_agent_node(install_agent))


# --------------- 3.  ROUTING FUNCTIONS ----------------------

# Helper for safe field access
def _get(state: WorkflowState, key: str):
    return state[key] if key in state else []

# A) After feasibility check
def route_after_feas(state: WorkflowState) -> str:
    try:
        answer = _get(state, "answers")[-1] if _get(state, "answers") else ""
        if "FEASIBILITY_YES" in answer:
            return "deep_searcher"
        elif "FEASIBILITY_NO" in answer or "GIVE_UP" in answer:
            return END
        else:                                   # Unexpected format
            return "feasibility_checker"        # Retry
    except Exception as e:
        print(f"[Router-feasibility] error: {e}")
        return END


# B) After search
def route_after_search(state: WorkflowState) -> str:
    try:
        answer = _get(state, "answers")[-1] if _get(state, "answers") else ""
        step_hist = _get(state, "step_name")
        retry_count = step_hist.count("deep_searcher")
        
        if "SEARCH_COMPLETE" in answer:
            return "extractor"
        elif "SEARCH_FAILURE" in answer and retry_count < 3:
            return "deep_searcher"              # retry
        elif "SEARCH_FAILURE" in answer:
            return END                          # give-up after 3
        elif "GIVE_UP" in answer:
            return END
        else:                                   # unexpected
            return "deep_searcher"
    except Exception as e:
        print(f"[Router-search] error: {e}")
        return END


# C) After extraction
def route_after_extract(state: WorkflowState) -> str:
    try:
        answer = _get(state, "answers")[-1] if _get(state, "answers") else ""
        step_hist = _get(state, "step_name")
        retry_count = step_hist.count("extractor")
        
        if "EXTRACTION_COMPLETE" in answer:
            return "validator"
        elif "EXTRACTION_FAILURE" in answer and retry_count < 3:
            return "deep_searcher"              # back to search for more info
        elif "EXTRACTION_FAILURE" in answer:
            return END
        elif "GIVE_UP" in answer:
            return END
        else:
            return "extractor"
    except Exception as e:
        print(f"[Router-extract] error: {e}")
        return END


# D) After validation
def route_after_validate(state: WorkflowState) -> str:
    try:
        answer = _get(state, "answers")[-1] if _get(state, "answers") else ""
        step_hist = _get(state, "step_name")
        retry_count = step_hist.count("validator")
        
        if "VALIDATION_PASS" in answer:
            return "installer"
        elif "VALIDATION_FAIL" in answer and retry_count < 2:
            return "extractor"                  # fix instructions then re-validate
        elif "VALIDATION_FAIL" in answer:
            return END
        elif "GIVE_UP" in answer:
            return END
        else:
            return "validator"
    except Exception as e:
        print(f"[Router-validate] error: {e}")
        return END


# E) After installation
def route_after_install(state: WorkflowState) -> str:
    try:
        answer = _get(state, "answers")[-1] if _get(state, "answers") else ""
        step_hist = _get(state, "step_name")
        retry_count = step_hist.count("installer")
        
        if "INSTALL_SUCCESS" in answer:
            return END
        elif "INSTALL_FAILURE" in answer and retry_count < 2:
            return "installer"                  # retry once
        else:
            return END
    except Exception as e:
        print(f"[Router-install] error: {e}")
        return END


# --------------- 4.  GRAPH EDGE WIRING ----------------------

workflow.add_edge(START, "feasibility_checker")

workflow.add_conditional_edges(
    "feasibility_checker",
    route_after_feas,
    {
        "deep_searcher": "deep_searcher",
        END: END
    }
)

workflow.add_conditional_edges(
    "deep_searcher",
    route_after_search,
    {
        "extractor": "extractor",
        "deep_searcher": "deep_searcher",   # retry path
        END: END
    }
)

workflow.add_conditional_edges(
    "extractor",
    route_after_extract,
    {
        "validator": "validator",
        "deep_searcher": "deep_searcher",   # fallback for more info
        "extractor": "extractor",           # retry self
        END: END
    }
)

workflow.add_conditional_edges(
    "validator",
    route_after_validate,
    {
        "installer": "installer",
        "extractor": "extractor",
        "validator": "validator",
        END: END
    }
)

workflow.add_conditional_edges(
    "installer",
    route_after_install,
    {
        END: END,
        "installer": "installer"            # retry install once
    }
)


# --------------- 5.  COMPILE WORKFLOW -----------------------
app = workflow.compile()

# The `app` object is now an executable LangGraph workflow.
# ============================================================