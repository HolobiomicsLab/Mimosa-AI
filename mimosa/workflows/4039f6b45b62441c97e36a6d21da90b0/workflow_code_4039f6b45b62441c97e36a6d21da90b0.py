# ============================================================
# LangGraph ‑ SmolAgent workflow :  “Deep search on using mzmind 
# with batch in command-line and create concrete usage examples”
# ============================================================

# ----------  PRE-LOADED CONTEXT  -----------------------------
# • WorkflowState  |  Action  |  Observation  classes
# • SmolAgentFactory
# • WorkflowNodeFactory
# • SHELL_TOOLS , BROWSER_TOOLS , CSV_TOOLS  (tool packages)
# -------------------------------------------------------------

from langgraph.graph import StateGraph, START, END

# ------------- 1) AGENT PROMPTS (single-purpose) -------------

# Agent-A : Web search specialist
prompt_search = """
You are an intensive web-research agent.

TASK
- Run broad and deep web searches about the CLI tool “mzmind”, focusing on
  “batch mode / batch processing / batch files”.
- Collect authoritative pages: documentation, blogs, forum threads, GitHub issues.
- Log every visited URL and 1-sentence description.

SUCCESS CASE
- When you have >=5 reliable sources, summarise key findings (bullets) + list URLs
- End your answer with the token: SEARCH_COMPLETE

FAILURE CASE
- If after exhaustive searching (<15 queries tried) you still lack info, explain
  your attempts & obstacles, then end with: SEARCH_FAILURE

ERROR CASE
- For any technical/tool error: explain and end with: GIVE_UP
"""

# Agent-B : Information extractor
prompt_extract = """
You are an extraction agent.

INPUT
- Previous agent’s summary & URLs (visible in observations)

TASK
- Parse content and extract:
  • concrete command-line invocations of mzmind in batch mode
  • references to batch configuration file syntax
- Produce a cleaned list of commands + any config snippets encountered
- Cite source URL beside each item

OUTPUT
- If >=3 distinct commands AND >=1 config snippet are extracted -> finish with: EXTRACT_COMPLETE
- Otherwise finish with: EXTRACT_FAILURE
- On technical issues finish with: GIVE_UP
"""

# Agent-C : Validator / gap-finder
prompt_validate = """
You are a validation agent.

TASK
- Inspect extracted commands & config snippets from observations.
CRITERIA
  • At least 3 unique usage cases
  • At least 1 full config file example
  • Sources are cited
ACTIONS
- If all criteria met, reply: VALIDATION_PASS
- If something missing, clearly list missing pieces, reply: VALIDATION_FAIL
- On technical error: GIVE_UP
"""

# Agent-D : Config & example generator
prompt_config = """
You are a generation agent.

INPUT
- Validated commands & snippets.

TASK
- Create:
  1) Three realistic use-case descriptions (what user is trying to do).
  2) For each use-case:
        • Full command-line example
        • Separate .batch configuration file content (proper syntax)
- Make sure files are self-contained and runnable.

OUTPUT
- When done: CONFIG_READY
- If cannot complete: CONFIG_FAILURE
- On technical issues: GIVE_UP
"""

# Agent-E : Final formatter
prompt_format = """
You are the final report writer.

TASK
- Combine prior data into a polished answer:
   • short introduction
   • bulleted key findings
   • the 3 use-cases with commands + config files (in fenced code blocks)
   • source list
- End ONLY with: FINAL_COMPLETE
"""

# ------------- 2) AGENT INSTANCES ----------------------------

agent_search   = SmolAgentFactory(prompt_search,   BROWSER_TOOLS)
agent_extract  = SmolAgentFactory(prompt_extract,  BROWSER_TOOLS)
agent_validate = SmolAgentFactory(prompt_validate, CSV_TOOLS)
agent_config   = SmolAgentFactory(prompt_config,   SHELL_TOOLS)
agent_format   = SmolAgentFactory(prompt_format,   CSV_TOOLS)

# ------------- 3) WORKFLOW GRAPH -----------------------------

workflow = StateGraph(WorkflowState)

# ---- Node registration
workflow.add_node("web_research",      WorkflowNodeFactory.create_agent_node(agent_search))
workflow.add_node("content_extract",   WorkflowNodeFactory.create_agent_node(agent_extract))
workflow.add_node("validator",         WorkflowNodeFactory.create_agent_node(agent_validate))
workflow.add_node("config_generator",  WorkflowNodeFactory.create_agent_node(agent_config))
workflow.add_node("final_formatter",   WorkflowNodeFactory.create_agent_node(agent_format))

# ------------- 4) ROUTING FUNCTIONS WITH ROBUST FALLBACK -----

MAX_RETRY = 3   # generic cap


def _count_attempt(state: WorkflowState, step: str) -> int:
    """utility – how many times we already attempted <step>"""
    return sum(1 for s in state.get("step_name", []) if s.startswith(step))


# ---- router after web research
def route_after_search(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        attempts = _count_attempt(state, "web_research")
        if "SEARCH_COMPLETE" in answer:
            return "content_extract"
        if "SEARCH_FAILURE" in answer and attempts < MAX_RETRY:
            return "web_research"   # retry same agent
        return "END_FAIL"            # emergency
    except Exception as e:
        print(f"RouterSearch error: {e}")
        return "END_FAIL"

# ---- router after extraction
def route_after_extract(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        attempts = _count_attempt(state, "content_extract")
        if "EXTRACT_COMPLETE" in answer:
            return "validator"
        if "EXTRACT_FAILURE" in answer and attempts < MAX_RETRY:
            # maybe broaden search again first
            return "web_research"
        return "END_FAIL"
    except Exception as e:
        print(f"RouterExtract error: {e}")
        return "END_FAIL"

# ---- router after validation
def route_after_validation(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        attempts_val = _count_attempt(state, "validator")
        if "VALIDATION_PASS" in answer:
            return "config_generator"
        if "VALIDATION_FAIL" in answer:
            # try extractor again if we still have room
            if attempts_val < MAX_RETRY:
                return "content_extract"
            else:
                return "web_research"
        return "END_FAIL"
    except Exception as e:
        print(f"RouterValidate error: {e}")
        return "END_FAIL"

# ---- router after config generation
def route_after_config(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        attempts = _count_attempt(state, "config_generator")
        if "CONFIG_READY" in answer:
            return "final_formatter"
        if "CONFIG_FAILURE" in answer and attempts < MAX_RETRY:
            return "validator"      # double-check data then re-gen
        return "END_FAIL"
    except Exception as e:
        print(f"RouterConfig error: {e}")
        return "END_FAIL"

# ---- router after final formatter
def route_after_final(state: WorkflowState) -> str:
    try:
        if "FINAL_COMPLETE" in state.get("answers", [""])[-1]:
            return END
        return "END_FAIL"
    except Exception as e:
        print(f"RouterFinal error: {e}")
        return "END_FAIL"

# ------------- 5) EDGES --------------------------------------

# start
workflow.add_edge(START, "web_research")

# conditional paths
workflow.add_conditional_edges(
    "web_research",
    route_after_search,
    {
        "content_extract": "content_extract",
        "web_research": "web_research",   # retry
        "END_FAIL": END
    }
)

workflow.add_conditional_edges(
    "content_extract",
    route_after_extract,
    {
        "validator": "validator",
        "web_research": "web_research",   # broaden search then loop
        "END_FAIL": END
    }
)

workflow.add_conditional_edges(
    "validator",
    route_after_validation,
    {
        "config_generator": "config_generator",
        "content_extract": "content_extract",
        "web_research": "web_research",
        "END_FAIL": END
    }
)

workflow.add_conditional_edges(
    "config_generator",
    route_after_config,
    {
        "final_formatter": "final_formatter",
        "validator": "validator",
        "END_FAIL": END
    }
)

workflow.add_conditional_edges(
    "final_formatter",
    route_after_final,
    {
        END: END,
        "END_FAIL": END
    }
)

# ------------- 6) COMPILE WORKFLOW ---------------------------

app = workflow.compile()