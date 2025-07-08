"""
LANGGRAPH – SMOLAGENT WORKFLOW
Goal: Produce a comprehensive report on “CNRS goals in AI for European sovereignty”

The workflow strictly follows the CORE ARCHITECTURE PRINCIPLES:
• 5 atomic SmolAgents (research → extract → validate → csv → report)
• Robust conditional routing with retries + emergency END fall-backs
• Multiple distinct tool-packages utilised (BROWSER / CSV / SHELL)
"""

# ===== MANDATORY LIB & SCHEMA IMPORTS =====
from langgraph.graph import StateGraph, START, END           # already available in context
# WorkflowState, Action, Observation already declared in context
# Tool packages already declared in context:
#   CSV_TOOL_TOOLS, BROWSER_TOOL_TOOLS, SHELL_TOOL_TOOLS
# SmolAgentFactory & WorkflowNodeFactory already declared in context


# ===== WORKFLOW INITIALISATION =====
workflow = StateGraph(WorkflowState)


# =====================================================================
# 1️⃣  WEB_RESEARCHER  – atomic task: perform deep web search
# =====================================================================
research_prompt = """
You are a specialised WEB RESEARCH agent.

OBJECTIVE
Search the web exhaustively for information about:
“CNRS (French National Centre for Scientific Research) ambitions / goals in Artificial Intelligence
with respect to guaranteeing or enhancing European technological sovereignty.”

INSTRUCTIONS
– Use every browser & search capability at your disposal
– Collect URLs, publication titles, dates, authors, official CNRS strategy documents, press releases,
  EU collaboration programmes, funding amounts, road-maps, etc.

OUTPUT PROTOCOL (MUST follow exactly one branch)
SUCCESS → End with the exact token: RESEARCH_COMPLETE
Provide:
• Bullet list of key findings (one fact per bullet)
• Full citation + URL for every finding

FAILURE (insufficient data) → End with: RESEARCH_FAILURE
Provide:
• What search terms, engines, filters you tried
• Gaps encountered
• New angles that could work

ERROR (technical) → End with: GIVE_UP
Provide:
• Nature of the error
• What human assistance is required
"""
research_agent   = SmolAgentFactory(research_prompt, BROWSER_TOOL_TOOLS)
workflow.add_node("web_researcher", WorkflowNodeFactory.create_agent_node(research_agent))


# =====================================================================
# 2️⃣  CONTENT_EXTRACTOR – atomic task: extract & consolidate facts
# =====================================================================
extract_prompt = """
You are a CONTENT EXTRACTION agent.

CONTEXT
You will receive the raw research summary produced by the previous agent.

TASK
– Parse the text
– Extract every unique data-point about CNRS AI goals & European sovereignty
– Produce a clean, de-duplicated list of JSON objects:
  [{ "theme": "...", "detail": "...", "source": "URL" }, …]

OUTPUT PROTOCOL
SUCCESS → End with: EXTRACT_COMPLETE
Provide only the JSON list (no explanations)

FAILURE (unparsable or missing info) → End with: EXTRACT_FAILURE
Explain the exact missing elements required

ERROR (technical) → End with: GIVE_UP
Explain the error
"""
extract_agent    = SmolAgentFactory(extract_prompt, BROWSER_TOOL_TOOLS)
workflow.add_node("content_extractor", WorkflowNodeFactory.create_agent_node(extract_agent))


# =====================================================================
# 3️⃣  INFO_VALIDATOR  – atomic task: quality / completeness check
# =====================================================================
validate_prompt = """
You are a VALIDATION agent.

INPUT
A JSON list of extracted facts about CNRS AI goals & EU sovereignty.

TASK
– Verify each item has "theme", "detail", "source"
– Check sources credibility (official CNRS / EU / recognised media).
– Determine if the list covers: (1) strategic goals, (2) funding / programmes,
  (3) partnerships, and (4) timeline/road-map.

OUTPUT PROTOCOL
If all four categories are covered with credible sources →
    VALIDATION_PASS (exact token) followed by 1-sentence confirmation.

If coverage is incomplete or sources not credible →
    VALIDATION_INCOMPLETE followed by
    • Missing categories list
    • Specific guidance for new research

Technical problem →
    GIVE_UP followed by error description
"""
validate_agent   = SmolAgentFactory(validate_prompt, BROWSER_TOOL_TOOLS)
workflow.add_node("info_validator", WorkflowNodeFactory.create_agent_node(validate_agent))


# =====================================================================
# 4️⃣  CSV_COMPILER   – atomic task: save sources to CSV
# =====================================================================
csv_prompt = """
You are a CSV COMPILATION agent.

INPUT
A validated JSON list of facts.

TASK
Generate a CSV with headers:
theme,detail,source
Each line = one JSON object.

OUTPUT PROTOCOL
SUCCESS →
    CSV_COMPLETE then the CSV text block
FAILURE (invalid JSON or other issue) →
    CSV_FAILURE then detailed reason
ERROR →
    GIVE_UP then error description
"""
csv_agent        = SmolAgentFactory(csv_prompt, CSV_TOOL_TOOLS)
workflow.add_node("csv_compiler", WorkflowNodeFactory.create_agent_node(csv_agent))


# =====================================================================
# 5️⃣  REPORT_WRITER  – atomic task: craft final comprehensive report
# =====================================================================
report_prompt = """
You are a REPORT WRITING agent.

INPUTS
1) The validated JSON fact list
2) The CSV of sources

TASK
Compose a detailed, well-structured report (>700 words) entitled:
“CNRS Strategy in Artificial Intelligence for European Sovereignty”.
Include:
• Executive summary
• Detailed sections per theme
• Proper in-text citations [1], [2]… matching the CSV order
• Concluding analysis on sovereignty impact.

OUTPUT PROTOCOL
SUCCESS → End with REPORT_COMPLETE
Provide the full report text.

If inputs are incomplete/invalid → REPORT_INCOMPLETE with explanation.
ERROR → GIVE_UP with error description.
"""
report_agent     = SmolAgentFactory(report_prompt, SHELL_TOOL_TOOLS)
workflow.add_node("report_writer", WorkflowNodeFactory.create_agent_node(report_agent))



# =====================================================================
# ========== ROUTING / ERROR-HANDLING FUNCTIONS ==========
# =====================================================================

MAX_RETRIES = 3     # global retry cap per agent


def count_attempts(state: WorkflowState, node_name: str) -> int:
    """Helper to count how many times we've tried a specific node"""
    return sum(1 for n in state.get("step_name", []) if n.startswith(node_name))


# ---------- ROUTER AFTER WEB_RESEARCHER ----------
def router_after_research(state: WorkflowState) -> str:
    try:
        answers = state.get("answers", [])
        if not answers:
            # Should never happen but safe-guard
            return "web_researcher"

        last_answer = answers[-1]

        if "RESEARCH_COMPLETE" in last_answer:
            return "content_extractor"

        # Retry logic
        retries = count_attempts(state, "web_researcher")
        if retries < MAX_RETRIES and "RESEARCH_FAILURE" in last_answer:
            return "web_researcher"

        # Emergency stop for GIVE_UP or exceeded retries
        return END
    except Exception as e:
        print(f"💥 router_after_research error: {e}")
        return END


# ---------- ROUTER AFTER CONTENT_EXTRACTOR ----------
def router_after_extract(state: WorkflowState) -> str:
    try:
        answers = state.get("answers", [])
        if not answers:
            return "content_extractor"
        last_answer = answers[-1]

        if "EXTRACT_COMPLETE" in last_answer:
            return "info_validator"

        retries = count_attempts(state, "content_extractor")
        if retries < MAX_RETRIES and "EXTRACT_FAILURE" in last_answer:
            return "content_extractor"

        # If extraction consistently fails, go back to research for new material
        if retries >= MAX_RETRIES:
            return "web_researcher"

        return END
    except Exception as e:
        print(f"💥 router_after_extract error: {e}")
        return END


# ---------- ROUTER AFTER INFO_VALIDATOR ----------
def router_after_validation(state: WorkflowState) -> str:
    try:
        answers = state.get("answers", [])
        if not answers:
            return "info_validator"
        last_answer = answers[-1]

        if "VALIDATION_PASS" in last_answer:
            return "csv_compiler"

        if "VALIDATION_INCOMPLETE" in last_answer:
            # Go back to research for additional info
            return "web_researcher"

        # If GIVE_UP or unknown
        return END
    except Exception as e:
        print(f"💥 router_after_validation error: {e}")
        return END


# ---------- ROUTER AFTER CSV_COMPILER ----------
def router_after_csv(state: WorkflowState) -> str:
    try:
        answers = state.get("answers", [])
        if not answers:
            return "csv_compiler"
        last_answer = answers[-1]

        if "CSV_COMPLETE" in last_answer:
            return "report_writer"

        retries = count_attempts(state, "csv_compiler")
        if retries < MAX_RETRIES and "CSV_FAILURE" in last_answer:
            return "csv_compiler"

        return END
    except Exception as e:
        print(f"💥 router_after_csv error: {e}")
        return END


# ---------- ROUTER AFTER REPORT_WRITER ----------
def router_after_report(state: WorkflowState) -> str:
    try:
        answers = state.get("answers", [])
        if not answers:
            return "report_writer"
        last_answer = answers[-1]

        if "REPORT_COMPLETE" in last_answer:
            return END

        retries = count_attempts(state, "report_writer")
        if retries < MAX_RETRIES and "REPORT_INCOMPLETE" in last_answer:
            return "report_writer"

        return END
    except Exception as e:
        print(f"💥 router_after_report error: {e}")
        return END



# =====================================================================
# ========== EDGE DEFINITIONS (with fallback paths) ==========
# =====================================================================

# START -> research
workflow.add_edge(START, "web_researcher")

# research -> extractor / retry / END
workflow.add_conditional_edges(
    "web_researcher",
    router_after_research,
    {
        "content_extractor": "content_extractor",
        "web_researcher": "web_researcher",
        END: END
    }
)

# extractor -> validator / retry / back to research / END
workflow.add_conditional_edges(
    "content_extractor",
    router_after_extract,
    {
        "info_validator": "info_validator",
        "content_extractor": "content_extractor",
        "web_researcher": "web_researcher",
        END: END
    }
)

# validator -> csv / research / END
workflow.add_conditional_edges(
    "info_validator",
    router_after_validation,
    {
        "csv_compiler": "csv_compiler",
        "web_researcher": "web_researcher",
        END: END
    }
)

# csv -> report / retry / END
workflow.add_conditional_edges(
    "csv_compiler",
    router_after_csv,
    {
        "report_writer": "report_writer",
        "csv_compiler": "csv_compiler",
        END: END
    }
)

# report -> END / retry
workflow.add_conditional_edges(
    "report_writer",
    router_after_report,
    {
        END: END,
        "report_writer": "report_writer"
    }
)


# =====================================================================
# ========== COMPILE WORKFLOW ==========
# =====================================================================
app = workflow.compile()

# The resulting `app` can be invoked with an initial (possibly empty) WorkflowState dict:
# result_state = app.invoke({})