# --------------  LangGraph × SmolAgent WORKFLOW -----------------
#
#  Goal: “Search for alternatives to ManusAI and save the findings in a CSV
#        file containing:  name | description | local (yes/no) | repo / website”
#
#  Agents (STRICT divide-and-conquer)
#    1⃣  feasibility_checker      – decide if the task is even possible
#    2⃣  web_search_agent         – perform focused web search
#    3⃣  extraction_agent         – transform raw search notes → structured rows
#    4⃣  csv_writer_agent         – write rows to ‘manusai_alternatives.csv’
#    5⃣  validation_agent         – verify CSV quality & decide final success
#
#  Fallback & retries
#    • web_search_agent   : max-3 attempts → else SEARCH_FAIL → END
#    • extraction_agent   : max-2 attempts → else PARSE_FAIL  → END
#    • validation_agent   : on failure → go back to csv_writer_agent once
#
# ----------------------------------------------------------------

from langgraph.graph import StateGraph, START, END

# =========== PROMPTS =========================================================
instruct_feasibility = """
You are an analytical feasibility assessor.

TASK
- Judge whether it is realistically possible to find at least three meaningful
  alternatives to the software “ManusAI” on the public internet.

CRITERIA
- An alternative must be a software or service that addresses a similar need.
- Information should be publicly accessible via normal web search.

COMPLETION PROTOCOL
If confident there is enough publicly available information:
  final_answer("FEASIBLE: Rationale [...]")
If confident the task is impossible or unreasonably difficult:
  final_answer("NOT_FEASIBLE: Explanation [...]")

IMPORTANT
Return EXACTLY one of the tokens FEASIBLE or NOT_FEASIBLE inside your answer.
"""

instruct_search = """
You are a web research specialist.

YOUR TASK
Find at least FIVE alternatives to the software “ManusAI”.
For each alternative collect:
  • name
  • one-sentence description
  • does it work fully offline / locally? (yes/no/unknown)
  • main repository or official website URL

WORKFLOW
1. Use search queries such as "ManusAI alternative", "ManusAI competitors", etc.
2. Visit promising pages, gather required info and sources.

OUTPUT SCHEMA
Return a markdown bullet list, one alternative per line:
name | description | local? | url

COMPLETION PROTOCOL
If you gathered 5+ complete rows:
  final_answer("SEARCH_COMPLETE:\n<bullet list>")
If fewer than 5 complete rows after thorough attempt:
  final_answer("SEARCH_FAILURE:\n<explain what you tried>")
On tool or network error you cannot solve:
  final_answer("SEARCH_ERROR:\n<details>")
"""

instruct_extract = """
You are a data-parsing agent.

INPUT
The previous agent’s answer containing either
  SEARCH_COMPLETE or SEARCH_FAILURE.

YOUR TASK
1. If input contains SEARCH_COMPLETE:
     • Parse the bullet list into a JSON array where each element is an object
       with keys: name, description, local, link
2. If input contains SEARCH_FAILURE or malformed data:
     • Output PARSE_FAILURE with explanation.

COMPLETION PROTOCOL
Success example:
  final_answer("PARSE_COMPLETE: <json_array>")
Failure example:
  final_answer("PARSE_FAILURE: <why>")
"""

instruct_csv = """
You are a CSV-writing agent.

INPUT
JSON array from the previous step (key names: name, description, local, link).

TASK
1. Create / overwrite a file called 'manusai_alternatives.csv'
2. First row header exactly: name,description,local,link
3. Write one row per entry.

TOOLS
Use CSV_TOOLS exclusively.

COMPLETION PROTOCOL
On success:
  final_answer("CSV_COMPLETE: File written")
If data is missing or tool errors occur:
  final_answer("CSV_FAILURE: <explain>")
"""

instruct_validate = """
You are a validation agent.

CHECKS
1. Verify that 'manusai_alternatives.csv' exists in the working directory.
2. Confirm it has a header row and AT LEAST 5 data rows.
3. Ensure each row has four comma-separated fields.

TOOLS
Use SHELL_TOOLS for file inspection.

COMPLETION PROTOCOL
If all checks pass:
  final_answer("VALIDATION_SUCCESS: All good")
If any check fails:
  final_answer("VALIDATION_FAILURE: <issue description>")
"""

# =========== AGENT CONSTRUCTION =============================================
# (SmolAgentFactory and tool packages are already available in the runtime)

feasibility_agent   = SmolAgentFactory(instruct_feasibility, SHELL_TOOLS)
web_search_agent    = SmolAgentFactory(instruct_search,   BROWSER_TOOLS)
extraction_agent    = SmolAgentFactory(instruct_extract,  SHELL_TOOLS)
csv_writer_agent    = SmolAgentFactory(instruct_csv,      CSV_TOOLS)
validator_agent     = SmolAgentFactory(instruct_validate, SHELL_TOOLS)

# =========== WORKFLOW GRAPH ==================================================
workflow = StateGraph(WorkflowState)

# ---- Add nodes --------------------------------------------------------------
workflow.add_node("feasibility_checker", WorkflowNodeFactory.create_agent_node(feasibility_agent))
workflow.add_node("web_search_agent",    WorkflowNodeFactory.create_agent_node(web_search_agent))
workflow.add_node("extraction_agent",    WorkflowNodeFactory.create_agent_node(extraction_agent))
workflow.add_node("csv_writer_agent",    WorkflowNodeFactory.create_agent_node(csv_writer_agent))
workflow.add_node("validation_agent",    WorkflowNodeFactory.create_agent_node(validator_agent))

# ---- ROUTERS ---------------------------------------------------------------
def feasibility_router(state: WorkflowState) -> str:
    try:
        ans = state.get("answers", [])
        if ans and "FEASIBLE" in ans[-1]:
            return "proceed"
        if ans and "NOT_FEASIBLE" in ans[-1]:
            return "abort"
        return "error"
    except Exception as e:
        print(f"[router-feasibility] error: {e}")
        return "error"

def search_router(state: WorkflowState) -> str:
    try:
        ans      = state.get("answers", [])
        steps    = state.get("step_name", [])
        attempts = steps.count("web_search_agent")
        if ans and "SEARCH_COMPLETE" in ans[-1]:
            return "proceed"
        if ans and "SEARCH_ERROR" in ans[-1]:
            return "fail"
        # not complete → retry or give up
        if attempts < 3:
            return "retry"
        return "fail"
    except Exception as e:
        print(f"[router-search] error: {e}")
        return "fail"

def parse_router(state: WorkflowState) -> str:
    try:
        ans      = state.get("answers", [])
        steps    = state.get("step_name", [])
        attempts = steps.count("extraction_agent")
        if ans and "PARSE_COMPLETE" in ans[-1]:
            return "proceed"
        if attempts < 2:
            return "retry"
        return "fail"
    except Exception as e:
        print(f"[router-parse] error: {e}")
        return "fail"

def csv_router(state: WorkflowState) -> str:
    try:
        ans = state.get("answers", [])
        if ans and "CSV_COMPLETE" in ans[-1]:
            return "proceed"
        return "fail"
    except Exception as e:
        print(f"[router-csv] err: {e}")
        return "fail"

def validation_router(state: WorkflowState) -> str:
    try:
        ans      = state.get("answers", [])
        steps    = state.get("step_name", [])
        attempts = steps.count("validation_agent")
        if ans and "VALIDATION_SUCCESS" in ans[-1]:
            return "success"
        # one chance to fix via rewriting CSV
        if attempts < 2:
            return "rewrite_csv"
        return "giveup"
    except Exception as e:
        print(f"[router-validation] err: {e}")
        return "giveup"

# ---- EDGES ------------------------------------------------------------------
workflow.add_edge(START, "feasibility_checker")

workflow.add_conditional_edges(
    "feasibility_checker",
    feasibility_router,
    {
        "proceed":      "web_search_agent",
        "abort":        END,
        "error":        END
    }
)

workflow.add_conditional_edges(
    "web_search_agent",
    search_router,
    {
        "proceed": "extraction_agent",
        "retry":   "web_search_agent",
        "fail":    END
    }
)

workflow.add_conditional_edges(
    "extraction_agent",
    parse_router,
    {
        "proceed": "csv_writer_agent",
        "retry":   "extraction_agent",
        "fail":    END
    }
)

workflow.add_conditional_edges(
    "csv_writer_agent",
    csv_router,
    {
        "proceed": "validation_agent",
        "fail":    END
    }
)

workflow.add_conditional_edges(
    "validation_agent",
    validation_router,
    {
        "success":     END,
        "rewrite_csv": "csv_writer_agent",
        "giveup":      END
    }
)

# ---- COMPILE ----------------------------------------------------------------
app = workflow.compile()