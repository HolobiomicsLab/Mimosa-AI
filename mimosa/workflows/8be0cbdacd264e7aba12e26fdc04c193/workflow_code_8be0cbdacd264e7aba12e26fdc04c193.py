# ============================================================
# MANUSAI ALTERNATIVES WORKFLOW
# ============================================================
#
# PRE-DECLARED RUNTIME CONTEXT (already loaded, DO NOT REDEFINE)
# - WorkflowState schema
# - SmolAgentFactory
# - WorkflowNodeFactory
# - Tool packages: SHELL_TOOLS, BROWSER_TOOLS, CSV_TOOLS
#
# ============================================================
from langgraph.graph import StateGraph, START, END

# ------------------------------------------------------------
# 1.  AGENT PROMPTS  (one atomic responsibility each)
# ------------------------------------------------------------

instruct_feasibility = """
You are a feasibility assessment agent.

## TASK
Determine whether sufficient public information about ManusAI and competing tools exists online and if extracting that information for analysis is legally permissible.

## HOW TO WORK
1. Use web-search tools to verify that ManusAI has public documentation or articles.
2. Quickly sample search results for terms like “ManusAI features”, “ManusAI comparison”.
3. Check if there are no explicit restrictions against scraping/public usage.

## COMPLETION PROTOCOL
- If information is publicly available AND permissible → respond with:
  FEASIBILITY_CONFIRMED: [one-sentence justification]

- If information is NOT available or scraping is disallowed → respond with:
  FEASIBILITY_DENIED: [reason]

- On unrecoverable technical error → explain the issue and finish with:
  GIVE_UP
"""

instruct_keyword = """
You are a keyword-search agent.

## TASK
Generate an INITIAL candidate list of alternatives to ManusAI by performing multiple web searches:
- “ManusAI alternatives”
- “tools similar to ManusAI”
- “ManusAI open-source replacement”
Aggregate names (only names!) that appear.

## COMPLETION PROTOCOL
- When at least 5 distinct candidates are collected → respond with:
  SEARCH_COMPLETE: [comma-separated list of names]

- If fewer than 5 viable candidates after exhaustive searching → respond with:
  SEARCH_FAILED: [explain attempts]

- On technical issues → respond with GIVE_UP
"""

instruct_scraper = """
You are a result-scraper agent.

## TASK
For every candidate name provided, visit the first 2 pages of search results.
Collect each candidate’s OFFICIAL website or main repository link (GitHub etc).

## OUTPUT FORMAT
SCRAPE_COMPLETE:
name1 | link1
name2 | link2
...

SCRAPE_FAILED: [explanation]

GIVE_UP (for technical errors)
"""

instruct_details = """
You are a detail-extractor agent.

## TASK
For each "name | link" pair provided:
1. Open the link.
2. Extract a concise 1-sentence description.
3. Decide if the software can run locally (YES if self-hostable/offline, else NO).

## OUTPUT FORMAT
DETAILS_COMPLETE:
name | description | local(YES/NO) | link
...

DETAILS_FAILED: [explanation]

GIVE_UP (technical errors)
"""

instruct_curator = """
You are a data-curator agent.

## TASK
Using the rows received, build an in-memory CSV with these exact headers:
name,description,local,link

## COMPLETION PROTOCOL
CURATION_COMPLETE: CSV_READY
CURATION_FAILED: [reason]
GIVE_UP
"""

instruct_writer = """
You are a CSV writer agent.

## TASK
Write (append) every curated row into the current CSV object.

## COMPLETION PROTOCOL
WRITE_COMPLETE: ROWS_WRITTEN
WRITE_FAILED: [reason]
GIVE_UP
"""

instruct_qc = """
You are a quality-checker agent.

## TASK
Validate the CSV:
- Correct headers and order (name,description,local,link)
- No empty fields
- UTF-8 safe

## COMPLETION PROTOCOL
QC_COMPLETE: CSV_VALID
QC_FAILED: [detailed issues]
GIVE_UP
"""

instruct_save = """
You are a file-saver agent.

## TASK
Persist the in-memory CSV to disk with filename manusai_alternatives.csv in the current working directory.

## COMPLETION PROTOCOL
SAVE_COMPLETE: FILE_SAVED
SAVE_FAILED: [reason]
GIVE_UP
"""

# ------------------------------------------------------------
# 2.  AGENT CREATION
# ------------------------------------------------------------
feasibility_agent   = SmolAgentFactory(instruct_feasibility, BROWSER_TOOLS)
keyword_agent       = SmolAgentFactory(instruct_keyword,   BROWSER_TOOLS)
scraper_agent       = SmolAgentFactory(instruct_scraper,   BROWSER_TOOLS)
details_agent       = SmolAgentFactory(instruct_details,   BROWSER_TOOLS)
curator_agent       = SmolAgentFactory(instruct_curator,   CSV_TOOLS)
writer_agent        = SmolAgentFactory(instruct_writer,    CSV_TOOLS)
qc_agent            = SmolAgentFactory(instruct_qc,        CSV_TOOLS)
save_agent          = SmolAgentFactory(instruct_save,      SHELL_TOOLS)

# ------------------------------------------------------------
# 3. WORKFLOW GRAPH INITIALISATION
# ------------------------------------------------------------
workflow = StateGraph(WorkflowState)

# ------------------------------------------------------------
# 4.  ADD NODES
# ------------------------------------------------------------
workflow.add_node("feasibility_checker", WorkflowNodeFactory.create_agent_node(feasibility_agent))
workflow.add_node("keyword_searcher",    WorkflowNodeFactory.create_agent_node(keyword_agent))
workflow.add_node("result_scraper",      WorkflowNodeFactory.create_agent_node(scraper_agent))
workflow.add_node("detail_extractor",    WorkflowNodeFactory.create_agent_node(details_agent))
workflow.add_node("data_curator",        WorkflowNodeFactory.create_agent_node(curator_agent))
workflow.add_node("data_writer",         WorkflowNodeFactory.create_agent_node(writer_agent))
workflow.add_node("quality_checker",     WorkflowNodeFactory.create_agent_node(qc_agent))
workflow.add_node("file_saver",          WorkflowNodeFactory.create_agent_node(save_agent))

# ------------------------------------------------------------
# 5.  ROUTING FUNCTIONS  (retry logic + fallback)
# ------------------------------------------------------------
MAX_RETRIES = 3   # universal retry cap

def _retry_count(state: WorkflowState, step: str) -> int:
    return state.get("step_name", []).count(step)

def router_feasibility(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        if "FEASIBILITY_CONFIRMED" in answer:
            return "keyword_searcher"
        if "FEASIBILITY_DENIED" in answer:
            return END
        # Unexpected output → retry with cap
        if _retry_count(state, "feasibility_checker") < MAX_RETRIES:
            return "feasibility_checker"
        return END
    except Exception as e:
        print(f"Routing Error (feasibility): {e}")
        return END

def router_keyword(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        if "SEARCH_COMPLETE" in answer:
            return "result_scraper"
        if "SEARCH_FAILED" in answer:
            # fallback: try again, else end
            if _retry_count(state, "keyword_searcher") < MAX_RETRIES:
                return "keyword_searcher"
            return END
        # Unknown response
        if _retry_count(state, "keyword_searcher") < MAX_RETRIES:
            return "keyword_searcher"
        return END
    except Exception as e:
        print(f"Routing Error (keyword): {e}")
        return END

def router_scraper(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        if "SCRAPE_COMPLETE" in answer:
            return "detail_extractor"
        if "SCRAPE_FAILED" in answer:
            # fallback to keyword_searcher to broaden search
            return "keyword_searcher"
        if _retry_count(state, "result_scraper") < MAX_RETRIES:
            return "result_scraper"
        return END
    except Exception as e:
        print(f"Routing Error (scraper): {e}")
        return END

def router_details(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        if "DETAILS_COMPLETE" in answer:
            return "data_curator"
        if "DETAILS_FAILED" in answer:
            # fallback to result_scraper (maybe links wrong)
            return "result_scraper"
        if _retry_count(state, "detail_extractor") < MAX_RETRIES:
            return "detail_extractor"
        return END
    except Exception as e:
        print(f"Routing Error (details): {e}")
        return END

def router_curator(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        if "CURATION_COMPLETE" in answer:
            return "data_writer"
        if "CURATION_FAILED" in answer:
            return "detail_extractor"
        if _retry_count(state, "data_curator") < MAX_RETRIES:
            return "data_curator"
        return END
    except Exception as e:
        print(f"Routing Error (curator): {e}")
        return END

def router_writer(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        if "WRITE_COMPLETE" in answer:
            return "quality_checker"
        if "WRITE_FAILED" in answer:
            return "data_curator"
        if _retry_count(state, "data_writer") < MAX_RETRIES:
            return "data_writer"
        return END
    except Exception as e:
        print(f"Routing Error (writer): {e}")
        return END

def router_qc(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        if "QC_COMPLETE" in answer:
            return "file_saver"
        if "QC_FAILED" in answer:
            # fallback: rewrite csv then re-qc
            return "data_writer"
        if _retry_count(state, "quality_checker") < MAX_RETRIES:
            return "quality_checker"
        return END
    except Exception as e:
        print(f"Routing Error (QC): {e}")
        return END

def router_save(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        if "SAVE_COMPLETE" in answer:
            return END
        if "SAVE_FAILED" in answer:
            return "file_saver" if _retry_count(state, "file_saver") < MAX_RETRIES else END
        # unknown
        if _retry_count(state, "file_saver") < MAX_RETRIES:
            return "file_saver"
        return END
    except Exception as e:
        print(f"Routing Error (save): {e}")
        return END

# ------------------------------------------------------------
# 6.  EDGES & CONDITIONAL ROUTES
# ------------------------------------------------------------
workflow.add_edge(START, "feasibility_checker")

workflow.add_conditional_edges("feasibility_checker", router_feasibility, {
    "keyword_searcher": "keyword_searcher",
    END: END,
    "feasibility_checker": "feasibility_checker"
})

workflow.add_conditional_edges("keyword_searcher", router_keyword, {
    "result_scraper": "result_scraper",
    "keyword_searcher": "keyword_searcher",
    END: END
})

workflow.add_conditional_edges("result_scraper", router_scraper, {
    "detail_extractor": "detail_extractor",
    "keyword_searcher": "keyword_searcher",
    "result_scraper": "result_scraper",
    END: END
})

workflow.add_conditional_edges("detail_extractor", router_details, {
    "data_curator": "data_curator",
    "result_scraper": "result_scraper",
    "detail_extractor": "detail_extractor",
    END: END
})

workflow.add_conditional_edges("data_curator", router_curator, {
    "data_writer": "data_writer",
    "detail_extractor": "detail_extractor",
    "data_curator": "data_curator",
    END: END
})

workflow.add_conditional_edges("data_writer", router_writer, {
    "quality_checker": "quality_checker",
    "data_curator": "data_curator",
    "data_writer": "data_writer",
    END: END
})

workflow.add_conditional_edges("quality_checker", router_qc, {
    "file_saver": "file_saver",
    "data_writer": "data_writer",
    "quality_checker": "quality_checker",
    END: END
})

workflow.add_conditional_edges("file_saver", router_save, {
    END: END,
    "file_saver": "file_saver"
})

# ------------------------------------------------------------
# 7.  COMPILE WORKFLOW
# ------------------------------------------------------------
manusai_workflow_app = workflow.compile()