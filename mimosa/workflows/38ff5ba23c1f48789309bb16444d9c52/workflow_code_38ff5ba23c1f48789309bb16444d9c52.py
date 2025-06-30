# LangGraph ‑ SmolAgent workflow: “Find ManusAI alternatives and store them in CSV”
# -------------------------------------------------------------------------------
# Assumptions:
# • SHELL_TOOLS, BROWSER_TOOLS, CSV_TOOLS    – already defined tool packages
# • SmolAgentFactory                          – already defined factory
# • WorkflowNodeFactory                       – already defined helper
# • WorkflowState                             – fixed typed-dict schema (see context)
# • We may only import from langgraph.graph   – everything else already available

from langgraph.graph import StateGraph, START, END



# ============ 1. AGENT PROMPTS ==================================================
# Every agent must finish with one of these EXACT trigger words:
#   STEP_SUCCESS   – task finished correctly
#   STEP_FAILURE   – task attempted but objective not met / info insufficient
#   STEP_ERROR     – technical / unexpected error

instruct_feasibility = """
You are an internet-research feasibility checker.

TASK
1. Verify that sufficient public information exists on ManusAI and comparable tools.
2. Confirm that nothing (license, robots.txt, etc.) prevents scraping and saving results to a CSV file.

WHEN DONE
• If both checks are positive, output a clear, concise statement of confirmation, then end with the EXACT word: STEP_SUCCESS
• If information is insufficient OR CSV storage seems disallowed, explain the problem and end with: STEP_FAILURE
• If you encounter a technical problem, explain it and end with: STEP_ERROR
"""

instruct_search = """
You are a focused web-search agent.

TASK
Run several distinct search queries such as:
  - "ManusAI alternative"
  - "tools like ManusAI"
  - "open-source ManusAI replacement"
Collect up to 20 unique result-page URLs that may list alternative tools.

OUTPUT FORMAT
• Provide the list of URLs, one per line.
• After the list, add: STEP_SUCCESS
• If no useful URLs are found, describe what you tried and add: STEP_FAILURE
• For technical issues, describe the error and add: STEP_ERROR
"""

instruct_scrape = """
You are a web-scraping agent.

INPUT
A list of candidate URLs obtained from the previous step.

TASK
Visit each URL. Extract names of tools that appear as alternatives to ManusAI and capture a primary link for each tool (official site or main repo).

OUTPUT FORMAT
Return lines formatted as  <tool_name> | <primary_link>
After listing all candidates:
• If at least one candidate extracted, finish with: STEP_SUCCESS
• If zero candidates, finish with: STEP_FAILURE
• On technical problems, finish with: STEP_ERROR
"""

instruct_collect_info = """
You are an information collector for software tools.

INPUT
A list of tool names with primary links.

TASK
For each tool:
1. Open its website or repository.
2. Extract a one-sentence description (max 30 words).
3. Determine if the software can be run fully locally without cloud services (answer YES or NO).

OUTPUT FORMAT
Return lines in this exact pipe-delimited schema:
   <name> | <description> | <local_yes_no> | <primary_link>
After all lines:
• End with: STEP_SUCCESS
• If any tool’s info cannot be retrieved, still output others and finish with STEP_SUCCESS.
• If none can be processed, finish with: STEP_FAILURE
• On technical problems, finish with: STEP_ERROR
"""

instruct_clean = """
You are a shell-style data cleaner.

INPUT
Multiple raw lines in the format
   <name> | <description> | <local_yes_no> | <primary_link>

TASK
1. Deduplicate by <name> (case-insensitive).
2. Trim whitespace, correct obvious formatting issues.
3. Ensure <local_yes_no> is either YES or NO.
4. Produce a clean list with the same four-column pipe-delimited schema.

OUTPUT FORMAT
Output only the cleaned lines, no header.
Then add: STEP_SUCCESS
If the input data is empty or unusable, add: STEP_FAILURE
For technical errors, add: STEP_ERROR
"""

instruct_csv_write = """
You are a CSV writer.

INPUT
Cleaned lines formatted as
   <name> | <description> | <local_yes_no> | <primary_link>

TASK
1. Create (or overwrite) a file named manusai_alternatives.csv.
2. Write the header: name,description,local,link
3. Append a row for each cleaned line (convert pipe delimiter to comma, keep fields quoted if needed).

UPON COMPLETION
• If writing succeeds, output STEP_SUCCESS
• On content problems (e.g., empty list), output STEP_FAILURE
• On technical errors, output STEP_ERROR
"""

instruct_final_verify = """
You are a CSV integrity verifier.

TASK
1. Open manusai_alternatives.csv.
2. Confirm the file exists, has header exactly 'name,description,local,link'
3. Ensure at least one data row exists and that every row has 4 columns.

OUTPUT
• If everything is correct, report summary statistics (# rows) and add: STEP_SUCCESS
• If any check fails, explain and add: STEP_FAILURE
• For technical errors, add: STEP_ERROR
"""

# ============ 2. AGENT INSTANTIATION ===========================================
feasibility_agent   = SmolAgentFactory(instruct_feasibility,   BROWSER_TOOLS)
search_agent        = SmolAgentFactory(instruct_search,        BROWSER_TOOLS)
scrape_agent        = SmolAgentFactory(instruct_scrape,        BROWSER_TOOLS)
collect_agent       = SmolAgentFactory(instruct_collect_info,  BROWSER_TOOLS)
clean_agent         = SmolAgentFactory(instruct_clean,         SHELL_TOOLS)
csv_writer_agent    = SmolAgentFactory(instruct_csv_write,     CSV_TOOLS)
verify_agent        = SmolAgentFactory(instruct_final_verify,  CSV_TOOLS)

# ============ 3. WORKFLOW GRAPH ===============================================
workflow = StateGraph(WorkflowState)

# ---- 3.1 Add agent nodes ------------------------------------------------------
workflow.add_node("feasibility_checker", WorkflowNodeFactory.create_agent_node(feasibility_agent))
workflow.add_node("searcher",            WorkflowNodeFactory.create_agent_node(search_agent))
workflow.add_node("scraper",             WorkflowNodeFactory.create_agent_node(scrape_agent))
workflow.add_node("info_collector",      WorkflowNodeFactory.create_agent_node(collect_agent))
workflow.add_node("data_cleaner",        WorkflowNodeFactory.create_agent_node(clean_agent))
workflow.add_node("csv_writer",          WorkflowNodeFactory.create_agent_node(csv_writer_agent))
workflow.add_node("final_verifier",      WorkflowNodeFactory.create_agent_node(verify_agent))

# ============ 4. ROUTING UTILITIES ============================================
def router_factory(current_step:str, success_phrase:str, next_step:str, fallback_step:str):
    """
    Returns a routing function specific to 'current_step'.
    Logic:
    • If last answer contains success_phrase  -> go 'next_step'
    • Otherwise:
        – retry same step if attempts < 3
        – else route to 'fallback_step'
    • On exception -> END  (emergency fallback)
    """
    def _router(state: WorkflowState) -> str:
        try:
            answers     = state.get("answers", [])
            step_names  = state.get("step_name", [])
            # Validate state
            if not answers or not step_names:
                print(f"[{current_step} router] ⚠️ Missing history → fallback")
                return "fallback"

            last_answer = answers[-1]
            attempts    = step_names.count(current_step)
            print(f"[{current_step} router] Attempt #{attempts} – evaluating…")

            if success_phrase in last_answer:
                print(f"[{current_step} router] ✅ Success detected")
                return "next"

            # FAILURE or ERROR from agent
            if attempts < 3:
                print(f"[{current_step} router] 🔄 Retrying (attempt {attempts+1})")
                return "retry"
            else:
                print(f"[{current_step} router] ⛔ Max retries – using fallback")
                return "fallback"
        except Exception as e:
            print(f"[{current_step} router] 💥 Exception {e} – emergency end")
            return "emergency"
    return _router

# ============ 5. EDGE DEFINITIONS WITH FALLBACKS ==============================
# Helper function to add conditional edges with standard labels
def add_edges(step_name:str, router, next_node:str, fallback_node:str):
    workflow.add_conditional_edges(
        step_name,
        router,
        {
            "next":      next_node,
            "retry":     step_name,           # self-loop retry
            "fallback":  fallback_node,
            "emergency": END                 # ultimate fail-safe
        }
    )

# START -> Feasibility ----------------------------------------------------------
workflow.add_edge(START, "feasibility_checker")
add_edges(
    "feasibility_checker",
    router_factory("feasibility_checker", "STEP_SUCCESS", "searcher", END),
    next_node="searcher",
    fallback_node=END
)

# Feasibility OK -> Search ------------------------------------------------------
add_edges(
    "searcher",
    router_factory("searcher", "STEP_SUCCESS", "scraper", "feasibility_checker"),
    next_node="scraper",
    fallback_node="feasibility_checker"     # fallback ① : go back one step
)

# Search -> Scrape --------------------------------------------------------------
add_edges(
    "scraper",
    router_factory("scraper", "STEP_SUCCESS", "info_collector", "searcher"),
    next_node="info_collector",
    fallback_node="searcher"                # fallback ②
)

# Scrape -> Info Collector ------------------------------------------------------
add_edges(
    "info_collector",
    router_factory("info_collector", "STEP_SUCCESS", "data_cleaner", "scraper"),
    next_node="data_cleaner",
    fallback_node="scraper"
)

# Info Collector -> Data Cleaner -----------------------------------------------
add_edges(
    "data_cleaner",
    router_factory("data_cleaner", "STEP_SUCCESS", "csv_writer", "info_collector"),
    next_node="csv_writer",
    fallback_node="info_collector"
)

# Data Cleaner -> CSV Writer ----------------------------------------------------
add_edges(
    "csv_writer",
    router_factory("csv_writer", "STEP_SUCCESS", "final_verifier", "data_cleaner"),
    next_node="final_verifier",
    fallback_node="data_cleaner"
)

# CSV Writer -> Final Verifier --------------------------------------------------
add_edges(
    "final_verifier",
    router_factory("final_verifier", "STEP_SUCCESS", END, "csv_writer"),
    next_node=END,
    fallback_node="csv_writer"
)

# ============ 6. COMPILE WORKFLOW =============================================
app = workflow.compile()