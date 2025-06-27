############################################################
# LANGGRAPH – SMOLAGENT WORKFLOW :  “mzmind batch CLI guide”
############################################################
# NOTE : Everything used below (StateGraph, WorkflowNodeFactory,
#        SmolAgentFactory, START, END, the 3 *TOOLS lists, and the 
#        WorkflowState schema) is ALREADY loaded in the execution
#        environment as per the system specification.               

from langgraph.graph import StateGraph, START, END

########################
# 1.  WORKFLOW OBJECT  #
########################
workflow = StateGraph(WorkflowState)     # <- fixed schema


##############################################
# 2.  AGENT PROMPTS  (single-responsibility) #
##############################################

# A)  Primary web researcher  ──────────────────────────────
prompt_web_primary = """
You are an intense web-research agent focused ONLY on finding
authoritative resources about **using “mzmind” in batch / command-line
mode**.

TASK STEPS
1. Run deep web searches with varied keywords: 
   "mzmind batch file", "mzmind command line", "mzmind .bat",
   "mzmind cli examples", etc.
2. Visit promising URLs and capture snippets of:
   - exact command invocations
   - explanation paragraphs
   - any sample *.bat or *.cmd* files shown
3. Save every URL you visited.

OUTPUT FORMAT (MANDATORY)
If you collected ≥3 distinct command-line examples, reply with:
    RESEARCH_COMPLETE
then in a Markdown code-block, list:
    • URL
    • the command(s)
    • any accompanying explanation
Else, reply with:
    RESEARCH_FAILURE
and explain exactly what you tried.

ERRORS
If browser tool errors occur, reply:
    GIVE_UP
"""
agent_web_primary = SmolAgentFactory(prompt_web_primary, BROWSER_TOOLS)

# B)  Secondary web researcher (fallback)  ─────────────────
prompt_web_secondary = """
You are a backup researcher. Your job is ONLY to gather resources
that the previous agent missed on “mzmind” batch CLI usage.

Use alternative search engines, language variants and file-type
filters (pdf, ppt, doc). Provide fresh links not already listed.

Follow same success / failure keywords:
    RESEARCH_COMPLETE  or  RESEARCH_FAILURE  or  GIVE_UP
"""
agent_web_secondary = SmolAgentFactory(prompt_web_secondary, BROWSER_TOOLS)

# C)  Content extractor (parses raw pages)  ────────────────
prompt_extractor = """
You are a concise extraction agent.

INPUT: previous observations containing raw web page html/text.

TASK
1. Identify every concrete `mzmind ...` command.
2. Extract any referenced *.bat* or *.cmd* content.
3. Normalise white-space, remove ads/nav text.
4. Store each example in CSV form columns:
    url, command, description
   (use the provided CSV tools).

Success keyword: EXTRACT_COMPLETE
Failure keyword: EXTRACT_FAILURE
Error keyword:   GIVE_UP
"""
agent_extractor = SmolAgentFactory(prompt_extractor, CSV_TOOLS)

# D)  Example generator (creates new .bat demos) ───────────
prompt_example_gen = """
You are an example-writer agent.

INPUT: a CSV list (url, command, description).

TASK
1. For **each** extracted command, write a self-contained *.bat*
   file example demonstrating realistic batch usage.
2. Add echo statements to explain each main step.
3. Include at least 3 different use-cases overall.
4. Output each example inside separate Markdown ```batch blocks.

Success keyword: EXAMPLES_COMPLETE
Failure keyword: EXAMPLES_FAILURE
Error keyword:   GIVE_UP
"""
agent_example_gen = SmolAgentFactory(prompt_example_gen, SHELL_TOOLS)

# E)  Validator (dry-runs / sanity checks)  ────────────────
prompt_validator = """
You are a validator agent.

INPUT: batch examples created in the previous step.

TASK
1. Perform dry-run syntax check with Windows cmd /Q /K to ensure
   commands parse (use shell-tools).
2. Flag any undefined variables.
3. Produce a summary table: file_name, valid?(yes/no), notes

Success keyword: VALIDATE_SUCCESS    (all examples valid)
Failure keyword: VALIDATE_FAILURE    (some invalid – list issues)
Error keyword:   GIVE_UP
"""
agent_validator = SmolAgentFactory(prompt_validator, SHELL_TOOLS)

# F)  Formatter (final answer builder)  ─────────────────────
prompt_formatter = """
You are the final report formatter.

INPUT: 
  - validated examples
  - earlier CSV with sources

TASK
Compose ONE comprehensive reply containing:
  ▸ Executive summary of how to use mzmind via batch.
  ▸ Bullet list of discovered commands & sources.
  ▸ The validated *.bat* examples (already provided).
  ▸ Tips / best-practices.

End with exact token: FORMAT_COMPLETE
"""
agent_formatter = SmolAgentFactory(prompt_formatter, [])   # no external tools


#########################################
# 3.  CONVERT AGENTS INTO GRAPH NODES   #
#########################################
workflow.add_node("web_primary",     WorkflowNodeFactory.create_agent_node(agent_web_primary))
workflow.add_node("web_secondary",   WorkflowNodeFactory.create_agent_node(agent_web_secondary))
workflow.add_node("extractor",       WorkflowNodeFactory.create_agent_node(agent_extractor))
workflow.add_node("example_gen",     WorkflowNodeFactory.create_agent_node(agent_example_gen))
workflow.add_node("validator",       WorkflowNodeFactory.create_agent_node(agent_validator))
workflow.add_node("formatter",       WorkflowNodeFactory.create_agent_node(agent_formatter))


##############################################
# 4.  ROUTING / ERROR-HANDLING FUNCTIONS     #
##############################################
# Helper: safe getter
def _get_latest(state: WorkflowState, key: str, default=None):
    lst = state.get(key, [])
    return lst[-1] if lst else default


# 4.1  Router after primary researcher
def route_after_primary(state: WorkflowState) -> str:
    try:
        answer = _get_latest(state, "answers", "")
        retries = state["step_name"].count("web_primary")
        if "RESEARCH_COMPLETE" in answer:
            return "extractor"
        if "GIVE_UP" in answer:
            print("Primary researcher gave up – switching to secondary")
            return "web_secondary"
        # failure but retries remaining?
        if retries < 2:
            print(f"Retrying primary research (attempt #{retries+1})")
            return "web_primary"
        print("Max retries reached – fallback to secondary researcher")
        return "web_secondary"
    except Exception as e:
        print(f"🚨 route_after_primary error: {e}")
        return "web_secondary"



# 4.2  Router after secondary researcher
def route_after_secondary(state: WorkflowState) -> str:
    try:
        answer = _get_latest(state, "answers", "")
        if "RESEARCH_COMPLETE" in answer:
            return "extractor"
        # if both researchers failed – abort workflow gracefully
        print("Both researchers failed – ending flow")
        return END
    except Exception as e:
        print(f"🚨 route_after_secondary error: {e}")
        return END



# 4.3  Router after extractor
def route_after_extractor(state: WorkflowState) -> str:
    try:
        answer = _get_latest(state, "answers", "")
        retries = state["step_name"].count("extractor")
        if "EXTRACT_COMPLETE" in answer:
            return "example_gen"
        if "GIVE_UP" in answer:
            return END
        if retries < 2:
            print("Retrying extractor")
            return "extractor"
        # fallback: gather more data by re-invoking secondary researcher
        print("Extractor stuck – going back to secondary research")
        return "web_secondary"
    except Exception as e:
        print(f"🚨 route_after_extractor error: {e}")
        return END



# 4.4  Router after example generator
def route_after_examples(state: WorkflowState) -> str:
    try:
        answer = _get_latest(state, "answers", "")
        retries = state["step_name"].count("example_gen")
        if "EXAMPLES_COMPLETE" in answer:
            return "validator"
        if "GIVE_UP" in answer:
            return END
        if retries < 2:
            return "example_gen"
        # cannot craft examples – end
        return END
    except Exception as e:
        print(f"🚨 route_after_examples error: {e}")
        return END



# 4.5  Router after validator
def route_after_validator(state: WorkflowState) -> str:
    try:
        answer = _get_latest(state, "answers", "")
        if "VALIDATE_SUCCESS" in answer:
            return "formatter"
        if "VALIDATE_FAILURE" in answer:
            # allow one fix attempt by generator
            gen_retries = state["step_name"].count("example_gen")
            if gen_retries < 3:
                print("Validation failed – regenerating examples")
                return "example_gen"
            # still failing, but let formatter mention issues
            return "formatter"
        return END
    except Exception as e:
        print(f"🚨 route_after_validator error: {e}")
        return END



# 4.6  Router after formatter (guaranteed END)
def route_after_formatter(state: WorkflowState) -> str:
    return END



#########################################
# 5.  GRAPH EDGES (with fallbacks)      #
#########################################
workflow.add_edge(START, "web_primary")

workflow.add_conditional_edges(
    "web_primary",
    route_after_primary,
    {
        "extractor":     "extractor",
        "web_primary":   "web_primary",      # retry
        "web_secondary": "web_secondary",
        END: END
    }
)

workflow.add_conditional_edges(
    "web_secondary",
    route_after_secondary,
    {
        "extractor": "extractor",
        END: END
    }
)

workflow.add_conditional_edges(
    "extractor",
    route_after_extractor,
    {
        "example_gen":   "example_gen",
        "extractor":     "extractor",        # retry
        "web_secondary": "web_secondary",
        END: END
    }
)

workflow.add_conditional_edges(
    "example_gen",
    route_after_examples,
    {
        "validator":   "validator",
        "example_gen": "example_gen",        # retry
        END: END
    }
)

workflow.add_conditional_edges(
    "validator",
    route_after_validator,
    {
        "formatter":   "formatter",
        "example_gen": "example_gen",
        END: END
    }
)

workflow.add_edge("formatter", END)   # formatter always goes to END


#############################
# 6.  COMPILE WORKFLOW      #
#############################
app = workflow.compile()

# The resulting `app` object can now be run with an initial WorkflowState:
# >>> app.invoke({"step_name":[], "actions":[], "observations":[],
#                 "rewards":[], "answers":[], "success":[]})