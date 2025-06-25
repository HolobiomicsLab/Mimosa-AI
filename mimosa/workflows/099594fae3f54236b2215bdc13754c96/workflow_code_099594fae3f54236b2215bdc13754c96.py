# =====================  MANDATORY IMPORTS  =====================
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Callable
import random                   # used for simple exponential back-off demo

# =====================  STATE SCHEMA  ==========================
class Action(TypedDict):
    tool: str
    inputs: dict

class Observation(TypedDict):
    data: str

class WorkflowState(TypedDict):
    step_name: List[str]
    actions: List[Action]
    observations: List[Observation]
    rewards: List[float]
    answers: List[str]
    success: List[bool]


# =====================  AGENT INSTRUCTIONS  ====================
# --- Search Agents -------------------------------------------------
instruct_search = """
You are a focused web search agent.

GOAL
- Locate the most reliable, up-to-date instructions for installing GPAW on macOS with Apple-Silicon (ARM).

TOOLS
- Use the provided browser tools to query the web and visit official GPAW / Conda / Homebrew resources.

OUTPUT FORMAT
- Summarise the 3–5 most relevant pages with their URLs.
- Propose a concise installation plan (max 6 shell commands).

ROUTING WORDS
If good instructions & plan ready:  SEARCH_COMPLETE
If you cannot locate reliable info: SEARCH_FAILURE
On unexpected tool error:          GIVE_UP
"""

instruct_alt_search = """
You are an ALTERNATIVE web search agent.

GOAL
- Use different keywords (e.g., “GPAW Apple M1”, “GPAW ARM64 brew”) to find installation guides.

Same OUTPUT FORMAT & ROUTING WORDS as primary search agent.
"""


# --- Extraction Agent ----------------------------------------------
instruct_extract = """
You are an extraction agent.

INPUT
- The last observation contains raw web page snippets & URLs.

TASK
- Extract ONLY the necessary shell/conda/brew commands required to install GPAW on Apple-Silicon.
- Preserve execution order.

OUTPUT
- A clean ordered list of commands.

ROUTING WORDS
If commands extracted:   EXTRACTION_COMPLETE
If extraction failed:    EXTRACTION_FAILURE
On tool problems:        GIVE_UP
"""


# --- Validation Agent ----------------------------------------------
instruct_validate = """
You are a validation agent.

INPUT
- Candidate installation command list from previous step.

TASK
1. Verify commands are compatible with macOS ARM64.
2. Check for mandatory prerequisites (python>=3.10, libxc, OpenBLAS, etc.)
3. Suggest fixes if any incompatibility detected.

OUTPUT
- Validated/fixed command list ready for execution.

ROUTING WORDS
Validation successful:   VALIDATION_COMPLETE
Validation failed:       VALIDATION_FAILURE
Unexpected error:        GIVE_UP
"""


# --- Installation Agent --------------------------------------------
instruct_install = """
You are an installation agent.

INPUT
- A validated, ordered list of shell commands.

TASK
- Execute each command with available shell tools.
- Capture output & error logs (truncate to 512 chars each to keep context small).

OUTPUT
- Installation logs summary.

ROUTING WORDS
Install ok:              INSTALL_COMPLETE
Install failed:          INSTALL_FAILURE
Tool error:              GIVE_UP
"""


# --- Verification Agent --------------------------------------------
instruct_verify = """
You are an installation verification agent.

TASK
- Run `python -c "import gpaw, platform, sys; print(gpaw.__version__)"` in shell.
- Confirm module import is successful and architecture is arm64.

OUTPUT
- Version string and architecture confirmation.

ROUTING WORDS
If ok:                   VERIFY_COMPLETE
If import fails:         VERIFY_FAILURE
Tool error:              GIVE_UP
"""


# --- Summariser -----------------------------------------------------
instruct_summary = """
You are a summarisation agent.

TASK
- Produce a final human-readable report containing:
  * Whether GPAW is installed and working.
  * Version & architecture info if available.
  * Any warnings, fallback steps, or manual actions left for the user.

Always finish with: SUMMARY_COMPLETE
"""


# =====================  AGENT CREATION  ============================
# NOTE: SmolAgentFactory is presumed to be already provided in the execution environment.
smol_search        = SmolAgentFactory(instruct_search,      BROWSER_TOOLS)
smol_alt_search    = SmolAgentFactory(instruct_alt_search,  BROWSER_TOOLS)
smol_extract       = SmolAgentFactory(instruct_extract,     BROWSER_TOOLS)
smol_validate      = SmolAgentFactory(instruct_validate,    BROWSER_TOOLS)
smol_install       = SmolAgentFactory(instruct_install,     SHELL_TOOLS)
smol_verify        = SmolAgentFactory(instruct_verify,      SHELL_TOOLS)
smol_summary       = SmolAgentFactory(instruct_summary,     [])              # no tools needed

# =====================  ROUTER FACTORY  ============================

def make_router(current: str,
                success_keyword: str,
                next_step: str,
                retry_step: str,
                alt_step: str = None,
                max_retries: int = 3) -> Callable[[WorkflowState], str]:
    """
    Generic router constructor with:
    - success path
    - retry path (same agent)
    - alternate path (optional)
    - emergency fallback -> summariser
    """
    def router(state: WorkflowState) -> str:
        print(f"---- ROUTER for {current} ----")
        try:
            answers = state.get("answers", [])
            steps   = state.get("step_name", [])
            last_answer = answers[-1] if answers else ""
            print(f"Last answer: {last_answer[:150]}")

            # SUCCESS PATH ---------------------------------------------------
            if success_keyword in last_answer:
                print(f"✅ Detected '{success_keyword}' → {next_step}")
                state["step_name"].append(next_step)
                return next_step

            # RETRY PATH -----------------------------------------------------
            retry_count = steps.count(current)
            if retry_count < max_retries:
                backoff = 2 ** retry_count + random.random()
                print(f"🔄 {current} failed, retry #{retry_count+1} after backoff {backoff:.2f}s")
                state["step_name"].append(retry_step)
                return retry_step

            # ALTERNATE PATH -------------------------------------------------
            if alt_step:
                print(f"➡️ Switching to alternate step: {alt_step}")
                state["step_name"].append(alt_step)
                return alt_step

            # EMERGENCY FALLBACK --------------------------------------------
            print("🚨 Max retries reached & no alternate — going to summariser")
            state["step_name"].append("summariser")
            return "summariser"

        except Exception as e:
            print(f"💥 Router exception {e} — emergency fallback to summariser")
            state["step_name"].append("summariser")
            return "summariser"
    return router


# =====================  WORKFLOW BUILD  ============================
workflow = StateGraph(WorkflowState)

# -------- Add Agent Nodes -----------------------------------------
workflow.add_node("web_searcher",            WorkflowNodeFactory.create_agent_node(smol_search))
workflow.add_node("alt_web_searcher",        WorkflowNodeFactory.create_agent_node(smol_alt_search))
workflow.add_node("instruction_extractor",   WorkflowNodeFactory.create_agent_node(smol_extract))
workflow.add_node("instruction_validator",   WorkflowNodeFactory.create_agent_node(smol_validate))
workflow.add_node("installer",               WorkflowNodeFactory.create_agent_node(smol_install))
workflow.add_node("verifier",                WorkflowNodeFactory.create_agent_node(smol_verify))
workflow.add_node("summariser",              WorkflowNodeFactory.create_agent_node(smol_summary))

# -------- Edges & Conditional Routing -----------------------------
workflow.add_edge(START, "web_searcher")

# 1. Search → Extract / Retry / Alt / Summary
workflow.add_conditional_edges(
    "web_searcher",
    make_router("web_searcher", "SEARCH_COMPLETE",
                next_step="instruction_extractor",
                retry_step="web_searcher",
                alt_step="alt_web_searcher",
                max_retries=3),
    {
        "instruction_extractor": "instruction_extractor",
        "web_searcher": "web_searcher",
        "alt_web_searcher": "alt_web_searcher",
        "summariser": "summariser"
    }
)

# 1b. Alt-Search → Extract / Retry Alt / Summary
workflow.add_conditional_edges(
    "alt_web_searcher",
    make_router("alt_web_searcher", "SEARCH_COMPLETE",
                next_step="instruction_extractor",
                retry_step="alt_web_searcher",
                alt_step=None,
                max_retries=2),
    {
        "instruction_extractor": "instruction_extractor",
        "alt_web_searcher": "alt_web_searcher",
        "summariser": "summariser"
    }
)

# 2. Extract → Validate / Retry / Summary
workflow.add_conditional_edges(
    "instruction_extractor",
    make_router("instruction_extractor", "EXTRACTION_COMPLETE",
                next_step="instruction_validator",
                retry_step="instruction_extractor",
                alt_step="web_searcher",
                max_retries=2),
    {
        "instruction_validator": "instruction_validator",
        "instruction_extractor": "instruction_extractor",
        "web_searcher": "web_searcher",
        "summariser": "summariser"
    }
)

# 3. Validate → Install / Retry / Summary
workflow.add_conditional_edges(
    "instruction_validator",
    make_router("instruction_validator", "VALIDATION_COMPLETE",
                next_step="installer",
                retry_step="instruction_validator",
                alt_step="web_searcher",
                max_retries=2),
    {
        "installer": "installer",
        "instruction_validator": "instruction_validator",
        "web_searcher": "web_searcher",
        "summariser": "summariser"
    }
)

# 4. Install → Verify / Retry / Summary
workflow.add_conditional_edges(
    "installer",
    make_router("installer", "INSTALL_COMPLETE",
                next_step="verifier",
                retry_step="installer",
                alt_step=None,
                max_retries=2),
    {
        "verifier": "verifier",
        "installer": "installer",
        "summariser": "summariser"
    }
)

# 5. Verify → Summary / Retry / Summary
workflow.add_conditional_edges(
    "verifier",
    make_router("verifier", "VERIFY_COMPLETE",
                next_step="summariser",
                retry_step="verifier",
                alt_step=None,
                max_retries=1),
    {
        "summariser": "summariser",
        "verifier": "verifier"
    }
)

# 6. Summariser → END
workflow.add_edge("summariser", END)

# =====================  COMPILE WORKFLOW  ==========================
app = workflow.compile()

# ------------- The compiled `app` can now be executed --------------