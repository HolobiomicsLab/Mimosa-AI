
We want to design a AI agent system that could in a automated way reproduce science research paper. It would automatically read the paper, install required tools, try to reproduce experiments and validate.

Given the vast numbers of tools and scientic method across different field, we believe using rigid agentic architecture is flawed.

We are exploring the idea of polymorphic meta-agent with a agent-within-agent pattern. based on the previous task result and next task goal, a meta-agent system will design an multi-agent system (or internal logic) specific for the task.

The multi-agent workflow would be a langraph of agents where nodes are either functions or Hugging Face "SmolAgent" instances. SmolAgent is a library for creating CodeAgent AI agents that generate Python tool calls to perform actions in multi-step processes.

There are different layer to the system.

# Langraph layer:

Langraph is used to make the overall multi-agent workflow, it would use smolAgent or python method as node.

Information between node is transfered throught a state dict:

```python
class WorkflowState(TypedDict):
    step_uuid: List[str] # Current step name
    answers: List[str] # List of raw agent answer
    success: List[bool] # List of success
```

To route between node we use "routing method", method that given state information choose the next node.

```python

def routing_function(state: WorkflowState) -> str:
    success_list = state.get("success", [])
    if not success_list:
        return "fallback_path"
    return "chart_maker"

workflow.add_node("web_surfer", WorkflowNodeFactory.create_agent_node(smolagent_web))
workflow.add_node("chart_maker", WorkflowNodeFactory.create_agent_node(smolagent_chart))

workflow.add_edge(START, "web_surfer")

workflow.add_conditional_edges(
    "web_surfer",
    advanced_router,
    {
        "chart_maker": "chart_maker",
        "retry_path": "web_surfer",
        "fallback_path": "chart_maker",
        "emergency_fallback": END
    }
)

workflow.add_conditional_edges(
    "chart_maker",
    advanced_router,
    {
        END: END,
        "retry_path": "chart_maker",
        "fallback_path": END,
        "emergency_fallback": END
    }
)
```

A node could be and should be in most case an agent, but it could also be a function:

```python
def data_transformation_node(self, state: WorkflowState) -> dict:
    state_observations = state.get("observations", [])
    # transform the data in last observation
    state_observations[-1] = parsing_function(state_observations)
    return {
        **state,
        "observations": state_observations,
    }
```

### Smolagent Layer: 

SmolAgent would be created by the WorkflowNodeFactory.create_agent_node method.

SmolAgent is a new library by hugging face that use a tool as python code approach
in a action/observations loop that continue until success. action is python tool usage, observations is tool feedback

```python
@tool
def go_to_url(url: str) -> bool:
    """Navigate to a specified URL.
    Args:
        url (str): The URL to navigate to.
    Returns:
        bool: True if navigation was successful, False otherwise.
    """
    return browser.go_to(url)
    
# more tools...

engine = HfApiModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=api_key,
    max_tokens=2048,
)

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(), go_to_url, get_page_text, get_navigable_links, is_link_valid, get_form_inputs, fill_form, screenshot],
    model=engine,
    max_steps=10,
)

instruct = """
Please search the web for latest news in NYC
"""
agent_output = agent.run(instruct)
```

### Tool layer:

The final layer, smolagent write python code to use tools, for example:

```python
while webdriver.page.is_loading():
    time.sleep(0.2)
    page_text = webdriver.page.get_text()
```

The meta-agent would be generated and executed by different components :

- A orchestration component which given a goal will call a LLM (like openai o3) to generate the multi-agent workflow logic for the composition layer. each workflow is generated for a task.

- A composition component which create a langraph workflow made of smolagents node or deterministic python function (for example to validate a previous node or transform data). 

- A execution component, which execute the generated langraph workflow in a sandbox environment

- A primitive component it is the set of tools that the smolagents could use, they might based on the MCP protocol, which would allow to have smolagent use tool as client, with the server-side implementation elsewhere (on a HPC computer, a scientific instrument, etc...) depending on the needs.
