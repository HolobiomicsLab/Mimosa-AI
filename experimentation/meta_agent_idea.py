from smolagents import CodeAgent, tool
from smolagents import HfApiModel
from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, List, Callable
import json
import pickle
from dataclasses import dataclass
from enum import Enum

# State management for the workflow
@dataclass
class AgentState:
    task: str
    result: Any = None
    performance_metrics: Dict = None
    generation: int = 0
    graph_definition: Dict = None
    mutation_log: List = None

class GraphFactory:
    """Wrapper class that provides LangGraph functionality as smolagent tools"""
    
    def __init__(self):
        self.current_workflow = None
        self.nodes = {}
        self.edges = []
        self.conditional_edges = []
    
    @tool
    def create_workflow(self, state_class_name: str = "AgentState") -> bool:
        """Initialize a new workflow graph"""
        self.current_workflow = StateGraph(AgentState)
        self.nodes = {}
        self.edges = []
        self.conditional_edges = []
        return True
    
    @tool
    def add_node(self, node_name: str, function_name: str) -> bool:
        """Add a node to the current workflow"""
        if self.current_workflow is None:
            return False
        
        # Store node reference for later compilation
        self.nodes[node_name] = function_name
        return True
    
    @tool
    def add_edge(self, from_node: str, to_node: str) -> bool:
        """Add a direct edge between nodes"""
        self.edges.append((from_node, to_node))
        return True
    
    @tool
    def add_conditional_edge(self, from_node: str, condition_func: str, mapping: str) -> bool:
        """Add conditional edge with routing logic"""
        # mapping should be JSON string like '{"pass": "next_node", "fail": "retry_node"}'
        try:
            mapping_dict = json.loads(mapping)
            self.conditional_edges.append((from_node, condition_func, mapping_dict))
            return True
        except:
            return False
    
    @tool
    def compile_workflow(self) -> bool:
        """Compile the current workflow into executable graph"""
        if not self.current_workflow:
            return False
        
        # Add all nodes (functions would be resolved at runtime)
        for node_name, func_name in self.nodes.items():
            self.current_workflow.add_node(node_name, lambda state: state)  # Placeholder
        
        # Add edges
        for from_node, to_node in self.edges:
            if from_node == "START":
                self.current_workflow.add_edge(START, to_node)
            elif to_node == "END":
                self.current_workflow.add_edge(from_node, END)
            else:
                self.current_workflow.add_edge(from_node, to_node)
        
        # Add conditional edges
        for from_node, condition_func, mapping in self.conditional_edges:
            self.current_workflow.add_conditional_edges(from_node, lambda state: "pass", mapping)
        
        return True
    
    @tool
    def save_graph_definition(self, filename: str) -> bool:
        """Save current graph definition for evolution"""
        graph_def = {
            "nodes": self.nodes,
            "edges": self.edges,
            "conditional_edges": self.conditional_edges
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(graph_def, f, indent=2)
            return True
        except:
            return False
    
    @tool
    def load_graph_definition(self, filename: str) -> bool:
        """Load graph definition from file"""
        try:
            with open(filename, 'r') as f:
                graph_def = json.load(f)
            
            self.nodes = graph_def.get("nodes", {})
            self.edges = graph_def.get("edges", [])
            self.conditional_edges = graph_def.get("conditional_edges", [])
            return True
        except:
            return False

class MetaAgent:
    """Self-evolving agent that can modify its own execution graph"""
    
    def __init__(self, model_config: Dict, generation: int = 0):
        self.generation = generation
        self.graph_factory = GraphFactory()
        self.performance_history = []
        
        # Initialize smolagent with graph construction tools
        self.engine = HfApiModel(**model_config)
        
        # Core tools for graph manipulation
        graph_tools = [
            self.graph_factory.create_workflow,
            self.graph_factory.add_node,
            self.graph_factory.add_edge,
            self.graph_factory.add_conditional_edge,
            self.graph_factory.compile_workflow,
            self.graph_factory.save_graph_definition,
            self.graph_factory.load_graph_definition,
            self._evaluate_performance,
            self._mutate_strategy
        ]
        
        self.agent = CodeAgent(
            tools=graph_tools,
            model=self.engine,
            max_steps=20
        )
    
    @tool
    def _evaluate_performance(self, result: str, target_metrics: str) -> float:
        """Evaluate how well the current agent performed"""
        # Simplified performance evaluation
        # In reality, this would be more sophisticated
        score = len(result) / 100.0  # Placeholder metric
        return min(score, 1.0)
    
    @tool
    def _mutate_strategy(self, current_performance: float, graph_definition: str) -> str:
        """Generate mutations for the next generation"""
        mutations = []
        
        if current_performance < 0.5:
            mutations.append("add_retry_logic")
            mutations.append("increase_max_steps")
        elif current_performance < 0.8:
            mutations.append("add_validation_node")
            mutations.append("optimize_node_order")
        else:
            mutations.append("add_parallel_processing")
            mutations.append("implement_caching")
        
        return json.dumps(mutations)
    
    def create_next_generation(self, task_performance: float, current_graph: Dict) -> 'MetaAgent':
        """Evolution instruction for the agent"""
        
        evolution_prompt = f"""
        You are generation {self.generation} meta-agent. Your performance on the last task was {task_performance:.2f}.
        
        Based on this performance, create an improved workflow for the next generation.
        
        Here are examples of how to use the graph construction tools:
        
        To create a new workflow:
        ```py
        create_workflow("AgentState")
        ```<end_code>
        
        To add nodes (functions will be implemented by next generation):
        ```py
        add_node("input_validator", "validate_input_function")
        add_node("task_decomposer", "decompose_task_function")
        add_node("parallel_executor", "execute_parallel_function")
        add_node("result_merger", "merge_results_function")
        add_node("quality_checker", "check_quality_function")
        add_node("create_next_generation", "evolution_function")
        ```<end_code>
        
        To connect nodes:
        ```py
        add_edge("START", "input_validator")
        add_edge("input_validator", "task_decomposer")
        add_edge("task_decomposer", "parallel_executor")
        add_edge("parallel_executor", "result_merger")
        ```<end_code>
        
        To add conditional logic based on performance:
        ```py
        # Route based on quality check
        quality_mapping = '{{"high": "create_next_generation", "low": "task_decomposer", "error": "input_validator"}}'
        add_conditional_edge("quality_checker", "evaluate_quality_score", quality_mapping)
        ```<end_code>
        
        To finalize and save:
        ```py
        compile_workflow()
        save_graph_definition(f"generation_{self.generation + 1}_architecture.json")
        ```<end_code>
        
        Analysis of current performance ({task_performance:.2f}):
        Current graph structure: {json.dumps(current_graph, indent=2)}
        
        Based on this analysis, design a better architecture that addresses the performance gaps.
        Focus on structural improvements like:
        - Better error handling if performance < 0.3
        - Parallel processing if performance 0.3-0.7  
        - Optimization nodes if performance > 0.7
        """
        
        # Execute evolution
        evolution_result = self.agent.run(evolution_prompt)
        
        # Create next generation agent
        next_gen = MetaAgent(
            model_config={"model_id": self.engine.model_id, "token": self.engine.token, "max_tokens": 5000},
            generation=self.generation + 1
        )
        
        return next_gen
    
    def execute_task(self, task: str) -> Dict:
        """Execute the current generation's workflow on a task"""
        
        execution_prompt = f"""
        Execute this task using your current workflow architecture: {task}
        
        Here are examples of how to use the graph construction tools:
        
        To create a new workflow:
        ```py
        create_workflow("AgentState")
        ```<end_code>
        
        To add nodes to your workflow:
        ```py
        add_node("analyze_task", "task_analyzer_function")
        add_node("execute_solution", "solution_executor") 
        add_node("evaluate_result", "result_evaluator")
        add_node("create_next_generation", "evolution_function")
        ```<end_code>
        
        To connect nodes with edges:
        ```py
        add_edge("START", "analyze_task")
        add_edge("analyze_task", "execute_solution")
        add_edge("execute_solution", "evaluate_result")
        add_edge("evaluate_result", "create_next_generation")
        add_edge("create_next_generation", "END")
        ```<end_code>
        
        To add conditional routing:
        ```py
        mapping = '{"success": "evaluate_result", "failure": "retry_solution", "error": "create_next_generation"}'
        add_conditional_edge("execute_solution", "check_solution_quality", mapping)
        ```<end_code>
        
        To compile and save your workflow:
        ```py
        compile_workflow()
        save_graph_definition(f"generation_{self.generation}_workflow.json")
        ```<end_code>
        
        Build and execute a workflow for this specific task: {task}
        Always end with 'create_next_generation' node for evolution.
        """
        
        result = self.agent.run(execution_prompt)
        
        # Evaluate performance
        performance = self._evaluate_performance(str(result), "default_metrics")
        
        self.performance_history.append({
            "task": task,
            "result": result,
            "performance": performance,
            "generation": self.generation
        })
        
        return {
            "result": result,
            "performance": performance,
            "generation": self.generation
        }

# Usage example
def demo_meta_agent_evolution():
    """Demonstrate the meta-agent evolution process"""
    
    model_config = {
        "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "token": "your_token_here",
        "max_tokens": 5000
    }
    
    # Initialize first generation
    agent_gen_0 = MetaAgent(model_config, generation=0)
    
    # Test tasks for evolution
    tasks = [
        "Create a joke and improve it if needed",
        "Analyze text sentiment and provide feedback",
        "Generate code and debug it automatically"
    ]
    
    current_agent = agent_gen_0
    
    for i, task in enumerate(tasks):
        print(f"\n=== Generation {current_agent.generation} executing task {i+1} ===")
        
        # Execute task
        result = current_agent.execute_task(task)
        print(f"Performance: {result['performance']:.2f}")
        
        # Evolve if performance is below threshold
        if result['performance'] < 0.9:
            print(f"Evolving to generation {current_agent.generation + 1}...")
            current_agent = current_agent.create_next_generation(
                result['performance'], 
                current_agent.graph_factory.nodes
            )
        
        # Show evolution progress
        print(f"Current generation: {current_agent.generation}")

if __name__ == "__main__":
    demo_meta_agent_evolution()