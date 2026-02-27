import json
import os
from dataclasses import dataclass
from typing import Any

from sources.utils.pricing import OpenRouterPricingClient


@dataclass
class AddressMCP:
    """Represents an MCP server address with port range."""

    ip: str
    port_min: int
    port_max: int

    def _validate_port(self, port_number: int) -> None:
        assert port_number >= 0 and port_number <= 65535, "Port not between 0 and 65535"

    def _validate_ip(self, ip: str) -> None:
        if not ip:
            raise ValueError("IP address cannot be empty")
        if not isinstance(ip, str):
            raise TypeError(f"IP address must be string, got {type(ip).__name__}")

    def __post_init__(self):
        """Validate the address and port range."""
        self._validate_ip(self.ip)
        self._validate_port(self.port_min)
        self._validate_port(self.port_max)
        if self.port_min > self.port_max:
            raise ValueError(f"port_min must be <= port_max for ip {self.ip}.")


class Config:
    """Configuration class for Mimosa AI Agent Framework."""

    def __init__(self):

        # workspace configuration
        self.workspace_dir = "/Users/cnrs/Documents/repository/toolomics/workspace"

        # MCPs server discovery
        self.discovery_addresses: list[AddressMCP] = [
            AddressMCP(ip="0.0.0.0", port_min=5000, port_max=5200)
        ]

        # LLMs choices
        self.planner_llm_model: str = "anthropic/claude-opus-4-5"
        self.prompts_llm_model: str = "anthropic/claude-sonnet-4-6"
        self.workflow_llm_model: str = "anthropic/claude-opus-4-6"
        self.smolagent_model_id: str = "openrouter/moonshotai/kimi-k2.5"
        self.judge_model = "openrouter/moonshotai/kimi-k2.5"
        self.engine_name: str = "litellm" # for smolagent

        # prompts for planner / workflow generator
        self.prompt_planner: str = "sources/prompts/planner_reproduction.md"
        self.prompt_workflow_creator: str = "sources/prompts/workflow_v8.md"

        # reasoning_effort: "minimal" (GPT-5 only, fastest), "low", "medium" (default), "high"
        self.reasoning_effort: str = "high"
        
        # max_tokens: Maximum number of tokens to generate for LLM responses
        self.max_tokens: int = 64000
        self._pricing_client = OpenRouterPricingClient()
        self._model_pricing_cache = None

        # DGM learning parameters
        self.learned_score_threshold = 0.8
        self.max_learning_dgm_iterations = 10

        # folder paths for workflow pre-defined code
        self.schema_code_path: str = "sources/modules/state_schema.py"
        self.smolagent_factory_code_path: str = "sources/modules/smolagent_factory.py"
        # folder path for cache
        self.runs_capsule_dir = "runs_capsule/"
        self.workflow_dir: str = "sources/workflows"
        self.memory_dir: str = "sources/memory"

        # runner settings
        self.runner_default_python_version: str = "3.10"
        self.runner_default_timeout: int = 3600*2
        self.runner_default_max_memory_mb: int = 1024
        self.runner_default_max_cpu_percent: int = 100
        self.runner_temp_dir: str = "./tmp"
        self.runner_requirements: list[str] = [
            "python-dotenv",
            "fastmcp==2.8.1",
            "requests>=2.31.0",
            # avoid optional extras that pull in packages like `helium`/`selenium`
            "pillow>=12.1.0",
            "smolagents[litellm,mlx-lm,telemetry,mcp]",
            "langgraph>=0.4.7",
            "matplotlib>=3.9.0",
            "numpy>=2.0.0",
            # correct PyPI package name
            "python-a2a",
            "opentelemetry-sdk",
            "opentelemetry-exporter-otlp",
            "openinference-instrumentation-smolagents",
        ]
        # notifications
        self.pushover_token: str | None = os.getenv("PUSHOVER_TOKEN")
        self.pushover_user: str | None = os.getenv("PUSHOVER_USER")


    @property
    def model_pricing(self) -> dict[str, dict[str, float]]:
        """Get model pricing with fallback to cached or default values."""
        if self._model_pricing_cache is None:
            # Try to fetch real-time pricing
            pricing_data = self._pricing_client.get_model_pricing_dict()
            if pricing_data:
                self._model_pricing_cache = pricing_data
            else:
                # Fallback to static pricing if API fails
                self._model_pricing_cache = self._pricing_client.get_fallback_pricing()
        return self._model_pricing_cache

    def refresh_pricing(self) -> None:
        """Force refresh of model pricing from OpenRouter API."""
        self._model_pricing_cache = None

    def create_paths(self) -> None:
        """Create necessary directories if they do not exist."""
        os.makedirs(self.workflow_dir, exist_ok=True)
        os.makedirs(self.memory_dir, exist_ok=True)
        os.makedirs(self.runner_temp_dir, exist_ok=True)

    def validate_paths(self) -> None:
        """Validate that all required paths exist."""
        assert os.path.exists(self.workflow_dir), (
            f"Workflow directory not found: {self.workflow_dir}"
        )
        assert os.path.exists(self.schema_code_path), (
            f"State schema file not found: {self.schema_code_path}"
        )
        assert os.path.exists(self.smolagent_factory_code_path), (
            f"SmolAgent factory file not found: {self.smolagent_factory_code_path}"
        )
        assert os.path.exists(self.prompt_workflow_creator), (
            f"System prompt file not found: {self.prompt_workflow_creator}"
        )
        assert os.path.exists(self.workspace_dir), (
            f"Workspace directory not found: {self.workspace_dir}"
        )

    def jsonify(
        self,
    ) -> dict[str, Any]:
        """Convert configuration to a JSON-serializable dictionary."""
        return {
            "workspace_dir": self.workspace_dir,
            "discovery_addresses": [
                {"ip": addr.ip, "port_min": addr.port_min, "port_max": addr.port_max}
                for addr in self.discovery_addresses
            ],
            "planner_llm_model": self.planner_llm_model,
            "prompts_llm_model": self.prompts_llm_model,
            "workflow_llm_model": self.workflow_llm_model,
            "smolagent_model_id": self.smolagent_model_id,
            "judge_model": self.judge_model,
            "engine_name": self.engine_name,
            "prompt_planner": self.prompt_planner,
            "prompt_workflow_creator": self.prompt_workflow_creator,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
            "learned_score_threshold": self.learned_score_threshold,
            "max_learning_dgm_iterations": self.max_learning_dgm_iterations,
            "schema_code_path": self.schema_code_path,
            "smolagent_factory_code_path": self.smolagent_factory_code_path,
            "runs_capsule_dir": self.runs_capsule_dir,
            "workflow_dir": self.workflow_dir,
            "memory_dir": self.memory_dir,
            "runner_default_python_version": self.runner_default_python_version,
            "runner_default_timeout": self.runner_default_timeout,
            "runner_default_max_memory_mb": self.runner_default_max_memory_mb,
            "runner_default_max_cpu_percent": self.runner_default_max_cpu_percent,
            "runner_temp_dir": self.runner_temp_dir,
            "runner_requirements": self.runner_requirements,
        }

    def from_json(self, data: dict[str, Any]) -> None:
        """Load configuration from a JSON-serializable dictionary."""
        self.workspace_dir = data.get("workspace_dir", self.workspace_dir)
        self.discovery_addresses = [
            AddressMCP(addr["ip"], addr["port_min"], addr["port_max"])
            for addr in data.get("discovery_addresses", [])
        ]
        self.planner_llm_model = data.get("planner_llm_model", self.planner_llm_model)
        self.prompts_llm_model = data.get(
            "prompts_llm_model", self.prompts_llm_model
        )
        self.workflow_llm_model = data.get(
            "workflow_llm_model", self.workflow_llm_model
        )
        self.smolagent_model_id = data.get("smolagent_model_id", self.smolagent_model_id)
        self.judge_model = data.get("judge_model", self.judge_model)
        self.engine_name = data.get("engine_name", self.engine_name)
        self.prompt_planner = data.get("prompt_planner", self.prompt_planner)
        self.prompt_workflow_creator = data.get(
            "prompt_workflow_creator", self.prompt_workflow_creator
        )
        self.reasoning_effort = data.get("reasoning_effort", self.reasoning_effort)
        self.max_tokens = data.get("max_tokens", self.max_tokens)
        self.learned_score_threshold = data.get(
            "learned_score_threshold", self.learned_score_threshold
        )
        self.max_learning_dgm_iterations = data.get(
            "max_learning_dgm_iterations", self.max_learning_dgm_iterations
        )
        self.schema_code_path = data.get("schema_code_path", self.schema_code_path)
        self.smolagent_factory_code_path = data.get(
            "smolagent_factory_code_path", self.smolagent_factory_code_path
        )
        self.runs_capsule_dir = data.get("runs_capsule_dir", self.runs_capsule_dir)
        self.workflow_dir = data.get("workflow_dir", self.workflow_dir)
        self.memory_dir = data.get("memory_dir", self.memory_dir)
        self.runner_default_python_version = data.get(
            "runner_default_python_version", self.runner_default_python_version
        )
        self.runner_default_timeout = data.get(
            "runner_default_timeout", self.runner_default_timeout
        )
        self.runner_default_max_memory_mb = data.get(
            "runner_default_max_memory_mb", self.runner_default_max_memory_mb
        )
        self.runner_default_max_cpu_percent = data.get(
            "runner_default_max_cpu_percent", self.runner_default_max_cpu_percent
        )
        self.runner_temp_dir = data.get("runner_temp_dir", self.runner_temp_dir)
        self.runner_requirements = data.get(
            "runner_requirements", self.runner_requirements
        )

    def dump(self, filepath: str) -> None:
        """Save configuration to a JSON file."""
        config_data = self.jsonify()
        with open(filepath, "w") as f:
            json.dump(config_data, f, indent=2)

    def load(self, filepath: str) -> None:
        """Load configuration from a JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        with open(filepath) as f:
            config_data = json.load(f)
        self.from_json(config_data)

    def __str__(self) -> str:
        """String representation of the configuration."""
        return (
            f"Config(workflow_dir={self.workflow_dir},\n"
            f"schema_code_path={self.schema_code_path},\n"
            f"smolagent_factory_code_path={self.smolagent_factory_code_path},\n"
            f"prompt_workflow_creator={self.prompt_workflow_creator}\n"
            f"workflow_llm_provider={self.workflow_llm_provider},\n"
            f"workflow_llm_model={self.workflow_llm_model},\n"
            f"prompts_llm_model={self.workflow_llm_model},\n"
            f"reasoning_effort={self.reasoning_effort},\n"
            f"runner_default_python_version={self.runner_default_python_version},\n"
            f"runner_default_timeout={self.runner_default_timeout},\n"
            f"runner_default_max_memory_mb={self.runner_default_max_memory_mb},\n"
            f"runner_default_max_cpu_percent={self.runner_default_max_cpu_percent},\n"
            f"runner_temp_dir={self.runner_temp_dir})\n"
        )
