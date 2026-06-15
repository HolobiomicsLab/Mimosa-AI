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
        self.workspace_dir = "/Users/mlg/Documents/CNRS/toolomics/workspace"

        # MCPs server discovery
        self.discovery_addresses: list[AddressMCP] = [
            AddressMCP(ip="0.0.0.0", port_min=5000, port_max=5200)
        ]

        # LLMs choices
        self.planner_llm_model: str = "openrouter/z-ai/glm-5.1"
        self.workflow_llm_model: str = "openrouter/z-ai/glm-5.1"
        self.smolagent_model_id: str = "openrouter/deepseek/deepseek-v4-pro"
        self.judge_model = "openrouter/qwen/qwen3.7-max"
        self.capsule_namer_model = "openrouter/deepseek/deepseek-v4-flash"
        self.engine_name: str = "litellm" # for smolagent


        # prompts for planner / workflow generator
        self.prompt_planner: str = "sources/prompts/planner_reproduction.md"
        self.prompt_workflow_creator: str = "sources/prompts/workflow_v11.md"
        self.prompt_smolagent: str = "sources/prompts/smolagent_sys_prompt.md"

        # reasoning_effort: "minimal" (GPT-5 only, fastest), "low", "medium" (default), "high"
        self.reasoning_effort: str = "medium"

        # max_tokens: Maximum number of tokens to generate for LLM responses
        self.max_tokens: int = 8192
        self._pricing_client = OpenRouterPricingClient()
        self._model_pricing_cache = None

        # learning parameters
        self.learned_score_threshold = 0.9
        self.max_learning_evolve_iterations = 20

        # evaluation concurrency settings
        self.max_concurrent_eval_tasks: int = 2  # Number of concurrent tasks for CSV evaluation mode

        # folder paths for workflow pre-defined code
        self.schema_code_path: str = "sources/modules/state_schema.py"
        self.smolagent_factory_code_path: str = "sources/modules/smolagent_factory.py"
        # folder path for cache
        self.runs_capsule_dir = "runs_capsule/"
        self.workflow_dir: str = "sources/workflows"
        self.memory_dir: str = "sources/memory"
        # When True, every child workflow's verifier eval anchors on the
        # earliest ancestor's cached rubric (claims + verify_*.py) so scores
        # are comparable across an evolved lineage. Disable for ablation.
        self.reuse_lineage_rubric: bool = True

        # openrouter providers
        self.openrouter_provider: list[str] | None = [
            "anthropic", "openai", "google-vertex", "google-ai-studio", "azure", "amazon-bedrock",
            "xai", "deepseek", "mistral", "cohere", "moonshotai", "z-ai", "alibaba", "minimax", "perplexity",
             "siliconflow", "novita", "deepinfra", "atlas-cloud", "parasail", "together", "fireworks", "nebius", "chutes",
             "groq", "cerebras", "sambanova", "nvidia"
        ]
        self.openrouter_provider_by_model: dict[str, list[str]] = {}
        self.openrouter_quantizations_by_model: dict[str, list[str] | None] = {}
        self.default_openrouter_quantizations: list[str] = ["bf16", "fp16", "fp8"]
        # runner settings
        self.runner_default_python_version: str = "3.12"
        self.runner_default_timeout: int = 10800
        # Per-agent (SmolAgentFactory) execution timeout in seconds. Injected into
        # the generated workflow as AGENT_EXECUTION_TIMEOUT. 3600 = 1 hour.
        self.agent_execution_timeout: int = 18000
        self.runner_default_max_memory_mb: int = 10000
        self.runner_default_max_cpu_percent: int = 100
        self.runner_temp_dir: str = "./tmp"
        self.runner_requirements: list[str] = [
            "setuptools>=70.0",
            "python-dotenv",
            "fastmcp==2.8.1",
            "requests>=2.31.0",
            # avoid optional extras that pull in packages like `helium`/`selenium`
            "pillow>=12.1.0",
            "smolagents[litellm,mlx-lm,telemetry,mcp]",
            "langgraph>=0.4.7",
            #"matplotlib>=3.9.0",
            "pandas==2.3.2",
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


    def openrouter_provider_for(self, model_id: str | None) -> list[str] | None:
        """Return the precheck-selected provider list for `model_id`, or the
        default `openrouter_provider` when no per-model selection exists.

        Callers that build an LLMConfig for an OpenRouter model should use
        this — passing the shared `openrouter_provider` directly can leave
        the runtime with no routable provider for that specific model.
        """
        if model_id and model_id in self.openrouter_provider_by_model:
            return self.openrouter_provider_by_model[model_id]
        return self.openrouter_provider

    def openrouter_quantizations_for(self, model_id: str | None) -> list[str] | None:
        """Return the OpenRouter `quantizations` filter to apply for `model_id`.

        `None` means omit the filter — this is the case when precheck
        selected at least one untagged first-party provider (google-vertex,
        google-ai-studio, etc.). Without precheck data, returns the default
        safety filter which blocks unsafe (int4/fp4) routing.
        """
        if model_id and model_id in self.openrouter_quantizations_by_model:
            return self.openrouter_quantizations_by_model[model_id]
        return self.default_openrouter_quantizations

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
            "workflow_llm_model": self.workflow_llm_model,
            "smolagent_model_id": self.smolagent_model_id,
            "judge_model": self.judge_model,
            "engine_name": self.engine_name,
            "openrouter_provider": self.openrouter_provider,
            "prompt_planner": self.prompt_planner,
            "prompt_workflow_creator": self.prompt_workflow_creator,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
            "learned_score_threshold": self.learned_score_threshold,
            "max_learning_evolve_iterations": self.max_learning_evolve_iterations,
            "schema_code_path": self.schema_code_path,
            "smolagent_factory_code_path": self.smolagent_factory_code_path,
            "runs_capsule_dir": self.runs_capsule_dir,
            "workflow_dir": self.workflow_dir,
            "memory_dir": self.memory_dir,
            "reuse_lineage_rubric": self.reuse_lineage_rubric,
            "runner_default_python_version": self.runner_default_python_version,
            "runner_default_timeout": self.runner_default_timeout,
            "agent_execution_timeout": self.agent_execution_timeout,
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
        self.workflow_llm_model = data.get(
            "workflow_llm_model", self.workflow_llm_model
        )
        self.smolagent_model_id = data.get("smolagent_model_id", self.smolagent_model_id)
        self.judge_model = data.get("judge_model", self.judge_model)
        self.engine_name = data.get("engine_name", self.engine_name)
        self.openrouter_provider = data.get("openrouter_provider", self.openrouter_provider)
        self.prompt_planner = data.get("prompt_planner", self.prompt_planner)
        self.prompt_workflow_creator = data.get(
            "prompt_workflow_creator", self.prompt_workflow_creator
        )
        self.reasoning_effort = data.get("reasoning_effort", self.reasoning_effort)
        self.max_tokens = data.get("max_tokens", self.max_tokens)
        self.learned_score_threshold = data.get(
            "learned_score_threshold", self.learned_score_threshold
        )
        self.max_learning_evolve_iterations = data.get(
            "max_learning_evolve_iterations", self.max_learning_evolve_iterations
        )
        self.schema_code_path = data.get("schema_code_path", self.schema_code_path)
        self.smolagent_factory_code_path = data.get(
            "smolagent_factory_code_path", self.smolagent_factory_code_path
        )
        self.runs_capsule_dir = data.get("runs_capsule_dir", self.runs_capsule_dir)
        self.workflow_dir = data.get("workflow_dir", self.workflow_dir)
        self.memory_dir = data.get("memory_dir", self.memory_dir)
        self.reuse_lineage_rubric = bool(
            data.get("reuse_lineage_rubric", self.reuse_lineage_rubric)
        )
        self.runner_default_python_version = data.get(
            "runner_default_python_version", self.runner_default_python_version
        )
        self.runner_default_timeout = data.get(
            "runner_default_timeout", self.runner_default_timeout
        )
        self.agent_execution_timeout = data.get(
            "agent_execution_timeout", self.agent_execution_timeout
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
        lines = ["Config("]
        lines.append(f"  workspace_dir={self.workspace_dir}")
        lines.append(f"  discovery_addresses={self.discovery_addresses}")
        lines.append(f"  planner_llm_model={self.planner_llm_model}")
        lines.append(f"  workflow_llm_model={self.workflow_llm_model}")
        lines.append(f"  smolagent_model_id={self.smolagent_model_id}")
        lines.append(f"  judge_model={self.judge_model}")
        lines.append(f"  capsule_namer_model={self.capsule_namer_model}")
        lines.append(f"  engine_name={self.engine_name}")
        lines.append(f"  prompt_planner={self.prompt_planner}")
        lines.append(f"  prompt_workflow_creator={self.prompt_workflow_creator}")
        lines.append(f"  prompt_smolagent={self.prompt_smolagent}")
        lines.append(f"  reasoning_effort={self.reasoning_effort}")
        lines.append(f"  max_tokens={self.max_tokens}")
        lines.append(f"  learned_score_threshold={self.learned_score_threshold}")
        lines.append(f"  max_learning_evolve_iterations={self.max_learning_evolve_iterations}")
        lines.append(f"  max_concurrent_eval_tasks={self.max_concurrent_eval_tasks}")
        lines.append(f"  schema_code_path={self.schema_code_path}")
        lines.append(f"  smolagent_factory_code_path={self.smolagent_factory_code_path}")
        lines.append(f"  runs_capsule_dir={self.runs_capsule_dir}")
        lines.append(f"  workflow_dir={self.workflow_dir}")
        lines.append(f"  memory_dir={self.memory_dir}")
        lines.append(f"  reuse_lineage_rubric={self.reuse_lineage_rubric}")
        lines.append(f"  openrouter_provider={self.openrouter_provider}")
        lines.append(f"  openrouter_provider_by_model={self.openrouter_provider_by_model}")
        lines.append(f"  openrouter_quantizations_by_model={self.openrouter_quantizations_by_model}")
        lines.append(f"  default_openrouter_quantizations={self.default_openrouter_quantizations}")
        lines.append(f"  runner_default_python_version={self.runner_default_python_version}")
        lines.append(f"  runner_default_timeout={self.runner_default_timeout}")
        lines.append(f"  agent_execution_timeout={self.agent_execution_timeout}")
        lines.append(f"  runner_default_max_memory_mb={self.runner_default_max_memory_mb}")
        lines.append(f"  runner_default_max_cpu_percent={self.runner_default_max_cpu_percent}")
        lines.append(f"  runner_temp_dir={self.runner_temp_dir}")
        lines.append(f"  runner_requirements={self.runner_requirements}")
        lines.append(f"  pushover_token={'***' if self.pushover_token else None}")
        lines.append(f"  pushover_user={'***' if self.pushover_user else None}")
        lines.append(")")
        return "\n".join(lines)

if __name__ == "__main__":
    config = Config()
    config.dump("config_default.json")
