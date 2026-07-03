import json
import os
from dataclasses import dataclass
from typing import Any

from sources.utils import paths
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

        ##############
        # Workspace Configuration
        ##############
        self.workspace_dir = "/Users/mlg/Documents/CNRS/toolomics/workspace"

        ##############
        # MCP Related
        ##############
        self.discovery_addresses: list[AddressMCP] = [
            AddressMCP(ip="0.0.0.0", port_min=5000, port_max=5200)
        ]

        ##############
        # AUdit / Export 
        ##############
        # When True, writes an ASTRA spec YAML after task completion.
        self.export_astra: bool = False

        ##############
        # LLM Related
        ##############
        self.planner_llm_model: str = "openrouter/z-ai/glm-5.2"
        self.workflow_llm_model: str = "openrouter/z-ai/glm-5.2"
        self.smolagent_model_id: str = "openrouter/deepseek/deepseek-v4-flash"
        self.judge_model = "openrouter/z-ai/glm-5.2"
        self.capsule_namer_model = "openrouter/deepseek/deepseek-v4-flash"

        ##############
        # ScienceAgentBench Concurrency settings
        ##############
        self.max_concurrent_eval_tasks: int = 1  # Number of concurrent tasks for CSV evaluation mode

        ##############
        # QD/Novelty / Learning parameters 
        # Touch with caution, for ablation studies and research only.
        ##############

        # strategy: "qd" (default), "tournament", "novelty"
        self.selection_strategy = "qd"
        # learning parameters
        self.learned_score_threshold = 0.92
        self.max_learning_evolve_iterations = 25
        # KNN settings for novelty
        self.novelty_comparison: str = "archive_knn"
        self.novelty_previous_n: int = 15
        # Length penalty: genotype size at which the penalty starts to grow
        self.length_penalty_baseline_chars: int = 8000
        self.length_penalty_lambda: float = 0.05
        # Selection pressure / archive settings
        self.min_improvement_threshold: float = 0.01
        self.population_size: int = 20
        self.novelty_k_neighbours: int = 15
        self.novelty_weight: float = 0.25
        self.admit_threshold: float = 0.3
        # Cold-start / parent-selection settings
        self.initial_population: int = 2
        self.crossover_rate: float = 0.4
        self.n_parents: int = 2
        # goal similarity thresholds before even considered by qd (need similar goal to avoid picking parents from a different task)
        self.parent_threshold_similarity: float = 0.8
        # bare minimum score for selection (don't pick failed parents)
        self.parent_threshold_score: float = 0.01

        ##############
        # LLM Related, advanced settings, touch with caution
        ##############
        self.engine_name: str = "litellm" # for smolagent
        # reasoning_effort: "minimal" (GPT-5 only, fastest), "low", "medium" (default), "high"
        self.reasoning_effort: str = "medium"
        # max_tokens: Maximum number of tokens to generate for LLM responses
        self.max_tokens: int = 8192
        self._pricing_client = OpenRouterPricingClient()
        self._model_pricing_cache = None
        # openrouter providers
        self.openrouter_provider: list[str] | None = [
            "anthropic", "openai", "google-vertex", "google-ai-studio", "azure", "amazon-bedrock",
            "xai", "deepseek", "mistral", "cohere", "moonshotai", "z-ai", "alibaba", "minimax", "perplexity",
             "siliconflow", "novita", "deepinfra", "atlas-cloud", "parasail", "together", "fireworks", "nebius", "chutes",
             "groq", "cerebras", "sambanova", "nvidia"
        ]
        # Request token logprobs and save them with agent memory (for ablations).
        # litellm engine only. With pinned providers, OpenRouter only routes to
        # those supporting them; models whose providers all lack logprobs then
        # fail routing (precheck probes with them too) — disable here if needed.
        # For direct (non-OpenRouter) providers whose litellm param map lacks
        # logprobs (e.g. mistral, anthropic), the request is dropped with a
        # warning instead of raising UnsupportedParamsError at run time.
        self.save_logprobs: bool = True

        ##############
        # Prompts and pre-defined code paths; Do not modify unless you know what you are doing.
        ##############
        self.prompt_planner: str = paths.resource_path("sources/prompts/planner_reproduction.md")
        self.prompt_workflow_creator: str = paths.resource_path("sources/prompts/workflow_v11.md")
        self.prompt_smolagent: str = paths.resource_path("sources/prompts/smolagent_sys_prompt.md")

        # folder paths for workflow pre-defined code
        self.schema_code_path: str = paths.resource_path("sources/modules/state_schema.py")
        self.smolagent_factory_code_path: str = paths.resource_path("sources/modules/smolagent_factory.py")
        # folder path for cache
        self.runs_capsule_dir = paths.default_runs_capsule_dir()
        self.workflow_dir: str = paths.default_workflow_dir()
        self.memory_dir: str = paths.default_memory_dir()

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
        self.runner_temp_dir: str = paths.default_tmp_dir()
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
        assert os.path.exists(self.prompt_planner), (
            f"Planner prompt file not found: {self.prompt_planner}"
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
            "capsule_namer_model": self.capsule_namer_model,
            "engine_name": self.engine_name,
            "openrouter_provider": self.openrouter_provider,
            "save_logprobs": self.save_logprobs,
            "prompt_planner": self.prompt_planner,
            "prompt_workflow_creator": self.prompt_workflow_creator,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
            "learned_score_threshold": self.learned_score_threshold,
            "selection_strategy": self.selection_strategy,
            "max_learning_evolve_iterations": self.max_learning_evolve_iterations,
            "novelty_comparison": self.novelty_comparison,
            "novelty_previous_n": self.novelty_previous_n,
            "length_penalty_baseline_chars": self.length_penalty_baseline_chars,
            "length_penalty_lambda": self.length_penalty_lambda,
            "min_improvement_threshold": self.min_improvement_threshold,
            "population_size": self.population_size,
            "novelty_k_neighbours": self.novelty_k_neighbours,
            "novelty_weight": self.novelty_weight,
            "admit_threshold": self.admit_threshold,
            "initial_population": self.initial_population,
            "crossover_rate": self.crossover_rate,
            "n_parents": self.n_parents,
            "parent_threshold_similarity": self.parent_threshold_similarity,
            "parent_threshold_score": self.parent_threshold_score,
            "schema_code_path": self.schema_code_path,
            "smolagent_factory_code_path": self.smolagent_factory_code_path,
            "runs_capsule_dir": self.runs_capsule_dir,
            "workflow_dir": self.workflow_dir,
            "memory_dir": self.memory_dir,
            "export_astra": self.export_astra,
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
        self.capsule_namer_model = data.get(
            "capsule_namer_model", self.capsule_namer_model
        )
        self.engine_name = data.get("engine_name", self.engine_name)
        self.openrouter_provider = data.get("openrouter_provider", self.openrouter_provider)
        self.save_logprobs = data.get("save_logprobs", self.save_logprobs)
        self.prompt_planner = data.get("prompt_planner", self.prompt_planner)
        self.prompt_workflow_creator = data.get(
            "prompt_workflow_creator", self.prompt_workflow_creator
        )
        self.reasoning_effort = data.get("reasoning_effort", self.reasoning_effort)
        self.max_tokens = data.get("max_tokens", self.max_tokens)
        self.learned_score_threshold = data.get(
            "learned_score_threshold", self.learned_score_threshold
        )
        self.selection_strategy = data.get("selection_strategy", self.selection_strategy)
        self.max_learning_evolve_iterations = data.get(
            "max_learning_evolve_iterations", self.max_learning_evolve_iterations
        )
        self.novelty_comparison = data.get("novelty_comparison", self.novelty_comparison)
        self.novelty_previous_n = int(
            data.get("novelty_previous_n", self.novelty_previous_n)
        )
        self.length_penalty_baseline_chars = int(
            data.get("length_penalty_baseline_chars", self.length_penalty_baseline_chars)
        )
        self.length_penalty_lambda = float(
            data.get("length_penalty_lambda", self.length_penalty_lambda)
        )
        self.min_improvement_threshold = float(
            data.get("min_improvement_threshold", self.min_improvement_threshold)
        )
        self.population_size = int(data.get("population_size", self.population_size))
        self.novelty_k_neighbours = int(
            data.get("novelty_k_neighbours", self.novelty_k_neighbours)
        )
        self.novelty_weight = float(data.get("novelty_weight", self.novelty_weight))
        self.admit_threshold = float(data.get("admit_threshold", self.admit_threshold))
        self.initial_population = int(
            data.get("initial_population", self.initial_population)
        )
        self.crossover_rate = float(data.get("crossover_rate", self.crossover_rate))
        self.n_parents = int(data.get("n_parents", self.n_parents))
        self.parent_threshold_similarity = float(
            data.get("parent_threshold_similarity", self.parent_threshold_similarity)
        )
        self.parent_threshold_score = float(
            data.get("parent_threshold_score", self.parent_threshold_score)
        )
        self.schema_code_path = data.get("schema_code_path", self.schema_code_path)
        self.smolagent_factory_code_path = data.get(
            "smolagent_factory_code_path", self.smolagent_factory_code_path
        )
        self.runs_capsule_dir = data.get("runs_capsule_dir", self.runs_capsule_dir)
        self.workflow_dir = data.get("workflow_dir", self.workflow_dir)
        self.memory_dir = data.get("memory_dir", self.memory_dir)
        self.export_astra = bool(
            data.get("export_astra", self.export_astra)
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
        self._reanchor_relative_paths()

    def _reanchor_relative_paths(self) -> None:
        """Resolve relative paths loaded from legacy config files.

        Older config files stored working-directory-relative paths
        ("sources/prompts/…", "./tmp"). Resolve them the same way the
        defaults are resolved so a persisted config keeps working when
        Mimosa is launched from any directory.
        """
        resource_fields = (
            "prompt_planner",
            "prompt_workflow_creator",
            "schema_code_path",
            "smolagent_factory_code_path",
        )
        for field in resource_fields:
            value = getattr(self, field)
            if value and not os.path.isabs(value):
                setattr(self, field, paths.resource_path(value))
        state_fields = ("workflow_dir", "memory_dir", "runs_capsule_dir")
        for field in state_fields:
            value = getattr(self, field)
            if value and not os.path.isabs(value):
                relative = os.path.normpath(value)
                setattr(self, field, paths.state_dir(relative, os.path.basename(relative)))
        tmp = self.runner_temp_dir
        if tmp and not os.path.isabs(tmp):
            relative = os.path.normpath(tmp)
            if paths.is_repo_checkout():
                self.runner_temp_dir = str(paths.PACKAGE_ROOT / relative)
            else:
                # Scratch data belongs in the cache dir, matching the default.
                self.runner_temp_dir = str(paths.cache_dir() / os.path.basename(relative))

    def dump(self, filepath: str) -> None:
        """Save configuration to a JSON file, creating parent dirs as needed."""
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
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
        lines.append(f"  save_logprobs={self.save_logprobs}")
        lines.append(f"  prompt_planner={self.prompt_planner}")
        lines.append(f"  prompt_workflow_creator={self.prompt_workflow_creator}")
        lines.append(f"  prompt_smolagent={self.prompt_smolagent}")
        lines.append(f"  reasoning_effort={self.reasoning_effort}")
        lines.append(f"  max_tokens={self.max_tokens}")
        lines.append(f"  learned_score_threshold={self.learned_score_threshold}")
        lines.append(f"  selection_strategy={self.selection_strategy}")
        lines.append(f"  max_learning_evolve_iterations={self.max_learning_evolve_iterations}")
        lines.append(f"  novelty_comparison={self.novelty_comparison}")
        lines.append(f"  novelty_previous_n={self.novelty_previous_n}")
        lines.append(f"  length_penalty_baseline_chars={self.length_penalty_baseline_chars}")
        lines.append(f"  length_penalty_lambda={self.length_penalty_lambda}")
        lines.append(f"  min_improvement_threshold={self.min_improvement_threshold}")
        lines.append(f"  population_size={self.population_size}")
        lines.append(f"  novelty_k_neighbours={self.novelty_k_neighbours}")
        lines.append(f"  novelty_weight={self.novelty_weight}")
        lines.append(f"  admit_threshold={self.admit_threshold}")
        lines.append(f"  initial_population={self.initial_population}")
        lines.append(f"  crossover_rate={self.crossover_rate}")
        lines.append(f"  n_parents={self.n_parents}")
        lines.append(f"  parent_threshold_similarity={self.parent_threshold_similarity}")
        lines.append(f"  parent_threshold_score={self.parent_threshold_score}")
        lines.append(f"  max_concurrent_eval_tasks={self.max_concurrent_eval_tasks}")
        lines.append(f"  schema_code_path={self.schema_code_path}")
        lines.append(f"  smolagent_factory_code_path={self.smolagent_factory_code_path}")
        lines.append(f"  runs_capsule_dir={self.runs_capsule_dir}")
        lines.append(f"  workflow_dir={self.workflow_dir}")
        lines.append(f"  memory_dir={self.memory_dir}")
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
