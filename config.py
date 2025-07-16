import os
from dataclasses import dataclass
from typing import Any


@dataclass
class AddressMCP:
    """Represents an MCP server address with port range."""

    ip: str
    port_min: int
    port_max: int

    def _validate_port(self, port_number: int) -> None:
        assert port_number >= 0 and port_number <= 65535, "Port number must be between 0 and 65535"
    
    def _validate_ip(self, ip: str) -> None:
        if not self.ip:
            raise ValueError("IP address cannot be empty")
        if not isinstance(self.ip, str):
            raise TypeError(f"IP address must be a string, got {type(self.ip).__name__} instead.")

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
        self.workflow_dir: str = "sources/workflows"
        self.memory_dir: str = "sources/memory"
        self.schema_code_path: str = "sources/modules/state_schema.py"
        self.smolagent_factory_code_path: str = "sources/modules/smolagent_factory.py"
        self.prompt_workflow_creator: str = "sources/prompts/workflow_creator.md"
        self.workflow_llm_provider: str = "openai"
        self.mcp_health_endpoint: str = "http://localhost:5000/health"
        self.runner_default_python_version: str = "3.10"
        self.runner_default_timeout: int = 3600
        self.runner_default_max_memory_mb: int = 1024
        self.runner_default_max_cpu_percent: int = 100
        self.runner_temp_dir: str = "./tmp"
        self.discovery_addresses: list[AddressMCP] = [
            AddressMCP(ip="localhost", port_min=5000, port_max=5250),
        ]
        self.runner_requirements: list[str] = [
            "python-dotenv",
            "fastmcp==2.8.1",
            "requests>=2.31.0",
            "smolagents[all]",
            "langgraph>=0.4.7",
            "matplotlib>=3.9.0",
            "numpy>=2.0.0",
            "python_a2a",
            "opentelemetry-sdk",
            "opentelemetry-exporter-otlp",
            "openinference-instrumentation-smolagents",
            "asyncio==3.4.3",
        ]
        self.pushover_token: str | None = os.getenv("PUSHOVER_TOKEN")
        self.pushover_user: str | None = os.getenv("PUSHOVER_USER")

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

    def jsonify(
        self,
    ) -> dict[str, str | int | str | None | list[dict[str, str | int]]]:
        """Convert configuration to a JSON-serializable dictionary."""
        return {
            "discovery_addresses": [
                {"ip": addr.ip, "port_min": addr.port_min, "port_max": addr.port_max}
                for addr in self.discovery_addresses
            ],
            "workflow_dir": self.workflow_dir,
            "schema_code_path": self.schema_code_path,
            "smolagent_factory_code_path": self.smolagent_factory_code_path,
            "prompt_workflow_creator": self.prompt_workflow_creator,
            "workflow_llm_provider": self.workflow_llm_provider,
            "mcp_health_endpoint": self.mcp_health_endpoint,
            "runner_default_python_version": self.runner_default_python_version,
            "runner_default_timeout": self.runner_default_timeout,
            "runner_default_max_memory_mb": self.runner_default_max_memory_mb,
            "runner_default_max_cpu_percent": self.runner_default_max_cpu_percent,
            "runner_temp_dir": self.runner_temp_dir,
            "runner_requirements": self.runner_requirements,
        }

    def from_json(self, data: dict[str, Any]) -> None:
        """Load configuration from a JSON-serializable dictionary."""
        self.workflow_dir = data.get("workflow_dir", self.workflow_dir)
        self.discovery_addresses = [
            AddressMCP(addr["ip"], addr["port_min"], addr["port_max"])
            for addr in data.get("discovery_addresses", [])
        ]
        self.schema_code_path = data.get("schema_code_path", self.schema_code_path)
        self.smolagent_factory_code_path = data.get(
            "smolagent_factory_code_path", self.smolagent_factory_code_path
        )
        self.prompt_workflow_creator = data.get(
            "prompt_workflow_creator", self.prompt_workflow_creator
        )
        self.workflow_llm_provider = data.get(
            "workflow_llm_provider", self.workflow_llm_provider
        )
        self.mcp_health_endpoint = data.get(
            "mcp_health_endpoint", self.mcp_health_endpoint
        )
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

    def __str__(self) -> str:
        """String representation of the configuration."""
        return (
            f"Config(workflow_dir={self.workflow_dir}, "
            f"schema_code_path={self.schema_code_path}, smolagent_factory_code_path={self.smolagent_factory_code_path}, "
            f"prompt_workflow_creator={self.prompt_workflow_creator}, workflow_llm_provider={self.workflow_llm_provider}, "
            f"mcp_health_endpoint={self.mcp_health_endpoint}, runner_default_python_version={self.runner_default_python_version}, "
            f"runner_default_timeout={self.runner_default_timeout}, runner_default_max_memory_mb={self.runner_default_max_memory_mb}, "
            f"runner_default_max_cpu_percent={self.runner_default_max_cpu_percent}, runner_temp_dir={self.runner_temp_dir})"
        )
