"""
NexusAI Configuration Management.

Supports YAML config files, environment variables, and programmatic config.
Auto-detects available API keys and selects best available provider.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from nexusai.core.exceptions import ConfigError


# Default config locations
CONFIG_DIR = Path.home() / ".nexusai"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
MEMORY_DIR = CONFIG_DIR / "memory"
PLUGINS_DIR = CONFIG_DIR / "plugins"
CACHE_DIR = CONFIG_DIR / "cache"


# Provider API key environment variable mappings
PROVIDER_ENV_KEYS: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "cohere": ["COHERE_API_KEY", "CO_API_KEY"],
    "together": ["TOGETHER_API_KEY"],
    "fireworks": ["FIREWORKS_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "huggingface": ["HUGGINGFACE_API_KEY", "HF_TOKEN"],
    "perplexity": ["PERPLEXITY_API_KEY"],
    "xai": ["XAI_API_KEY"],
    "cerebras": ["CEREBRAS_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "ollama": [],  # No key needed — local
    "g4f": [],     # No key needed — free
    "pollinations": [], # No key needed - completely free cloud API
    "needle": [],  # Local on-device inference via WASM
}

# Default models per provider
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
    "mistral": "mistral-large-latest",
    "cohere": "command-r-plus",
    "together": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "fireworks": "accounts/fireworks/models/llama-v3p1-70b-instruct",
    "deepseek": "deepseek-chat",
    "huggingface": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "perplexity": "llama-3.1-sonar-large-128k-online",
    "xai": "grok-2",
    "cerebras": "llama-3.3-70b",
    "ollama": "llama3.2",
}


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""

    name: str
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def default_model(self) -> str:
        return self.model or DEFAULT_MODELS.get(self.name, "")


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_provider: str = "sentence-transformers"
    vector_store: str = "faiss"  # faiss, chroma, memory
    top_k: int = 5
    rerank: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class AgentConfig:
    """Configuration for the agent pipeline."""

    max_iterations: int = 25
    max_tool_calls: int = 50
    sandbox_code: bool = True
    auto_approve_tools: bool = False
    verbose: bool = True
    pipeline_roles: list[str] = field(
        default_factory=lambda: [
            "planner",
            "researcher",
            "tool_manager",
            "coder",
            "reviewer",
            "auditor",
            "verifier",
        ]
    )


@dataclass
class WebConfig:
    """Configuration for web dashboard."""

    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class NexusConfig:
    """
    Main NexusAI configuration.

    Priority: programmatic > environment variables > config file > defaults.
    """

    # Primary provider
    provider: str = "auto"
    model: str | None = None

    # Sub-configs
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    rag: RAGConfig = field(default_factory=RAGConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    web: WebConfig = field(default_factory=WebConfig)

    # General
    verbose: bool = True
    log_level: str = "INFO"
    memory_enabled: bool = True
    plugins_enabled: bool = True

    def __post_init__(self):
        """Auto-detect available providers from environment."""
        self._detect_providers()

    def _detect_providers(self):
        """Scan environment for API keys and register available providers."""
        for provider_name, env_vars in PROVIDER_ENV_KEYS.items():
            if provider_name in self.providers:
                continue

            # Local/Free providers are always available
            if provider_name in ("ollama", "g4f", "pollinations", "needle"):
                self.providers[provider_name] = ProviderConfig(name=provider_name)
                continue

            # Check environment variables
            for env_var in env_vars:
                key = os.environ.get(env_var)
                if key:
                    self.providers[provider_name] = ProviderConfig(
                        name=provider_name,
                        api_key=key,
                    )
                    break

    def get_provider_config(self, name: str | None = None) -> ProviderConfig:
        """Get config for a specific provider, or the best available one."""
        target = name or self.provider

        if target == "auto":
            target = self._auto_select_provider()

        if target not in self.providers:
            available = list(self.providers.keys())
            raise ConfigError(
                f"Provider '{target}' not available. "
                f"Available: {available}. "
                f"Set the appropriate API key environment variable."
            )

        config = self.providers[target]
        if self.model and not config.model:
            config.model = self.model
        return config

    def _auto_select_provider(self) -> str:
        """Auto-select the best available provider."""
        # Priority order
        priority = [
            "anthropic",
            "openai",
            "gemini",
            "groq",
            "deepseek",
            "mistral",
            "cohere",
            "together",
            "fireworks",
            "huggingface",
            "perplexity",
            "xai",
            "cerebras",
            "ollama",
        ]

        for p in priority:
            if p in self.providers:
                return p

        raise ConfigError(
            "No LLM providers available. Set an API key or install Ollama for local models. "
            "Example: export OPENAI_API_KEY=sk-... or export GROQ_API_KEY=gsk_..."
        )

    @property
    def available_providers(self) -> list[str]:
        """List all available providers."""
        return list(self.providers.keys())

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "NexusConfig":
        """Load config from YAML file."""
        path = Path(path) if path else CONFIG_FILE
        if not path.exists():
            return cls()

        try:
            import yaml

            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except Exception as e:
            raise ConfigError(f"Failed to load config from {path}: {e}")

    @classmethod
    def from_env(cls) -> "NexusConfig":
        """Create config purely from environment variables."""
        config = cls()
        config.provider = os.environ.get("NEXUSAI_PROVIDER", "auto")
        config.model = os.environ.get("NEXUSAI_MODEL")
        config.verbose = os.environ.get("NEXUSAI_VERBOSE", "true").lower() == "true"
        config.log_level = os.environ.get("NEXUSAI_LOG_LEVEL", "INFO")
        return config

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dict (hiding API keys)."""
        return {
            "provider": self.provider,
            "model": self.model,
            "available_providers": self.available_providers,
            "verbose": self.verbose,
            "log_level": self.log_level,
            "agent": {
                "max_iterations": self.agent.max_iterations,
                "pipeline_roles": self.agent.pipeline_roles,
            },
            "rag": {
                "chunk_size": self.rag.chunk_size,
                "vector_store": self.rag.vector_store,
                "top_k": self.rag.top_k,
            },
        }

    def ensure_dirs(self):
        """Create necessary directories."""
        for d in [CONFIG_DIR, MEMORY_DIR, PLUGINS_DIR, CACHE_DIR]:
            d.mkdir(parents=True, exist_ok=True)
