"""
NexusAI Registry — Dynamic provider and tool discovery.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Type

from nexusai.core.exceptions import NexusAIError


class ProviderRegistry:
    """
    Registry for LLM providers. Supports dynamic registration and lazy loading.
    """

    _providers: dict[str, Type] = {}
    _instances: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type):
        """Register a provider class."""
        cls._providers[name] = provider_class

    @classmethod
    def get(cls, name: str, **kwargs) -> Any:
        """Get or create a provider instance."""
        if name in cls._instances:
            return cls._instances[name]

        if name not in cls._providers:
            # Try lazy loading
            cls._lazy_load(name)

        if name not in cls._providers:
            raise NexusAIError(
                f"Provider '{name}' not registered. "
                f"Available: {list(cls._providers.keys())}"
            )

        instance = cls._providers[name](**kwargs)
        cls._instances[name] = instance
        return instance

    @classmethod
    def _lazy_load(cls, name: str):
        """Attempt to import and register a provider by name."""
        module_map = {
            "openai": "nexusai.providers.openai_provider",
            "anthropic": "nexusai.providers.anthropic_provider",
            "gemini": "nexusai.providers.gemini_provider",
            "groq": "nexusai.providers.groq_provider",
            "ollama": "nexusai.providers.ollama_provider",
            "mistral": "nexusai.providers.mistral_provider",
            "cohere": "nexusai.providers.cohere_provider",
            "together": "nexusai.providers.together_provider",
            "deepseek": "nexusai.providers.deepseek_provider",
            "fireworks": "nexusai.providers.fireworks_provider",
            "huggingface": "nexusai.providers.huggingface_provider",
        }

        module_name = module_map.get(name)
        if module_name:
            try:
                importlib.import_module(module_name)
            except ImportError:
                pass

    @classmethod
    def available(cls) -> list[str]:
        """List all registered providers."""
        return list(cls._providers.keys())

    @classmethod
    def clear(cls):
        """Clear all registrations (for testing)."""
        cls._providers.clear()
        cls._instances.clear()


class ToolRegistry:
    """
    Registry for tools. Tools are discovered and registered automatically.
    """

    _tools: dict[str, Type] = {}
    _tool_instances: dict[str, Any] = {}
    _tool_metadata: dict[str, dict] = {}

    @classmethod
    def register(cls, name: str, tool_class: Type, metadata: dict | None = None):
        """Register a tool class with optional metadata."""
        cls._tools[name] = tool_class
        cls._tool_metadata[name] = metadata or {
            "name": name,
            "description": getattr(tool_class, "description", f"Tool: {name}"),
            "category": getattr(tool_class, "category", "general"),
        }

    @classmethod
    def get(cls, name: str) -> Any:
        """Get or create a tool instance."""
        if name in cls._tool_instances:
            return cls._tool_instances[name]

        if name not in cls._tools:
            cls._lazy_load_tools()

        if name not in cls._tools:
            raise NexusAIError(
                f"Tool '{name}' not registered. "
                f"Available: {list(cls._tools.keys())}"
            )

        instance = cls._tools[name]()
        cls._tool_instances[name] = instance
        return instance

    @classmethod
    def _lazy_load_tools(cls):
        """Auto-discover and load all tool modules."""
        tool_modules = [
            "nexusai.tools.file_tools",
            "nexusai.tools.git_tools",
            "nexusai.tools.github_tools",
            "nexusai.tools.web_tools",
            "nexusai.tools.shell_tools",
            "nexusai.tools.python_tools",
            "nexusai.tools.data_tools",
            "nexusai.tools.math_tools",
            "nexusai.tools.image_tools",
            "nexusai.tools.pdf_tools",
            "nexusai.tools.crypto_tools",
            "nexusai.tools.network_tools",
            "nexusai.tools.docker_tools",
            "nexusai.tools.database_tools",
            "nexusai.tools.email_tools",
            "nexusai.tools.archive_tools",
            "nexusai.tools.text_tools",
            "nexusai.tools.api_tools",
            "nexusai.tools.system_tools",
            "nexusai.tools.calendar_tools",
        ]

        for module_name in tool_modules:
            try:
                importlib.import_module(module_name)
            except ImportError:
                pass

    @classmethod
    def available(cls) -> list[str]:
        """List all registered tools."""
        if not cls._tools:
            cls._lazy_load_tools()
        return list(cls._tools.keys())

    @classmethod
    def search(cls, query: str) -> list[dict]:
        """Search tools by name or description."""
        if not cls._tools:
            cls._lazy_load_tools()

        query_lower = query.lower()
        results = []
        for name, meta in cls._tool_metadata.items():
            if (
                query_lower in name.lower()
                or query_lower in meta.get("description", "").lower()
                or query_lower in meta.get("category", "").lower()
            ):
                results.append(meta)
        return results

    @classmethod
    def by_category(cls) -> dict[str, list[str]]:
        """Group tools by category."""
        if not cls._tools:
            cls._lazy_load_tools()

        categories: dict[str, list[str]] = {}
        for name, meta in cls._tool_metadata.items():
            cat = meta.get("category", "general")
            categories.setdefault(cat, []).append(name)
        return categories

    @classmethod
    def get_schema(cls, name: str) -> dict:
        """Get the JSON schema for a tool's parameters."""
        if name not in cls._tools:
            cls._lazy_load_tools()

        tool_class = cls._tools.get(name)
        if not tool_class:
            return {}

        return getattr(tool_class, "schema", cls._tool_metadata.get(name, {}))

    @classmethod
    def count(cls) -> int:
        """Total number of registered tools."""
        if not cls._tools:
            cls._lazy_load_tools()
        return len(cls._tools)

    @classmethod
    def clear(cls):
        """Clear all registrations (for testing)."""
        cls._tools.clear()
        cls._tool_instances.clear()
        cls._tool_metadata.clear()
