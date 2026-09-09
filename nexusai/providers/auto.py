"""Auto Provider — Automatically selects the best available LLM provider."""

from __future__ import annotations
from typing import Iterator

from nexusai.providers.base import BaseProvider, Message, Response, EmbeddingResponse
from nexusai.core.config import NexusConfig
from nexusai.core.registry import ProviderRegistry
from nexusai.core.exceptions import ProviderError


class AutoProvider(BaseProvider):
    """
    Meta-provider that automatically selects the best available provider
    based on installed packages and configured API keys.
    """

    name = "auto"

    def __init__(self, config: NexusConfig | None = None, **kwargs):
        self._config = config or NexusConfig.from_env()
        self._delegate: BaseProvider | None = None
        self._kwargs = kwargs
        self._resolve()

    def _resolve(self):
        """Find and instantiate the best available provider."""
        provider_config = self._config.get_provider_config()
        self._delegate = ProviderRegistry.get(
            provider_config.name,
            config=provider_config,
            **self._kwargs,
        )

    def invoke(self, messages, **kwargs) -> Response:
        assert self._delegate
        return self._delegate.invoke(messages, **kwargs)

    def stream(self, messages, **kwargs) -> Iterator[str]:
        assert self._delegate
        return self._delegate.stream(messages, **kwargs)

    async def ainvoke(self, messages, **kwargs) -> Response:
        assert self._delegate
        return await self._delegate.ainvoke(messages, **kwargs)

    def embed(self, texts: list[str], **kwargs) -> EmbeddingResponse:
        assert self._delegate
        return self._delegate.embed(texts, **kwargs)

    @property
    def model(self) -> str:
        return self._delegate._model if self._delegate else "auto"

    @property
    def provider_name(self) -> str:
        return self._delegate.name if self._delegate else "auto"

    def __repr__(self) -> str:
        return f"<AutoProvider(delegate={self._delegate})>"


ProviderRegistry.register("auto", AutoProvider)
