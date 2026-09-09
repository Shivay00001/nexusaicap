"""
NexusAI Base Provider — Abstract interface for all LLM providers.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

from nexusai.core.config import ProviderConfig
from nexusai.core.exceptions import ProviderError


@dataclass
class Message:
    """A chat message."""

    role: str  # system, user, assistant, tool
    content: str
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class Response:
    """LLM response wrapper."""

    content: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[dict] | None = None
    finish_reason: str | None = None
    raw: Any = None

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


@dataclass
class EmbeddingResponse:
    """Embedding response wrapper."""

    embeddings: list[list[float]]
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)


class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers.

    Every provider must implement:
    - invoke(): Synchronous completion
    - stream(): Streaming completion
    - ainvoke(): Async completion
    """

    name: str = "base"
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_vision: bool = False
    supports_embeddings: bool = False

    def __init__(self, config: ProviderConfig | None = None, **kwargs):
        self.config = config or ProviderConfig(name=self.name)
        self._model = kwargs.get("model") or self.config.default_model
        self._temperature = kwargs.get("temperature", self.config.temperature)
        self._max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        self._api_key = kwargs.get("api_key") or self.config.api_key
        self._base_url = kwargs.get("base_url") or self.config.base_url
        self._timeout = kwargs.get("timeout", self.config.timeout)
        self._client: Any = None
        self._async_client: Any = None

    @abstractmethod
    def invoke(
        self,
        messages: list[Message] | str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Response:
        """Send a completion request and return the full response."""
        ...

    def stream(
        self,
        messages: list[Message] | str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """Stream completion tokens."""
        # Default: fall back to non-streaming
        response = self.invoke(
            messages, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        yield response.content

    async def ainvoke(
        self,
        messages: list[Message] | str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Response:
        """Async completion. Default wraps sync invoke."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.invoke(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                **kwargs,
            ),
        )

    async def astream(
        self,
        messages: list[Message] | str,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Async streaming. Default wraps sync stream."""
        loop = asyncio.get_event_loop()
        for chunk in self.stream(messages, **kwargs):
            yield chunk

    def embed(self, texts: list[str], **kwargs) -> EmbeddingResponse:
        """Generate embeddings. Override in providers that support it."""
        raise ProviderError(
            self.name, f"Provider '{self.name}' does not support embeddings."
        )

    def _normalize_messages(self, messages: list[Message] | str) -> list[Message]:
        """Convert string input to message list."""
        if isinstance(messages, str):
            return [Message(role="user", content=messages)]
        return messages

    def _messages_to_dicts(self, messages: list[Message]) -> list[dict]:
        """Convert Message objects to dicts for API calls."""
        return [m.to_dict() for m in messages]

    @property
    def model(self) -> str:
        return self._model

    @property
    def available_models(self) -> list[str]:
        """List available models for this provider. Override per provider."""
        return [self._model]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(model={self._model})>"
