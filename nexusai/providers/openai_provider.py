"""OpenAI Provider — GPT-4o, GPT-4, GPT-3.5, o1, o3, etc."""

from __future__ import annotations

from typing import Iterator

from nexusai.providers.base import BaseProvider, Message, Response, EmbeddingResponse
from nexusai.core.config import ProviderConfig
from nexusai.core.registry import ProviderRegistry
from nexusai.core.exceptions import ProviderError


class OpenAIProvider(BaseProvider):
    name = "openai"
    supports_streaming = True
    supports_tools = True
    supports_vision = True
    supports_embeddings = True

    MODELS = [
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4",
        "gpt-3.5-turbo", "o1", "o1-mini", "o1-preview",
        "o3", "o3-mini", "o4-mini",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )
        except ImportError:
            raise ProviderError(self.name, "Install openai: pip install nexusai[openai]")

    def invoke(self, messages, *, model=None, temperature=None, max_tokens=None, tools=None, **kwargs) -> Response:
        msgs = self._messages_to_dicts(self._normalize_messages(messages))
        params = {
            "model": model or self._model,
            "messages": msgs,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens or self._max_tokens,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")

        try:
            resp = self._client.chat.completions.create(**params)
            choice = resp.choices[0]
            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in choice.message.tool_calls
                ]
            return Response(
                content=choice.message.content or "",
                model=resp.model,
                provider=self.name,
                usage={
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                    "total_tokens": resp.usage.total_tokens if resp.usage else 0,
                },
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason,
                raw=resp,
            )
        except Exception as e:
            raise ProviderError(self.name, str(e))

    def stream(self, messages, *, model=None, temperature=None, max_tokens=None, **kwargs) -> Iterator[str]:
        msgs = self._messages_to_dicts(self._normalize_messages(messages))
        try:
            stream = self._client.chat.completions.create(
                model=model or self._model,
                messages=msgs,
                temperature=temperature if temperature is not None else self._temperature,
                max_tokens=max_tokens or self._max_tokens,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise ProviderError(self.name, str(e))

    def embed(self, texts: list[str], **kwargs) -> EmbeddingResponse:
        model = kwargs.get("model", "text-embedding-3-small")
        try:
            resp = self._client.embeddings.create(input=texts, model=model)
            return EmbeddingResponse(
                embeddings=[e.embedding for e in resp.data],
                model=model,
                provider=self.name,
                usage={"total_tokens": resp.usage.total_tokens if resp.usage else 0},
            )
        except Exception as e:
            raise ProviderError(self.name, str(e))

    @property
    def available_models(self) -> list[str]:
        return self.MODELS


ProviderRegistry.register("openai", OpenAIProvider)
