"""Mistral Provider — Mistral Large, Medium, Small, Codestral."""

from __future__ import annotations
from typing import Iterator
from nexusai.providers.base import BaseProvider, Message, Response, EmbeddingResponse
from nexusai.core.registry import ProviderRegistry
from nexusai.core.exceptions import ProviderError


class MistralProvider(BaseProvider):
    name = "mistral"
    supports_streaming = True
    supports_tools = True
    supports_embeddings = True

    MODELS = ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest", "codestral-latest", "open-mistral-nemo", "pixtral-large-latest"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            from mistralai import Mistral
            self._client = Mistral(api_key=self._api_key)
        except ImportError:
            raise ProviderError(self.name, "Install mistralai: pip install nexusai[mistral]")

    def invoke(self, messages, *, model=None, temperature=None, max_tokens=None, tools=None, **kwargs) -> Response:
        msgs = self._messages_to_dicts(self._normalize_messages(messages))
        params = {"model": model or self._model, "messages": msgs, "temperature": temperature if temperature is not None else self._temperature, "max_tokens": max_tokens or self._max_tokens}
        if tools:
            params["tools"] = tools
        try:
            resp = self._client.chat.complete(**params)
            choice = resp.choices[0]
            tc = None
            if choice.message.tool_calls:
                tc = [{"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in choice.message.tool_calls]
            return Response(content=choice.message.content or "", model=resp.model, provider=self.name,
                usage={"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens, "total_tokens": resp.usage.total_tokens} if resp.usage else {},
                tool_calls=tc, finish_reason=choice.finish_reason, raw=resp)
        except Exception as e:
            raise ProviderError(self.name, str(e))

    def stream(self, messages, *, model=None, temperature=None, max_tokens=None, **kwargs) -> Iterator[str]:
        msgs = self._messages_to_dicts(self._normalize_messages(messages))
        try:
            stream = self._client.chat.stream(model=model or self._model, messages=msgs, temperature=temperature if temperature is not None else self._temperature, max_tokens=max_tokens or self._max_tokens)
            for event in stream:
                if event.data.choices and event.data.choices[0].delta.content:
                    yield event.data.choices[0].delta.content
        except Exception as e:
            raise ProviderError(self.name, str(e))

    def embed(self, texts: list[str], **kwargs) -> EmbeddingResponse:
        try:
            resp = self._client.embeddings.create(model="mistral-embed", inputs=texts)
            return EmbeddingResponse(embeddings=[e.embedding for e in resp.data], model="mistral-embed", provider=self.name)
        except Exception as e:
            raise ProviderError(self.name, str(e))

    @property
    def available_models(self) -> list[str]:
        return self.MODELS

ProviderRegistry.register("mistral", MistralProvider)
