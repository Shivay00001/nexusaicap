"""Groq Provider — Llama, Mixtral, Gemma on Groq's ultra-fast inference."""

from __future__ import annotations
from typing import Iterator
from nexusai.providers.base import BaseProvider, Message, Response
from nexusai.core.registry import ProviderRegistry
from nexusai.core.exceptions import ProviderError


class GroqProvider(BaseProvider):
    name = "groq"
    supports_streaming = True
    supports_tools = True

    MODELS = [
        "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
        "llama-3.2-90b-vision-preview", "mixtral-8x7b-32768",
        "gemma2-9b-it", "qwen-qwq-32b",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            from groq import Groq
            self._client = Groq(api_key=self._api_key, timeout=self._timeout)
        except ImportError:
            raise ProviderError(self.name, "Install groq: pip install nexusai[groq]")

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
        try:
            resp = self._client.chat.completions.create(**params)
            choice = resp.choices[0]
            tc = None
            if choice.message.tool_calls:
                tc = [{"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in choice.message.tool_calls]
            return Response(
                content=choice.message.content or "", model=resp.model, provider=self.name,
                usage={"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens, "total_tokens": resp.usage.total_tokens} if resp.usage else {},
                tool_calls=tc, finish_reason=choice.finish_reason, raw=resp,
            )
        except Exception as e:
            raise ProviderError(self.name, str(e))

    def stream(self, messages, *, model=None, temperature=None, max_tokens=None, **kwargs) -> Iterator[str]:
        msgs = self._messages_to_dicts(self._normalize_messages(messages))
        try:
            stream = self._client.chat.completions.create(
                model=model or self._model, messages=msgs,
                temperature=temperature if temperature is not None else self._temperature,
                max_tokens=max_tokens or self._max_tokens, stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise ProviderError(self.name, str(e))

    @property
    def available_models(self) -> list[str]:
        return self.MODELS


ProviderRegistry.register("groq", GroqProvider)
