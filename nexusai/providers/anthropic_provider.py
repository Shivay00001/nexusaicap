"""Anthropic Provider — Claude Sonnet, Opus, Haiku."""

from __future__ import annotations

from typing import Iterator

from nexusai.providers.base import BaseProvider, Message, Response
from nexusai.core.registry import ProviderRegistry
from nexusai.core.exceptions import ProviderError


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    supports_streaming = True
    supports_tools = True
    supports_vision = True

    MODELS = [
        "claude-sonnet-4-20250514", "claude-opus-4-20250514",
        "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229", "claude-3-haiku-20240307",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=self._api_key,
                timeout=self._timeout,
            )
        except ImportError:
            raise ProviderError(self.name, "Install anthropic: pip install nexusai[anthropic]")

    def invoke(self, messages, *, model=None, temperature=None, max_tokens=None, tools=None, **kwargs) -> Response:
        msgs = self._normalize_messages(messages)

        # Extract system message
        system_msg = ""
        chat_msgs = []
        for m in msgs:
            if m.role == "system":
                system_msg = m.content
            else:
                chat_msgs.append({"role": m.role, "content": m.content})

        params = {
            "model": model or self._model,
            "messages": chat_msgs,
            "max_tokens": max_tokens or self._max_tokens,
        }
        if system_msg:
            params["system"] = system_msg
        if temperature is not None:
            params["temperature"] = temperature
        else:
            params["temperature"] = self._temperature
        if tools:
            params["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"].get("parameters", {}),
                }
                for t in tools
            ]

        try:
            resp = self._client.messages.create(**params)
            content = ""
            tool_calls = []
            for block in resp.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {"name": block.name, "arguments": str(block.input)},
                    })

            return Response(
                content=content,
                model=resp.model,
                provider=self.name,
                usage={
                    "prompt_tokens": resp.usage.input_tokens,
                    "completion_tokens": resp.usage.output_tokens,
                    "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
                },
                tool_calls=tool_calls if tool_calls else None,
                finish_reason=resp.stop_reason,
                raw=resp,
            )
        except Exception as e:
            raise ProviderError(self.name, str(e))

    def stream(self, messages, *, model=None, temperature=None, max_tokens=None, **kwargs) -> Iterator[str]:
        msgs = self._normalize_messages(messages)
        system_msg = ""
        chat_msgs = []
        for m in msgs:
            if m.role == "system":
                system_msg = m.content
            else:
                chat_msgs.append({"role": m.role, "content": m.content})

        params = {
            "model": model or self._model,
            "messages": chat_msgs,
            "max_tokens": max_tokens or self._max_tokens,
            "temperature": temperature if temperature is not None else self._temperature,
        }
        if system_msg:
            params["system"] = system_msg

        try:
            with self._client.messages.stream(**params) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            raise ProviderError(self.name, str(e))

    @property
    def available_models(self) -> list[str]:
        return self.MODELS


ProviderRegistry.register("anthropic", AnthropicProvider)
