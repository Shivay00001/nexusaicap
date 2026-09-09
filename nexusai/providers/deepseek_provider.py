"""DeepSeek Provider — DeepSeek Chat, DeepSeek Coder."""

from __future__ import annotations
from typing import Iterator
import httpx
from nexusai.providers.base import BaseProvider, Message, Response
from nexusai.core.registry import ProviderRegistry
from nexusai.core.exceptions import ProviderError


class DeepSeekProvider(BaseProvider):
    name = "deepseek"
    supports_streaming = True
    supports_tools = True
    MODELS = ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._base_url = self._base_url or "https://api.deepseek.com/v1"
        self._http = httpx.Client(base_url=self._base_url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=self._timeout)

    def invoke(self, messages, *, model=None, temperature=None, max_tokens=None, tools=None, **kwargs) -> Response:
        msgs = self._messages_to_dicts(self._normalize_messages(messages))
        payload = {"model": model or self._model, "messages": msgs, "temperature": temperature if temperature is not None else self._temperature, "max_tokens": max_tokens or self._max_tokens, "stream": False}
        if tools:
            payload["tools"] = tools
        try:
            resp = self._http.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            tc = choice.get("message", {}).get("tool_calls")
            usage = data.get("usage", {})
            return Response(content=choice["message"].get("content", ""), model=data.get("model", self._model), provider=self.name,
                usage={"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0), "total_tokens": usage.get("total_tokens", 0)},
                tool_calls=tc, finish_reason=choice.get("finish_reason"), raw=data)
        except Exception as e:
            raise ProviderError(self.name, str(e))

    def stream(self, messages, *, model=None, temperature=None, max_tokens=None, **kwargs) -> Iterator[str]:
        msgs = self._messages_to_dicts(self._normalize_messages(messages))
        payload = {"model": model or self._model, "messages": msgs, "temperature": temperature if temperature is not None else self._temperature, "max_tokens": max_tokens or self._max_tokens, "stream": True}
        try:
            with self._http.stream("POST", "/chat/completions", json=payload) as resp:
                for line in resp.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        import json
                        data = json.loads(line[6:])
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        if delta.get("content"):
                            yield delta["content"]
        except Exception as e:
            raise ProviderError(self.name, str(e))

    @property
    def available_models(self) -> list[str]:
        return self.MODELS

ProviderRegistry.register("deepseek", DeepSeekProvider)
