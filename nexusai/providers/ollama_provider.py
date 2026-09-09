"""Ollama Provider — Local LLM inference (no API key required)."""

from __future__ import annotations
import json
from typing import Iterator
import httpx
from nexusai.providers.base import BaseProvider, Message, Response, EmbeddingResponse
from nexusai.core.registry import ProviderRegistry
from nexusai.core.exceptions import ProviderError


class OllamaProvider(BaseProvider):
    name = "ollama"
    supports_streaming = True
    supports_tools = True
    supports_embeddings = True

    MODELS = [
        "llama3.2", "llama3.1", "llama3", "mistral", "mixtral",
        "gemma2", "qwen2.5", "phi3", "codellama", "deepseek-coder-v2",
        "command-r", "starcoder2", "vicuna", "neural-chat",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._base_url = self._base_url or "http://localhost:11434"
        self._http = httpx.Client(base_url=self._base_url, timeout=self._timeout)

    def invoke(self, messages, *, model=None, temperature=None, max_tokens=None, tools=None, **kwargs) -> Response:
        msgs = self._messages_to_dicts(self._normalize_messages(messages))
        payload = {
            "model": model or self._model,
            "messages": msgs,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self._temperature,
            },
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        if tools:
            payload["tools"] = tools

        try:
            resp = self._http.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("message", {})
            tc = None
            if msg.get("tool_calls"):
                tc = [{"id": f"call_{i}", "type": "function", "function": {"name": t["function"]["name"], "arguments": json.dumps(t["function"].get("arguments", {}))}} for i, t in enumerate(msg["tool_calls"])]
            return Response(
                content=msg.get("content", ""), model=data.get("model", self._model), provider=self.name,
                usage={"prompt_tokens": data.get("prompt_eval_count", 0), "completion_tokens": data.get("eval_count", 0), "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)},
                tool_calls=tc, finish_reason="stop", raw=data,
            )
        except httpx.ConnectError:
            raise ProviderError(self.name, f"Cannot connect to Ollama at {self._base_url}. Is Ollama running?")
        except Exception as e:
            raise ProviderError(self.name, str(e))

    def stream(self, messages, *, model=None, temperature=None, max_tokens=None, **kwargs) -> Iterator[str]:
        msgs = self._messages_to_dicts(self._normalize_messages(messages))
        payload = {
            "model": model or self._model, "messages": msgs, "stream": True,
            "options": {"temperature": temperature if temperature is not None else self._temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        try:
            with self._http.stream("POST", "/api/chat", json=payload) as resp:
                for line in resp.iter_lines():
                    if line:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
        except httpx.ConnectError:
            raise ProviderError(self.name, f"Cannot connect to Ollama at {self._base_url}.")
        except Exception as e:
            raise ProviderError(self.name, str(e))

    def embed(self, texts: list[str], **kwargs) -> EmbeddingResponse:
        model = kwargs.get("model", "nomic-embed-text")
        try:
            embeddings = []
            for text in texts:
                resp = self._http.post("/api/embed", json={"model": model, "input": text})
                resp.raise_for_status()
                data = resp.json()
                embeddings.append(data.get("embeddings", [[]])[0])
            return EmbeddingResponse(embeddings=embeddings, model=model, provider=self.name)
        except Exception as e:
            raise ProviderError(self.name, str(e))

    @property
    def available_models(self) -> list[str]:
        try:
            resp = self._http.get("/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return self.MODELS


ProviderRegistry.register("ollama", OllamaProvider)
