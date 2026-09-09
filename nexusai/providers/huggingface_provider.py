"""HuggingFace Provider — Inference API for open-source models."""

from __future__ import annotations
from typing import Iterator
import httpx
from nexusai.providers.base import BaseProvider, Message, Response, EmbeddingResponse
from nexusai.core.registry import ProviderRegistry
from nexusai.core.exceptions import ProviderError


class HuggingFaceProvider(BaseProvider):
    name = "huggingface"
    supports_streaming = True
    supports_embeddings = True
    MODELS = ["meta-llama/Meta-Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-2-9b-it", "Qwen/Qwen2.5-7B-Instruct", "microsoft/Phi-3-mini-4k-instruct"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._base_url = self._base_url or "https://api-inference.huggingface.co"
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self._http = httpx.Client(base_url=self._base_url, headers=headers, timeout=self._timeout)

    def invoke(self, messages, *, model=None, temperature=None, max_tokens=None, tools=None, **kwargs) -> Response:
        msgs = self._normalize_messages(messages)
        # Build prompt from messages
        prompt = ""
        for m in msgs:
            if m.role == "system":
                prompt += f"<|system|>\n{m.content}\n"
            elif m.role == "user":
                prompt += f"<|user|>\n{m.content}\n"
            elif m.role == "assistant":
                prompt += f"<|assistant|>\n{m.content}\n"
        prompt += "<|assistant|>\n"

        mdl = model or self._model
        payload = {"inputs": prompt, "parameters": {"temperature": temperature if temperature is not None else self._temperature, "max_new_tokens": max_tokens or self._max_tokens, "return_full_text": False}}
        try:
            resp = self._http.post(f"/models/{mdl}", json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data[0].get("generated_text", "") if isinstance(data, list) else data.get("generated_text", "")
            return Response(content=text, model=mdl, provider=self.name, finish_reason="stop", raw=data)
        except Exception as e:
            raise ProviderError(self.name, str(e))

    def stream(self, messages, *, model=None, temperature=None, max_tokens=None, **kwargs) -> Iterator[str]:
        # HF Inference API doesn't natively support SSE streaming for all models; fall back
        resp = self.invoke(messages, model=model, temperature=temperature, max_tokens=max_tokens)
        yield resp.content

    def embed(self, texts: list[str], **kwargs) -> EmbeddingResponse:
        model = kwargs.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        try:
            resp = self._http.post(f"/models/{model}", json={"inputs": texts})
            resp.raise_for_status()
            data = resp.json()
            return EmbeddingResponse(embeddings=data if isinstance(data[0], list) else [data], model=model, provider=self.name)
        except Exception as e:
            raise ProviderError(self.name, str(e))

    @property
    def available_models(self) -> list[str]:
        return self.MODELS

ProviderRegistry.register("huggingface", HuggingFaceProvider)
