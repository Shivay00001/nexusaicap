"""Google Gemini Provider — Gemini 2.0, 1.5 Pro, Flash, etc."""

from __future__ import annotations

from typing import Iterator

from nexusai.providers.base import BaseProvider, Message, Response, EmbeddingResponse
from nexusai.core.registry import ProviderRegistry
from nexusai.core.exceptions import ProviderError


class GeminiProvider(BaseProvider):
    name = "gemini"
    supports_streaming = True
    supports_tools = True
    supports_vision = True
    supports_embeddings = True

    MODELS = [
        "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash",
        "gemini-2.0-flash-lite", "gemini-1.5-pro", "gemini-1.5-flash",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._genai = genai
            self._gen_model = genai.GenerativeModel(self._model)
        except ImportError:
            raise ProviderError(self.name, "Install google-generativeai: pip install nexusai[google]")

    def invoke(self, messages, *, model=None, temperature=None, max_tokens=None, tools=None, **kwargs) -> Response:
        msgs = self._normalize_messages(messages)
        if model and model != self._model:
            gen_model = self._genai.GenerativeModel(model)
        else:
            gen_model = self._gen_model

        # Convert messages to Gemini format
        contents = []
        for m in msgs:
            if m.role == "system":
                contents.append({"role": "user", "parts": [m.content]})
                contents.append({"role": "model", "parts": ["Understood. I will follow these instructions."]})
            elif m.role == "user":
                contents.append({"role": "user", "parts": [m.content]})
            elif m.role == "assistant":
                contents.append({"role": "model", "parts": [m.content]})

        generation_config = {
            "temperature": temperature if temperature is not None else self._temperature,
            "max_output_tokens": max_tokens or self._max_tokens,
        }

        try:
            resp = gen_model.generate_content(
                contents,
                generation_config=generation_config,
            )
            return Response(
                content=resp.text if resp.text else "",
                model=model or self._model,
                provider=self.name,
                usage={
                    "prompt_tokens": getattr(resp.usage_metadata, "prompt_token_count", 0) if hasattr(resp, "usage_metadata") else 0,
                    "completion_tokens": getattr(resp.usage_metadata, "candidates_token_count", 0) if hasattr(resp, "usage_metadata") else 0,
                    "total_tokens": getattr(resp.usage_metadata, "total_token_count", 0) if hasattr(resp, "usage_metadata") else 0,
                },
                finish_reason="stop",
                raw=resp,
            )
        except Exception as e:
            raise ProviderError(self.name, str(e))

    def stream(self, messages, *, model=None, temperature=None, max_tokens=None, **kwargs) -> Iterator[str]:
        msgs = self._normalize_messages(messages)
        if model and model != self._model:
            gen_model = self._genai.GenerativeModel(model)
        else:
            gen_model = self._gen_model

        contents = []
        for m in msgs:
            if m.role == "system":
                contents.append({"role": "user", "parts": [m.content]})
                contents.append({"role": "model", "parts": ["Understood."]})
            elif m.role == "user":
                contents.append({"role": "user", "parts": [m.content]})
            elif m.role == "assistant":
                contents.append({"role": "model", "parts": [m.content]})

        try:
            response = gen_model.generate_content(
                contents,
                generation_config={"temperature": temperature or self._temperature, "max_output_tokens": max_tokens or self._max_tokens},
                stream=True,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise ProviderError(self.name, str(e))

    def embed(self, texts: list[str], **kwargs) -> EmbeddingResponse:
        try:
            result = self._genai.embed_content(
                model="models/text-embedding-004",
                content=texts,
            )
            embeddings = result["embedding"] if isinstance(result["embedding"][0], list) else [result["embedding"]]
            return EmbeddingResponse(embeddings=embeddings, model="text-embedding-004", provider=self.name)
        except Exception as e:
            raise ProviderError(self.name, str(e))

    @property
    def available_models(self) -> list[str]:
        return self.MODELS


ProviderRegistry.register("gemini", GeminiProvider)
