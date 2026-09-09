"""Cohere Provider — Command R+, Command R, Embed."""

from __future__ import annotations
from typing import Iterator
from nexusai.providers.base import BaseProvider, Message, Response, EmbeddingResponse
from nexusai.core.registry import ProviderRegistry
from nexusai.core.exceptions import ProviderError


class CohereProvider(BaseProvider):
    name = "cohere"
    supports_streaming = True
    supports_tools = True
    supports_embeddings = True

    MODELS = ["command-r-plus", "command-r", "command-light", "command"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            import cohere
            self._client = cohere.ClientV2(api_key=self._api_key)
        except ImportError:
            raise ProviderError(self.name, "Install cohere: pip install nexusai[cohere]")

    def invoke(self, messages, *, model=None, temperature=None, max_tokens=None, tools=None, **kwargs) -> Response:
        msgs = self._messages_to_dicts(self._normalize_messages(messages))
        try:
            resp = self._client.chat(model=model or self._model, messages=msgs, temperature=temperature if temperature is not None else self._temperature, max_tokens=max_tokens or self._max_tokens)
            return Response(content=resp.message.content[0].text if resp.message.content else "", model=model or self._model, provider=self.name,
                usage={"prompt_tokens": resp.usage.tokens.input_tokens if resp.usage else 0, "completion_tokens": resp.usage.tokens.output_tokens if resp.usage else 0, "total_tokens": (resp.usage.tokens.input_tokens + resp.usage.tokens.output_tokens) if resp.usage else 0},
                finish_reason=resp.finish_reason, raw=resp)
        except Exception as e:
            raise ProviderError(self.name, str(e))

    def stream(self, messages, *, model=None, temperature=None, max_tokens=None, **kwargs) -> Iterator[str]:
        msgs = self._messages_to_dicts(self._normalize_messages(messages))
        try:
            stream = self._client.chat_stream(model=model or self._model, messages=msgs, temperature=temperature if temperature is not None else self._temperature, max_tokens=max_tokens or self._max_tokens)
            for event in stream:
                if hasattr(event, "delta") and hasattr(event.delta, "message") and event.delta.message and event.delta.message.content and event.delta.message.content.text:
                    yield event.delta.message.content.text
        except Exception as e:
            raise ProviderError(self.name, str(e))

    def embed(self, texts: list[str], **kwargs) -> EmbeddingResponse:
        try:
            resp = self._client.embed(texts=texts, model="embed-english-v3.0", input_type="search_document", embedding_types=["float"])
            return EmbeddingResponse(embeddings=resp.embeddings.float_ or [], model="embed-english-v3.0", provider=self.name)
        except Exception as e:
            raise ProviderError(self.name, str(e))

    @property
    def available_models(self) -> list[str]:
        return self.MODELS

ProviderRegistry.register("cohere", CohereProvider)
