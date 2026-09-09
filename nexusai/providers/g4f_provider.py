"""GPT4Free Provider — Free access to various LLM models."""
from __future__ import annotations
from typing import Iterator
from nexusai.providers.base import BaseProvider, Message, Response
from nexusai.core.registry import ProviderRegistry
from nexusai.core.exceptions import ProviderError

class G4FProvider(BaseProvider):
    name = "g4f"
    supports_streaming = True
    supports_tools = False  # Most G4F providers don't natively support function calling robustly
    
    MODELS = ["gpt-4", "gpt-4o", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet", "gemini-pro", "mixtral-8x7b"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            import g4f
            self._g4f = g4f
        except ImportError:
            raise ProviderError(self.name, "Install g4f: pip install -U g4f")

    def invoke(self, messages, *, model=None, temperature=None, max_tokens=None, tools=None, **kwargs) -> Response:
        msgs = self._messages_to_dicts(self._normalize_messages(messages))
        try:
            response = self._g4f.ChatCompletion.create(
                model=model or self._model or self._g4f.models.default,
                messages=msgs,
            )
            return Response(
                content=response,
                model=model or "g4f-default",
                provider=self.name,
                finish_reason="stop"
            )
        except Exception as e:
            raise ProviderError(self.name, str(e))

    def stream(self, messages, *, model=None, temperature=None, max_tokens=None, **kwargs) -> Iterator[str]:
        msgs = self._messages_to_dicts(self._normalize_messages(messages))
        try:
            response = self._g4f.ChatCompletion.create(
                model=model or self._model or self._g4f.models.default,
                messages=msgs,
                stream=True
            )
            for message in response:
                if message:
                    yield message
        except Exception as e:
            raise ProviderError(self.name, str(e))
            
    @property
    def available_models(self) -> list[str]:
        return self.MODELS

ProviderRegistry.register("g4f", G4FProvider)
