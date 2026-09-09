"""Pollinations.ai Provider — Completely free text generation API used heavily in WA agents."""
from __future__ import annotations
import json
import urllib.request
import urllib.parse
from typing import Iterator

from nexusai.providers.base import BaseProvider, Message, Response
from nexusai.core.registry import ProviderRegistry
from nexusai.core.exceptions import ProviderError

class PollinationsProvider(BaseProvider):
    name = "pollinations"
    supports_streaming = False
    supports_tools = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://text.pollinations.ai/"

    @property
    def available_models(self) -> list[str]:
        return [
            "openai",        # GPT-4o / GPT-4o-mini
            "mistral",       # Mistral Nemo
            "mistral-large", # Mistral Large
            "llama",         # Llama 3.1
            "command-r",     # Cohere Command-R
            "unity",         # Unity with search
            "midjourney",    # Midjourney simulator
            "rtist",         # Artist simulator
            "searchdoge",    # Search Doge
            "p1"             # Pollinations 1 (Reasoning)
        ]

    def invoke(self, messages: list[Message], *, model: str = None, temperature: float = None, max_tokens: int = None, tools: list[dict] = None, **kwargs) -> Response:
        try:
            # Pollinations accepts a simple POST request with a JSON body matching OpenAI format
            url = self.base_url
            if model:
                url += f"?model={model}"
                
            headers = {"Content-Type": "application/json"}
            data = {
                "messages": self._messages_to_dicts(self._normalize_messages(messages)),
                "jsonMode": False
            }
            
            req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers, method="POST")
            
            with urllib.request.urlopen(req, timeout=kwargs.get('timeout', 30)) as response:
                result_text = response.read().decode('utf-8')
                
            return Response(
                content=result_text,
                model=model or "pollinations-default",
                provider=self.name,
                finish_reason="stop"
            )
        except Exception as e:
            raise ProviderError(self.name, f"Pollinations API request failed: {e}")

    def stream(self, messages: list[Message], **kwargs) -> Iterator[str]:
        # Pollinations doesn't support streaming natively via the simple endpoint, so we fallback to invoke
        resp = self.invoke(messages, **kwargs)
        yield resp.content

ProviderRegistry.register("pollinations", PollinationsProvider)
