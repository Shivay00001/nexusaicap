import os
import json
import urllib.request
from pathlib import Path
from typing import Iterator

from nexusai.providers.base import BaseProvider, Message, Response
from nexusai.core.registry import ProviderRegistry
from nexusai.core.exceptions import ProviderError

class NeedleProvider(BaseProvider):
    """Local inference provider using Cactus Needle v2 (via needle-rs) for tool-calling."""
    name = "needle"
    supports_streaming = False
    supports_tools = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            import needle_rs
        except ImportError:
            raise ProviderError(
                self.name, 
                "needle-rs is not installed. Run `pip install needle-rs` to use Needle v2 locally."
            )
        
        self.needle_rs = needle_rs
        self.model_path = Path.home() / ".nexusai" / "models" / "needle2.cact"
        self._engine = None

    def _ensure_model(self):
        """Downloads the needle2.cact model if it doesn't exist."""
        if self._engine is not None:
            return self._engine
            
        if not self.model_path.exists():
            print(f"[Needle] Downloading Needle v2 model (13.7MB) to {self.model_path}...")
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://huggingface.co/Cactus-Compute/needle2/resolve/main/needle2.cact"
            urllib.request.urlretrieve(url, self.model_path)
            print("[Needle] Download complete!")
            
        try:
            self._engine = self.needle_rs.V2Engine.load(str(self.model_path))
            return self._engine
        except Exception as e:
            raise ProviderError(self.name, f"Failed to load Needle v2 engine: {e}")

    @property
    def available_models(self) -> list[str]:
        return ["needle-v2"]

    def _format_tools_json(self, tools: list[dict]) -> str:
        """Format OpenAI-style tools to JSON string for Needle."""
        if not tools:
            return "[]"
        return json.dumps([tool["function"] for tool in tools])

    def invoke(self, messages: list[Message], *, model: str = None, temperature: float = 0.0, max_tokens: int = 128, tools: list[dict] = None, **kwargs) -> Response:
        engine = self._ensure_model()
        
        # Combine messages into a single query string for Needle
        query = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        tools_json = self._format_tools_json(tools)
        
        try:
            out = engine.run(query, tools_json)
            # 'out' is an object with 'tool_call' or 'text' properties
            content = ""
            tool_calls = []
            
            # Since V2Engine.run() returns a decoded text or tool call depending on the wrapper structure,
            # Let's inspect the output type or properties. If it has a tool_call property, we map it.
            if hasattr(out, "tool_call") and out.tool_call:
                tool_data = json.loads(out.tool_call) if isinstance(out.tool_call, str) else out.tool_call
                
                # Check for single dict vs list of dicts
                if isinstance(tool_data, dict):
                    tool_calls = [{
                        "id": f"call_{tool_data.get('name', 'unknown')}",
                        "function": {
                            "name": tool_data.get("name"),
                            "arguments": json.dumps(tool_data.get("arguments", {}))
                        }
                    }]
            elif hasattr(out, "text") and out.text:
                content = out.text
            else:
                content = str(out) # Fallback if run returns simple string
                
            return Response(
                content=content,
                model="needle-v2",
                provider=self.name,
                tool_calls=tool_calls,
                finish_reason="tool_calls" if tool_calls else "stop"
            )
        except Exception as e:
            raise ProviderError(self.name, f"Needle generation failed: {e}")

    def stream(self, messages: list[Message], **kwargs) -> Iterator[str]:
        resp = self.invoke(messages, **kwargs)
        yield resp.content

ProviderRegistry.register("needle", NeedleProvider)
