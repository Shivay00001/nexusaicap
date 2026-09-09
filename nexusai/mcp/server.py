"""NexusAI MCP Server — Expose NexusAI tools via Model Context Protocol."""
from __future__ import annotations
import json, asyncio
from nexusai.core.registry import ToolRegistry

class MCPServer:
    """Simple MCP server that exposes NexusAI tools."""
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host; self.port = port
        self._tools_loaded = False

    def _ensure_tools(self):
        if not self._tools_loaded:
            import nexusai.tools.file_tools, nexusai.tools.shell_tools, nexusai.tools.web_tools
            import nexusai.tools.python_tools, nexusai.tools.data_tools, nexusai.tools.math_tools
            self._tools_loaded = True

    def get_tool_list(self) -> list[dict]:
        self._ensure_tools()
        return [{"name": name, **meta} for name, meta in ToolRegistry._tool_metadata.items()]

    async def handle_request(self, request: dict) -> dict:
        method = request.get("method", "")
        if method == "tools/list":
            return {"tools": self.get_tool_list()}
        elif method == "tools/call":
            name = request.get("params", {}).get("name", "")
            args = request.get("params", {}).get("arguments", {})
            try:
                self._ensure_tools()
                if "." in name:
                    cls_name, method_name = name.split(".", 1)
                    tool = ToolRegistry.get(cls_name)
                    result = tool.execute(method_name, **args)
                    return {"content": [{"type": "text", "text": str(result)}]}
                return {"error": f"Tool '{name}' requires class.method format"}
            except Exception as e:
                return {"error": str(e)}
        elif method == "initialize":
            return {"protocolVersion": "2024-11-05", "serverInfo": {"name": "nexusai-mcp", "version": "1.0.0"}, "capabilities": {"tools": {}}}
        return {"error": f"Unknown method: {method}"}

    async def start(self):
        """Start the MCP server using websockets."""
        try:
            import websockets
            async def handler(ws):
                async for msg in ws:
                    request = json.loads(msg)
                    response = await self.handle_request(request)
                    response["id"] = request.get("id")
                    response["jsonrpc"] = "2.0"
                    await ws.send(json.dumps(response))
            async with websockets.serve(handler, self.host, self.port):
                print(f"🔌 NexusAI MCP server running on ws://{self.host}:{self.port}")
                await asyncio.Future()
        except ImportError:
            print("websockets not installed. pip install nexusai[mcp]")
