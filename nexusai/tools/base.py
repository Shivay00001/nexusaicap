"""
NexusAI Base Tool — Abstract tool interface and decorator.

Every tool class inherits from BaseTool and registers its methods.
The @tool decorator can also be used to create standalone tools.
"""

from __future__ import annotations

import functools
import inspect
import json
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Callable

from nexusai.core.registry import ToolRegistry
from nexusai.core.exceptions import ToolError


@dataclass
class ToolResult:
    """Result from a tool execution."""

    success: bool
    output: Any
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.success:
            return str(self.output)
        return f"Error: {self.error}"

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output": str(self.output) if self.output is not None else None,
            "error": self.error,
        }


class BaseTool(ABC):
    """
    Base class for all NexusAI tools.

    Subclasses define tool methods and register them automatically.
    Each tool class represents a category (e.g., FileTool, GitTool).
    """

    name: str = "base_tool"
    description: str = "Base tool"
    category: str = "general"
    version: str = "1.0.0"

    # Schema for the tool's parameters (OpenAI function calling format)
    schema: dict = {}

    def __init_subclass__(cls, **kwargs):
        """Auto-register tool subclasses."""
        super().__init_subclass__(**kwargs)
        if cls.name != "base_tool":
            # Register all public methods as individual tools
            for method_name in dir(cls):
                if method_name.startswith("_"):
                    continue
                method = getattr(cls, method_name, None)
                if callable(method) and hasattr(method, "_tool_meta"):
                    meta = method._tool_meta
                    tool_name = f"{cls.name}.{method_name}"
                    ToolRegistry.register(
                        tool_name,
                        cls,
                        metadata={
                            "name": tool_name,
                            "description": meta.get("description", ""),
                            "category": cls.category,
                            "parameters": meta.get("parameters", {}),
                            "class": cls.name,
                            "method": method_name,
                        },
                    )

            # Also register the class itself
            ToolRegistry.register(
                cls.name,
                cls,
                metadata={
                    "name": cls.name,
                    "description": cls.description,
                    "category": cls.category,
                },
            )

    def execute(self, method_name: str, **kwargs) -> ToolResult:
        """Execute a named method on this tool."""
        method = getattr(self, method_name, None)
        if not method:
            return ToolResult(success=False, output=None, error=f"Method '{method_name}' not found")
        try:
            result = method(**kwargs)
            return ToolResult(success=True, output=result)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    def get_methods(self) -> list[dict]:
        """List all tool methods with their metadata."""
        methods = []
        for name in dir(self):
            if name.startswith("_"):
                continue
            method = getattr(self, name, None)
            if callable(method) and hasattr(method, "_tool_meta"):
                methods.append({
                    "name": name,
                    "full_name": f"{self.name}.{name}",
                    **method._tool_meta,
                })
        return methods

    def to_openai_tools(self) -> list[dict]:
        """Convert tool methods to OpenAI function calling format."""
        tools = []
        for m in self.get_methods():
            tools.append({
                "type": "function",
                "function": {
                    "name": m["full_name"].replace(".", "_"),
                    "description": m.get("description", ""),
                    "parameters": m.get("parameters", {"type": "object", "properties": {}}),
                },
            })
        return tools


def tool(
    description: str = "",
    parameters: dict | None = None,
    returns: str = "",
):
    """
    Decorator to mark a method as a tool.

    Usage:
        class MyTool(BaseTool):
            @tool(description="Read a file", parameters={...})
            def read_file(self, path: str) -> str:
                ...
    """

    def decorator(func: Callable) -> Callable:
        # Auto-generate parameters from type hints if not provided
        params = parameters
        if params is None:
            sig = inspect.signature(func)
            props = {}
            required = []
            for pname, param in sig.parameters.items():
                if pname == "self":
                    continue
                ptype = "string"
                if param.annotation == int:
                    ptype = "integer"
                elif param.annotation == float:
                    ptype = "number"
                elif param.annotation == bool:
                    ptype = "boolean"
                elif param.annotation == list:
                    ptype = "array"
                elif param.annotation == dict:
                    ptype = "object"

                props[pname] = {"type": ptype, "description": pname}
                if param.default == inspect.Parameter.empty:
                    required.append(pname)

            params = {
                "type": "object",
                "properties": props,
                "required": required,
            }

        func._tool_meta = {
            "description": description or func.__doc__ or "",
            "parameters": params,
            "returns": returns,
        }

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._tool_meta = func._tool_meta
        return wrapper

    return decorator
