"""NexusAI Exception Hierarchy."""


class NexusAIError(Exception):
    """Base exception for all NexusAI errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ProviderError(NexusAIError):
    """Raised when an LLM provider encounters an error."""

    def __init__(self, provider: str, message: str, **kwargs):
        super().__init__(f"[{provider}] {message}", **kwargs)
        self.provider = provider


class ToolError(NexusAIError):
    """Raised when a tool execution fails."""

    def __init__(self, tool_name: str, message: str, **kwargs):
        super().__init__(f"[Tool:{tool_name}] {message}", **kwargs)
        self.tool_name = tool_name


class AgentError(NexusAIError):
    """Raised when an agent pipeline step fails."""

    def __init__(self, role: str, message: str, **kwargs):
        super().__init__(f"[Agent:{role}] {message}", **kwargs)
        self.role = role


class RAGError(NexusAIError):
    """Raised when RAG pipeline encounters an error."""
    pass


class ConfigError(NexusAIError):
    """Raised when configuration is invalid."""
    pass


class PluginError(NexusAIError):
    """Raised when a plugin fails to load or execute."""

    def __init__(self, plugin_name: str, message: str, **kwargs):
        super().__init__(f"[Plugin:{plugin_name}] {message}", **kwargs)
        self.plugin_name = plugin_name


class MCPError(NexusAIError):
    """Raised when MCP communication fails."""
    pass
