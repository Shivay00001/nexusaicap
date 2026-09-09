"""
NexusAI Agent Orchestrator — The brain of the 7-role pipeline.

Pipeline: Planner → Researcher → Tool Manager → Coder → Reviewer → Auditor → Verifier

This is the main entry point for autonomous task execution.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from nexusai.core.config import NexusConfig
from nexusai.core.exceptions import AgentError
from nexusai.providers.base import BaseProvider, Message, Response
from nexusai.tools.base import ToolResult


@dataclass
class TaskContext:
    """Shared context passed through the pipeline."""

    task: str
    plan: str = ""
    research: str = ""
    selected_tools: list[str] = field(default_factory=list)
    code: str = ""
    code_output: str = ""
    review_feedback: str = ""
    audit_report: str = ""
    verification: str = ""
    messages: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    iterations: int = 0
    status: str = "pending"  # pending, running, completed, failed
    start_time: float = 0
    end_time: float = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        return (self.end_time or time.time()) - self.start_time

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "status": self.status,
            "plan": self.plan[:500],
            "code": self.code[:1000],
            "iterations": self.iterations,
            "tool_calls": len(self.tool_results),
            "errors": self.errors,
            "elapsed_seconds": round(self.elapsed, 2),
        }


class AgentOrchestrator:
    """
    Main orchestrator for the 7-role autonomous agent pipeline.

    Usage:
        orchestrator = AgentOrchestrator(config)
        result = orchestrator.run("Build a REST API with FastAPI")
    """

    SYSTEM_PROMPT = """You are NexusAI, a powerful autonomous AI agent with access to tools and code execution.
You operate through a sophisticated 7-role pipeline to complete tasks:

1. PLANNER: Decompose the task into actionable steps
2. RESEARCHER: Gather relevant context and information
3. TOOL MANAGER: Select the right tools for each step
4. CODER: Write and execute code to accomplish the task
5. REVIEWER: Review the code for quality and correctness
6. AUDITOR: Check for security issues and best practices
7. VERIFIER: Validate the output meets requirements

You have access to 2000+ tools across categories: file operations, git, GitHub, web, shell, Python execution, data processing, math, cryptography, networking, Docker, databases, email, archives, text processing, APIs, and system management.

IMPORTANT RULES:
- Think step by step
- Use tools when needed
- Write clean, working code
- Handle errors gracefully
- Verify your work before declaring success
"""

    def __init__(self, config: NexusConfig | None = None, provider: BaseProvider | None = None, verbose: bool = True):
        self.config = config or NexusConfig.from_env()
        self._provider = provider
        self.verbose = verbose
        self._tools_loaded = False

    def _get_provider(self) -> BaseProvider:
        """Get or create the LLM provider."""
        if self._provider:
            return self._provider

        from nexusai.core.registry import ProviderRegistry

        provider_config = self.config.get_provider_config()

        # Import all providers to register them
        provider_modules = [
            "nexusai.providers.openai_provider",
            "nexusai.providers.anthropic_provider",
            "nexusai.providers.gemini_provider",
            "nexusai.providers.groq_provider",
            "nexusai.providers.ollama_provider",
            "nexusai.providers.mistral_provider",
            "nexusai.providers.cohere_provider",
            "nexusai.providers.together_provider",
            "nexusai.providers.deepseek_provider",
            "nexusai.providers.fireworks_provider",
            "nexusai.providers.huggingface_provider",
        ]
        for mod in provider_modules:
            try:
                __import__(mod)
            except (ImportError, Exception):
                pass

        self._provider = ProviderRegistry.get(provider_config.name, config=provider_config)
        return self._provider

    def _load_tools(self):
        """Load all tool classes."""
        if self._tools_loaded:
            return
        tool_modules = [
            "nexusai.tools.file_tools",
            "nexusai.tools.shell_tools",
            "nexusai.tools.web_tools",
            "nexusai.tools.git_tools",
            "nexusai.tools.github_tools",
            "nexusai.tools.python_tools",
            "nexusai.tools.data_tools",
            "nexusai.tools.math_tools",
            "nexusai.tools.crypto_tools",
            "nexusai.tools.network_tools",
            "nexusai.tools.system_tools",
        ]
        for mod in tool_modules:
            try:
                __import__(mod)
            except (ImportError, Exception):
                pass
        self._tools_loaded = True

    def _get_tools_description(self) -> str:
        """Build a description of available tools for the LLM."""
        from nexusai.core.registry import ToolRegistry

        self._load_tools()
        categories = ToolRegistry.by_category()
        desc = "## Available Tools\n\n"
        for cat, tools in categories.items():
            desc += f"### {cat.title()}\n"
            for t in tools[:20]:
                meta = ToolRegistry._tool_metadata.get(t, {})
                desc += f"- **{t}**: {meta.get('description', '')}\n"
            desc += "\n"
        return desc

    def _execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        from nexusai.core.registry import ToolRegistry

        self._load_tools()

        # Parse class.method format
        if "." in tool_name:
            class_name, method_name = tool_name.split(".", 1)
        else:
            class_name = tool_name
            method_name = None

        try:
            tool_instance = ToolRegistry.get(class_name)
            if method_name:
                return tool_instance.execute(method_name, **kwargs)
            # If no method specified, try to find a default
            return ToolResult(success=True, output=f"Tool '{tool_name}' loaded. Specify a method.")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    def run(self, task: str, **kwargs) -> TaskContext:
        """
        Execute a task through the full 7-role pipeline.

        Args:
            task: Natural language task description
            **kwargs: Additional parameters

        Returns:
            TaskContext with all results
        """
        ctx = TaskContext(task=task, start_time=time.time(), status="running")

        if self.verbose:
            self._log("🚀", f"Starting task: {task}")
            self._log("🔧", f"Provider: {self.config.provider}")

        try:
            provider = self._get_provider()

            # Phase 1: Planning
            ctx = self._phase_plan(provider, ctx)

            # Phase 2: Research
            ctx = self._phase_research(provider, ctx)

            # Phase 3: Tool Selection
            ctx = self._phase_tool_selection(provider, ctx)

            # Phase 4: Code Generation & Execution
            ctx = self._phase_code(provider, ctx)

            # Phase 5: Review
            ctx = self._phase_review(provider, ctx)

            # Phase 6: Audit
            ctx = self._phase_audit(provider, ctx)

            # Phase 7: Verification
            ctx = self._phase_verify(provider, ctx)

            ctx.status = "completed"

        except Exception as e:
            ctx.status = "failed"
            ctx.errors.append(str(e))
            if self.verbose:
                self._log("❌", f"Task failed: {e}")

        ctx.end_time = time.time()

        if self.verbose:
            self._log("✅" if ctx.status == "completed" else "❌",
                      f"Task {ctx.status} in {ctx.elapsed:.1f}s ({ctx.iterations} iterations)")

        return ctx

    def _phase_plan(self, provider: BaseProvider, ctx: TaskContext) -> TaskContext:
        """Phase 1: Planner — decompose task into steps."""
        if self.verbose:
            self._log("📋", "Phase 1: Planning...")

        messages = [
            Message(role="system", content=self.SYSTEM_PROMPT),
            Message(role="user", content=f"""You are the PLANNER. Decompose this task into clear, actionable steps.

TASK: {ctx.task}

Create a numbered plan with specific steps. Be concrete and actionable.
Include what tools might be needed for each step.
Format your plan clearly."""),
        ]

        response = provider.invoke(messages)
        ctx.plan = response.content
        ctx.messages.append({"role": "planner", "content": response.content})
        ctx.iterations += 1

        if self.verbose:
            self._log("📋", f"Plan created ({len(ctx.plan)} chars)")

        return ctx

    def _phase_research(self, provider: BaseProvider, ctx: TaskContext) -> TaskContext:
        """Phase 2: Researcher — gather context."""
        if self.verbose:
            self._log("🔍", "Phase 2: Researching...")

        messages = [
            Message(role="system", content=self.SYSTEM_PROMPT),
            Message(role="user", content=f"""You are the RESEARCHER. Based on the plan, identify what context and information is needed.

TASK: {ctx.task}
PLAN: {ctx.plan}

Analyze what information, dependencies, or context would be needed to execute this plan.
Note any assumptions, requirements, or potential issues."""),
        ]

        response = provider.invoke(messages)
        ctx.research = response.content
        ctx.messages.append({"role": "researcher", "content": response.content})
        ctx.iterations += 1

        if self.verbose:
            self._log("🔍", "Research complete")

        return ctx

    def _phase_tool_selection(self, provider: BaseProvider, ctx: TaskContext) -> TaskContext:
        """Phase 3: Tool Manager — select appropriate tools."""
        if self.verbose:
            self._log("🔧", "Phase 3: Selecting tools...")

        tools_desc = self._get_tools_description()

        messages = [
            Message(role="system", content=self.SYSTEM_PROMPT),
            Message(role="user", content=f"""You are the TOOL MANAGER. Select the right tools for this task.

TASK: {ctx.task}
PLAN: {ctx.plan}

{tools_desc}

List the tools that should be used for each step of the plan.
Format: tool_name: reason for selection"""),
        ]

        response = provider.invoke(messages)
        ctx.messages.append({"role": "tool_manager", "content": response.content})
        ctx.iterations += 1

        if self.verbose:
            self._log("🔧", "Tools selected")

        return ctx

    def _phase_code(self, provider: BaseProvider, ctx: TaskContext) -> TaskContext:
        """Phase 4: Coder — write and execute code."""
        if self.verbose:
            self._log("💻", "Phase 4: Coding...")

        messages = [
            Message(role="system", content=self.SYSTEM_PROMPT + "\n\nYou are the CODER. Write clean, working code to accomplish the task. Output ONLY the code needed, wrapped in ```python blocks."),
            Message(role="user", content=f"""TASK: {ctx.task}

PLAN: {ctx.plan}

RESEARCH: {ctx.research}

Write the complete code to accomplish this task. Make it production-quality.
Use Python. Include error handling. Make it complete and runnable.

Output the code in a ```python code block."""),
        ]

        response = provider.invoke(messages)
        ctx.code = response.content

        # Extract code from markdown code blocks
        import re
        code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response.content, re.DOTALL)
        if code_blocks:
            ctx.code = "\n\n".join(code_blocks)

        ctx.messages.append({"role": "coder", "content": response.content})
        ctx.iterations += 1

        # Execute the code
        if ctx.code and self.config.agent.sandbox_code:
            if self.verbose:
                self._log("⚡", "Executing code...")
            from nexusai.tools.python_tools import PythonTool

            py = PythonTool()
            result = py.execute(ctx.code)
            ctx.code_output = json.dumps(result, default=str)
            ctx.tool_results.append({"tool": "python.execute", "result": result})

            if self.verbose:
                status = "✅" if result.get("success") else "❌"
                self._log(status, f"Code execution: {'success' if result.get('success') else 'failed'}")

        return ctx

    def _phase_review(self, provider: BaseProvider, ctx: TaskContext) -> TaskContext:
        """Phase 5: Reviewer — review code quality."""
        if self.verbose:
            self._log("👁️", "Phase 5: Reviewing...")

        messages = [
            Message(role="system", content=self.SYSTEM_PROMPT),
            Message(role="user", content=f"""You are the REVIEWER. Review this code for quality, correctness, and completeness.

TASK: {ctx.task}
CODE:
```python
{ctx.code[:3000]}
```

EXECUTION RESULT: {ctx.code_output[:1000]}

Review for:
1. Correctness — does it solve the task?
2. Quality — is it clean and well-structured?
3. Completeness — is anything missing?
4. Edge cases — are they handled?

Provide specific feedback."""),
        ]

        response = provider.invoke(messages)
        ctx.review_feedback = response.content
        ctx.messages.append({"role": "reviewer", "content": response.content})
        ctx.iterations += 1

        if self.verbose:
            self._log("👁️", "Review complete")

        return ctx

    def _phase_audit(self, provider: BaseProvider, ctx: TaskContext) -> TaskContext:
        """Phase 6: Auditor — security and safety check."""
        if self.verbose:
            self._log("🛡️", "Phase 6: Auditing...")

        messages = [
            Message(role="system", content=self.SYSTEM_PROMPT),
            Message(role="user", content=f"""You are the AUDITOR. Check this code for security issues and best practices.

CODE:
```python
{ctx.code[:3000]}
```

Check for:
1. Security vulnerabilities (injection, exposure, etc.)
2. Resource leaks
3. Error handling gaps
4. Best practice violations
5. Potential dangerous operations

Provide a brief security audit report."""),
        ]

        response = provider.invoke(messages)
        ctx.audit_report = response.content
        ctx.messages.append({"role": "auditor", "content": response.content})
        ctx.iterations += 1

        if self.verbose:
            self._log("🛡️", "Audit complete")

        return ctx

    def _phase_verify(self, provider: BaseProvider, ctx: TaskContext) -> TaskContext:
        """Phase 7: Verifier — validate output."""
        if self.verbose:
            self._log("✔️", "Phase 7: Verifying...")

        messages = [
            Message(role="system", content=self.SYSTEM_PROMPT),
            Message(role="user", content=f"""You are the VERIFIER. Validate that the task was completed successfully.

TASK: {ctx.task}
PLAN: {ctx.plan[:500]}
CODE EXECUTION RESULT: {ctx.code_output[:1000]}
REVIEW: {ctx.review_feedback[:500]}
AUDIT: {ctx.audit_report[:500]}

Determine:
1. Was the task completed? (YES/NO)
2. Quality score (1-10)
3. Any remaining issues?
4. Final summary

Be concise."""),
        ]

        response = provider.invoke(messages)
        ctx.verification = response.content
        ctx.messages.append({"role": "verifier", "content": response.content})
        ctx.iterations += 1

        if self.verbose:
            self._log("✔️", "Verification complete")

        return ctx

    def chat(self, message: str, history: list[Message] | None = None) -> str:
        """Simple chat mode (no pipeline, just LLM + tools)."""
        provider = self._get_provider()
        messages = [Message(role="system", content=self.SYSTEM_PROMPT)]
        if history:
            messages.extend(history)
        messages.append(Message(role="user", content=message))

        response = provider.invoke(messages)
        return response.content

    def _log(self, emoji: str, message: str):
        """Log with rich formatting."""
        try:
            from rich.console import Console
            console = Console()
            console.print(f"  {emoji}  [bold]{message}[/bold]")
        except ImportError:
            print(f"  {emoji}  {message}")
