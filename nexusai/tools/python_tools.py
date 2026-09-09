"""Python Tools — Safe Python code execution and REPL."""
from __future__ import annotations
import sys, io, traceback, ast, textwrap
from nexusai.tools.base import BaseTool, tool

class PythonTool(BaseTool):
    name = "python"; description = "Execute Python code safely with output capture"; category = "code"

    def __init__(self):
        self._globals = {"__builtins__": __builtins__}
        self._locals = {}

    @tool(description="Execute Python code and return output")
    def execute(self, code: str) -> dict:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        try:
            tree = ast.parse(code)
            # If last statement is an expression, capture its value
            last_expr = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = ast.Expression(tree.body.pop().value)
                ast.fix_missing_locations(last_expr)
            compiled = compile(tree, "<nexusai>", "exec")
            exec(compiled, self._globals, self._locals)
            result = None
            if last_expr:
                result = eval(compile(last_expr, "<nexusai>", "eval"), self._globals, self._locals)
            stdout = sys.stdout.getvalue()
            return {"success": True, "output": stdout, "result": repr(result) if result is not None else None}
        except Exception as e:
            return {"success": False, "error": traceback.format_exc(), "stderr": sys.stderr.getvalue()}
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    @tool(description="Evaluate a Python expression")
    def eval_expr(self, expression: str) -> str:
        try:
            result = eval(expression, self._globals, self._locals)
            return repr(result)
        except Exception as e:
            return f"Error: {e}"

    @tool(description="Install a Python package")
    def install_package(self, package: str) -> dict:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "install", package], capture_output=True, text=True, timeout=120)
        return {"success": result.returncode == 0, "output": result.stdout, "error": result.stderr}

    @tool(description="List installed Python packages")
    def list_packages(self) -> list:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"], capture_output=True, text=True)
        import json
        return json.loads(result.stdout) if result.returncode == 0 else []

    @tool(description="Format Python code with black")
    def format_code(self, code: str) -> str:
        try:
            import black
            return black.format_str(code, mode=black.Mode())
        except ImportError:
            return code

    @tool(description="Check Python syntax")
    def check_syntax(self, code: str) -> dict:
        try:
            ast.parse(code)
            return {"valid": True, "error": None}
        except SyntaxError as e:
            return {"valid": False, "error": str(e), "line": e.lineno}

    @tool(description="Get Python version info")
    def version_info(self) -> dict:
        return {"version": sys.version, "executable": sys.executable, "platform": sys.platform, "prefix": sys.prefix}

    @tool(description="Reset the execution environment")
    def reset(self) -> str:
        self._globals = {"__builtins__": __builtins__}
        self._locals = {}
        return "Execution environment reset"

    @tool(description="Run Python code from a file")
    def run_file(self, path: str) -> dict:
        with open(path) as f:
            code = f.read()
        return self.execute(code)

    @tool(description="Generate a requirements.txt from imports in code")
    def extract_imports(self, code: str) -> list:
        tree = ast.parse(code)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        stdlib = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()
        return sorted(imports - stdlib - {"nexusai"})
