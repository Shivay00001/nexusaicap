"""Shell Tools — Execute shell commands safely."""

from __future__ import annotations
import subprocess
import os
import platform
from nexusai.tools.base import BaseTool, tool


class ShellTool(BaseTool):
    name = "shell"
    description = "Execute shell commands, manage processes, and interact with the operating system"
    category = "system"

    @tool(description="Execute a shell command and return output")
    def run(self, command: str, cwd: str = None, timeout: int = 120, shell: bool = True) -> dict:
        try:
            result = subprocess.run(
                command, shell=shell, capture_output=True, text=True,
                cwd=cwd or os.getcwd(), timeout=timeout,
            )
            return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode, "success": result.returncode == 0}
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": f"Command timed out after {timeout}s", "returncode": -1, "success": False}
        except Exception as e:
            return {"stdout": "", "stderr": str(e), "returncode": -1, "success": False}

    @tool(description="Run a command and stream output line by line")
    def run_stream(self, command: str, cwd: str = None) -> list:
        lines = []
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd)
        for line in proc.stdout:
            lines.append(line.rstrip())
        proc.wait()
        return lines

    @tool(description="Get environment variable")
    def get_env(self, name: str, default: str = "") -> str:
        return os.environ.get(name, default)

    @tool(description="Set environment variable")
    def set_env(self, name: str, value: str) -> str:
        os.environ[name] = value
        return f"Set {name}={value}"

    @tool(description="Get system information")
    def system_info(self) -> dict:
        return {
            "os": platform.system(), "os_release": platform.release(), "os_version": platform.version(),
            "architecture": platform.machine(), "processor": platform.processor(),
            "python_version": platform.python_version(), "hostname": platform.node(),
        }

    @tool(description="List running processes")
    def list_processes(self, name_filter: str = None) -> list:
        try:
            import psutil
            procs = []
            for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                info = p.info
                if name_filter and name_filter.lower() not in info["name"].lower():
                    continue
                procs.append(info)
            return procs[:50]
        except ImportError:
            result = subprocess.run("tasklist" if platform.system() == "Windows" else "ps aux", shell=True, capture_output=True, text=True)
            return [{"output": result.stdout[:3000]}]

    @tool(description="Kill a process by PID")
    def kill_process(self, pid: int) -> str:
        try:
            import psutil
            p = psutil.Process(pid)
            p.terminate()
            return f"Terminated process {pid} ({p.name()})"
        except Exception as e:
            return f"Failed to kill process {pid}: {e}"

    @tool(description="Get disk usage")
    def disk_usage(self, path: str = "/") -> dict:
        try:
            import psutil
            usage = psutil.disk_usage(path)
            return {"total_gb": round(usage.total / 1e9, 2), "used_gb": round(usage.used / 1e9, 2), "free_gb": round(usage.free / 1e9, 2), "percent": usage.percent}
        except ImportError:
            import shutil
            total, used, free = shutil.disk_usage(path)
            return {"total_gb": round(total / 1e9, 2), "used_gb": round(used / 1e9, 2), "free_gb": round(free / 1e9, 2)}

    @tool(description="Get memory usage")
    def memory_usage(self) -> dict:
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {"total_gb": round(mem.total / 1e9, 2), "available_gb": round(mem.available / 1e9, 2), "used_gb": round(mem.used / 1e9, 2), "percent": mem.percent}
        except ImportError:
            return {"error": "psutil not installed"}

    @tool(description="Check if a command is available in PATH")
    def which(self, command: str) -> str:
        import shutil
        result = shutil.which(command)
        return result or f"{command} not found in PATH"

    @tool(description="Run a Python script")
    def run_python(self, script_path: str, args: list = None) -> dict:
        import sys
        cmd = [sys.executable, script_path] + (args or [])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}

    @tool(description="Run pip install")
    def pip_install(self, package: str) -> dict:
        import sys
        result = subprocess.run([sys.executable, "-m", "pip", "install", package], capture_output=True, text=True, timeout=300)
        return {"stdout": result.stdout, "stderr": result.stderr, "success": result.returncode == 0}
