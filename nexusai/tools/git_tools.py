"""Git Tools — Git operations via subprocess."""
from __future__ import annotations
import subprocess, os
from nexusai.tools.base import BaseTool, tool

class GitTool(BaseTool):
    name = "git"; description = "Git version control operations"; category = "vcs"

    def _run(self, args: str, cwd: str = None) -> str:
        r = subprocess.run(f"git {args}", shell=True, capture_output=True, text=True, cwd=cwd or os.getcwd())
        if r.returncode != 0 and r.stderr:
            return f"Error: {r.stderr.strip()}"
        return r.stdout.strip()

    @tool(description="Initialize a git repository")
    def init(self, path: str = ".") -> str: return self._run("init", path)
    @tool(description="Clone a repository")
    def clone(self, url: str, dest: str = None) -> str: return self._run(f"clone {url}" + (f" {dest}" if dest else ""))
    @tool(description="Get git status")
    def status(self, cwd: str = None) -> str: return self._run("status --short", cwd)
    @tool(description="Stage files")
    def add(self, files: str = ".", cwd: str = None) -> str: return self._run(f"add {files}", cwd)
    @tool(description="Commit changes")
    def commit(self, message: str, cwd: str = None) -> str: return self._run(f'commit -m "{message}"', cwd)
    @tool(description="Push to remote")
    def push(self, remote: str = "origin", branch: str = "main", cwd: str = None) -> str: return self._run(f"push {remote} {branch}", cwd)
    @tool(description="Pull from remote")
    def pull(self, remote: str = "origin", branch: str = "main", cwd: str = None) -> str: return self._run(f"pull {remote} {branch}", cwd)
    @tool(description="Show git log")
    def log(self, n: int = 10, cwd: str = None) -> str: return self._run(f"log --oneline -n {n}", cwd)
    @tool(description="Show diff")
    def diff(self, cwd: str = None) -> str: return self._run("diff", cwd)
    @tool(description="List branches")
    def branches(self, cwd: str = None) -> str: return self._run("branch -a", cwd)
    @tool(description="Checkout a branch")
    def checkout(self, branch: str, create: bool = False, cwd: str = None) -> str: return self._run(f"checkout {'-b ' if create else ''}{branch}", cwd)
    @tool(description="Stash changes")
    def stash(self, cwd: str = None) -> str: return self._run("stash", cwd)
    @tool(description="Apply stashed changes")
    def stash_pop(self, cwd: str = None) -> str: return self._run("stash pop", cwd)
    @tool(description="Show git blame for a file")
    def blame(self, path: str, cwd: str = None) -> str: return self._run(f"blame {path}", cwd)
    @tool(description="Show git tags")
    def tags(self, cwd: str = None) -> str: return self._run("tag -l", cwd)
    @tool(description="Create a tag")
    def create_tag(self, name: str, message: str = "", cwd: str = None) -> str: return self._run(f'tag -a {name} -m "{message or name}"', cwd)
    @tool(description="Get current branch name")
    def current_branch(self, cwd: str = None) -> str: return self._run("rev-parse --abbrev-ref HEAD", cwd)
    @tool(description="Get remote URL")
    def remote_url(self, cwd: str = None) -> str: return self._run("remote get-url origin", cwd)
