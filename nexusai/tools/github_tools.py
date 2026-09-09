"""GitHub Tools — GitHub API operations."""
from __future__ import annotations
import os, json
from nexusai.tools.base import BaseTool, tool

class GitHubTool(BaseTool):
    name = "github"; description = "GitHub API operations — repos, issues, PRs, gists"; category = "vcs"

    def _api(self, endpoint: str, method: str = "GET", data: dict = None) -> dict:
        import httpx
        token = os.environ.get("GITHUB_TOKEN", "")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            headers["Authorization"] = f"token {token}"
        url = f"https://api.github.com{endpoint}"
        resp = httpx.request(method, url, headers=headers, json=data, timeout=30)
        return resp.json() if resp.status_code < 400 else {"error": resp.text, "status": resp.status_code}

    @tool(description="Get repository info")
    def repo_info(self, owner: str, repo: str) -> dict: return self._api(f"/repos/{owner}/{repo}")
    @tool(description="List repository issues")
    def list_issues(self, owner: str, repo: str, state: str = "open") -> list: return self._api(f"/repos/{owner}/{repo}/issues?state={state}")
    @tool(description="Create an issue")
    def create_issue(self, owner: str, repo: str, title: str, body: str = "") -> dict: return self._api(f"/repos/{owner}/{repo}/issues", "POST", {"title": title, "body": body})
    @tool(description="List pull requests")
    def list_prs(self, owner: str, repo: str, state: str = "open") -> list: return self._api(f"/repos/{owner}/{repo}/pulls?state={state}")
    @tool(description="Get user info")
    def user_info(self, username: str) -> dict: return self._api(f"/users/{username}")
    @tool(description="Search repositories")
    def search_repos(self, query: str, sort: str = "stars") -> dict: return self._api(f"/search/repositories?q={query}&sort={sort}")
    @tool(description="List repository contents")
    def list_contents(self, owner: str, repo: str, path: str = "") -> list: return self._api(f"/repos/{owner}/{repo}/contents/{path}")
    @tool(description="Create a gist")
    def create_gist(self, filename: str, content: str, description: str = "", public: bool = True) -> dict: return self._api("/gists", "POST", {"description": description, "public": public, "files": {filename: {"content": content}}})
    @tool(description="Star a repository")
    def star_repo(self, owner: str, repo: str) -> dict: return self._api(f"/user/starred/{owner}/{repo}", "PUT")
    @tool(description="List repository releases")
    def list_releases(self, owner: str, repo: str) -> list: return self._api(f"/repos/{owner}/{repo}/releases")
    @tool(description="List user repositories")
    def list_user_repos(self, username: str) -> list: return self._api(f"/users/{username}/repos?sort=updated")
