"""Web Tools — HTTP requests, web scraping, URL handling."""

from __future__ import annotations
import json
import re
from urllib.parse import urlparse, urljoin
from nexusai.tools.base import BaseTool, tool


class WebTool(BaseTool):
    name = "web"
    description = "HTTP requests, web scraping, URL manipulation, and internet operations"
    category = "web"

    @tool(description="Make an HTTP GET request")
    def get(self, url: str, headers: dict = None, params: dict = None, timeout: int = 30) -> dict:
        import httpx
        resp = httpx.get(url, headers=headers, params=params, timeout=timeout, follow_redirects=True)
        return {"status": resp.status_code, "headers": dict(resp.headers), "body": resp.text[:10000], "url": str(resp.url)}

    @tool(description="Make an HTTP POST request")
    def post(self, url: str, data: dict = None, json_data: dict = None, headers: dict = None, timeout: int = 30) -> dict:
        import httpx
        resp = httpx.post(url, data=data, json=json_data, headers=headers, timeout=timeout, follow_redirects=True)
        return {"status": resp.status_code, "headers": dict(resp.headers), "body": resp.text[:10000]}

    @tool(description="Make an HTTP PUT request")
    def put(self, url: str, json_data: dict = None, headers: dict = None, timeout: int = 30) -> dict:
        import httpx
        resp = httpx.put(url, json=json_data, headers=headers, timeout=timeout)
        return {"status": resp.status_code, "body": resp.text[:10000]}

    @tool(description="Make an HTTP DELETE request")
    def delete_request(self, url: str, headers: dict = None, timeout: int = 30) -> dict:
        import httpx
        resp = httpx.delete(url, headers=headers, timeout=timeout)
        return {"status": resp.status_code, "body": resp.text[:5000]}

    @tool(description="Download a file from URL")
    def download(self, url: str, save_path: str, timeout: int = 120) -> str:
        import httpx
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as resp:
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
        return f"Downloaded to {save_path} ({Path(save_path).stat().st_size} bytes)"

    @tool(description="Extract text from HTML (basic scraping)")
    def scrape_text(self, url: str) -> str:
        import httpx
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)[:10000]
        except ImportError:
            text = re.sub(r"<[^>]+>", " ", resp.text)
            return re.sub(r"\s+", " ", text).strip()[:10000]

    @tool(description="Extract all links from a webpage")
    def extract_links(self, url: str) -> list:
        import httpx
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            links = []
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"])
                links.append({"url": href, "text": a.get_text(strip=True)[:100]})
            return links
        except ImportError:
            return re.findall(r'href=["\']([^"\']+)["\']', resp.text)

    @tool(description="Parse URL components")
    def parse_url(self, url: str) -> dict:
        parsed = urlparse(url)
        return {"scheme": parsed.scheme, "host": parsed.hostname, "port": parsed.port, "path": parsed.path, "query": parsed.query, "fragment": parsed.fragment}

    @tool(description="Check if a URL is reachable")
    def ping_url(self, url: str, timeout: int = 10) -> dict:
        import httpx
        import time
        start = time.time()
        try:
            resp = httpx.head(url, timeout=timeout, follow_redirects=True)
            return {"reachable": True, "status": resp.status_code, "latency_ms": round((time.time() - start) * 1000, 2)}
        except Exception as e:
            return {"reachable": False, "error": str(e), "latency_ms": round((time.time() - start) * 1000, 2)}

    @tool(description="Fetch JSON from URL")
    def get_json(self, url: str, headers: dict = None) -> dict:
        import httpx
        resp = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
        return resp.json()

    @tool(description="URL encode a string")
    def url_encode(self, text: str) -> str:
        from urllib.parse import quote
        return quote(text)

    @tool(description="URL decode a string")
    def url_decode(self, text: str) -> str:
        from urllib.parse import unquote
        return unquote(text)

    @tool(description="Submit a form (POST with form data)")
    def submit_form(self, url: str, form_data: dict, headers: dict = None) -> dict:
        import httpx
        resp = httpx.post(url, data=form_data, headers=headers, timeout=30, follow_redirects=True)
        return {"status": resp.status_code, "body": resp.text[:5000]}
