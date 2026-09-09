"""Search Tools — Perplexity-like web search capabilities."""
from __future__ import annotations
import urllib.parse
from nexusai.tools.base import BaseTool, tool

class SearchTool(BaseTool):
    name = "search"; description = "Advanced web search and intelligence gathering like Perplexity"; category = "web"

    @tool(description="Search the web and return summarized answers with citations")
    def web_search(self, query: str, num_results: int = 5) -> dict:
        """
        Performs a search using DuckDuckGo HTML (no API key required) and 
        returns parsed search results.
        """
        import httpx
        from bs4 import BeautifulSoup
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        try:
            response = httpx.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            
            for a in soup.find_all('a', class_='result__snippet'):
                link = a.get('href')
                # DDG routes URLs through their own redirect
                if link and link.startswith('//duckduckgo.com/l/?'):
                    parsed_link = urllib.parse.parse_qs(urllib.parse.urlparse(link).query).get('uddg', [''])[0]
                    if parsed_link:
                        results.append({
                            "url": parsed_link,
                            "snippet": a.text.strip()
                        })
                if len(results) >= num_results:
                    break
                    
            return {
                "query": query,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}
            
    @tool(description="Deep dive into a specific URL to extract main content")
    def read_webpage(self, url: str) -> str:
        """Extract readable text content from a webpage."""
        import httpx
        try:
            from bs4 import BeautifulSoup
            headers = {"User-Agent": "Mozilla/5.0 (compatible; NexusAIBot/1.0)"}
            resp = httpx.get(url, headers=headers, timeout=20, follow_redirects=True)
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Remove junk
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
                
            text = soup.get_text(separator="\n\n", strip=True)
            # Limit to a reasonable size
            return text[:15000]
        except Exception as e:
            return f"Failed to read webpage: {str(e)}"
            
    @tool(description="Synthesize an answer using web search (Perplexity style)")
    def answer_with_search(self, query: str) -> str:
        """Searches the web, reads top pages, and synthesizes an answer."""
        search_results = self.web_search(query, num_results=3)
        if "error" in search_results:
            return search_results["error"]
            
        context = []
        for i, res in enumerate(search_results.get("results", [])):
            page_content = self.read_webpage(res["url"])
            context.append(f"Source [{i+1}] ({res['url']}):\nSnippet: {res['snippet']}\nContent: {page_content[:2000]}...")
            
        return "\n\n---\n\n".join(context)
