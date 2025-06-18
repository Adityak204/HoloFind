import aiohttp
import asyncio
from typing import List, Dict, Optional
from trafilatura import Trafilatura, extract


SEARCH_ENGINE_API = "https://ddg-api.herokuapp.com/search?q="  # DuckDuckGo lite proxy
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_ENDPOINT = "https://api.tavily.com/search"


async def search_via_tavily(query: str, num_results: int = 5) -> List[str]:
    headers = {"Content-Type": "application/json"}
    body = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "max_results": num_results,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                TAVILY_ENDPOINT, json=body, headers=headers, timeout=10
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Tavily error: {resp.status}")
                data = await resp.json()
                return [item["url"] for item in data.get("results", [])[:num_results]]
    except Exception as e:
        print(f"[search_via_tavily] Failed: {e}")
        raise


async def search_via_duckduckgo(query: str, num_results: int = 5) -> List[str]:
    url = SEARCH_ENGINE_API + query.replace(" ", "+")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                data = await resp.json()
                return [item["href"] for item in data.get("results", [])[:num_results]]
    except Exception as e:
        print(f"[search_via_duckduckgo] Failed for query '{query}': {e}")
        return []


async def search_urls(query: str, num_results: int = 5) -> List[str]:
    try:
        return await search_via_tavily(query, num_results)
    except Exception:
        print("Falling back to DuckDuckGo...")
        return await search_via_duckduckgo(query, num_results)


async def fetch_and_parse_url(url: str) -> Optional[Dict]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=15) as resp:
                if resp.status != 200:
                    raise Exception(f"Non-200 response: {resp.status}")
                html = await resp.text()
                extracted_text = extract(html)
                if not extracted_text:
                    raise Exception("Trafilatura failed to extract content")
                return {
                    "source_url": url,
                    "raw_html": html,
                    "text": extracted_text[:5000],  # Optional truncation
                }
    except Exception as e:
        print(f"[fetch_and_parse_url] Failed for {url}: {e}")
        return None
