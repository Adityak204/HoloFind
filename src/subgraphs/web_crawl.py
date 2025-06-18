from typing import List, Dict, Optional, TypedDict
import asyncio
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

from utils.search_utils import search_urls
from utils.web_scraper import fetch_and_parse_url
from utils.dedup import is_url_already_indexed, register_indexed_url
from utils.chunk_embed_store import chunk_embed_store_documents

# -----------------------------
# Shared in-memory URL cache
# -----------------------------
indexed_url_cache = set()


# -----------------------------
# State Definition
# -----------------------------
class WebCrawlState(TypedDict):
    sub_queries: List[str]
    candidate_urls: List[str]
    crawled_docs: List[Dict[str, any]]


# -----------------------------
# Step 1: Web Search
# -----------------------------
async def search_step(state: WebCrawlState) -> WebCrawlState:
    sub_queries = state["sub_queries"]
    query_to_urls = await asyncio.gather(*(search_urls(query) for query in sub_queries))
    all_urls = [url for sublist in query_to_urls for url in sublist]
    return {"sub_queries": sub_queries, "candidate_urls": all_urls, "crawled_docs": []}


# -----------------------------
# Step 2: Crawl & Parse with Retry/Fallback
# -----------------------------
async def crawl_step(state: WebCrawlState) -> WebCrawlState:
    urls = state["candidate_urls"]
    docs: List[Dict] = []

    async def crawl_single(url: str) -> Optional[Dict]:
        result = await fetch_and_parse_url(url)
        return result

    results = await asyncio.gather(*(crawl_single(url) for url in urls))
    for res in results:
        if res:
            docs.append(res)

    return {
        "sub_queries": state["sub_queries"],
        "candidate_urls": urls,
        "crawled_docs": docs,
    }


# -----------------------------
# Step 3: Deduplication Check
# -----------------------------
def dedup_step(state: WebCrawlState) -> WebCrawlState:
    filtered_docs = []
    for doc in state["crawled_docs"]:
        url = doc.get("source_url")
        if not is_url_already_indexed(url):
            register_indexed_url(url)
            filtered_docs.append(doc)
    return {
        "sub_queries": state["sub_queries"],
        "candidate_urls": state["candidate_urls"],
        "crawled_docs": filtered_docs,
    }


# -----------------------------
# Step 4: Chunk + Embed + Store
# -----------------------------
async def store_step(state: WebCrawlState) -> WebCrawlState:
    await chunk_embed_store_documents(state["crawled_docs"])
    return state


# -----------------------------
# Define Web Crawl Subgraph
# -----------------------------
web_crawl_builder = StateGraph(WebCrawlState)

web_crawl_builder.add_node("search", RunnableLambda(search_step))
web_crawl_builder.add_node("crawl", RunnableLambda(crawl_step))
web_crawl_builder.add_node("dedup", RunnableLambda(dedup_step))
web_crawl_builder.add_node("store", RunnableLambda(store_step))

web_crawl_builder.set_entry_point("search")
web_crawl_builder.add_edge("search", "crawl")
web_crawl_builder.add_edge("crawl", "dedup")
web_crawl_builder.add_edge("dedup", "store")
web_crawl_builder.add_edge("store", END)

web_crawl_subgraph = web_crawl_builder.compile()

# Example invocation:
# output = await web_crawl_subgraph.ainvoke({"sub_queries": ["Iran oil sanctions", "Israel cyber attacks"]})
