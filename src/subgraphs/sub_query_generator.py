from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import JsonOutputParser
from langchain.chains import LLMChain
from langchain.tools.tavily_search import TavilySearchResults
from ..prompts.sub_query_templates import (
    INITIAL_SUBQUERY_PROMPT,
    REFINE_SUBQUERY_PROMPT,
    FILTER_SUBQUERY_PROMPT,
)

from typing import TypedDict, List
import asyncio


class SubQueryState(TypedDict):
    user_query: str
    sub_queries: List[str]
    refined_sub_queries: List[str]
    final_sub_queries: List[str]
    snippets: dict


llm = ChatOpenAI(model="gpt-4", temperature=0)
parser = JsonOutputParser()

initial_subquery_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(INITIAL_SUBQUERY_PROMPT),
    output_parser=parser,
)


async def generate_initial_subqueries(state: SubQueryState) -> SubQueryState:
    result = await initial_subquery_chain.arun(query=state["user_query"])
    sub_queries = [x["sub_query"] for x in result]
    return {**state, "sub_queries": sub_queries}


tavily_tool = TavilySearchResults(k=3)


async def fetch_web_snippets(state: SubQueryState) -> SubQueryState:
    snippets = {}

    async def fetch(query):
        result = await tavily_tool.arun(query)
        return result

    tasks = [fetch(q) for q in state["sub_queries"]]
    results = await asyncio.gather(*tasks)
    for q, res in zip(state["sub_queries"], results):
        snippets[q] = res
    return {**state, "snippets": snippets}


refine_prompt = PromptTemplate.from_template(REFINE_SUBQUERY_PROMPT)

refiner_chain = LLMChain(
    llm=llm,
    prompt=refine_prompt,
)


async def refine_sub_queries(state: SubQueryState) -> SubQueryState:
    user_query = state["user_query"]
    snippets = state["snippets"]

    async def refine_one(sub_query):
        snippet_text = "\n".join(s["content"] for s in snippets.get(sub_query, []))
        return await refiner_chain.arun(
            user_query=user_query,
            sub_query=sub_query,
            snippets=snippet_text,
        )

    tasks = [refine_one(q) for q in state["sub_queries"]]
    refined = await asyncio.gather(*tasks)
    return {**state, "refined_sub_queries": refined}


filter_prompt = PromptTemplate.from_template(FILTER_SUBQUERY_PROMPT)

filter_chain = LLMChain(
    llm=llm,
    prompt=filter_prompt,
)


async def filter_relevant_subqueries(state: SubQueryState) -> SubQueryState:
    final_queries = []
    for q in state["refined_sub_queries"]:
        result = await filter_chain.arun(user_query=state["user_query"], sub_query=q)
        if "yes" in result.lower():
            final_queries.append(q)
    return {**state, "final_sub_queries": final_queries}


def build_subquery_subgraph() -> StateGraph:
    builder = StateGraph(SubQueryState)

    builder.add_node("generate_initial", generate_initial_subqueries)
    builder.add_node("fetch_snippets", fetch_web_snippets)
    builder.add_node("refine_subqueries", refine_sub_queries)
    builder.add_node("filter_subqueries", filter_relevant_subqueries)

    builder.set_entry_point("generate_initial")
    builder.add_edge("generate_initial", "fetch_snippets")
    builder.add_edge("fetch_snippets", "refine_subqueries")
    builder.add_edge("refine_subqueries", "filter_subqueries")
    builder.add_edge("filter_subqueries", END)

    return builder.compile()


if __name__ == "__main__":
    graph = build_subquery_subgraph()
    output = await graph.ainvoke(
        {"user_query": "Why are tensions rising between Israel and Iran in 2025?"}
    )
    print(output["final_sub_queries"])
