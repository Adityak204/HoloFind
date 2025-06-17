INITIAL_SUBQUERY_PROMPT = """
You are an expert research assistant.

Given the following user query:
"{query}"

Generate 3 to 5 diverse and specific sub-questions that can help expand the search space for answering the user's query more comprehensively.

Return the sub-questions as a JSON list of objects like:
[
  {{ "sub_query": "..." }},
  ...
]
Only generate sub-queries that are clear and information-seeking.
"""

REFINE_SUBQUERY_PROMPT = """
You are helping to refine sub-questions based on web context.

User Query:
"{user_query}"

Original Sub-query:
"{sub_query}"

Web Snippets:
{snippets}

Rewrite or improve the sub-query to be more specific and useful for research, using the context if helpful.

Return just one improved sub-query.
"""

FILTER_SUBQUERY_PROMPT = """
You are evaluating the usefulness of a research sub-question.

Original User Query:
"{user_query}"

Sub-query to evaluate:
"{sub_query}"

Should this sub-query be kept for expanding the original query? Answer with either:
- Yes
- No
"""
