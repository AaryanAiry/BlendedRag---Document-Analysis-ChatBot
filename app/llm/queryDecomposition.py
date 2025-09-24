# app/llm/queryDecomposition.py
import json
from app.llm.llmClient import llmClient

# Keywords indicating a query might need decomposition
COMPLEX_QUERY_KEYWORDS = [
    "and", "combine", "join", "across", "from", "to", 
    "table", "plot", "visualize", "merge", "calculate"
]

def needs_decomposition(query: str) -> bool:
    """Return True if query is complex enough to require decomposition."""
    q = query.lower()
    if any(k in q for k in COMPLEX_QUERY_KEYWORDS):
        return True
    if len(q.split()) > 8:  # optional length heuristic
        return True
    return False

def decompose(query: str, temperature: float = 0.3) -> list[str]:
    """
    Decompose a complex query into multiple sub-queries using LLM.
    
    Returns a list of sub-queries (or [query] if no decomposition is needed).
    """
    if not needs_decomposition(query):
        return [query]

    prompt = f"""
You are a query decomposition assistant.

Take the following complex question and break it down into multiple smaller,
executable sub-queries. Each sub-query should be something that could be run
independently against a knowledge base.

⚠️ Output must be a valid JSON array of strings and nothing else.

Example:
Question: "Fetch sales data from 2020 to 2022"
Output: [
"Fetch sales data for 2020", 
"Fetch sales data for 2021", 
"Fetch sales data for 2022",
"Combine all into a single result"]

Question:
"{query}"
"""
    response = llmClient.generateAnswer(prompt, temperature=temperature)

    # Ensure valid JSON output
    try:
        sub_queries = json.loads(response)
        if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
            return sub_queries
    except Exception:
        pass

    # Fallback: treat whole response as a single query
    return [query]

# Optional: simple test
if __name__ == "__main__":
    test_queries = [
        "How many pages are in the pdf?",
        "Fetch sales data from 2012 to 2016 and create a table with year and sales",
        "Combine revenue and expenses from 2020 to 2022 into one chart"
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        print("Sub-queries:", decompose(q))
