# app/llm/queryExecutor.py
from app.llm.queryDecomposition import decompose
from app.llm.querySessionManager import QuerySessionManager
from app.rag.ragPipeline import execute_rag_query  # placeholder, your RAG execution function

session_manager = QuerySessionManager()

def execute_query(session_id: str, query: str):
    """
    Execute a user query:
    - Decompose if needed
    - Run RAG for each sub-query
    - Store sub-query results
    Returns the result of the last sub-query.
    """
    sub_queries = decompose(query)
    final_result = None

    for sub_query in sub_queries:
        result = execute_rag_query(sub_query)  # your existing RAG pipeline function
        session_manager.store_subquery_result(session_id, sub_query, result)
        final_result = result

    return final_result
