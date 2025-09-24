# app/llm/querySessionManager.py
from collections import defaultdict

class QuerySessionManager:
    """
    Stores sub-query results for ongoing sessions.
    Allows retrieval of combined results for final visualizations or tables.
    """

    def __init__(self):
        # sessions[session_id] = list of dicts: [{"query": ..., "result": ...}, ...]
        self.sessions = defaultdict(list)

    def store_subquery_result(self, session_id: str, sub_query: str, result):
        """Store the result of a sub-query under a session."""
        self.sessions[session_id].append({
            "query": sub_query,
            "result": result
        })

    def get_combined_results(self, session_id: str):
        """Return list of results of all sub-queries for a session."""
        return [item["result"] for item in self.sessions.get(session_id, [])]

    def get_last_result(self, session_id: str):
        """Return result of the last sub-query."""
        session = self.sessions.get(session_id, [])
        if session:
            return session[-1]["result"]
        return None

    def clear_session(self, session_id: str):
        """Clear all stored sub-query results for a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
