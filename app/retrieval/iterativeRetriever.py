# app/retrieval/iterativeRetriever.py
from app.retrieval.blendedRetriever import BlendedRetriever
from app.rag.queryRefiner import refine_query_intelligent

class IterativeRetriever:
    def __init__(self, retriever: BlendedRetriever, max_rounds: int = 2, threshold: float = 0.3):
        self.retriever = retriever
        self.max_rounds = max_rounds
        self.threshold = threshold

    def retrieve(self, query: str, doc_id: str, collection_name: str, top_k: int = 5):
        round = 0
        retrieved = []
        variants = [query]

        while round < self.max_rounds:
            round += 1
            docs = self.retriever.retrieve(doc_id, collection_name, queries=variants, keywords=[], top_k_final=top_k)
            if self._confidence_ok(docs):
                return docs
            variants = refine_query_intelligent(query)["variants"]  # expand/refine query
        return docs

    def _confidence_ok(self, docs):
        if not docs:
            return False
        # Placeholder: check average score or metadata (can improve later)
        return True
