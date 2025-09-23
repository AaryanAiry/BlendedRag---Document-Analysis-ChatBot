# app/retrievers/blendedRetriever.py
from typing import List, Dict
from app.retrieval.denseRetriever import DenseRetriever
from app.retrieval.sparseRetriever import SparseRetriever
from app.utils.logger import getLogger
from app.chromaClient import chromaClient
from app.embeddings.embeddingClient import EmbeddingClient

logger = getLogger(__name__)

class BlendedRetriever:
    def __init__(self, alpha: float = 0.3):  # Lowered from 0.6 to favor sparse (keyword) matches
        """
        alpha: weight for dense retriever (0.3 = 30% dense, 70% sparse)
        """
        self.alpha = alpha
        embedding_client = EmbeddingClient()
        self.dense = DenseRetriever(
            chroma_client=chromaClient,
            embedding_fn=embedding_client.generateEmbedding
        )
        self.sparse = SparseRetriever()  # Adjust if needs params

    def _normalize(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
        min_s, max_s = min(scores), max(scores)
        if max_s - min_s == 0:
            return [0.5] * len(scores)  # Neutral if all scores are the same
        normalized = [(s - min_s) / (max_s - min_s) for s in scores]
        logger.debug(f"Normalized scores: {normalized}")  # Log normalized scores
        return normalized

    def query(self, doc_id: str, query: str, top_k: int = 10) -> List[Dict]:  # Increased top_k to 10
        """
        Blends dense (semantic) and sparse (keyword) scores.
        Returns ranked chunks.
        """
        logger.info(f"Querying doc_id: {doc_id} with query: {query}, top_k: {top_k}")
        dense_results = self.dense.query(doc_id, query, top_k=top_k)
        sparse_results = self.sparse.query(doc_id, query, top_k=top_k)

        # Extract scores & chunks
        dense_scores = self._normalize([r["score"] for r in dense_results])
        sparse_scores = self._normalize([r["score"] for r in sparse_results])

        combined = {}

        # Merge dense first
        for i, r in enumerate(dense_results):
            chunk_data = r["chunk"]
            cid = chunk_data.get("id") if isinstance(chunk_data, dict) else None
            combined[cid] = {
                "chunk": chunk_data,
                "score": self.alpha * dense_scores[i]
            }
            logger.debug(f"Dense: chunk_id={cid}, score={self.alpha * dense_scores[i]}")

        # Merge sparse (add weighted)
        for i, r in enumerate(sparse_results):
            chunk_data = r.get("chunk")
            if isinstance(chunk_data, dict):
                cid = chunk_data.get("id")
                text = chunk_data.get("text", "")
            else:  # it's a string
                cid = None
                text = chunk_data
            if cid in combined:
                combined[cid]["score"] += (1 - self.alpha) * sparse_scores[i]
            else:
                combined[cid] = {
                    "chunk": chunk_data,
                    "score": (1 - self.alpha) * sparse_scores[i]
                }
            logger.debug(f"Sparse: chunk_id={cid}, score={(1 - self.alpha) * sparse_scores[i]}")

        ranked = sorted(
            combined.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        logger.info(f"Ranked chunks: {[r['score'] for r in ranked]}")  # Log final scores

        return ranked[:top_k]

# Singleton instance
blendedRetriever = BlendedRetriever()