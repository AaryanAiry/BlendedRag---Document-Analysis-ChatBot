# app/retrieval/reranker.py
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except Exception:
    CROSS_ENCODER_AVAILABLE = False

from app.embeddings.embeddingClient import EmbeddingClient
import numpy as np
from numpy.linalg import norm

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        if CROSS_ENCODER_AVAILABLE:
            try:
                self.model = CrossEncoder(self.model_name)
                logger.info(f"CrossEncoder loaded: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load CrossEncoder '{self.model_name}': {e}")
                self.model = None

        # embedding fallback
        self.embedder = EmbeddingClient()

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        candidates: list of {"chunk": {"id":..., "text":..., "meta":...}, "score": ...}
        Returns candidates with added 'rerank_score' sorted desc.
        """
        if not candidates:
            return []

        texts = []
        valid_candidates = []
        for c in candidates:
            chunk = c.get("chunk")
            text = chunk.get("text") if isinstance(chunk, dict) else str(chunk)
            texts.append(text)
            valid_candidates.append(c)

        if self.model is not None:
            pairs = [(query, t) for t in texts]
            scores = self.model.predict(pairs)
            for cand, s in zip(valid_candidates, scores):
                cand["rerank_score"] = float(s)
            reranked = sorted(valid_candidates, key=lambda x: x["rerank_score"], reverse=True)
            logger.info("Reranker: used CrossEncoder")
            return reranked[:top_k]
        else:
            # fallback: use embedding cosine similarity
            try:
                emb_q = self.embedder.generateEmbedding(query)
                emb_docs = self.embedder.generateEmbeddings(texts)
                # compute cosine
                sims = []
                qv = np.array(emb_q, dtype=float)
                qnorm = norm(qv) + 1e-12
                for v in emb_docs:
                    vv = np.array(v, dtype=float)
                    sims.append(float(np.dot(qv, vv) / (qnorm * (norm(vv) + 1e-12))))
                for cand, s in zip(valid_candidates, sims):
                    cand["rerank_score"] = float(s)
                reranked = sorted(valid_candidates, key=lambda x: x["rerank_score"], reverse=True)
                logger.info("Reranker: used embedding fallback")
                return reranked[:top_k]
            except Exception as e:
                logger.exception(f"Reranker fallback failed: {e}")
                # last resort: return original order
                return valid_candidates[:top_k]


# singleton
reranker = Reranker()
