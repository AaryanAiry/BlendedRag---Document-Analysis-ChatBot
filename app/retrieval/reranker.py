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
from scipy.special import expit  # sigmoid for normalizing CrossEncoder scores

class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        normalize_scores: bool = True,  # optional: map scores to 0..1
    ):
        self.model_name = model_name
        self.model = None
        self.normalize_scores = normalize_scores

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
        Returns candidates with added 'rerank_score' sorted descending.
        """
        if not candidates:
            return []

        # Prepare texts and valid candidates
        texts = []
        valid_candidates = []
        for c in candidates:
            chunk = c.get("chunk")
            text = chunk.get("text") if isinstance(chunk, dict) else str(chunk)
            texts.append(text)
            valid_candidates.append(c)

        try:
            if self.model is not None:
                # CrossEncoder scoring
                pairs = [(query, t) for t in texts]
                scores = self.model.predict(pairs)
                if self.normalize_scores:
                    scores = [float(expit(s)) for s in scores]  # map to 0..1
                for cand, s in zip(valid_candidates, scores):
                    cand["rerank_score"] = float(s)
                    logger.debug(f"Candidate {cand['chunk'].get('id')}: rerank_score={s}")
                reranked = sorted(valid_candidates, key=lambda x: x["rerank_score"], reverse=True)
                logger.info("Reranker: used CrossEncoder")
                return reranked[:top_k]
            else:
                # Embedding fallback
                emb_q = self.embedder.generateEmbedding(query)
                emb_docs = self.embedder.generateEmbeddings(texts)
                sims = []
                qv = np.array(emb_q, dtype=float)
                qnorm = norm(qv) + 1e-12
                for v in emb_docs:
                    vv = np.array(v, dtype=float)
                    sim = float(np.dot(qv, vv) / (qnorm * (norm(vv) + 1e-12)))
                    if self.normalize_scores:
                        sim = 0.5 + 0.5 * sim  # map cosine from [-1,1] to [0,1]
                    sims.append(sim)
                for cand, s in zip(valid_candidates, sims):
                    cand["rerank_score"] = float(s)
                    logger.debug(f"Candidate {cand['chunk'].get('id')}: rerank_score={s}")
                reranked = sorted(valid_candidates, key=lambda x: x["rerank_score"], reverse=True)
                logger.info("Reranker: used embedding fallback")
                return reranked[:top_k]
        except Exception as e:
            logger.exception(f"Reranker failed: {e}")
            # fallback: return original order
            return valid_candidates[:top_k]

# singleton instance
reranker = Reranker()

