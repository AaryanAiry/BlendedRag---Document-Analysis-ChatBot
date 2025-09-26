# app/retrieval/blendedRetriever.py
from typing import List, Dict, Optional
from app.retrieval.denseRetriever import DenseRetriever
from app.retrieval.sparseRetriever import SparseRetriever
from app.utils.logger import getLogger
from app.chromaClient import chromaClient
from app.retrieval.reranker import reranker
from app.embeddings.embeddingClient import EmbeddingClient
import hashlib
import re
from collections import Counter

logger = getLogger(__name__)

class BlendedRetriever:
    def __init__(self, alpha: float = 0.3, diversity_penalty: float = 0.12):
        """
        alpha: weight for dense retriever (0.3 = 30% dense, 70% sparse)
        diversity_penalty: penalty applied per extra chunk from the same page (tunable)
        """
        self.alpha = alpha
        self.diversity_penalty = diversity_penalty
        embedding_client = EmbeddingClient()
        self.dense = DenseRetriever(
            chroma_client=chromaClient,
            embedding_fn=embedding_client.generateEmbedding
        )
        self.sparse = SparseRetriever()

    def _joint_normalize(self, dense_scores: List[float], sparse_scores: List[float]):
        """Normalize dense + sparse scores together instead of separately."""
        all_scores = (dense_scores or []) + (sparse_scores or [])
        if not all_scores:
            return [], []
        min_s, max_s = min(all_scores), max(all_scores)
        if max_s - min_s == 0:
            return [0.5] * len(dense_scores), [0.5] * len(sparse_scores)

        def scale(x): return (x - min_s) / (max_s - min_s)
        return [scale(s) for s in dense_scores], [scale(s) for s in sparse_scores]

    def _generate_key(self, chunk_data) -> str:
        """Generate stable ID for chunks using provided id or hash of text."""
        if isinstance(chunk_data, dict):
            cid = chunk_data.get("id")
            if cid:
                return str(cid)
            text = chunk_data.get("text", "")
        else:
            text = str(chunk_data)
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _get_page_for_chunk(self, chunk_data) -> Optional[int]:
        """
        Try multiple strategies to extract page number from chunk metadata or chunk id.
        """
        if not chunk_data:
            return None
        # if chunk_data is dict with meta
        if isinstance(chunk_data, dict):
            meta = chunk_data.get("meta") or chunk_data.get("metadata") or {}
            if isinstance(meta, dict) and "page" in meta:
                try:
                    return int(meta["page"])
                except Exception:
                    pass
            # also support chunk_data with "id" like docId_page{n}_chunk{i}
            cid = chunk_data.get("id")
            if isinstance(cid, str):
                m = re.search(r"_page(\d+)_", cid)
                if m:
                    return int(m.group(1))
        # chunk_data might be a plain string id
        if isinstance(chunk_data, str):
            m = re.search(r"_page(\d+)_", chunk_data)
            if m:
                return int(m.group(1))
        return None

    def _apply_diversity_penalty(self, ranked_list: List[Dict]) -> List[Dict]:
        """
        If many chunks come from the same page, apply a small penalty to later ones.
        This promotes diversity across pages/sections.
        """
        pages = [self._get_page_for_chunk(item.get("chunk")) for item in ranked_list]
        counts = Counter(pages)
        # if everything is None or only single page, skip penalty
        if len([p for p in pages if p is not None]) <= 1:
            return ranked_list

        # Walk through list and apply penalty grows with how many times that page has appeared so far
        seen = Counter()
        for item in ranked_list:
            page = self._get_page_for_chunk(item.get("chunk"))
            if page is None:
                continue
            seen[page] += 1
            # apply penalty for second+ appearance
            if seen[page] > 1:
                penalty = self.diversity_penalty * (seen[page] - 1)
                item["score"] = item.get("score", 0.0) - penalty
                logger.debug(f"Applied diversity penalty to page {page}: -{penalty} (now {item['score']})")
        return ranked_list

    def query(self, doc_id: str, query: str, top_k: int = 10, rerank: bool = True) -> List[Dict]:
        logger.info(f"Querying doc_id: {doc_id} with query: {query}, top_k: {top_k}")

        # Get dense and sparse results (each entry: {"chunk": {...}, "score": float})
        dense_results = self.dense.query(doc_id, query, top_k=top_k)
        sparse_results = self.sparse.query(doc_id, query, top_k=top_k)

        # Extract scores (guard for empty)
        dense_scores = [float(r.get("score", 0.0)) for r in dense_results] if dense_results else []
        sparse_scores = [float(r.get("score", 0.0)) for r in sparse_results] if sparse_results else []

        # Joint normalization to keep them comparable
        dense_scores, sparse_scores = self._joint_normalize(dense_scores, sparse_scores)

        combined = {}

        # Merge dense results
        for i, r in enumerate(dense_results or []):
            chunk_data = r.get("chunk") if isinstance(r, dict) else r
            key = self._generate_key(chunk_data)
            score = self.alpha * (dense_scores[i] if i < len(dense_scores) else 0.0)
            if key in combined:
                combined[key]["score"] += score
            else:
                combined[key] = {"chunk": chunk_data, "score": score}
            logger.debug(f"Dense merged: key={key}, add_score={score}")

        # Merge sparse results
        for i, r in enumerate(sparse_results or []):
            # sparse returns {"chunk": text-or-dict, "score": s, "id": maybe}
            chunk_data = r.get("chunk") if isinstance(r, dict) else r
            key = self._generate_key(chunk_data)
            score = (1 - self.alpha) * (sparse_scores[i] if i < len(sparse_scores) else 0.0)
            if key in combined:
                combined[key]["score"] += score
            else:
                combined[key] = {"chunk": chunk_data, "score": score}
            logger.debug(f"Sparse merged: key={key}, add_score={score}")

        ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)

        # Debug: log top-5 before diversity/rerank
        logger.info("Top-5 candidates before penalty/rerank:")
        for idx, c in enumerate(ranked[:5]):
            chunk = c.get("chunk")
            page = self._get_page_for_chunk(chunk)
            text_snippet = (chunk.get("text")[:120] if isinstance(chunk, dict) else str(chunk)[:120])
            logger.info(f"  #{idx+1}: score={c['score']:.4f} page={page} snippet={text_snippet!r}")

        # Apply diversity penalty
        ranked = self._apply_diversity_penalty(ranked)

        # Re-sort after penalty
        ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)

        # Optionally rerank with cross-encoder (if configured)
        if rerank and reranker is not None:
            try:
                reranked = reranker.rerank(query, ranked, top_k=top_k)
                logger.info("Top-5 after rerank:")
                for idx, c in enumerate(reranked[:5]):
                    logger.info(f"  #{idx+1}: rerank_score={c.get('rerank_score')} blended_score={c.get('score'):.4f}")
                return reranked
            except Exception as e:
                logger.error(f"Reranker failed: {e}. Falling back to blended ranks.")

        # return top_k
        return ranked[:top_k]


# Singleton instance
blendedRetriever = BlendedRetriever()

