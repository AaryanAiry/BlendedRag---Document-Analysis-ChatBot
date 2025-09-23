# app/retrievers/sparseRetriever.py
import os
import pickle
from rank_bm25 import BM25Okapi
from typing import List, Dict
from app.utils.logger import getLogger

logger = getLogger(__name__)

CACHE_DIR = "pythonService/data/cache/bm25"
os.makedirs(CACHE_DIR, exist_ok=True)

class SparseRetriever:
    def __init__(self):
        self.indices = {}  # in-memory cache {doc_id: BM25Okapi}
        self._cached_chunks = {}  # {doc_id: chunks}
        self._cached_ids = {}     # {doc_id: ids}

    def _get_cache_path(self, doc_id: str) -> str:
        return os.path.join(CACHE_DIR, f"{doc_id}.pkl")


    
    def indexDocument(self, doc_id: str, chunks: List[str], ids: List[str]):
        """
        Build BM25 index for a document and cache it.
        """
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)

        # Store chunks, ids, and BM25 model in cache
        cache_data = {"chunks": chunks, "ids": ids, "bm25": bm25}
        self.indices[doc_id] = bm25
        self._cached_chunks[doc_id] = chunks
        self._cached_ids[doc_id] = ids

        with open(self._get_cache_path(doc_id), "wb") as f:
            pickle.dump(cache_data, f)

        logger.info(f"BM25 index built and cached for document {doc_id}")


    def _load_index(self, doc_id: str):
        if doc_id in self.indices:
            return self.indices[doc_id]

        path = self._get_cache_path(doc_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No BM25 cache found for doc_id={doc_id}")

        with open(path, "rb") as f:
            data = pickle.load(f)
            bm25 = data["bm25"]
            self.indices[doc_id] = bm25
            self._cached_chunks[doc_id] = data["chunks"]
            self._cached_ids[doc_id] = data["ids"]
            return bm25

    def query(self, doc_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top chunks for a query using BM25.
        Returns list of dicts: [{"chunk": str, "score": float, "id": str}, ...]
        """
        bm25 = self._load_index(doc_id)
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

        chunks = self._cached_chunks[doc_id]
        ids = self._cached_ids[doc_id]

        ranked = sorted(
            [{"chunk": c, "score": s, "id": i} for c, s, i in zip(chunks, scores, ids)],
            key=lambda x: x["score"],
            reverse=True
        )

        return ranked[:top_k]


# Singleton instance
sparseRetriever = SparseRetriever()
