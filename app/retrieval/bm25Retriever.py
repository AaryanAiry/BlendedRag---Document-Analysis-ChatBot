# pythonService/app/retrieval/bm25Retriever.py
from rank_bm25 import BM25Okapi
from typing import List, Dict
import re

class BM25Store:
    def __init__(self):
        self.index = {}

    def build(self, doc_id: str, chunks: List[Dict]):
        tokenized = [re.findall(r"[A-Za-z0-9]+", c["text"].lower()) for c in chunks]
        self.index[doc_id] = {
            "bm25": BM25Okapi(tokenized),
            "chunks": chunks,
            "tokens": tokenized
        }

    def query(self, doc_id: str, q: str, top_k: int=20, keywords: List[str]=None) -> List[Dict]:
        store = self.index.get(doc_id)
        if not store:
            return []
        scores = store["bm25"].get_scores(q.split())
        if keywords:
            kwset = set(keywords)
            for i, toks in enumerate(store["tokens"]):
                overlap = len(kwset.intersection(toks))
                scores[i] += 0.1 * overlap
        ranked = sorted(
            [{"chunk": c, "score": float(s)} for c, s in zip(store["chunks"], scores)],
            key=lambda x: x["score"], reverse=True
        )
        return ranked[:top_k]

bm25Store = BM25Store()
