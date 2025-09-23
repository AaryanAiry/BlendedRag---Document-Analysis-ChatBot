# pythonService/app/retrieval/denseRetriever.py
from typing import List, Dict

class DenseRetriever:
    def __init__(self, chroma_client, embedding_fn):
        self.chroma = chroma_client
        self.embed = embedding_fn

    def query(self, collection_name: str, q: str, top_k: int=20) -> List[Dict]:
        col = self.chroma.get_or_create_collection(collection_name)
        res = col.query(query_texts=[q], n_results=top_k,
                        include=["documents","metadatas","distances"])
        out = []
        for i, cid in enumerate(res["ids"][0]):
            out.append({
                "chunk": {
                    "id": cid,
                    "text": res["documents"][0][i],
                    "meta": res["metadatas"][0][i],
                },
                "score": 1.0 - float(res["distances"][0][i]) if "distances" in res else 0.0,
            })
        return out
