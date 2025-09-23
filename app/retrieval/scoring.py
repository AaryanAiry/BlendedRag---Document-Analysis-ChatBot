# pythonService/app/retrieval/scoring.py
from collections import defaultdict
from typing import List, Dict

def rrf_fuse(ranklists: List[List[Dict]], k: float = 60.0) -> List[Dict]:
    rrf = defaultdict(float)
    seen = {}
    for rl in ranklists:
        for rank, item in enumerate(rl, start=1):
            cid = item["chunk"]["id"]
            rrf[cid] += 1.0 / (k + rank)
            seen[cid] = item["chunk"]
    fused = [{"chunk": seen[cid], "score": s} for cid, s in rrf.items()]
    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused
