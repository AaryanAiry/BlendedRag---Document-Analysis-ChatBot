from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import re
from app.retrieval.queryRefiner import refine_query_intelligent
from app.embeddings.embeddingClient import EmbeddingClient
from app.storage.documentStore import documentStore
from app.utils.logger import getLogger
from app.chromaClient import chromaClient, collection  # Shared Chroma client & collection

router = APIRouter()
logger = getLogger(__name__)
embedding_client = EmbeddingClient()

# --- Models ---
class QueryRequest(BaseModel):
    docId: str
    query: str
    topK: int = 5
    refine: bool = True

class RetrievedChunk(BaseModel):
    chunkIndex: int
    text: str
    score: float
    snippet: str = ""

class QueryResponse(BaseModel):
    docId: str
    query: str
    refinedQueries: List[str]
    results: List[RetrievedChunk]
    mergedBlocks: List[str]

# --- Helper functions ---
def mergeTopChunks(chunks: list[dict], maxTokens: int = 500):
    merged = []
    current_block = ""
    current_tokens = 0
    for c in chunks:
        tokens = len(c["text"].split())
        if current_tokens + tokens > maxTokens:
            if current_block:
                merged.append(current_block)
            current_block = c["text"]
            current_tokens = tokens
        else:
            current_block += " " + c["text"]
            current_tokens += tokens
    if current_block:
        merged.append(current_block)
    return merged

def getTopSentences(text: str, query: str, top_n: int = 3):
    sentences = re.split(r'(?<=[.!?]) +', text)
    query_terms = set(query.lower().split())
    scores = [(len(set(s.lower().split()) & query_terms), s) for s in sentences]
    scores.sort(reverse=True)
    return " ".join([s for _, s in scores[:top_n]])

def chromaRetrieveTopK(doc_id: str, query: str, topK: int = 5):
    """Perform similarity search using ChromaDB for a specific document."""
    query_embedding = embedding_client.generateEmbedding(query)
    query_embedding_2d = query_embedding.reshape(1, -1).tolist()
    results = collection.query(
        query_embeddings=query_embedding_2d,
        where={"docId": doc_id},
        n_results=topK
    )

    chunks = []
    if results and len(results.get("documents", [])) > 0:
        for i, text in enumerate(results["documents"][0]):
            chunks.append({
                "chunkIndex": results["metadatas"][0][i]["chunkIndex"],
                "text": text,
                "score": float(results["distances"][0][i])
            })
    return chunks

# --- API Endpoint ---
@router.post("/api/query", response_model=QueryResponse)
def queryEndpoint(req: QueryRequest):
    doc = documentStore.getDocument(req.docId)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if req.refine:
        rq = refine_query_intelligent(req.query)
        refinedQueries = rq["variants"]
    else:
        refinedQueries = [req.query]

    resultsList = [chromaRetrieveTopK(req.docId, q, topK=req.topK) for q in refinedQueries]
    fusedChunks = resultsList[0] if resultsList else []

    for chunk in fusedChunks:
        chunk["snippet"] = getTopSentences(chunk["text"], req.query, top_n=3)

    mergedBlocks = mergeTopChunks(fusedChunks, maxTokens=500)

    logger.info(f"Query for docId={req.docId} returned {len(fusedChunks)} chunks and {len(mergedBlocks)} merged blocks")

    return QueryResponse(
        docId=req.docId,
        query=req.query,
        refinedQueries=refinedQueries,
        results=[
            RetrievedChunk(
                chunkIndex=item["chunkIndex"],
                text=item["text"],
                score=item["score"],
                snippet=item["snippet"]
            )
            for item in fusedChunks
        ],
        mergedBlocks=mergedBlocks
    )