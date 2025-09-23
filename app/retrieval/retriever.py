# app/retrieval/retriever.py
# Chroma-based semantic search

from typing import List, Dict, Any
from app.embeddings.embeddingClient import EmbeddingClient
from app.chromaClient import collection  # Shared collection only

# Singleton embedding client
embeddingClient = EmbeddingClient()

def retrieveTopK(docId: str, queryText: str, topK: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve top-K most similar chunks for a given query from Chroma.

    Args:
        docId (str): ID of the document to query.
        queryText (str): User query text.
        topK (int): Number of top chunks to return.

    Returns:
        List[Dict[str, Any]]: List of chunks with 'chunkIndex', 'text', and 'score'.
    """
    # Generate embedding for the query
    queryVec = embeddingClient.generateEmbedding(queryText)

    # Query Chroma collection
    results = collection.query(
        query_embeddings=[queryVec],  # no .tolist()
        n_results=topK,
        where={"docId": docId}
    )

    # Handle empty results
    if not results or len(results.get("metadatas", [])) == 0 or len(results["metadatas"][0]) == 0:
        return []

    # Process results
    top_chunks = []
    for i, metadata in enumerate(results["metadatas"][0]):
        top_chunks.append({
            "chunkIndex": metadata.get("chunkIndex"),
            "text": metadata.get("text", ""),
            "score": float(results["distances"][0][i])
        })

    return top_chunks
