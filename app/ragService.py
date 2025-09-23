# app/rag/ragService.py
import os
from app.utils.logger import getLogger
from app.storage.documentStore import documentStore
from app.retrieval.queryRefiner import refine_query_intelligent
from app.retrieval.blendedRetriever import blendedRetriever
from app.embeddings.embeddingClient import EmbeddingClient
from app.llm.llmClient import llmClient  # Qwen wrapper
from app.llm.postProcessor import post_process_answer 
from app.routes.queryRoutes import getTopSentences

logger = getLogger(__name__)

# -------------------------------
# Initialize Embedding Client
# -------------------------------
embedding_client = EmbeddingClient()

# -------------------------------
# Helper Functions
# -------------------------------
def build_rag_prompt(query: str, chunks: list, max_context_tokens: int = 300) -> str:
    accumulated_tokens = 0
    context_parts = []
    for chunk in chunks:
        text = chunk.get("text", "")
        est_tokens = len(text) // 4
        if accumulated_tokens + est_tokens > max_context_tokens:
            break
        context_parts.append(text)
        accumulated_tokens += est_tokens

    context = "\n\n".join(context_parts)
    return (
        f"Answer the following question using only the provided context.\n"
        f"If the answer is not directly in the context, give your best summary based on it.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )


def generate_answer(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """
    Generate answer from Qwen model.
    """
    try:
        return llmClient.generateAnswer(prompt, max_tokens=max_tokens, temperature=temperature)
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Error generating answer"

# -------------------------------
# Blended RAG Service
# -------------------------------
def query_document(docId: str, user_query: str, topK: int = 10) -> dict:
    """
    Query a document using Query Refinement + Blended Retriever + Qwen + PostProcessor
    """
    doc = documentStore.getDocument(docId)
    if not doc:
        logger.warning(f"Document not found: {docId}")
        return {"error": "Document not found", "docId": docId}

    # Step 1: Refine Query
    rq = refine_query_intelligent(user_query)

    # Step 2: Blended Retrieval
    retrieved_docs = blendedRetriever.query(
        doc_id=docId,
        query=rq.get("refinedQuery", user_query),
        top_k=topK
    )

    # Extract text chunks
    # top_chunks = [{"text": d.get("chunk")} for d in retrieved_docs[:3] if d.get("chunk")]
    top_chunks =     [
        {"text": getTopSentences(d.get("chunk"), user_query, top_n=2)}
        for d in retrieved_docs[:5] if d.get("chunk")
    ]

    # Step 3: Generate Raw Answer
    prompt = build_rag_prompt(user_query, top_chunks)
    raw_answer = generate_answer(prompt,max_tokens=120)

    # Step 4: Post-process Answer
    final_answer = post_process_answer(
        raw_answer,
        query=user_query,
        context_chunks=top_chunks
    )

    return {
        "docId": docId,
        "originalQuery": user_query,
        "queryRefinement": rq,
        "retrievedChunks": retrieved_docs,
        "rawAnswer": raw_answer,          # Keep for debugging
        "finalAnswer": final_answer       # Use this for production
    }

# -------------------------------
# Optional: Refinement function
# -------------------------------
def refine_answer(query: str, initial_answer: str, top_chunks: list) -> str:
    """
    Refine an initial answer by re-prompting Qwen with context
    """
    context = "\n\n".join([chunk.get("text", "") for chunk in top_chunks if chunk.get("text")])
    prompt = (
        f"Given the following context:\n{context}\n\n"
        f"The previous answer to the question '{query}' was: '{initial_answer}'.\n"
        "Improve or refine this answer if necessary, correcting mistakes and adding missing information.\nAnswer:"
    )
    return generate_answer(prompt)
