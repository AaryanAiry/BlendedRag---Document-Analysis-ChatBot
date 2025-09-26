# app/rag/hybridRagPipeline.py
from typing import List, Dict, Any
from app.utils.logger import getLogger
from app.retrieval.queryRefiner import refine_query_intelligent
from app.retrieval.blendedRetriever import blendedRetriever
from app.retrieval.reranker import reranker
from app.llm.llmClient import llmClient
from app.rag.postProcessor import post_process_answer
from app.rag.answerJudge import answerJudge
from app.storage.documentStore import documentStore
from app.rag.answerRefiner import refine_final_answer, normalize_chunks
from app.llm import sourceCiter

logger = getLogger(__name__)

# Optional iterative retriever
try:
    from app.retrieval.iterativeRetriever import IterativeRetriever
    iterative_available = True
    iterative_retriever = IterativeRetriever(retriever=blendedRetriever)
except Exception:
    iterative_available = False
    iterative_retriever = None


def _build_prompt(query: str, context_chunks: List[Dict], max_context_tokens: int = 800) -> str:
    """
    Builds a prompt using normalized chunks only.
    Assumes each chunk has 'chunk' dict with 'text' and a top-level 'page'.
    """
    accumulated = 0
    parts = []
    for i, c in enumerate(context_chunks, start=1):
        snippet = c["chunk"]["text"][:2000]
        est_tokens = len(snippet) // 4
        if accumulated + est_tokens > max_context_tokens:
            break
        page = c.get("page", "?")
        parts.append(f"[{i}] (page={page})\n{snippet}")
        accumulated += est_tokens

    context_block = "\n\n".join(parts)
    prompt = (
        "Answer the question using ONLY the provided context. If not present, say you don't know.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    return prompt


def run_pipeline(
    doc_id: str,
    user_query: str,
    top_k: int = 5,
    rerank: bool = True,
    judge_threshold: float = 0.7,
    iterative: bool = True,
    debug: bool = False
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"docId": doc_id, "originalQuery": user_query, "finalAnswer": None}
    doc_meta = documentStore.getDocument(doc_id)
    if not doc_meta:
        return {"error": "Document not found", "docId": doc_id}

    # Step 1: Query refinement
    try:
        rq = refine_query_intelligent(user_query)
        refined = rq.get("refinedQuery") or rq.get("variants", [user_query])[0] or user_query
    except Exception as e:
        logger.warning(f"Query refinement failed: {e}")
        refined = user_query
    result["refinedQuery"] = refined

    # Step 2: Retrieve
    retrieve_k = max(top_k * 3, 10)
    retrieved = blendedRetriever.query(doc_id, refined, top_k=retrieve_k)
    if iterative and iterative_available:
        try:
            iterative_docs = iterative_retriever.retrieve(refined, doc_id, collection_name=None, top_k=retrieve_k)
            if iterative_docs:
                retrieved = iterative_docs
        except Exception as e:
            logger.debug(f"Iterative retrieval failed/ignored: {e}")

    # Step 3: Rerank
    try:
        reranked = reranker.rerank(refined, retrieved, top_k=top_k) if rerank and retrieved else retrieved[:top_k]
    except Exception as e:
        logger.exception(f"Reranker failed, falling back: {e}")
        reranked = retrieved[:top_k]

    # Step 4: Normalize chunks
    context_chunks: List[Dict] = normalize_chunks(reranked)
    result["chunksUsed"] = [
        {"id": c['chunk'].get('id'), "page": c.get("page"), "score": c.get("score")}
        for c in context_chunks
    ]

    # Step 5: Build prompt and generate answer
    prompt = _build_prompt(user_query, context_chunks, max_context_tokens=1200)
    try:
        raw_answer = llmClient.generateAnswer(prompt, max_tokens=512, temperature=0.7)
    except Exception as e:
        logger.exception(f"LLM generation failed: {e}")
        return {"error": "LLM generation failed", "details": str(e)}
    result["rawAnswer"] = raw_answer

    # Step 6: Judge answer
    judge = answerJudge.score_answer(user_query, raw_answer, context_chunks)
    result["judge"] = judge

    # Step 7: Refinement loop
    attempts = 0
    max_attempts = 2
    while judge.get("score", 0) < judge_threshold and attempts < max_attempts:
        attempts += 1
        logger.info(f"Low judge score ({judge['score']}). Attempting refinement #{attempts}")
        refine_prompt = (
            prompt
            + "\n\nThe previous answer was low confidence. "
              "Please re-check the context and produce a concise answer focusing only on facts present in the context."
        )
        try:
            raw_answer = llmClient.generateAnswer(refine_prompt, max_tokens=512, temperature=0.7)
            result[f"rawAnswer_attempt_{attempts}"] = raw_answer
            judge = answerJudge.score_answer(user_query, raw_answer, context_chunks)
            result.setdefault("judge_attempts", []).append(judge)
            if judge.get("score", 0) >= judge_threshold:
                break
        except Exception as e:
            logger.debug(f"Refinement attempt failed: {e}")
            break

    # Step 8: Post-process and attach citations
    final_answer = refine_final_answer(raw_answer, user_query, context_chunks)
    final_answer = sourceCiter.sourceCiter.cite_sources(user_query, final_answer, context_chunks)
    result["finalAnswer"] = final_answer
    result["attempts"] = attempts

    citations = [
        {"rank": i, "chunk_id": c["chunk"].get("id"), "page": c.get("page")}
        for i, c in enumerate(context_chunks, start=1)
    ]
    result["citations"] = citations

    if debug:
        result["context_chunks_debug"] = context_chunks

    return result

