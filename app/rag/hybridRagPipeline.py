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

logger = getLogger(__name__)

# Optional iterative retriever - import if available
try:
    from app.retrieval.iterativeRetriever import IterativeRetriever
    iterative_available = True
    # create a simple instance using blendedRetriever
    iterative_retriever = IterativeRetriever(retriever=blendedRetriever)
except Exception:
    iterative_available = False
    iterative_retriever = None


def _build_prompt(query: str, context_chunks: List[Dict], max_context_tokens: int = 800) -> str:
    """
    Simple prompt builder: include numbered chunks up to token limit.
    """
    accumulated = 0
    parts = []
    for i, c in enumerate(context_chunks, start=1):
        text = c.get("chunk", {}).get("text") if isinstance(c.get("chunk"), dict) else c.get("chunk", str(c))
        snippet = (text or "")[:2000]
        est_tokens = len(snippet) // 4
        if accumulated + est_tokens > max_context_tokens:
            break
        parts.append(f"[{i}] (page={c.get('chunk', {}).get('meta', {}).get('page','?') if isinstance(c.get('chunk'), dict) else c.get('page','?')})\n{snippet}")
        accumulated += est_tokens
    context_block = "\n\n".join(parts)
    prompt = (
        f"Answer the question using ONLY the provided context. If not present, say you don't know.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"
    )
    return prompt


def run_pipeline(doc_id: str, user_query: str, top_k: int = 5,
                 rerank: bool = True,
                 judge_threshold: int = 70,
                 iterative: bool = True,
                 debug: bool = False) -> Dict[str, Any]:
    """
    Orchestrate full hybrid RAG.
    Returns dict with keys:
      - finalAnswer, rawAnswer, judge (score+reason), chunksUsed, citations (if any), debug
    """
    result: Dict[str, Any] = {"docId": doc_id, "originalQuery": user_query, "finalAnswer": None}
    # sanity: doc exists?
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

    # Step 2: Retrieve (blended)
    # retrieve more than needed to allow reranker to pick the best
    retrieve_k = max(top_k * 3, 10)
    retrieved = blendedRetriever.query(doc_id, refined, top_k=retrieve_k)
    if debug:
        result["retrieved_raw"] = retrieved

    # Optional Iterative retrieval if enabled and available
    if iterative and iterative_available:
        try:
            iterative_docs = iterative_retriever.retrieve(refined, doc_id, collection_name=None, top_k=retrieve_k)
            if iterative_docs:
                retrieved = iterative_docs  # use iterative results if provided
        except Exception as e:
            logger.debug(f"Iterative retrieval failed/ignored: {e}")

    # Step 3: Rerank (cross-encoder / fallback)
    candidates = retrieved  # candidates format expected by reranker
    if rerank and candidates:
        try:
            reranked = reranker.rerank(refined, candidates, top_k=top_k)
        except Exception as e:
            logger.exception(f"Reranker failed, falling back: {e}")
            reranked = candidates[:top_k]
    else:
        reranked = candidates[:top_k]

    # Normalize final selected chunks into a consistent shape
    context_chunks: List[Dict] = []
    for r in reranked:
        chunk_obj = r.get("chunk") if isinstance(r, dict) else r
        # normalize
        item = {
            "chunk": chunk_obj,
            "score": r.get("rerank_score", r.get("score", 0)),
        }
        # attempt to expose page/id for citations
        if isinstance(chunk_obj, dict):
            # keep meta if present
            item["page"] = chunk_obj.get("meta", {}).get("page") or chunk_obj.get("meta", {}).get("page_num") or chunk_obj.get("meta", {}).get("pageNumber")
        else:
            item["page"] = None
        context_chunks.append(item)

    result["chunksUsed"] = [
        {"id": (c['chunk'].get('id') if isinstance(c['chunk'], dict) else None),
         "page": c.get("page"),
         "score": c.get("score")}
        for c in context_chunks
    ]

    # Step 4: Build prompt from top chunks and generate answer
    prompt = _build_prompt(user_query, context_chunks, max_context_tokens=1200)
    try:
        raw_answer = llmClient.generateAnswer(prompt, max_tokens=512, temperature=0.0)
    except Exception as e:
        logger.exception(f"LLM generation failed: {e}")
        return {"error": "LLM generation failed", "details": str(e)}

    result["rawAnswer"] = raw_answer

    # Step 5: Judge the raw answer
    judge = answerJudge.score_answer(user_query, raw_answer, context_chunks)
    result["judge"] = judge

    # If judge says low confidence, optionally try a simple refinement loop
    attempts = 0
    max_attempts = 2
    while judge.get("score", 0) < judge_threshold and attempts < max_attempts:
        attempts += 1
        logger.info(f"Low judge score ({judge['score']}). Attempting refinement #{attempts}")
        # try slightly different prompt: ask to be concise and cite chunk indices if present
        refine_prompt = prompt + "\n\nThe previous answer was low confidence. Please re-check the context and produce a concise answer focusing only on facts present in the context."
        try:
            raw_answer = llmClient.generateAnswer(refine_prompt, max_tokens=512, temperature=0.0)
            result["rawAnswer_attempt_{}".format(attempts)] = raw_answer
            judge = answerJudge.score_answer(user_query, raw_answer, context_chunks)
            result.setdefault("judge_attempts", []).append(judge)
            if judge.get("score", 0) >= judge_threshold:
                break
        except Exception as e:
            logger.debug(f"Refinement attempt failed: {e}")
            break

    # Step 6: Post-process (cleanup & optional citations inserted by post_process_answer)
    final_answer = post_process_answer(raw_answer, user_query, context_chunks)
    result["finalAnswer"] = final_answer
    result["attempts"] = attempts

    # Optionally attach citations: your post_processor may already add them
    # Provide a compact citations list for the API consumer
    citations = []
    for i, c in enumerate(context_chunks, start=1):
        cid = None
        if isinstance(c["chunk"], dict):
            cid = c["chunk"].get("id")
            page = c["chunk"].get("meta", {}).get("page")
        else:
            page = c.get("page")
        citations.append({"rank": i, "chunk_id": cid, "page": page})
    result["citations"] = citations

    # Debug info
    if debug:
        result["context_chunks_debug"] = context_chunks

    return result

