# app/rag/answerRefiner.py

import re
from app.rag.postProcessor import post_process_answer

def normalize_chunks(context_chunks: list) -> list:
    """
    Ensure every chunk is a dict with 'chunk' containing a dict with 'text' and 'meta',
    and a top-level 'page' field.
    """
    normalized = []
    for i, c in enumerate(context_chunks, start=1):
        if isinstance(c, dict):
            chunk_obj = c.get("chunk")
            score = c.get("score", 0.0)
            if isinstance(chunk_obj, dict):
                meta = chunk_obj.get("meta", {})
                page = meta.get("page") or meta.get("page_num") or meta.get("pageNumber")
            else:
                chunk_obj = {"text": str(chunk_obj or ""), "meta": {}}
                page = None
        else:
            chunk_obj = {"text": str(c), "meta": {}}
            score = 0.0
            page = None
        normalized.append({"chunk": chunk_obj, "score": score, "page": page})
    return normalized


def refine_final_answer(raw_answer: str, query: str, context_chunks: list) -> str:
    """
    Refine LLM output before sending to frontend:
    - Remove AI instructions/boilerplate
    - Deduplicate repeated sentences
    - Clean excessive whitespace
    - Optionally truncate overly long outputs
    - Normalize context_chunks internally
    """

    if not raw_answer:
        return "No answer could be generated."

    # Step 0: Normalize context chunks
    context_chunks = normalize_chunks(context_chunks)

    # Step 1: Remove AI instructions
    cleaned = re.sub(
        r"(You are an AI assistant.*?for this task\.|Do not use information from previous questions and answers.*?task\.)",
        "",
        raw_answer,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Step 2: Deduplicate repeated sentences
    sentences = re.split(r'(?<=[.!?])\s+', cleaned.strip())
    seen = set()
    unique_sentences = []
    for s in sentences:
        s_clean = s.strip()
        if s_clean and s_clean not in seen:
            seen.add(s_clean)
            unique_sentences.append(s_clean)
    cleaned = " ".join(unique_sentences)

    # Step 3: Remove excessive whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Step 4: Optionally truncate if too long
    max_chars = 3000
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "…"

    # Step 5: Pass through postProcessor for scoring & citations
    final_answer = post_process_answer(cleaned, query, context_chunks)

    return final_answer
