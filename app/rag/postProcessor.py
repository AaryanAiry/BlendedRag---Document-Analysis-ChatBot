# app/rag/postProcessor.py
from app.llm.sourceCiter import sourceCiter
from app.rag.answerJudge import answerJudge

def post_process_answer(raw_answer: str, query: str, context_chunks: list) -> str:
    if not raw_answer:
        return "No answer could be generated."

    cleaned = raw_answer.strip()
    # Judge scoring
    result = answerJudge.score_answer(query, cleaned, context_chunks)
    score_str = f"\n\n---\nAnswer Confidence: {result['score']}/100\nReason: {result['reason']}"

    # Cite sources
    final_with_sources = sourceCiter.cite_sources(query, cleaned, context_chunks)
    return final_with_sources + score_str
