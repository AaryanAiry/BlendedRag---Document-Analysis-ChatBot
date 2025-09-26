from app.llm.sourceCiter import sourceCiter

def post_process_answer(raw_answer: str, query: str, context_chunks: list) -> str:
    """
    Cleans, scores, and optionally cites sources.
    """
    if not raw_answer:
        return "No answer could be generated."

    cleaned = raw_answer.strip()

    # Add confidence scoring
    from app.rag.answerJudge import answerJudge
    result = answerJudge.score_answer(query, cleaned, context_chunks)
    score_str = f"\n\n---\nAnswer Confidence: {result['score']}/100\nReason: {result['reason']}"

    # Add sources only if requested
    final_with_sources = sourceCiter.cite_sources(query, cleaned, context_chunks)

    return final_with_sources + score_str
