# pythonService/app/rag/prompts.py
RQ_PROMPT = """You are a retrieval query refiner for a RAG system.
Given a user question, produce:
1) A single precise reformulation (RefinedQuery)
2) 1–3 focused SubQueries covering different aspects
3) 5–10 Keywords (single words or short noun phrases)
4) An Intent tag: one of [fact, summary, compare, howto, error, meta]
Output JSON with keys: refinedQuery, subQueries, keywords, intent.

Rules:
- Keep subQueries short and non-overlapping.
- Keywords should be document-search friendly (no stopwords).
- If the question is broad or ambiguous, make the refinedQuery specific but faithful.

Question:
{question}

Return ONLY JSON.
"""
