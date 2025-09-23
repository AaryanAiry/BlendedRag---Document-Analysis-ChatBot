# app/rag/queryRefiner.py
import json
import re
from typing import Dict, List
from app.llm.llmClient import llmClient
from app.rag.prompts import RQ_PROMPT

_stopwords = {
    "the","is","at","which","on","a","an","and","or","in","for","to","of","by","with","as","that","this"
}

_synonymMap = {
    "price": ["cost", "pricing"],
    "error": ["issue", "problem", "fault"],
    "summary": ["overview", "abstract"]
}

def _basic_preprocess(query: str) -> str:
    q = query.strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q

def _fallback_variants(query: str) -> List[str]:
    refined = _basic_preprocess(query)
    tokens = [t for t in re.split(r"\W+", refined) if t and t not in _stopwords]
    variants = {refined}
    for tok in tokens:
        if tok in _synonymMap:
            for syn in _synonymMap[tok]:
                variants.add(refined.replace(tok, syn))
    if len(tokens) > 1:
        variants.add(" ".join(tokens))
    return list(variants)

def _intent_to_weights(intent: str) -> Dict[str, float]:
    intent = (intent or "generic").lower()
    if intent in ("fact", "compare", "meta"):
        return {"bm25": 0.6, "dense": 0.4}
    if intent in ("summary", "howto"):
        return {"bm25": 0.4, "dense": 0.6}
    if intent == "error":
        return {"bm25": 0.5, "dense": 0.5}
    return {"bm25": 0.5, "dense": 0.5}

def _cheap_keywords(s: str, cap: int = 10) -> List[str]:
    toks = [t for t in re.findall(r"[a-z0-9\-]+", s.lower()) if t not in _stopwords and len(t) > 2]
    uniq = []
    for t in toks:
        if t not in uniq:
            uniq.append(t)
    return uniq[:cap]

def _extract_json(s: str) -> str:
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in LLM output.")
    return s[start:end+1]

def refine_query_intelligent(query: str) -> Dict:
    original = query
    refinedQuery = None
    subQueries = []
    keywords = []
    intent = "generic"

    # 1) Try LLM-based refinement
    try:
        prompt = RQ_PROMPT.format(question=original)
        raw = llmClient.generateAnswer(prompt, max_tokens=256)
        data = json.loads(_extract_json(raw))
        refinedQuery = data.get("refinedQuery") or data.get("refined_query")
        subQueries = data.get("subQueries") or data.get("sub_queries") or []
        keywords = data.get("keywords") or []
        intent = (data.get("intent") or "generic").lower()
    except Exception:
        # Fall back to heuristic
        refinedQuery = _basic_preprocess(original)
        tokens = [t for t in re.split(r"\W+", refinedQuery) if t and t not in _stopwords]
        if tokens:
            subQueries = [" ".join(tokens)]
        keywords = _cheap_keywords(refinedQuery)

    weightingHint = _intent_to_weights(intent)
    variants = list({refinedQuery, *_fallback_variants(refinedQuery), *subQueries})

    return {
        "original": original,
        "refinedQuery": refinedQuery,
        "subQueries": subQueries[:3],
        "keywords": keywords,
        "intent": intent,
        "weightingHint": weightingHint,
        "variants": variants
    }
