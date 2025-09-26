# app/llm/sourceCiter.py

import json
import re
from typing import List, Dict
from app.llm.llmClient import llmClient
from app.utils.logger import getLogger

logger = getLogger(__name__)

class SourceCiter:
    def __init__(self):
        self.llm = llmClient

    def safe_parse_json(self, text: str):
        try:
            return json.loads(text)
        except:
            matches = re.findall(r"\{.*\}", text, re.S)
            if matches:
                try:
                    return json.loads(matches[-1])
                except Exception as e:
                    logger.debug(f"Safe JSON parse failed: {e}")
                    return None
        return None

    def cite_sources(self, query: str, answer: str, context_chunks: List[Dict]) -> str:
        """
        Annotates answer with citations if requested by user query.
        Assumes all chunks are normalized dicts with 'chunk' and 'page'.
        """
        if "cite" not in query.lower() and "source" not in query.lower():
            return answer

        # Build context text safely
        context_text = "\n\n".join(
            [f"[Chunk {c['chunk'].get('id', i)}] (Page {c.get('page', '?')}) {c['chunk']['text']}"
             for i, c in enumerate(context_chunks, start=1)]
        )

        prompt = f"""
The user asked: "{query}"
The assistant answered: "{answer}"

Here are the available source chunks with IDs and page numbers:
{context_text}

Task:
- Identify which chunks support each part of the answer.
- Provide citations as a list of chunk IDs and page numbers.
- Respond ONLY in JSON with keys "citations" (list of objects with "chunk_id" and "page") and "reason".

Example:
{{
    "citations": [
        {{"chunk_id": "2", "page": 5}},
        {{"chunk_id": "4", "page": 6}}
    ],
    "reason": "The answer is based mainly on chunk 2 (page 5) and chunk 4 (page 6)."
}}
"""

        try:
            response = self.llm.generateAnswer(prompt, max_tokens=250)
            data = self.safe_parse_json(response)
            if data:
                citations = data.get("citations", [])
                reason = data.get("reason", "")
                if citations:
                    cite_str = "\n\n---\nSources:\n"
                    for c in citations:
                        cite_str += f"- Chunk {c.get('chunk_id', '?')} (Page {c.get('page', '?')})\n"
                    cite_str += f"\nReason: {reason}"
                    return answer + cite_str
        except Exception as e:
            logger.error(f"Source citation failed: {e}")

        # Fallback heuristic: simple string overlap
        overlap_chunks = []
        for i, c in enumerate(context_chunks, start=1):
            text = c['chunk']['text']
            if any(word in text.lower() for word in query.lower().split()):
                overlap_chunks.append({
                    "chunk_id": c['chunk'].get("id", i),
                    "page": c.get("page", "?")
                })

        if overlap_chunks:
            cite_str = "\n\n---\nSources (heuristic):\n"
            for c in overlap_chunks:
                cite_str += f"- Chunk {c['chunk_id']} (Page {c['page']})\n"
            return answer + cite_str

        return answer + "\n\n---\nSources: None identified."


# Singleton instance
sourceCiter = SourceCiter()

