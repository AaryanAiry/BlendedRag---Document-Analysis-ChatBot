import json
import re
from typing import List, Dict
from app.llm.llmClient import llmClient
from app.utils.logger import getLogger

logger = getLogger(__name__)

class SourceCiter:
    def __init__(self):
        self.llm = llmClient

    def cite_sources(self, query: str, answer: str, context_chunks: List[Dict]) -> str:
        """
        Annotates answer with citations if requested by user query.
        """
        if "cite" not in query.lower() and "source" not in query.lower():
            return answer  # no citation request

        context_text = "\n\n".join(
            [f"[Chunk {i}] (Page {c.get('page', '?')}) {c['chunk']['text'] if isinstance(c['chunk'], dict) else str(c['chunk'])}"
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
                {{"chunk_id": 2, "page": 5}},
                {{"chunk_id": 4, "page": 6}}
            ],
            "reason": "The answer is based mainly on chunk 2 (page 5) and chunk 4 (page 6)."
        }}
        """

        try:
            response = self.llm.generate(prompt, max_new_tokens=250)
            match = re.search(r"\{.*\}", response, re.S)
            if match:
                data = json.loads(match.group(0))
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

        # fallback heuristic: string overlap
        overlap_chunks = []
        for i, c in enumerate(context_chunks, start=1):
            text = c['chunk']['text'] if isinstance(c['chunk'], dict) else str(c['chunk'])
            if any(word in text.lower() for word in query.lower().split()):
                overlap_chunks.append((i, c.get("page", "?")))

        if overlap_chunks:
            cite_str = "\n\n---\nSources (heuristic):\n"
            for cid, page in overlap_chunks:
                cite_str += f"- Chunk {cid} (Page {page})\n"
            return answer + cite_str

        return answer + "\n\n---\nSources: None identified."


# Singleton
sourceCiter = SourceCiter()
