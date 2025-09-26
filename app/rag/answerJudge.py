# app/rag/answerJudge.py
from typing import List, Dict
import json
import re
from app.llm.llmClient import llmClient  # Qwen2.5-3B
from app.llm.mistralClient import mistralClient  # Mistral for judging
from app.utils.logger import getLogger

logger = getLogger(__name__)

class AnswerJudge:
    def __init__(self):
        self.llm_primary = llmClient
        self.llm_judge = mistralClient  # Mistral-7B

    def _parse_json_from_text(self, text: str):
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception as e:
            logger.debug(f"AnswerJudge JSON parse failed: {e}")
            return None

    def score_answer(self, query: str, answer: str, context_chunks: List[Dict], max_tokens: int = 200) -> Dict:
        """
        Returns {"score": float(0-1), "method": str, "reason": str}
        """
        if not answer:
            return {"score": 0.0, "method": "none", "reason": "No answer provided"}

        # --- Heuristic: fraction of chunks mentioning query tokens ---
        try:
            query_tokens = set(query.lower().split())
            overlap_count = 0
            for c in context_chunks:
                text = c.get("chunk", {}).get("text") if isinstance(c.get("chunk"), dict) else str(c.get("chunk", ""))
                text_tokens = set(text.lower().split())
                if query_tokens & text_tokens:
                    overlap_count += 1
            heuristic_score = overlap_count / max(1, len(context_chunks))
        except Exception as e:
            logger.debug(f"Heuristic overlap scoring failed: {e}")
            heuristic_score = 0.5

        # --- LLM judgment using Mistral ---
        final_score = heuristic_score
        method = "overlap_only"
        llm_judged = "N/A"

        try:
            prompt = f"""
You are a judge. Check if the assistant's answer is fully supported by the provided context chunks.
Answer only "Y" if fully supported, else "N".
Question: "{query}"
Answer: "{answer}"
Context chunks (text only, short snippets):
{chr(10).join([c.get('chunk', {}).get('text', '')[:200] if isinstance(c.get('chunk'), dict) else str(c.get('chunk',''))[:200] for c in context_chunks])}
"""
            llm_out = self.llm_judge.generateAnswer(prompt, max_tokens=64, temperature=0.0)
            llm_out_clean = llm_out.strip().upper()
            llm_judged = "Y" if "Y" in llm_out_clean else "N"

            if llm_judged == "Y":
                final_score = max(heuristic_score, 0.8)
                method = "llm+overlap"
            else:
                final_score = min(heuristic_score, 0.7)
                method = "llm+overlap"

        except Exception as e:
            logger.debug(f"Mistral judge failed, fallback to heuristic: {e}")
            final_score = heuristic_score
            method = "overlap_only"

        reason = f"Heuristic overlap fraction={heuristic_score:.2f}, LLM judged supported={llm_judged}"
        return {"score": round(final_score, 3), "method": method, "reason": reason}

# singleton
answerJudge = AnswerJudge()
