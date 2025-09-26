# app/rag/answerJudge.py
from typing import List, Dict
import json
import re
from app.llm.llmClient import llmClient
from app.utils.logger import getLogger

logger = getLogger(__name__)

class AnswerJudge:
    def __init__(self):
        self.llm = llmClient

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
        Ask the LLM to score the answer 0-100 and explain the reason.
        Returns {"score": int, "reason": str}
        """
        if not answer:
            return {"score": 0, "reason": "No answer provided"}

        # Build compact context summary
        ctx_lines = []
        for i, c in enumerate(context_chunks, start=1):
            text = c.get("chunk", {}).get("text") if isinstance(c.get("chunk"), dict) else c.get("chunk", str(c))
            page = c.get("chunk", {}).get("meta", {}).get("page") if isinstance(c.get("chunk"), dict) else c.get("page", "?")
            ctx_lines.append(f"[{i}] page={page} text_snip=\"{(text or '')[:200].replace(chr(10),' ')}\"")
        ctx_text = "\n".join(ctx_lines)[:4000]

        prompt = f"""
You are a judge that scores an assistant's answer on a scale 0-100.
User question:
\"\"\"{query}\"\"\"

Assistant's answer:
\"\"\"{answer}\"\"\"

Context chunks (short snips):
{ctx_text}

Task:
1) Score the answer 0-100 where 100 == fully correct, precise, supported by context and not hallucinated. 
2) Provide a short reason (1-2 sentences).
3) Output ONLY valid JSON like: {{"score": 85, "reason": "reason text..."}}

Give concise reasoning.
"""

        try:
            llm_out = self.llm.generateAnswer(prompt, max_tokens=max_tokens, temperature=0.0)
            parsed = self._parse_json_from_text(llm_out)
            if parsed and isinstance(parsed.get("score"), (int, float)):
                score = int(round(parsed.get("score")))
                reason = parsed.get("reason", "")
                logger.info(f"AnswerJudge: score={score}, reason={reason}")
                return {"score": max(0, min(100, score)), "reason": reason}
            else:
                logger.debug(f"AnswerJudge: LLM output not JSON or missing score. Raw: {llm_out}")
        except Exception as e:
            logger.error(f"AnswerJudge LLM error: {e}")

        # Fallback heuristic: overlap between answer and context
        try:
            answer_tokens = set((answer or "").lower().split())
            scores = []
            for c in context_chunks:
                text = c.get("chunk", {}).get("text") if isinstance(c.get("chunk"), dict) else c.get("chunk", "")
                text_tokens = set((text or "").lower().split())
                overlap = len(answer_tokens & text_tokens)
                scores.append(overlap)
            if scores:
                # normalize to 0-100
                max_overlap = max(scores)
                score = int(min(100, (max_overlap / (max(1, sum(len((c.get('chunk',{}).get('text','') or '').split()) for c in context_chunks)))) * 100 * 5))
                reason = "Heuristic overlap scoring (fallback)"
                logger.info(f"AnswerJudge fallback score={score}")
                return {"score": score, "reason": reason}
        except Exception as e:
            logger.debug(f"AnswerJudge fallback failed: {e}")

        return {"score": 50, "reason": "Unable to judge reliably; returning neutral score."}


# singleton
answerJudge = AnswerJudge()
