# app/llm/llmClient.py
from llama_cpp import Llama
from app.utils.logger import getLogger
import os

logger = getLogger(__name__)

class LLMClient:
    def __init__(self, model_path: str = "models/qwen2.5-3b-instruct-q5_k_m.gguf"):
        """
        Initializes the LLM client for local Qwen model inference.
        Uses CPU by default. If compiled with GPU support in llama_cpp, will use GPU automatically.
        """
        self.model_path = model_path

        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {self.model_path}")
        logger.info(f"Loading Qwen model from: {self.model_path} ...")

        # Load model
        try:
            self.llm = Llama(model_path=self.model_path)
            logger.info("Qwen LLM loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            raise e

    # def generateAnswer(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    #     """
    #     Generate an answer for the given prompt using Qwen.
    #     """
    #     try:
    #         output = self.llm(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
    #         # Llama_cpp returns a dict with 'choices' -> 'text'
    #         if 'choices' in output and len(output['choices']) > 0:
    #             return output['choices'][0]['text'].strip()
    #         return ""
    #     except Exception as e:
    #         logger.error(f"Qwen generation failed: {e}")
    #         return "Error: Failed to generate answer."

    def generateAnswer(self, prompt: str, max_tokens: int = None, temperature: float = 0.7) -> str:
        """
        Generate an answer for the given prompt using Qwen.
        Dynamically adjusts max_tokens based on prompt length if not provided.
        """
        try:
            # Estimate tokens in prompt (roughly 1 token ≈ 4 characters)
            est_prompt_tokens = len(prompt) // 4
            if max_tokens is None:
                # Total target ~512, leave buffer
                max_tokens = max(128, 512 - est_prompt_tokens - 50)

            output = self.llm(prompt=prompt, max_tokens=max_tokens, temperature=temperature)

            if 'choices' in output and len(output['choices']) > 0:
                return output['choices'][0]['text'].strip()
            return ""
        except Exception as e:
            logger.error(f"Qwen generation failed: {e}")
            return "Error: Failed to generate answer."



# Singleton instance for reuse
llmClient = LLMClient()
