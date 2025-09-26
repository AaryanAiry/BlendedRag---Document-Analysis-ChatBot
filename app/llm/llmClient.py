# app/llm/llmClient.py

import os
from llama_cpp import Llama
from app.utils.logger import getLogger

logger = getLogger(__name__)


class LLMClient:
    def __init__(self, model_path: str = "app/llm/models/qwen2.5-3b-instruct-q5_k_m.gguf"):
        """
        Initializes the LLM client for local Qwen model inference.
        Uses CPU by default. If compiled with GPU support in llama_cpp, will use GPU automatically.
        """

        self.model_path = model_path

        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {self.model_path}")

        logger.info(f"Loading Qwen model from: {self.model_path} ...")

        try:
            # bump context length to 2048 and enable verbose logging
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=2048,
                verbose=True
            )
            logger.info("Qwen LLM loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            raise e

    def generateAnswer(self, prompt: str, max_tokens: int = None, temperature: float = 0.7) -> str:
        """
        Generate an answer for the given prompt using Qwen.
        Dynamically adjusts max_tokens based on prompt length if not provided.
        """
        try:
            # Estimate tokens in prompt (roughly 1 token ≈ 4 characters)
            est_prompt_tokens = len(prompt) // 4

            if max_tokens is None:
                # Leave buffer to stay within n_ctx=2048
                max_tokens = max(128, 2048 - est_prompt_tokens - 50)

            output = self.llm(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

            if "choices" in output and len(output["choices"]) > 0:
                return output["choices"][0]["text"].strip()
            return ""
        except Exception as e:
            logger.error(f"Qwen generation failed: {e}")
            return "Error: Failed to generate answer."


# Singleton instance for reuse
llmClient = LLMClient()




# # app/llm/llmClient.py
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from app.utils.logger import getLogger
# import os

# logger = getLogger(__name__)

# class LLMClient:
#     def __init__(self, model_dir: str = "app/llm/models/Mistral-7B-v0.1"):
#         """
#         Initializes the LLM client for local Mistral 7B (PyTorch) inference.
#         Uses GPU if available.
#         """
#         self.model_dir = model_dir
#         if not os.path.exists(self.model_dir):
#             raise ValueError(f"Model path does not exist: {self.model_dir}")

#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         logger.info(f"Loading Mistral 7B model from: {self.model_dir} on {self.device} ...")

#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 self.model_dir,
#                 device_map="auto" if self.device == "cuda" else None,
#                 torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#             )
#             logger.info("Mistral 7B loaded successfully (PyTorch).")
#         except Exception as e:
#             logger.error(f"Failed to load Mistral 7B model: {e}")
#             raise e

#     def generateAnswer(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
#         """
#         Generate an answer for the given prompt using Mistral 7B.
#         """
#         try:
#             inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
#             output_ids = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_tokens,
#                 temperature=temperature,
#                 do_sample=True,
#                 pad_token_id=self.tokenizer.eos_token_id
#             )
#             return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
#         except Exception as e:
#             logger.error(f"Mistral generation failed: {e}")
#             return "Error: Failed to generate answer."


# # Singleton instance for reuse
# llmClient = LLMClient()
