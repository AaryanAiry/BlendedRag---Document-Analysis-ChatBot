# app/llm/llmClient.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.utils.logger import getLogger
import os

logger = getLogger(__name__)

class MistralClient:
    def __init__(self, model_dir: str = "app/llm/models/Mistral-7B-v0.1"):
        """
        Initializes the LLM client for local Mistral 7B (PyTorch) inference.
        Uses GPU if available.
        """
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            raise ValueError(f"Model path does not exist: {self.model_dir}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Mistral 7B model from: {self.model_dir} on {self.device} ...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            logger.info("Mistral 7B loaded successfully (PyTorch).")
        except Exception as e:
            logger.error(f"Failed to load Mistral 7B model: {e}")
            raise e

    def generateAnswer(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Generate an answer for the given prompt using Mistral 7B.
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "pad_token_id": self.tokenizer.eos_token_id,
            }

            # If user sets temperature <= 0, switch to greedy decoding
            if temperature and temperature > 0:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["do_sample"] = True
            else:
                generation_kwargs["do_sample"] = False  # greedy mode

            output_ids = self.model.generate(**inputs, **generation_kwargs)
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        except Exception as e:
            logger.error(f"Mistral generation failed: {e}")
            return "Error: Failed to generate answer."

    # def generateAnswer(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    #     """
    #     Generate an answer for the given prompt using Mistral 7B.
    #     """
    #     try:
    #         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    #         output_ids = self.model.generate(
    #             **inputs,
    #             max_new_tokens=max_tokens,
    #             temperature=temperature,
    #             do_sample=True,
    #             pad_token_id=self.tokenizer.eos_token_id
    #         )
    #         return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    #     except Exception as e:
    #         logger.error(f"Mistral generation failed: {e}")
    #         return "Error: Failed to generate answer."


# Singleton instance for reuse
mistralClient = MistralClient()
