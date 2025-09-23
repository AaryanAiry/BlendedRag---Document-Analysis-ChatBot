from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # HuggingFace repo
local_dir = "./app/models/mistral7b"         

# Download and save tokenizer locally
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
tokenizer.save_pretrained(local_dir)

# Download and save model locally
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
model.save_pretrained(local_dir)

print(f"Mistral-7B downloaded successfully to {local_dir}")
