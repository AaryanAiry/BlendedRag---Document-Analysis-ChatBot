from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/mnt/MyLinuxSpace/hf_cache/Mistral-7B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto", device_map="auto")

inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
