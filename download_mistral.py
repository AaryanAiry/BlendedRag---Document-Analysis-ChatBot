from huggingface_hub import snapshot_download
import os

# Model repo
model_name = "mistralai/Mistral-7B-v0.1"

# Local path where you want to store the model
local_dir = os.path.join("app", "llm", "models", "Mistral-7B-v0.1")
os.makedirs(local_dir, exist_ok=True)

print(f"Downloading {model_name} to {local_dir}...")

# Download the full model
snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    local_dir_use_symlinks=False  # ensures files are fully copied
)

print("✅ Download complete!")
