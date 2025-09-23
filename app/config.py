
import os
from pathlib import Path

# === Base Directories ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Ensure required directories exist
for d in [UPLOADS_DIR, CHUNKS_DIR, EMBEDDINGS_DIR]:
    os.makedirs(d, exist_ok=True)

# === Embedding Settings ===
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # local model
EMBEDDING_DIMENSION = 384  # for all-MiniLM-L6-v2

# === Chunking Settings ===
CHUNK_SIZE = 300  # characters per chunk
CHUNK_OVERLAP = 50  # characters overlap to maintain context

# === Retrieval Settings ===
TOP_K = 5  # number of chunks to retrieve during search
COSINE_SIMILARITY_THRESHOLD = 0.3  # minimum relevance for a match

# === LLM Settings (to be integrated later) ===
LLM_MODEL_NAME = "models/qwen2.5-3b-instruct-q5_k_m.gguf"  # placeholder for local LLM

# === Miscellaneous ===
ALLOWED_FILE_TYPES = [".pdf"]
MAX_UPLOAD_SIZE_MB = 25
