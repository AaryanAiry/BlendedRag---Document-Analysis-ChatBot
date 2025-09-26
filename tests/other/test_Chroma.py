# test_chroma.py
from chromadb import Client
from chromadb.config import Settings
import uuid

# --- Initialize Chroma client (in-memory) ---
chroma = Client(Settings(
    anonymized_telemetry=False
))

# --- Get or create collection ---
chunks_collection = chroma.get_or_create_collection("chunks")

# --- Create a single chunk ---
doc_id = str(uuid.uuid4())
chunk_id = f"{doc_id}_page1_chunk0"
text = "This is a test chunk for Chroma ingestion."
embedding = [0.1] * 384  # dummy embedding

# --- Add the chunk ---
chunks_collection.add(
    ids=[chunk_id],
    embeddings=[embedding],
    documents=[text],
    metadatas=[{"doc_id": doc_id, "page": 1, "type": "text"}]
)

print(f"✅ Chunk added: {chunk_id}")

# --- Query it back ---
results = chunks_collection.query(
    query_embeddings=[embedding],
    n_results=1
)

print("🔎 Query results:", results)

# --- Check total count ---
print("Total chunks in collection:", chunks_collection.count())
