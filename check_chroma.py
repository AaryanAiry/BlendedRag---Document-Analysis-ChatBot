# check_chroma_peek.py
from app.chromaClient import chromaClient

# ----------------- Collections -----------------
print("✅ Chroma collections:")
for col in chromaClient.list_collections():
    print(" -", col.name)

# ----------------- Chunks -----------------
print("\n--- Chunks ---")
chunks = chromaClient.chunks.get()  # get all chunks
print(f"Total chunks stored: {len(chunks['ids'])}")
if len(chunks['ids']) > 0:
    for i in range(min(5, len(chunks['ids']))):  # show first 5 for brevity
        print(f"{i+1}. ID: {chunks['ids'][i]}, doc_id: {chunks['metadatas'][i]['doc_id']}, text snippet: {chunks['documents'][i][:50]}...")

# ----------------- Tables -----------------
print("\n--- Tables ---")
tables = chromaClient.tables.get()
print(f"Total tables stored: {len(tables['ids'])}")
if len(tables['ids']) > 0:
    for i in range(min(5, len(tables['ids']))):
        print(f"{i+1}. ID: {tables['ids'][i]}, doc_id: {tables['metadatas'][i]['doc_id']}, content snippet: {tables['documents'][i][:50]}...")

# ----------------- Images -----------------
print("\n--- Images ---")
images = chromaClient.images.get()
print(f"Total images stored: {len(images['ids'])}")
if len(images['ids']) > 0:
    for i in range(min(5, len(images['ids']))):
        print(f"{i+1}. ID: {images['ids'][i]}, doc_id: {images['metadatas'][i]['doc_id']}, ref: {images['documents'][i]}")
