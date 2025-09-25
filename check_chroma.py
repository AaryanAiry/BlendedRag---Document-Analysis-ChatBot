from app.chromaClient import ChromaClient

# Use the same ChromaClient as the server
cc = ChromaClient()  

print("📂 Available collections:")
for col in [cc.chunks, cc.tables, cc.images]:
    try:
        print(f" - {col.name}: {len(col.get()['documents'])} items")
    except Exception as e:
        print(f"⚠️ Collection '{col.name}' not found or empty: {e}")
