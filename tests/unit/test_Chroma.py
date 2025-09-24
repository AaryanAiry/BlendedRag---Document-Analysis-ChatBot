from app.chromaClient import ChromaClient

cc = ChromaClient()

print("Chunk count:", cc.chunks.count())  # Should print 1
results = cc.query_chunks(query_embedding=[0.1]*384, n_results=1)
print("Query results:", results)