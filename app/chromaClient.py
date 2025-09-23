# app/chromaClient.py
import chromadb

# Persistent Chroma client using new API
chromaClient = chromadb.PersistentClient(path="./data/chroma")

# Shared collection
collection = chromaClient.get_or_create_collection(name="documents")
