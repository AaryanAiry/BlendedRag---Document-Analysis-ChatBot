import os
from chromadb import Client
from chromadb.config import Settings

# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# DB_DIR = os.path.join(ROOT_DIR, "app", "data", "chroma")

DB_DIR = "data/chroma"  # short path to avoid "File name too long"

class ChromaClient:
    def __init__(self, db_dir: str = DB_DIR):
        # os.makedirs(db_dir, exist_ok=True)

        self.client = Client(Settings(
            anonymized_telemetry=False  # Disable telemetry
        ))

        # Initialize collections
        self.chunks = self.client.get_or_create_collection("chunks")
        self.tables = self.client.get_or_create_collection("tables")
        self.images = self.client.get_or_create_collection("images")

    def get_or_create_collection(self, name: str):
        return self.client.get_or_create_collection(name)

    # ---------------- Chunks ----------------
    def add_chunk(self, chunk_id: str, embedding: list, text: str, doc_id: str, page: int, type_: str = "text"):
        metadata = {"doc_id": doc_id, "page": page, "type": type_}
        self.chunks.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )
        print(f"✅ Chunk added: {chunk_id}, text len={len(text)}, embedding len={len(embedding)}")

    def query_chunks(self, query_embedding: list, n_results: int = 5):
        return self.chunks.query(query_embeddings=[query_embedding], n_results=n_results)

    # ---------------- Tables ----------------
    def add_table(self, table_id: str, embedding: list, table_json: str, doc_id: str, page: int):
        metadata = {"doc_id": doc_id, "page": page, "type": "table"}
        self.tables.add(
            ids=[table_id],
            embeddings=[embedding],
            documents=[table_json],
            metadatas=[metadata]
        )

    def query_tables(self, query_embedding: list, n_results: int = 3):
        return self.tables.query(query_embeddings=[query_embedding], n_results=n_results)

    # ---------------- Images ----------------
    def add_image(self, image_id: str, embedding: list, doc_id: str, page: int, document_ref: str = None):
        metadata = {"doc_id": doc_id, "page": page, "type": "image"}
        doc_str = document_ref if document_ref is not None else image_id
        self.images.add(
            ids=[image_id],
            embeddings=[embedding],
            documents=[doc_str],
            metadatas=[metadata]
        )

    def query_images(self, query_embedding: list, n_results: int = 3):
        return self.images.query(query_embeddings=[query_embedding], n_results=n_results)

    def list_collections(self):
        return self.client.list_collections()
    
    def count_chunks(self):
        return self.chunks.count()

# ---------------- Debug run ----------------
if __name__ == "__main__":
    cc = ChromaClient()

    print("✅ Chroma collections initialized:")
    print(" -", cc.chunks.name)
    print(" -", cc.tables.name)
    print(" -", cc.images.name)

    # --- Test inserting a dummy chunk ---
    cc.add_chunk(
        chunk_id="test1",
        embedding=[0.1] * 384,
        text="Hello world, this is a test chunk.",
        doc_id="doc1",
        page=1
    )

    # --- Query it back ---
    results = cc.query_chunks(query_embedding=[0.1] * 384, n_results=1)
    print("🔎 Query Results:", results)

chromaClient = ChromaClient()