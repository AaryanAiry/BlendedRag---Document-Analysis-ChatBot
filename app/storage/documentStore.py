# app/storage/documentStore.py
from typing import Dict, Any, List
import threading
from app.chromaClient import chromaClient
from app.utils.logger import getLogger

logger = getLogger(__name__)

EMBEDDING_DIM = 384  # Must match the embedding model

class DocumentStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.collection = chromaClient.get_or_create_collection(
            name="documents",
            metadata={"description": "PDF chunks with embeddings"}
        )
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def saveDocument(self, docId: str, data: Dict[str, Any]) -> None:
        with self.lock:
            chunks = data["chunks"]
            embeddings = data.get("embeddings")
            if embeddings is not None and len(chunks) != len(embeddings):
                raise ValueError("Number of chunks and embeddings must match")

            if chunks and isinstance(chunks[0], str):
                chunks = [{"text": c} for c in chunks]

            if embeddings is not None and len(embeddings) > 0:  # Safe check for non-empty embeddings
                ids = [f"{docId}_{i}" for i in range(len(chunks))]
                metadatas = [{
                    "docId": docId,
                    "chunkIndex": i,
                    "fileName": data.get("fileName", "unknown"),
                    "pageCount": data.get("pageCount", 0)
                } for i, chunk in enumerate(chunks)]
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings.tolist(),
                    metadatas=metadatas,
                    documents=[c["text"] for c in chunks]
                )

            # ✅ Always persist metadata in memory
            self._metadata[docId] = {
                "fileName": data.get("fileName", "unknown"),
                "pageCount": data.get("pageCount", 0),
                "numChunks": len(chunks)
            }
            logger.info(f"Saved metadata for docId={docId}: {self._metadata[docId]}")

    def getDocument(self, docId: str) -> Dict[str, Any] | None:
        try:
            results = self.collection.get(where={"docId": docId}, include=["metadatas", "documents"])
            if results and results.get("documents"):
                chunks = [
                    {"chunkIndex": md["chunkIndex"], "text": doc, "score": 0.0}
                    for md, doc in zip(results["metadatas"], results["documents"])
                ]
                metadata = self._metadata.get(docId, {})
                return {
                    "docId": docId,
                    "fileName": metadata.get("fileName", "unknown"),
                    "pageCount": metadata.get("pageCount", 0),
                    "chunks": chunks
                }
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve document {docId}: {e}")
            return None

    def listDocuments(self) -> List[Dict[str, Any]]:
        try:
            results = self.collection.get(include=["metadatas"])
            docs: Dict[str, Dict[str, Any]] = {}
            if results and results.get("metadatas"):
                for md in results["metadatas"]:
                    if not md or not md.get("docId"):
                        continue
                    docId = md["docId"]
                    if docId not in docs:
                        docs[docId] = {
                            "docId": docId,
                            "fileName": md.get("fileName", "unknown"),
                            "pageCount": md.get("pageCount", 0),
                            "numChunks": 0
                        }
                    docs[docId]["numChunks"] += 1
            return list(docs.values())
        except Exception as e:
            logger.error(f"Failed to list documents from Chroma: {e}")
            return []

    def deleteDocument(self, docId: str) -> bool:
        with self.lock:
            if docId in self._metadata:
                del self._metadata[docId]
            self.collection.delete(where={"docId": docId})
            return True

documentStore = DocumentStore()
