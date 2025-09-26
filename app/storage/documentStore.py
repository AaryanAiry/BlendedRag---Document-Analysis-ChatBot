from typing import Dict, Any, List
import threading
from app.utils.logger import getLogger

logger = getLogger(__name__)

class DocumentStore:
    """
    Keeps lightweight metadata in memory.
    All heavy storage is delegated to ChromaClient.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def saveDocument(self, docId: str, data: Dict[str, Any]) -> None:
        with self.lock:
            self._metadata[docId] = {
                "fileName": data.get("fileName", "unknown"),
                "pageCount": data.get("pageCount", 0),
                "numChunks": len(data.get("chunks", []))
            }
            logger.info(f"Saved metadata for docId={docId}: {self._metadata[docId]}")

    def getDocument(self, docId: str) -> Dict[str, Any] | None:
        with self.lock:
            return self._metadata.get(docId)

    def listDocuments(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [
                {"docId": docId, **meta}
                for docId, meta in self._metadata.items()
            ]

    def deleteDocument(self, docId: str) -> bool:
        with self.lock:
            if docId in self._metadata:
                del self._metadata[docId]
                return True
            return False

documentStore = DocumentStore()

