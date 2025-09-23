from fastapi import APIRouter, HTTPException
from app.storage.documentStore import documentStore
from app.utils.logger import getLogger

router = APIRouter()
logger = getLogger(__name__)

@router.get("/admin/documents")
def listDocuments():
    return {"documents": documentStore.listDocuments()}

@router.get("/admin/documents/{docId}")
def getDocument(docId: str):
    doc = documentStore.getDocument(docId)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    # return metadata + chunk count (avoid returning embeddings by default)
    return {
        "docId": docId,
        "fileName": doc.get("fileName"),
        "pageCount": doc.get("pageCount"),
        "numChunks": len(doc.get("chunks", []))
    }

@router.delete("/admin/documents/{docId}")
def deleteDocument(docId: str):
    ok = documentStore.deleteDocument(docId)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")
    logger.info(f"Deleted document {docId}")
    return {"deleted": True}