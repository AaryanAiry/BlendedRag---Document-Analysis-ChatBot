# app/routes/documentRoutes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from app.storage.documentStore import documentStore
from app.utils.logger import getLogger

router = APIRouter()
logger = getLogger(__name__)

# Response models
class DocumentMetadata(BaseModel):
    docId: str
    fileName: str
    pageCount: int
    numChunks: int

class ListDocumentsResponse(BaseModel):
    documents: List[DocumentMetadata]

class DeleteDocumentResponse(BaseModel):
    docId: str
    deleted: bool

# In-memory metadata cache (for listing)
# Can be persisted later if needed
metadataIndex: Dict[str, Dict] = {}

@router.get("/api/documents", response_model=ListDocumentsResponse)
def listDocuments():
    """
    Return list of uploaded documents with basic metadata.
    """
    documents = []
    for docId, meta in metadataIndex.items():
        documents.append(DocumentMetadata(
            docId=docId,
            fileName=meta.get("fileName", "unknown"),
            pageCount=meta.get("pageCount", 0),
            numChunks=meta.get("numChunks", 0)
        ))
    return ListDocumentsResponse(documents=documents)

@router.delete("/api/documents/{docId}", response_model=DeleteDocumentResponse)
def deleteDocument(docId: str):
    """
    Delete a document by docId from Chroma and metadata index.
    """
    if docId not in metadataIndex:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete from Chroma
    success = documentStore.deleteDocument(docId)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document {docId} not found")
    logger.info(f"Deleted document {docId}")
    return {"message": f"Document {docId} deleted successfully"}