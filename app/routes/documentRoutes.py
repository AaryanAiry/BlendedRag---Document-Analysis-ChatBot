# app/routes/documentRoutes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path
import os

router = APIRouter()

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "uploads"))

# Response models
class DocumentMetadata(BaseModel):
    docId: str
    fileName: str
    pageCount: int = 0   # Optional
    numChunks: int = 0   # Optional

class ListDocumentsResponse(BaseModel):
    documents: List[DocumentMetadata]

class DeleteDocumentResponse(BaseModel):
    docId: str
    deleted: bool

@router.get("/api/documents", response_model=ListDocumentsResponse)
def listDocuments():
    """
    List all PDFs in data/uploads automatically.
    """
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    documents = []
    for file in os.listdir(UPLOAD_DIR):
        if file.lower().endswith(".pdf"):
            documents.append(DocumentMetadata(
                docId=file,       # Using filename as docId
                fileName=file
            ))
    return ListDocumentsResponse(documents=documents)


@router.delete("/api/documents/{docId}", response_model=DeleteDocumentResponse)
def deleteDocument(docId: str):
    """
    Delete a document by filename from data/uploads.
    """
    file_path = Path(UPLOAD_DIR) / docId
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    os.remove(file_path)
    return DeleteDocumentResponse(docId=docId, deleted=True)


# # app/routes/documentRoutes.py
# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel
# from typing import List, Dict
# import os
# import json
# from app.utils.logger import getLogger

# router = APIRouter()
# logger = getLogger(__name__)

# UPLOAD_DIR = "data/uploads"  # folder where PDFs are saved

# # Response models
# class DocumentMetadata(BaseModel):
#     docId: str
#     fileName: str
#     pageCount: int
#     numChunks: int

# class ListDocumentsResponse(BaseModel):
#     documents: List[DocumentMetadata]

# class DeleteDocumentResponse(BaseModel):
#     docId: str
#     deleted: bool

# # In-memory metadata cache
# metadataIndex: Dict[str, Dict] = {}

# def load_metadata_index():
#     """Scan UPLOAD_DIR and rebuild metadata index from JSON metadata files if any"""
#     global metadataIndex
#     metadataIndex = {}
#     if not os.path.exists(UPLOAD_DIR):
#         os.makedirs(UPLOAD_DIR)
#         return

#     for fname in os.listdir(UPLOAD_DIR):
#         # Expecting files like: <docId>_<filename>.pdf
#         if fname.endswith(".pdf"):
#             docId = fname.split("_")[0]
#             # Try to load lightweight metadata if available
#             meta_file = os.path.join(UPLOAD_DIR, f"{docId}_metadata.json")
#             if os.path.exists(meta_file):
#                 with open(meta_file, "r") as f:
#                     meta = json.load(f)
#             else:
#                 # fallback metadata
#                 meta = {
#                     "fileName": fname,
#                     "pageCount": 0,
#                     "numChunks": 0
#                 }
#             metadataIndex[docId] = meta

# # Initialize metadata at startup
# load_metadata_index()

# @router.get("/api/documents", response_model=ListDocumentsResponse)
# def listDocuments():
#     """
#     Return list of uploaded documents with basic metadata.
#     """
#     documents = []
#     for docId, meta in metadataIndex.items():
#         documents.append(DocumentMetadata(
#             docId=docId,
#             fileName=meta.get("fileName", "unknown"),
#             pageCount=meta.get("pageCount", 0),
#             numChunks=meta.get("numChunks", 0)
#         ))
#     return ListDocumentsResponse(documents=documents)

# @router.delete("/api/documents/{docId}", response_model=DeleteDocumentResponse)
# def deleteDocument(docId: str):
#     """
#     Delete a document by docId directly from UPLOAD_DIR and remove metadata.
#     """
#     if docId not in metadataIndex:
#         raise HTTPException(status_code=404, detail="Document not found")

#     # Delete PDF file
#     pdf_file = None
#     for fname in os.listdir(UPLOAD_DIR):
#         if fname.startswith(docId + "_") and fname.endswith(".pdf"):
#             pdf_file = os.path.join(UPLOAD_DIR, fname)
#             break

#     if pdf_file and os.path.exists(pdf_file):
#         os.remove(pdf_file)
#         logger.info(f"Deleted PDF file: {pdf_file}")

#     # Delete metadata JSON if exists
#     meta_file = os.path.join(UPLOAD_DIR, f"{docId}_metadata.json")
#     if os.path.exists(meta_file):
#         os.remove(meta_file)
#         logger.info(f"Deleted metadata file: {meta_file}")

#     # Remove from in-memory index
#     del metadataIndex[docId]

#     return {"docId": docId, "deleted": True}
