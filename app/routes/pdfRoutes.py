# app/routes/pdfRoutes.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.pdfParser.ingestor import processPdf
from app.storage.documentStore import documentStore
from app.utils.logger import getLogger
from pydantic import BaseModel

router = APIRouter()
logger = getLogger(__name__)

class PDFResponse(BaseModel):
    docId: str
    fileName: str
    pageCount: int
    chunks: list

@router.post("")  # Explicit route without trailing slash
async def processPdfEndpoint(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        uploadResult = await processPdf(file)
        logger.info(f"Stored {len(uploadResult['chunks'])} chunks with embeddings in ChromaDB for docId: {uploadResult['docId']}")
        return PDFResponse(
            docId=uploadResult["docId"],
            fileName=uploadResult["fileName"],
            pageCount=uploadResult["pageCount"],
            chunks=[{"text": c["text"]} for c in uploadResult["chunks"]]
        )
    except Exception as e:
        logger.error(f"Failed to process PDF {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))