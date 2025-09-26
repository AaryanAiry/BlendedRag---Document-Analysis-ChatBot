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
    tables: list = []      # NEW: include tables
    images: list = []      # NEW: include images

@router.post("")
async def processPdfEndpoint(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        uploadResult = await processPdf(file)
        logger.info(f"Stored {len(uploadResult['chunks'])} text chunks with embeddings in ChromaDB for docId: {uploadResult['docId']}")

        # Retrieve structured elements from Chroma or return from uploadResult if stored
        # Here we assume processPdf now returns images/tables alongside chunks
        return PDFResponse(
            docId=uploadResult["docId"],
            fileName=uploadResult["fileName"],
            pageCount=uploadResult["pageCount"],
            chunks=[{"text": c["text"]} for c in uploadResult.get("chunks", [])],
            tables=uploadResult.get("tables", []),
            images=uploadResult.get("images", [])
        )
    except Exception as e:
        logger.error(f"Failed to process PDF {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

