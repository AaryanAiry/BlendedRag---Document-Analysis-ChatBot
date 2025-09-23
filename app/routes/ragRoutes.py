from fastapi import APIRouter
from pydantic import BaseModel
from app.ragService import query_document

router = APIRouter()

class RAGRequest(BaseModel):
    docId: str
    query: str
    topK: int = 5

@router.post("/api/ask")
async def ask_rag(req: RAGRequest):
    result = query_document(req.docId, req.query, req.topK)
    return result
