from fastapi import APIRouter
from pydantic import BaseModel
from app.ragService import query_document
from app.rag.hybridRagPipeline import run_pipeline

router = APIRouter()

class RAGRequest(BaseModel):
    docId: str
    query: str
    topK: int = 5

# @router.post("/api/ask")
# async def ask_rag(req: RAGRequest):
#     result = query_document(req.docId, req.query, req.topK)
#     return result
@router.post("/api/ask")
async def ask_rag(req: RAGRequest):
    out = run_pipeline(req.docId, req.query, top_k=req.topK, debug=True)
    return out

