from pydantic import BaseModel
from typing import List

class Chunk(BaseModel):
    text: str

class PDFResponse(BaseModel):
    docId: str
    fileName: str
    pageCount: int
    chunks: List[Chunk]

class QueryRequest(BaseModel):
    docId: str
    query: str

class QueryResult(BaseModel):
    matchedChunks: List[Chunk]
    answer: str

class DocumentMetadata(BaseModel):
    docId: str
    fileName: str
    pageCount: int
    chunkCount: int

class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]
