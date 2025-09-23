from pydantic import BaseModel
from typing import List

class chunk(BaseModel):
    text:str

class pdfResponse(BaseModel):
    docId: str
    fileName: str
    pageCount: int
    chunks: List[chunk]