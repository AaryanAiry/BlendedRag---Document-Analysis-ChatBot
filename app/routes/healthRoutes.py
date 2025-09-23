from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def healthCheck():
    return{"status":"ok","service":"Document AI Engine"}
