import requests
import json
import time
import os

BASE_URL = "http://127.0.0.1:8000"

def uploadSample(pdfPath: str):
    """Uploads a sample PDF to the FastAPI service and returns the JSON response."""
    if not os.path.exists(pdfPath):
        raise FileNotFoundError(f"PDF file not found: {pdfPath}")

    with open(pdfPath, "rb") as f:
        # Correct multipart/form-data upload for FastAPI
        files = {"file": (os.path.basename(pdfPath), f, "application/pdf")}
        resp = requests.post(f"{BASE_URL}/processPdf/", files=files)

    if resp.status_code != 200:
        raise RuntimeError(f"Upload failed with status {resp.status_code}: {resp.text}")

    try:
        return resp.json()
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON received from server: {resp.text}")

def queryDoc(docId: str, query: str):
    """Sends a query to the processed document and returns the JSON response."""
    payload = {
        "docId": docId,
        "query": query,
        "topK": 5,
        "refine": True
    }
    resp = requests.post(f"{BASE_URL}/api/query/", json=payload)

    if resp.status_code != 200:
        raise RuntimeError(f"Query failed [{resp.status_code}]: {resp.text}")

    try:
        return resp.json()
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON received: {resp.text}")

if __name__ == "__main__":
    pdf_path = "sample.pdf"  # Adjust if stored in a different folder

    print(f"Uploading {pdf_path} ...")
    upload_response = uploadSample(pdf_path)
    print("Upload response:")
    print(json.dumps(upload_response, indent=2))

    # Extract docId
    docId = upload_response.get("docId")
    if not docId:
        raise KeyError(f"No 'docId' found in upload response: {upload_response}")

    time.sleep(1)  # wait briefly for indexing
    print(f"Querying document with docId={docId} ...")
    query_response = queryDoc(docId, "What is the main purpose of the library?")
    print("Query response:")
    print(json.dumps(query_response, indent=2))
