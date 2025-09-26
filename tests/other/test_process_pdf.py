import requests
import json
from app.chromaClient import chromaClient
from app.storage.documentStore import documentStore

# --- CONFIG ---
BASE_URL = "http://127.0.0.1:8000"
PDF_FILE = "nvidia_sample_removed.pdf"  # Path to your test PDF

# --- 1️⃣ Test /processPdf endpoint ---
print("🔹 Uploading PDF to /processPdf endpoint...")
with open(PDF_FILE, "rb") as f:
    files = {"file": (PDF_FILE, f, "application/pdf")}
    response = requests.post(f"{BASE_URL}/processPdf", files=files)

if response.status_code != 200:
    print("❌ processPdf failed:", response.text)
    exit(1)

result = response.json()
docId = result["docId"]
print(f"✅ PDF ingested successfully. docId={docId}")
print(f"Pages: {result['pageCount']}, Chunks: {len(result['chunks'])}\n")

# --- 2️⃣ Verify documentStore metadata ---
print("🔹 Checking documentStore metadata...")
doc_meta = documentStore.getDocument(docId)
print(json.dumps(doc_meta, indent=2))
print()

# --- 3️⃣ Verify Chroma chunks ---
print("🔹 Querying Chroma for chunks...")
query_embedding = [0]*384  # Dummy vector
chroma_results = chromaClient.chunks.query(
    query_embeddings=[query_embedding],
    where={"doc_id": docId},
    n_results=5
)

print("Top 5 chunks from Chroma:")
for i, text in enumerate(chroma_results["documents"][0]):
    metadata = chroma_results["metadatas"][0][i]
    distance = chroma_results["distances"][0][i]
    print(f"  [{i}] score={distance}, chunkIndex={metadata.get('chunkIndex')}")
    print(f"       text: {text[:80]}...\n")

# --- 4️⃣ Optional: test /api/query endpoint ---
TEST_QUERY = "NVIDIA GPU performance"
print(f"🔹 Testing /api/query with query: '{TEST_QUERY}'")
resp = requests.post(f"{BASE_URL}/api/query", json={
    "docId": docId,
    "query": TEST_QUERY,
    "topK": 3,
    "refine": True
})

if resp.status_code == 200:
    query_result = resp.json()
    print(f"✅ Retrieved {len(query_result['results'])} chunks for query '{TEST_QUERY}'")
    for r in query_result["results"]:
        print(f"  chunkIndex={r['chunkIndex']}, score={r['score']}")
        print(f"  snippet: {r['snippet']}\n")
else:
    print("❌ /api/query failed:", resp.text)

print("🎉 All checks done!")
