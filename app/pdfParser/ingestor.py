import uuid
import os
from fastapi import UploadFile
from app.pdfParser.parser import extractTextFromPdf
from app.pdfParser.chunker import chunkText
from app.embeddings.embeddingClient import EmbeddingClient
from app.storage.documentStore import documentStore
from app.utils.logger import getLogger
from app.retrieval.sparseRetriever import sparseRetriever
from app.chromaClient import chromaClient

uploadDir = "data/uploads"
logger = getLogger(__name__)
embeddingClient = EmbeddingClient()
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

async def processPdf(file: UploadFile):
    os.makedirs(uploadDir, exist_ok=True)
    try:
        # Generate unique docId and file path
        docId = str(uuid.uuid4())
        filePath = os.path.join(uploadDir, f"{docId}_{file.filename}")
        logger.info(f"Starting ingestion for: {file.filename}, saving as: {filePath}")

        # Save uploaded file
        contents = await file.read()
        if not contents:
            raise ValueError(f"Uploaded file is empty: {file.filename}")
        with open(filePath, "wb") as f:
            f.write(contents)

        # --- Step 1: Extract plain text ---
        text, pageCount = extractTextFromPdf(filePath)
        if not text:
            raise ValueError(f"No text extracted from PDF: {file.filename}")
        logger.info(f"Extracted text length: {len(text)} characters")

        # --- Step 2: Chunk text ---
        chunks = chunkText(text, chunkSize=CHUNK_SIZE, chunkOverlap=CHUNK_OVERLAP)
        logger.info(f"Generated {len(chunks)} chunks")

        # --- Step 3: Embeddings for chunks ---
        embeddings = embeddingClient.generateEmbeddings(chunks)
        logger.info(f"Generated embeddings for {len(chunks)} chunks")

        ids = [f"{docId}_chunk_{i}" for i in range(len(chunks))]

        # --- Step 4: BM25 sparse index ---
        sparseRetriever.indexDocument(docId, chunks, ids)
        logger.info(f"BM25 index built for docId={docId}")

        # --- Step 5: Save chunks to Chroma ---
        for i, chunk in enumerate(chunks):
            chromaClient.add_chunk(
                chunk_id=ids[i],
                embedding=embeddings[i].tolist(),
                text=chunk,
                doc_id=docId,
                page=1  # TODO: replace with real page info when using pdfToJson
            )
        logger.info(f"Saved {len(chunks)} chunks to Chroma for docId={docId}")

        # --- Step 6: Save in memory store ---
        documentStore.saveDocument(docId, {
            "fileName": file.filename,
            "pageCount": pageCount,
            "chunks": [{"text": c} for c in chunks],
            "embeddings": embeddings
        })

        # --- Step 7: (future) Add tables & images ---
        # Once pdfToJson extracts tables/images, add:
        # chromaClient.add_table(...)
        # chromaClient.add_image(...)

        return {
            "docId": docId,
            "fileName": file.filename,
            "pageCount": pageCount,
            "chunks": [{"text": chunk} for chunk in chunks]
        }

    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}")
        raise
