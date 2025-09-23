# app/pdfParser/ingestor.py
import uuid
import os
from fastapi import UploadFile
from app.pdfParser.parser import extractTextFromPdf
from app.pdfParser.chunker import chunkText
from app.embeddings.embeddingClient import EmbeddingClient
from app.storage.documentStore import documentStore
from app.utils.logger import getLogger
from app.retrieval.sparseRetriever import sparseRetriever

# Import the shared Chroma client
from app.chromaClient import chromaClient

uploadDir = "data/uploads"
logger = getLogger(__name__)
embeddingClient = EmbeddingClient()
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200  # overlap between chunks to preserve context

async def processPdf(file: UploadFile):
    os.makedirs(uploadDir, exist_ok=True)
    try:
        # Generate unique docId and file path
        docId = str(uuid.uuid4())
        filePath = os.path.join(uploadDir, f"{docId}_{file.filename}")
        logger.info(f"Starting ingestion for: {file.filename}, saving as: {filePath}")

        # Save the uploaded file
        contents = await file.read()
        if not contents:
            raise ValueError(f"Uploaded file is empty: {file.filename}")
        with open(filePath, "wb") as f:
            f.write(contents)

        # Extract text and page count
        text, pageCount = extractTextFromPdf(filePath)
        if not text:
            raise ValueError(f"No text extracted from PDF: {file.filename}")
        logger.info(f"Extracted text length: {len(text)} characters")

        # Chunk text
        chunks = chunkText(text, chunkSize=CHUNK_SIZE, chunkOverlap=CHUNK_OVERLAP)
        logger.info(f"Generated {len(chunks)} chunks")

        # Generate embeddings
        embeddings = embeddingClient.generateEmbeddings(chunks)
        logger.info(f"Generated embeddings for {len(chunks)} chunks")

        # Prepare metadata and IDs for Chroma
        metadatas = [
            {
                "docId": docId,
                "chunkIndex": i,
                "text": chunks[i],
                "fileName": file.filename or "unknown.pdf",
                "pageCount": pageCount or 0
            } for i in range(len(chunks))
        ]
        ids = [f"{docId}_{i}" for i in range(len(chunks))]

        # Build and cache BM25 index
        sparseRetriever.indexDocument(docId, chunks, ids)
        logger.info(f"BM25 index built for docId={docId}")

        # Save document in memory and Chroma
        documentStore.saveDocument(docId, {
            "fileName": file.filename,
            "pageCount": pageCount,
            "chunks": [{"text": chunks[i]} for i in range(len(chunks))],
            "embeddings": embeddings
        })

        # Add chunks to Chroma collection (use documentStore's collection to avoid duplication)
        collection = documentStore.collection
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),  # Convert to list
            documents=chunks,                # List of strings
            metadatas=metadatas
        )
        logger.info(f"Saved {len(chunks)} chunks to Chroma for docId={docId}")

        return {
            "docId": docId,
            "fileName": file.filename,
            "pageCount": pageCount,
            "chunks": [{"text": chunk} for chunk in chunks]
        }

    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}")
        raise
    # finally:
    #     # Clean up temporary file if it exists
    #     if os.path.exists(filePath):
    #         os.remove(filePath)