# app/pdfParser/ingestor.py
import uuid
import os
from fastapi import UploadFile
from app.pdfParser.pdfToJson import extract_pdf_layout
from app.pdfParser.chunker import chunkText
from app.embeddings.embeddingClient import EmbeddingClient
from app.storage.documentStore import documentStore
from app.utils.logger import getLogger
from app.retrieval.sparseRetriever import sparseRetriever
from app.chromaClient import chromaClient

uploadDir = "data/uploads"
logger = getLogger(__name__)
embeddingClient = EmbeddingClient()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

async def processPdf(file: UploadFile):
    os.makedirs(uploadDir, exist_ok=True)
    try:
        docId = str(uuid.uuid4())
        filePath = os.path.join(uploadDir, f"{docId}_{file.filename}")
        logger.info(f"Starting ingestion for: {file.filename}, saving as: {filePath}")

        contents = await file.read()
        if not contents:
            raise ValueError(f"Uploaded file is empty: {file.filename}")
        with open(filePath, "wb") as f:
            f.write(contents)

        pdf_json = extract_pdf_layout(filePath, docId=docId, chromaClient=chromaClient, save_file=False)
        logger.info(f"Extracted structured PDF layout with {len(pdf_json['pages'])} pages")

        all_chunks = []

        for page in pdf_json["pages"]:
            page_text = " ".join([e["content"] for e in page["elements"] if e["type"] == "textbox"])

            page_chunks = chunkText(
                page_text,
                chunkSize=CHUNK_SIZE,
                chunkOverlap=CHUNK_OVERLAP,
                docId=docId,
                page_number=page["page_number"]
            )

            chunk_texts = [c["text"] for c in page_chunks]
            chunk_ids = [c["id"] for c in page_chunks]
            embeddings = embeddingClient.generateEmbeddings(chunk_texts)

            # BM25 indexing
            sparseRetriever.indexDocument(docId, chunk_texts, chunk_ids)

            # Chroma storage with extended metadata
            for i, chunk in enumerate(page_chunks):
                chromaClient.add_chunk(
                    chunk_id=chunk["id"],
                    embedding=embeddings[i].tolist(),
                    text=chunk["text"],
                    doc_id=docId,
                    page=page["page_number"],
                    type_="text"
                )

            all_chunks.extend(page_chunks)

        logger.info(f"Processed {len(all_chunks)} text chunks for docId={docId}")

        documentStore.saveDocument(docId, {
            "fileName": file.filename,
            "pageCount": len(pdf_json["pages"]),
            "chunks": all_chunks
        })

        return {
            "docId": docId,
            "fileName": file.filename,
            "pageCount": len(pdf_json["pages"]),
            "chunks": all_chunks
        }

    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}")
        raise




# import uuid
# import os
# from fastapi import UploadFile
# from app.pdfParser.pdfToJson import extract_pdf_layout
# from app.pdfParser.chunker import chunkText
# from app.embeddings.embeddingClient import EmbeddingClient
# from app.storage.documentStore import documentStore
# from app.utils.logger import getLogger
# from app.retrieval.sparseRetriever import sparseRetriever
# from app.chromaClient import chromaClient

# uploadDir = "data/uploads"
# logger = getLogger(__name__)
# embeddingClient = EmbeddingClient()
# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200

# async def processPdf(file: UploadFile):
#     os.makedirs(uploadDir, exist_ok=True)
#     try:
#         # --- Step 0: Generate docId and save PDF ---
#         docId = str(uuid.uuid4())
#         filePath = os.path.join(uploadDir, f"{docId}_{file.filename}")
#         logger.info(f"Starting ingestion for: {file.filename}, saving as: {filePath}")

#         contents = await file.read()
#         if not contents:
#             raise ValueError(f"Uploaded file is empty: {file.filename}")
#         with open(filePath, "wb") as f:
#             f.write(contents)

#         # --- Step 1: Extract structured layout from PDF ---
#         pdf_json = extract_pdf_layout(filePath, docId=docId, chromaClient=chromaClient, save_file=False)
#         logger.info(f"Extracted structured PDF layout with {len(pdf_json['pages'])} pages")

#         # --- Step 2: Chunk text per page and generate embeddings ---
#         all_chunks = []
#         all_embeddings = []
#         all_chunk_ids = []

#         for page in pdf_json["pages"]:
#             page_text = " ".join([e["content"] for e in page["elements"] if e["type"] == "textbox"])
#             chunks = chunkText(page_text, chunkSize=CHUNK_SIZE, chunkOverlap=CHUNK_OVERLAP)

#             embeddings = embeddingClient.generateEmbeddings(chunks)
#             page_chunk_ids = [f"{docId}_page{page['page_number']}_chunk{i}" for i in range(len(chunks))]

#             # BM25 indexing
#             sparseRetriever.indexDocument(docId, chunks, page_chunk_ids)

#             # Save chunks to Chroma
#             for i, chunk in enumerate(chunks):
#                 chromaClient.add_chunk(
#                     chunk_id=page_chunk_ids[i],
#                     embedding=embeddings[i].tolist(),
#                     text=chunk,
#                     doc_id=docId,
#                     page=page["page_number"]
#                 )

#             all_chunks.extend(chunks)
#             all_embeddings.extend(embeddings)
#             all_chunk_ids.extend(page_chunk_ids)

#         logger.info(f"Processed {len(all_chunks)} text chunks for docId={docId}")

#         # --- Step 3: Save document metadata ---
#         documentStore.saveDocument(docId, {
#             "fileName": file.filename,
#             "pageCount": len(pdf_json["pages"]),
#             "chunks": [{"text": c} for c in all_chunks],
#             "embeddings": all_embeddings
#         })

#         return {
#             "docId": docId,
#             "fileName": file.filename,
#             "pageCount": len(pdf_json["pages"]),
#             "chunks": [{"text": c} for c in all_chunks]
#         }

#     except Exception as e:
#         logger.error(f"Ingestion failed for {file.filename}: {e}")
#         raise





# import uuid
# import os
# from fastapi import UploadFile
# from app.pdfParser.parser import extractTextFromPdf
# from app.pdfParser.chunker import chunkText
# from app.embeddings.embeddingClient import EmbeddingClient
# from app.storage.documentStore import documentStore
# from app.utils.logger import getLogger
# from app.retrieval.sparseRetriever import sparseRetriever
# from app.chromaClient import chromaClient

# uploadDir = "data/uploads"
# logger = getLogger(__name__)
# embeddingClient = EmbeddingClient()
# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200

# async def processPdf(file: UploadFile):
#     os.makedirs(uploadDir, exist_ok=True)
#     try:
#         # Generate unique docId and file path
#         docId = str(uuid.uuid4())
#         filePath = os.path.join(uploadDir, f"{docId}_{file.filename}")
#         logger.info(f"Starting ingestion for: {file.filename}, saving as: {filePath}")

#         # Save uploaded file
#         contents = await file.read()
#         if not contents:
#             raise ValueError(f"Uploaded file is empty: {file.filename}")
#         with open(filePath, "wb") as f:
#             f.write(contents)

#         # --- Step 1: Extract plain text ---
#         text, pageCount = extractTextFromPdf(filePath)
#         if not text:
#             raise ValueError(f"No text extracted from PDF: {file.filename}")
#         logger.info(f"Extracted text length: {len(text)} characters")

#         # --- Step 2: Chunk text ---
#         chunks = chunkText(text, chunkSize=CHUNK_SIZE, chunkOverlap=CHUNK_OVERLAP)
#         logger.info(f"Generated {len(chunks)} chunks")

#         # --- Step 3: Embeddings for chunks ---
#         embeddings = embeddingClient.generateEmbeddings(chunks)
#         logger.info(f"Generated embeddings for {len(chunks)} chunks")

#         ids = [f"{docId}_chunk_{i}" for i in range(len(chunks))]

#         # --- Step 4: BM25 sparse index ---
#         sparseRetriever.indexDocument(docId, chunks, ids)
#         logger.info(f"BM25 index built for docId={docId}")

#         # --- Step 5: Save chunks to Chroma ---
#         for i, chunk in enumerate(chunks):
#             chromaClient.add_chunk(
#                 chunk_id=ids[i],
#                 embedding=embeddings[i].tolist(),
#                 text=chunk,
#                 doc_id=docId,
#                 page=1  # TODO: replace with real page info when using pdfToJson
#             )
#         logger.info(f"Saved {len(chunks)} chunks to Chroma for docId={docId}")

#         # --- Step 6: Save in memory store ---
#         documentStore.saveDocument(docId, {
#             "fileName": file.filename,
#             "pageCount": pageCount,
#             "chunks": [{"text": c} for c in chunks],
#             "embeddings": embeddings
#         })

#         # --- Step 7: (future) Add tables & images ---
#         # Once pdfToJson extracts tables/images, add:
#         # chromaClient.add_table(...)
#         # chromaClient.add_image(...)

#         return {
#             "docId": docId,
#             "fileName": file.filename,
#             "pageCount": pageCount,
#             "chunks": [{"text": chunk} for chunk in chunks]
#         }

#     except Exception as e:
#         logger.error(f"Ingestion failed for {file.filename}: {e}")
#         raise
