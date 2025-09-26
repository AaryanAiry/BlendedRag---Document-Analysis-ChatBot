def chunkText(text: str, chunkSize: int = 200, chunkOverlap: int = 100, docId: str = None, page_number: int = None):
    """
    Splits text into overlapping chunks and assigns unique IDs.
    chunkSize: number of words per chunk
    chunkOverlap: number of words to overlap between chunks
    Returns list of dicts: {"id": ..., "text": ...}
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_index = 0

    if chunkSize <= 0:
        raise ValueError("chunkSize must be > 0")

    while start < len(words):
        end = min(start + chunkSize, len(words))
        chunk_text = " ".join(words[start:end])

        # Assign unique chunk ID
        if docId is not None and page_number is not None:
            chunk_id = f"{docId}_page{page_number}_chunk{chunk_index}"
        else:
            chunk_id = f"chunk{chunk_index}"

        chunks.append({
            "id": chunk_id,
            "text": chunk_text
        })

        start += max(1, chunkSize - chunkOverlap)
        chunk_index += 1

    return chunks

