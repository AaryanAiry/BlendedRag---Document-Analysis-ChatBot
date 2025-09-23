def chunkText(text: str, chunkSize: int = 200, chunkOverlap: int = 50):
    """
    Splits text into overlapping chunks.
    chunkSize: number of words per chunk
    chunkOverlap: number of words to overlap between chunks
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunkSize, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunkSize - chunkOverlap
    return chunks
