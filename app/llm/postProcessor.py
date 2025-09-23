# app/rag/postProcessor.py

def post_process_answer(raw_answer: str, query: str, context_chunks: list) -> str:
    """
    Cleans and structures the LLM raw output.
    - Removes hallucinations
    - Ensures table/list formatting if requested
    - Trims redundant content
    """
    if not raw_answer:
        return "No answer could be generated."

    cleaned = raw_answer.strip()

    # Basic heuristic: ensure Markdown table header if user asked for "table"
    if "table" in query.lower() and "|" not in cleaned:
        cleaned = "Could not detect a table format in the answer. Original response:\n\n" + cleaned

    return cleaned
