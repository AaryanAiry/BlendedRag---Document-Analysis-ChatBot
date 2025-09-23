# parser.py
import fitz  # PyMuPDF

def extractTextFromPdf(filePath: str):
    """Extracts text from a PDF and returns (text, pageCount)"""
    text_list = []
    with fitz.open(filePath) as doc:
        pageCount = len(doc)
        for page in doc:
            text_list.append(page.get_text())
    return "\n".join(text_list), pageCount
