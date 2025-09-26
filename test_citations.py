# test_citations.py
from app.llm.sourceCiter import sourceCiter

mock_chunks = [
    {"chunk": {"text": "AI is a field of computer science.", "id": "c1", "meta": {"page": 2}}, "id": "c1", "page": 2},
    {"chunk": {"text": "Machine learning is a subset of AI.", "id": "c2", "meta": {"page": 5}}, "id": "c2", "page": 5},
    {"chunk": {"text": "Deep learning uses neural networks.", "id": "c3", "meta": {"page": 8}}, "id": "c3", "page": 8},
]

query = "Cite sources about AI and Machine Learning"
answer = "AI is a field of computer science, and machine learning is a subset of AI."

final_answer = sourceCiter.cite_sources(query, answer, mock_chunks)
print("\n=== Final Answer with Citations ===\n")
print(final_answer)
