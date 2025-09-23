from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingClient:
    def __init__(self, modelName: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(modelName)
    
    def generateEmbeddings(self, texts: list[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True))
    
    def generateEmbedding(self, text: str) -> np.ndarray:
        return self.generateEmbeddings([text])[0]