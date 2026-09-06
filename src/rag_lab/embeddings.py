from collections.abc import Sequence
import numpy as np
from .models import Chunk

class Embedder:
    """A thin, deliberate wrapper around SentenceTransformer only."""
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        getter = getattr(self.model, "get_embedding_dimension", self.model.get_sentence_embedding_dimension)
        return int(getter())

    def embed_documents(self, chunks: Sequence[Chunk]) -> np.ndarray:
        texts = [chunk.text for chunk in chunks]
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.asarray(self.model.encode(texts, convert_to_numpy=True), dtype=np.float32)

    def embed_query(self, question: str) -> np.ndarray:
        if not question.strip():
            raise ValueError("question must not be empty")
        return np.asarray(self.model.encode(question, convert_to_numpy=True), dtype=np.float32)
