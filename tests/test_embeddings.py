import sys
import types
import numpy as np
from rag_lab.embeddings import Embedder
from rag_lab.models import Chunk

class FakeModel:
    def __init__(self, name): self.name = name
    def get_sentence_embedding_dimension(self): return 4
    def encode(self, texts, convert_to_numpy=True):
        count = len(texts) if isinstance(texts, list) else 1
        result = np.ones((count, 4), dtype=np.float32)
        return result if isinstance(texts, list) else result[0]

def test_embedding_dimensions_and_shape(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeModel))
    embedder = Embedder("fake")
    vectors = embedder.embed_documents([Chunk(0, "text", 0, 4), Chunk(1, "more", 5, 9)])
    assert embedder.dimension == 4
    assert vectors.shape == (2, 4)
    assert embedder.embed_query("question").shape == (4,)
