import numpy as np
from conftest import TinyEmbedder
from rag_lab.models import Chunk
from rag_lab.retrieval import Retriever
from rag_lab.vector_store import VectorStore

def test_retrieval_preserves_rank_id_text_and_top_k():
    chunks = [Chunk(0, "red", 0, 3), Chunk(1, "blue", 4, 8), Chunk(2, "other", 9, 14)]
    embedder = TinyEmbedder(); store = VectorStore(3, "cosine"); store.add(embedder.embed_documents(chunks))
    results = Retriever(embedder, store, chunks).retrieve("blue", top_k=2)
    assert len(results) == 2
    assert (results[0].rank, results[0].chunk_id, results[0].text) == (1, 1, "blue")
