from .embeddings import Embedder
from .models import Chunk, SearchResult
from .vector_store import VectorStore

class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, chunks: list[Chunk]):
        if vector_store.count != len(chunks):
            raise ValueError("FAISS vector count does not match saved chunks")
        self.embedder, self.vector_store, self.chunks = embedder, vector_store, chunks
        self.last_query_vector = None

    def retrieve(self, question: str, top_k: int = 5) -> list[SearchResult]:
        self.last_query_vector = self.embedder.embed_query(question)
        scores, indices = self.vector_store.search(self.last_query_vector, top_k)
        return [SearchResult(rank, self.chunks[int(i)].id, self.chunks[int(i)].text, float(score))
                for rank, (score, i) in enumerate(zip(scores, indices), start=1) if i >= 0]
