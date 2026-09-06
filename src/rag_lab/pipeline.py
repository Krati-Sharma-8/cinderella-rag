import json
from pathlib import Path
import numpy as np
from .chunking import chunk_document
from .cleaner import clean_document
from .config import DEFAULTS
from .context_builder import build_context
from .embeddings import Embedder
from .generation import Generator, generator_from_environment
from .loader import load_document
from .models import Chunk
from .retrieval import Retriever
from .vector_store import VectorStore

def build_index(strategy=DEFAULTS.chunk_strategy, chunk_size=DEFAULTS.chunk_size,
                overlap=DEFAULTS.chunk_overlap, model_name=DEFAULTS.embedding_model,
                metric=DEFAULTS.similarity_metric, story_path=DEFAULTS.story_path,
                artifacts_dir=DEFAULTS.artifacts_dir) -> dict:
    text = clean_document(load_document(story_path))
    chunks = chunk_document(text, strategy, chunk_size, overlap)
    embedder = Embedder(model_name)
    store = VectorStore(embedder.dimension, metric)
    store.add(embedder.embed_documents(chunks))
    artifacts = Path(artifacts_dir); artifacts.mkdir(parents=True, exist_ok=True)
    store.save(artifacts / "index.faiss")
    (artifacts / "chunks.json").write_text(json.dumps([c.to_dict() for c in chunks], indent=2), encoding="utf-8")
    config = {"chunk_strategy": strategy, "chunk_size": chunk_size, "chunk_overlap": overlap,
              "embedding_model": model_name, "embedding_dimension": embedder.dimension,
              "similarity_metric": metric, "number_of_chunks": len(chunks)}
    (artifacts / "index_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config

def load_retriever(artifacts_dir=DEFAULTS.artifacts_dir) -> tuple[Retriever, dict]:
    artifacts = Path(artifacts_dir)
    required = [artifacts / n for n in ("index.faiss", "chunks.json", "index_config.json")]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("Index artifacts are missing. Run `python -m rag_lab.cli index`.")
    config = json.loads(required[2].read_text(encoding="utf-8"))
    chunks = [Chunk.from_dict(item) for item in json.loads(required[1].read_text(encoding="utf-8"))]
    embedder = Embedder(config["embedding_model"])
    if embedder.dimension != config["embedding_dimension"]:
        raise ValueError("Current embedding model dimension is incompatible with the saved index")
    store = VectorStore.load(required[0], config["similarity_metric"])
    return Retriever(embedder, store, chunks), config

def ask(question: str, top_k=DEFAULTS.top_k, generator: Generator | None = None,
        artifacts_dir=DEFAULTS.artifacts_dir) -> dict:
    retriever, config = load_retriever(artifacts_dir)
    results = retriever.retrieve(question, top_k)
    context = build_context(results)
    answer = (generator or generator_from_environment()).generate(question, context)
    return {"question": question, "results": results, "context": context, "answer": answer,
            "config": config, "query_vector_norm": float(np.linalg.norm(retriever.last_query_vector))}
