from pathlib import Path
import numpy as np

METRICS = ("cosine", "dot_product", "l2")

class VectorStore:
    def __init__(self, dimension: int, metric: str = "cosine"):
        import faiss
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        self.dimension, self.metric = dimension, metric
        # Cosine is inner product after unit normalization; raw dot product is not normalized.
        # L2 uses squared Euclidean distance, where smaller values are better.
        self.index = faiss.IndexFlatL2(dimension) if metric == "l2" else faiss.IndexFlatIP(dimension)

    @property
    def count(self) -> int:
        return int(self.index.ntotal)

    @staticmethod
    def normalize(vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=np.float32).copy()
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.where(norms == 0, 1, norms)

    def add(self, vectors: np.ndarray) -> None:
        values = self._matrix(vectors)
        self.index.add(self.normalize(values) if self.metric == "cosine" else values)

    def search(self, query: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        values = self._matrix(query)
        if self.metric == "cosine":
            values = self.normalize(values)
        scores, ids = self.index.search(values, min(top_k, self.count))
        return scores[0], ids[0]

    def _matrix(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"expected vectors with dimension {self.dimension}")
        return values

    def save(self, path: str | Path) -> None:
        import faiss
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: str | Path, metric: str) -> "VectorStore":
        import faiss
        index = faiss.read_index(str(path))
        store = cls(index.d, metric)
        store.index = index
        return store
