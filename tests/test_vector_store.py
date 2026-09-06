import numpy as np
from rag_lab.vector_store import VectorStore

def test_index_count_top_k_and_cosine_normalization():
    store = VectorStore(2, "cosine")
    store.add(np.array([[10, 0], [1, 1], [0, 4]], dtype=np.float32))
    scores, ids = store.search(np.array([2, 0]), 2)
    assert store.count == 3 and len(ids) == 2
    assert ids[0] == 0 and scores[0] == 1.0

def test_l2_returns_nearest_with_squared_distance():
    store = VectorStore(2, "l2"); store.add(np.array([[0, 0], [4, 4]], dtype=np.float32))
    scores, ids = store.search(np.array([1, 1]), 1)
    assert ids.tolist() == [0] and scores.tolist() == [2.0]

def test_saved_index_loading(tmp_path):
    path = tmp_path / "index.faiss"; store = VectorStore(2, "dot_product")
    store.add(np.array([[1, 0], [0, 1]], dtype=np.float32)); store.save(path)
    loaded = VectorStore.load(path, "dot_product")
    assert loaded.count == 2
    assert loaded.search(np.array([0, 2]), 1)[1].tolist() == [1]
