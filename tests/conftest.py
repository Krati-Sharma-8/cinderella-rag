import numpy as np

class TinyEmbedder:
    model_name = "tiny-test-embedder"
    dimension = 3
    def embed_query(self, text):
        return np.array({"red": [1,0,0], "blue": [0,1,0]}.get(text, [0,0,1]), dtype=np.float32)
    def embed_documents(self, chunks):
        return np.stack([self.embed_query(chunk.text) for chunk in chunks])
