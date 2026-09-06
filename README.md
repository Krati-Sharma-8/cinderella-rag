# Cinderella RAG Lab

An intentionally small, framework-free playground for seeing every part of Retrieval-Augmented Generation (RAG). It uses one local Cinderella story. Retrieval works without an API key; generation is optional.

## The idea

RAG first **retrieves** relevant source passages and then optionally asks a language model to answer from those passages. This separates finding evidence from writing an answer.

```text
                        INGESTION
cinderella.txt → Loader → Cleaner → Chunker → Embedding Model → FAISS Index

                          QUERY
Question → Embedding Model → Query Vector → FAISS Search → Top-K Chunks
                                                        ├→ display retrieval
                                                        ↓
                                                  Build Context
                                                        ↓
                                                   Optional LLM
                                                        ↓
                                                      Answer
```

Each stage lives in a small module rather than LangChain or another RAG framework.

## Setup and complete command guide

Python 3.12 or newer is required. The first embedding command downloads the model once; later queries use its local cache.

```bash
# install runtime and test dependencies
python -m pip install -e '.[dev]'

# obtain/refresh the public-domain story (a usable local copy is already included)
python scripts/download_story.py

# inspect loading and lightweight cleaning
python -m rag_lab.cli document

# inspect every chunk
python -m rag_lab.cli chunks --strategy characters --chunk-size 500 --overlap 100

# inspect one real embedding
python -m rag_lab.cli embed --chunk-id 3

# create artifacts/index.faiss, chunks.json, and index_config.json
python -m rag_lab.cli index --strategy characters --chunk-size 500 --overlap 100 --metric cosine

# retrieval only: this never calls an LLM
python -m rag_lab.cli search "Why did Cinderella leave the ball?" --top-k 5

# retrieval plus optional generation
python -m rag_lab.cli ask "What happened when midnight arrived?" --top-k 5

# tests and learning UI
pytest
streamlit run streamlit_app.py
```

To try another experiment, rebuild with (for example) `--chunk-size 300 --overlap 50 --metric l2`, then repeat the same search. `--top-k` changes only the number returned and therefore does **not** require rebuilding.

## Learn the mechanics

**Chunks** make a long document retrievable in focused pieces. Character size counts characters; word size counts whitespace-delimited words; sentence size counts simple punctuation-delimited sentences. Overlap repeats the end of one window at the start of the next so facts across a boundary are not separated. More overlap means more vectors and redundancy. It must be smaller than size so the window always advances.

An **embedding** is a fixed-length numerical vector whose direction captures patterns of meaning learned by the model. It is not readable prose, but nearby semantic ideas tend to have nearby vectors. This model produces 384 dimensions. The `embed` command exposes values, shape, and norm rather than hiding them.

**FAISS** stores vectors and performs exact nearest-neighbor search here:

* **Cosine similarity** compares direction, ignoring magnitude. Both stored and query vectors are normalized, then inner product is used. Larger is better.
* **Dot product** uses raw inner product, so direction and magnitude matter. Larger is better.
* **L2** is squared Euclidean distance. Smaller is better. Its FAISS index is structurally different.

**Top-K** means return the best K indexed chunks. Raising K offers generation more evidence but also more irrelevant context.

## Diagnosing failure

An answer can exist in the story but retrieval can miss it: a question and passage may embed differently, a boundary may split the fact, chunks may be too broad, or K may be too small. Inspect IDs, exact text, order, and scores before blaming generation. A **retrieval failure** means useful evidence never reached the context. A **generation failure** means good evidence was retrieved but the optional model misunderstood, ignored, or embellished it. The CLI and UI expose the exact context to distinguish these.

Changing strategy, size, overlap, embedding model, or metric requires cleaning/chunking again, recomputing every document embedding, and rebuilding FAISS. The saved `index_config.json` records those choices; the UI refuses to search stale settings. Changing only question or Top-K does not rebuild: just the question embedding and search are repeated.

## Optional answer generation

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` to use the deliberately tiny OpenAI-compatible provider. `OPENAI_MODEL` and `OPENAI_BASE_URL` are configurable. With no key, `ask` explicitly reports that generation is disabled while search remains fully functional. The prompt requires only retrieved context, an insufficiency admission, and supporting chunk IDs.

## Reading order

Start at `src/rag_lab/loader.py`, then follow:

```text
loader.py → cleaner.py → chunking.py → embeddings.py → vector_store.py
→ retrieval.py → context_builder.py → generation.py → pipeline.py
→ streamlit_app.py
```

`cli.py` shows how individual stages are exposed. `models.py` contains the two plain data records, and `config.py` centralizes defaults. Tests document boundary and mathematical guarantees. Evaluation prompts and expected keywords are in `evals/questions.json`.

## Future Experiments

After understanding this version, possible separate experiments include semantic or recursive chunking, BM25/hybrid retrieval, reranking, query rewriting, multi-query and parent-child retrieval, metadata filters, multiple formats, vector databases, approximate/HNSW indexes, RAGAS-style evaluation, conversation memory, and deployment. They are deliberately not implemented here.
