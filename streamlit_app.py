import json
import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from rag_lab.config import DEFAULTS
from rag_lab.context_builder import build_context
from rag_lab.generation import DisabledGenerator, generator_from_environment
from rag_lab.pipeline import build_index, load_retriever

load_dotenv()
st.set_page_config(page_title="Cinderella RAG Lab", page_icon="👠", layout="wide")
st.title("Cinderella RAG Learning Playground")
st.caption("Inspect what a small retrieval pipeline does at every step.")

config_path = DEFAULTS.artifacts_dir / "index_config.json"
saved = json.loads(config_path.read_text()) if config_path.exists() else {}
with st.sidebar:
    st.header("RAG Configuration")
    strategy = st.selectbox("Chunk strategy", ["characters", "words", "sentences"], index=["characters", "words", "sentences"].index(saved.get("chunk_strategy", DEFAULTS.chunk_strategy)))
    size = st.selectbox("Chunk size", [100, 300, 500, 800], index=[100, 300, 500, 800].index(saved.get("chunk_size", 500)) if saved.get("chunk_size", 500) in [100,300,500,800] else 2)
    overlap = st.selectbox("Chunk overlap", [0, 50, 100, 200], index=[0,50,100,200].index(saved.get("chunk_overlap", 100)) if saved.get("chunk_overlap",100) in [0,50,100,200] else 2)
    model = st.text_input("Embedding model", saved.get("embedding_model", DEFAULTS.embedding_model))
    metric = st.selectbox("Similarity metric", ["cosine", "dot_product", "l2"], index=["cosine","dot_product","l2"].index(saved.get("similarity_metric", "cosine")))
    top_k = st.selectbox("Top K", [1, 3, 5, 10], index=2)
    generation = st.checkbox("Generation enabled", value=False)
    wanted = {"chunk_strategy": strategy, "chunk_size": size, "chunk_overlap": overlap,
              "embedding_model": model, "similarity_metric": metric}
    stale = not saved or any(saved.get(k) != v for k, v in wanted.items())
    if stale: st.warning("Index settings changed. Rebuild before searching.")
    if overlap >= size: st.error("Overlap must be smaller than chunk size.")
    if st.button("Rebuild Index", disabled=overlap >= size):
        with st.spinner("Chunking, embedding, and indexing..."):
            saved = build_index(strategy, size, overlap, model, metric)
        st.success(f"Indexed {saved['number_of_chunks']} chunks."); st.rerun()

question = st.text_input("Ask Cinderella:", "Why did Cinderella leave the ball?")
if st.button("Search", type="primary", disabled=stale):
    try:
        retriever, config = load_retriever(); results = retriever.retrieve(question, top_k)
        context = build_context(results)
        st.header("Retrieved Chunks")
        for result in results:
            label = "Distance" if metric == "l2" else "Similarity"
            with st.expander(f"#{result.rank} | Chunk {result.chunk_id} | {label} {result.score:.4f}", expanded=True): st.write(result.text)
        st.header("Generated Answer")
        generator = generator_from_environment(generation)
        if isinstance(generator, DisabledGenerator): st.info("Generation disabled or OPENAI_API_KEY is not configured. Retrieval still worked.")
        else:
            with st.spinner("Generating..."): st.write(generator.generate(question, context))
        with st.expander("View exact context sent to LLM"): st.code(context)
        st.header("Debug Information")
        st.json({"question": question, "embedding_model": config["embedding_model"],
                 "embedding_dimensions": config["embedding_dimension"],
                 "query_vector_norm": float(np.linalg.norm(retriever.last_query_vector)),
                 "number_of_indexed_chunks": config["number_of_chunks"], "similarity_metric": metric,
                 "top_k": top_k, "chunk_strategy": strategy, "chunk_size": size, "chunk_overlap": overlap})
    except Exception as error:
        st.error(str(error))
