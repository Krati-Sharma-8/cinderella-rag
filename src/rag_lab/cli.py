import argparse
import json
import numpy as np
from dotenv import load_dotenv
from .chunking import STRATEGIES, chunk_document
from .cleaner import clean_document
from .config import DEFAULTS
from .context_builder import build_context
from .embeddings import Embedder
from .generation import generator_from_environment
from .loader import load_document
from .pipeline import ask, build_index, load_retriever
from .vector_store import METRICS

def local_chunks(args):
    return chunk_document(clean_document(load_document(DEFAULTS.story_path)), args.strategy, args.chunk_size, args.overlap)

def command_document(args):
    raw = load_document(DEFAULTS.story_path); cleaned = clean_document(raw)
    print(f"Raw character count: {len(raw):,}\nClean character count: {len(cleaned):,}\n\n{cleaned[:args.first]}")

def command_chunks(args):
    text = clean_document(load_document(DEFAULTS.story_path)); chunks = chunk_document(text, args.strategy, args.chunk_size, args.overlap)
    print(f"Document length: {len(text):,} characters\nStrategy: {args.strategy}\nChunk size: {args.chunk_size}\nOverlap: {args.overlap}\nTotal chunks: {len(chunks)}")
    for chunk in chunks:
        print(f"\n{'-'*34}\nChunk {chunk.id}\nCharacters: {chunk.start} - {chunk.end}\n{'-'*34}\n\n{chunk.text}")

def command_embed(args):
    chunks = local_chunks(args)
    if args.chunk_id < 0 or args.chunk_id >= len(chunks): raise SystemExit(f"chunk ID must be between 0 and {len(chunks)-1}")
    chunk = chunks[args.chunk_id]; embedder = Embedder(args.model); vector = embedder.embed_documents([chunk])[0]
    print(f"Chunk ID: {chunk.id}\n\nText:\n{chunk.text}\n\nEmbedding model:\n{embedder.model_name}\n\nDimensions:\n{len(vector)}\n\nVector norm: {np.linalg.norm(vector):.6f}\n\nFirst 10 dimensions:\n{vector[:10].tolist()}")

def command_index(args):
    config = build_index(args.strategy, args.chunk_size, args.overlap, args.model, args.metric)
    print("Index created:\n" + json.dumps(config, indent=2))

def show_results(question, results, config, top_k):
    print(f"QUESTION\n{question}\n\nRETRIEVAL CONFIGURATION\n\nChunk strategy: {config['chunk_strategy']}\nChunk size: {config['chunk_size']}\nChunk overlap: {config['chunk_overlap']}\nEmbedding: {config['embedding_model']}\nMetric: {config['similarity_metric']}\nTop K: {top_k}")
    for result in results:
        label = "Distance" if config["similarity_metric"] == "l2" else "Score"
        print(f"\n\nRESULT #{result.rank}\nChunk ID: {result.chunk_id}\n{label}: {result.score:.6f}\n\n{result.text}")

def command_search(args):
    retriever, config = load_retriever(); results = retriever.retrieve(args.question, args.top_k)
    show_results(args.question, results, config, args.top_k)

def command_ask(args):
    output = ask(args.question, args.top_k, generator_from_environment(True))
    show_results(args.question, output["results"], output["config"], args.top_k)
    print(f"\n\nRETRIEVED CONTEXT\n{output['context']}\n\nANSWER\n{output['answer']}\n\nSUPPORTING CHUNKS\n" + ", ".join(str(r.chunk_id) for r in output["results"]))

def add_chunk_args(parser):
    parser.add_argument("--strategy", choices=STRATEGIES, default=DEFAULTS.chunk_strategy)
    parser.add_argument("--chunk-size", type=int, default=DEFAULTS.chunk_size)
    parser.add_argument("--overlap", type=int, default=DEFAULTS.chunk_overlap)

def main(argv=None):
    load_dotenv(); parser = argparse.ArgumentParser(prog="rag-lab"); sub = parser.add_subparsers(required=True)
    p=sub.add_parser("document"); p.add_argument("--first", type=int, default=1000); p.set_defaults(func=command_document)
    p=sub.add_parser("chunks"); add_chunk_args(p); p.set_defaults(func=command_chunks)
    p=sub.add_parser("embed"); add_chunk_args(p); p.add_argument("--chunk-id", type=int, required=True); p.add_argument("--model", default=DEFAULTS.embedding_model); p.set_defaults(func=command_embed)
    p=sub.add_parser("index"); add_chunk_args(p); p.add_argument("--model", default=DEFAULTS.embedding_model); p.add_argument("--metric", choices=METRICS, default=DEFAULTS.similarity_metric); p.set_defaults(func=command_index)
    for name, function in (("search", command_search), ("ask", command_ask)):
        p=sub.add_parser(name); p.add_argument("question"); p.add_argument("--top-k", type=int, default=DEFAULTS.top_k); p.set_defaults(func=function)
    args=parser.parse_args(argv)
    try: args.func(args)
    except (FileNotFoundError, ValueError) as error: parser.error(str(error))

if __name__ == "__main__": main()
