from .models import SearchResult

def build_context(results: list[SearchResult]) -> str:
    """Preserve exact retrieved text and make provenance visible."""
    return "\n\n".join(f"[Chunk {result.chunk_id}]\n{result.text}" for result in results)
