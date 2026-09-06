import re
from .models import Chunk

STRATEGIES = ("characters", "words", "sentences")

def _validate(strategy: str, chunk_size: int, overlap: int) -> None:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

def chunk_document(text: str, strategy: str = "characters", chunk_size: int = 500,
                   overlap: int = 100) -> list[Chunk]:
    _validate(strategy, chunk_size, overlap)
    if not text:
        return []
    if strategy == "characters":
        return _character_chunks(text, chunk_size, overlap)
    pattern = r"\S+" if strategy == "words" else r"[^\s.!?][^.!?]*(?:[.!?]+[\"']?|$)"
    units = list(re.finditer(pattern, text, flags=re.MULTILINE))
    return _unit_chunks(text, units, chunk_size, overlap)

def _character_chunks(text: str, size: int, overlap: int) -> list[Chunk]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(Chunk(len(chunks), text[start:end], start, end))
        if end == len(text):
            break
        start = end - overlap
    return chunks

def _unit_chunks(text: str, units: list[re.Match], size: int, overlap: int) -> list[Chunk]:
    chunks, start_unit, step = [], 0, size - overlap
    while start_unit < len(units):
        group = units[start_unit:start_unit + size]
        start, end = group[0].start(), group[-1].end()
        chunks.append(Chunk(len(chunks), text[start:end], start, end))
        if start_unit + size >= len(units):
            break
        start_unit += step
    return chunks
