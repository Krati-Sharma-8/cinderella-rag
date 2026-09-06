from pathlib import Path

def load_document(path: str | Path) -> str:
    """Read one local UTF-8 document; ingestion never accesses the network."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Story not found at {path}. Run scripts/download_story.py first.")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Story file is empty: {path}")
    return text
