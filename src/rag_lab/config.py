from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Settings:
    story_path: Path = PROJECT_ROOT / "data/raw/cinderella.txt"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    chunk_strategy: str = "characters"
    chunk_size: int = 500
    chunk_overlap: int = 100
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_metric: str = "cosine"
    top_k: int = 5

DEFAULTS = Settings()
