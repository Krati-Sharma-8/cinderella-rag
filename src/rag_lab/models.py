from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class Chunk:
    id: int
    text: str
    start: int | None
    end: int | None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "Chunk":
        return cls(**value)

@dataclass(frozen=True)
class SearchResult:
    rank: int
    chunk_id: int
    text: str
    score: float
