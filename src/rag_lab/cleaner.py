import re

def clean_document(text: str) -> str:
    """Normalize whitespace without changing punctuation or wording."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    cleaned = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()
