"""Download Project Gutenberg's Cinderella text, with a bundled-file fallback.

Runtime retrieval never invokes this script or accesses the network.
"""
from pathlib import Path
import re
from urllib.request import Request, urlopen

URL = "https://www.gutenberg.org/cache/epub/77727/pg77727.txt"
DESTINATION = Path(__file__).resolve().parents[1] / "data/raw/cinderella.txt"

def strip_gutenberg(text: str) -> str:
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    if start_marker in text:
        text = text.split(start_marker, 1)[1].split("***", 1)[1]
    if end_marker in text:
        text = text.split(end_marker, 1)[0]
    # eBook 77727 contains Cinderella followed by The Three Bears. Start at the
    # story's second title and stop before the unrelated story.
    story_start = re.search(r"\nCINDERELLA\s*\n\s*This is the story", text)
    story_end = re.search(r"\nTHE THREE BEARS\s*\n", text)
    if not story_start or not story_end or story_end.start() <= story_start.start():
        raise ValueError("could not locate Cinderella boundaries in Gutenberg text")
    text = text[story_start.start() + 1:story_end.start()]
    text = re.sub(r"\n?\[Illustration(?::.*?\n)?\]\n?", "\n", text, flags=re.DOTALL)
    return text.strip()

def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    try:
        request = Request(URL, headers={"User-Agent": "cinderella-rag-lab/0.1"})
        with urlopen(request, timeout=30) as response:
            story = strip_gutenberg(response.read().decode("utf-8-sig"))
        if "Cinderella" not in story or len(story) < 2000:
            raise ValueError("download did not resemble the expected Cinderella story")
        DESTINATION.write_text(story + "\n", encoding="utf-8")
        print(f"Saved {len(story):,} characters to {DESTINATION}")
    except Exception as error:
        if DESTINATION.exists():
            print(f"Download failed ({error}); keeping bundled local story at {DESTINATION}")
        else:
            raise SystemExit(f"Download failed: {error}\nPlace a UTF-8 story at {DESTINATION}")

if __name__ == "__main__": main()
