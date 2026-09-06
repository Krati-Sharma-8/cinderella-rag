import runpy
from pathlib import Path


strip_gutenberg = runpy.run_path(
    Path(__file__).parents[1] / "scripts/download_story.py"
)["strip_gutenberg"]


def test_download_extracts_cinderella_and_excludes_the_following_story():
    ebook = """*** START OF THE PROJECT GUTENBERG EBOOK STORIES ***
front matter
CINDERELLA

This is the story of Cinderella at the ball.

[Illustration]

She left when the clock struck midnight.

THE THREE BEARS

Goldilocks entered the bears' cottage.
*** END OF THE PROJECT GUTENBERG EBOOK STORIES ***"""

    story = strip_gutenberg(ebook)

    assert story.startswith("CINDERELLA")
    assert "midnight" in story
    assert "Illustration" not in story
    assert "THE THREE BEARS" not in story
    assert "Goldilocks" not in story
