import pytest
from rag_lab.chunking import chunk_document

def test_character_boundaries_and_final_short_chunk():
    chunks = chunk_document("abcdefghij", "characters", 4, 1)
    assert [(c.text, c.start, c.end) for c in chunks] == [
        ("abcd", 0, 4), ("defg", 3, 7), ("ghij", 6, 10)]

def test_word_overlap_repeats_units():
    chunks = chunk_document("one two three four five", "words", 3, 1)
    assert chunks[0].text == "one two three"
    assert chunks[1].text == "three four five"

def test_sentence_strategy_and_empty_text():
    assert [c.text for c in chunk_document("One. Two! Three?", "sentences", 2, 1)] == ["One. Two!", "Two! Three?"]
    assert chunk_document("", "characters", 10, 0) == []

@pytest.mark.parametrize("size,overlap", [(5, 5), (5, 6), (5, -1)])
def test_invalid_overlap(size, overlap):
    with pytest.raises(ValueError): chunk_document("hello", "characters", size, overlap)
