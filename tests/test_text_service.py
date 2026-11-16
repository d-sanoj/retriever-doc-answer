"""Unit tests for text processing (port of the original script's helpers)."""

from app.services.text_service import (
    STOP,
    chunk_words,
    clean_markdown,
    tokenize,
)


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The quick Brown fox jumps over the lazy dog") == [
        "quick",
        "brown",
        "fox",
        "jumps",
        "lazy",
        "dog",
    ]


def test_tokenize_keeps_numbers_and_splits_on_punctuation():
    assert tokenize("GPT-4 costs $0.015/post; v2.0!") == [
        "gpt",
        "4",
        "costs",
        "0",
        "015",
        "post",
        "v2",
        "0",
    ]


def test_stopword_list_matches_original():
    # Spot-checks from the original script's list (including its duplicates).
    assert "the" in STOP and "is" in STOP and "needn" in STOP
    assert len(STOP) == 138
    assert "retrieval" not in STOP and "model" not in STOP


def test_clean_markdown_strips_frontmatter():
    md = "---\ntitle: Hello\nslug: hello\n---\n\nBody text here.\n"
    assert clean_markdown(md) == "Body text here."


def test_clean_markdown_skips_fences_images_and_describe_blocks():
    md = (
        "Intro line\n"
        "```python\n"
        "print('kept code line')\n"
        "```\n"
        "![alt text](image.png)\n"
        ":::describe\n"
        "A long description.\n"
        ":::"
    )
    cleaned = clean_markdown(md)
    assert "Intro line" in cleaned
    assert "kept code line" in cleaned  # fence lines skipped, code kept
    assert "alt text" not in cleaned
    assert "describe" not in cleaned


def test_clean_markdown_tables_and_links():
    md = (
        "| col A | col B |\n| --- | --- |\n| x | y |\n"
        "See [the docs](https://example.com) for more."
    )
    cleaned = clean_markdown(md)
    assert "|" not in cleaned
    assert "the docs" in cleaned
    assert "https://example.com" not in cleaned


def test_clean_markdown_strips_markup_chars():
    assert clean_markdown("# Title **bold** `code` > quote") == (
        "Title bold code quote"
    )


def test_chunk_words_windows_and_overlap():
    words = [f"w{i}" for i in range(50)]
    chunks = chunk_words(words, 20, 4)  # step 16; short tail dropped
    assert len(chunks) == 2
    assert chunks[0].split() == words[0:20]
    assert chunks[1].split() == words[16:36]


def test_chunk_words_drops_short_trailing_chunk():
    words = [f"w{i}" for i in range(45)]
    chunks = chunk_words(words, 20, 0)  # step 20: 20, 20, 5 -> last dropped
    assert len(chunks) == 2
    assert chunks[1].split() == words[20:40]


def test_chunk_words_keeps_single_short_chunk():
    words = ["only", "ten", "words", "here", "in", "this", "tiny", "post"]
    assert chunk_words(words, 200, 40) == [" ".join(words)]
