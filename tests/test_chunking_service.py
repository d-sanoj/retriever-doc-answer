"""Unit tests for section splitting, sentence splitting, and the five
chunking strategies (port of measure-chunking.py's chunkers)."""

import pytest

from app.services.chunking_service import (
    CHUNKING_STRATEGIES,
    ChunkingService,
)
from app.services.text_service import (
    clean_markdown,
    split_into_sentences,
    split_sections,
    split_sentences,
)

# One sample post: 100-word intro, 100-word first section, 500-word
# second section (one-word sentences, so sentence boundaries are exact).
INTRO = " ".join(
    ["Intro", "paragraph", "before", "any", "heading."]
    + [f"filler{i}." for i in range(95)]
)
FIRST = " ".join(
    ["This", "is", "the", "first", "section."]
    + [f"alpha{i}." for i in range(95)]
)
SECOND = " ".join(f"word{i}." for i in range(500))

SAMPLE_MD = (
    "---\ntitle: Sample Post\nslug: sample-post\n---\n\n"
    f"{INTRO}\n\n# First Section\n{FIRST}\n\n# Second Section\n{SECOND}\n"
)


def _posts(md: str = SAMPLE_MD) -> dict:
    """One-post corpus in repository shape (sections cleaned)."""
    return {
        "sample-post": {
            "title": "sample-post",
            "text": clean_markdown(md),
            "words": len(clean_markdown(md).split()),
            "sections": [
                (h, clean_markdown(b)) for h, b in split_sections(md)
            ],
        }
    }


def test_split_sections_first_gets_none_heading():
    sections = split_sections(SAMPLE_MD)
    assert len(sections) == 3
    assert sections[0][0] is None
    assert "Intro paragraph" in sections[0][1]
    assert sections[1][0] == "First Section"
    assert sections[2][0] == "Second Section"


def test_split_sections_no_headings_is_one_section():
    sections = split_sections("Just body text.\nMore body.\n")
    assert sections == [(None, "Just body text.\nMore body.")]


def test_split_into_sentences():
    assert split_into_sentences("One. Two! Three? Four.") == [
        "One.",
        "Two!",
        "Three?",
        "Four.",
    ]


def test_split_sentences_caps_at_max_words():
    text = " ".join(f"Sentence {i} here." for i in range(10))
    pieces = split_sentences(text, 10)
    assert all(len(p.split()) <= 10 for p in pieces)
    assert " ".join(pieces) == text
    assert len(pieces) > 1


def test_split_sentences_short_text_unchanged():
    assert split_sentences("Short text.", 400) == ["Short text."]


def test_all_five_strategies_known():
    assert CHUNKING_STRATEGIES == (
        "fixed200",
        "fixed400",
        "fixed200+head",
        "section",
        "para200",
    )


def test_build_index_unknown_strategy():
    with pytest.raises(ValueError, match="unknown chunking strategy"):
        ChunkingService.build_index(_posts(), "sliding")


def test_fixed200_windows_and_overlap():
    chunks = ChunkingService.build_index(_posts(), "fixed200")
    texts = [c["text"] for c in chunks]
    assert len(texts) == 5  # 704 words, step 160
    assert texts[0].split()[:3] == ["Intro", "paragraph", "before"]
    assert all(len(t.split()) <= 200 for t in texts)
    # 40-word overlap: second chunk starts 160 words in.
    assert texts[1].split()[0] == texts[0].split()[160]
    assert all(c["slug"] == "sample-post" for c in chunks)
    assert all(c["tokens"] for c in chunks)


def test_fixed400_is_wider_than_fixed200():
    chunks200 = ChunkingService.build_index(_posts(), "fixed200")
    chunks400 = ChunkingService.build_index(_posts(), "fixed400")
    assert len(chunks400) == 2
    assert len(chunks400) < len(chunks200)
    assert len(chunks400[0]["text"].split()) == 400


def test_fixed200_head_carries_section_heading():
    chunks = ChunkingService.build_index(_posts(), "fixed200+head")
    texts = [c["text"] for c in chunks]
    assert texts[0].startswith("sample post — ")
    assert any(t.startswith("First Section — ") for t in texts)
    assert any(t.startswith("Second Section — ") for t in texts)


def test_section_one_chunk_per_section_with_heading():
    chunks = ChunkingService.build_index(_posts(), "section")
    texts = [c["text"] for c in chunks]
    assert texts[0].startswith("sample post — Intro paragraph")
    assert any(t.startswith("First Section — ") for t in texts)
    assert any(t.startswith("Second Section — ") for t in texts)


def test_section_splits_long_section_at_sentences():
    chunks = ChunkingService.build_index(_posts(), "section")
    second = [c for c in chunks if c["text"].startswith("Second Section — ")]
    # 500 words in the second section -> split at ~400 words.
    assert len(second) == 2
    # 400-word piece + 2 heading words + the em dash.
    assert all(len(c["text"].split()) <= 403 for c in second)
    # Nothing is lost: the section's words are all in the two chunks.
    joined = " ".join(c["text"] for c in second)
    assert "word0." in joined and "word499." in joined


def test_section_empty_body_is_just_the_heading():
    posts = {
        "empty": {
            "title": "empty",
            "text": "",
            "words": 0,
            "sections": [("Lonely Heading", "")],
        }
    }
    chunks = ChunkingService.build_index(posts, "section")
    assert chunks == [
        {
            "slug": "empty",
            "text": "Lonely Heading",
            "tokens": ["lonely", "heading"],
        }
    ]


def test_para200_accumulates_sentences_to_200_words():
    chunks = ChunkingService.build_index(_posts(), "para200")
    texts = [c["text"] for c in chunks]
    # Intro and first section (100 words each) stay whole; the 500-word
    # second section accumulates to 200 + 200 + 100 words.
    assert len(texts) == 5
    second = [t for t in texts if t.startswith("Second Section — ")]
    assert len(second) == 3
    # 200-word piece + 2 heading words + the em dash.
    assert all(len(t.split()) <= 203 for t in texts)
    joined = " ".join(second)
    assert "word0." in joined and "word499." in joined
    assert any(t.startswith("First Section — ") for t in texts)
