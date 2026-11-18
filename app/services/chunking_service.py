"""Structure-aware chunking: five strategies from the 2026-11-18 post.

Faithful port of ``measure-chunking.py``. Every strategy is reproduced
byte-for-byte — window sizes, overlaps, the heading carry (``" — "``), the
sentence-boundary split, and the short-tail rules — so the API reproduces
the reference ``data/chunking/results-chunking.json`` exactly.

Strategies (``posts`` carry ``title``, ``text``, and ``sections`` — the
``(heading, cleaned body)`` pairs from the repository):

- ``fixed200``: 200-word windows, 40 words overlap (the 2026-11-16
  baseline).
- ``fixed400``: 400-word windows, 80 words overlap.
- ``fixed200+head``: 200-word windows on a 160-word step, each carrying
  the heading of the section its first word belongs to.
- ``section``: one chunk per markdown section, heading carried; long
  sections split at sentence boundaries, max ~400 words.
- ``para200``: whole paragraphs accumulated to ~200 words, heading
  carried, never split mid-paragraph.
"""

from app.services.text_service import (
    chunk_words,
    split_into_sentences,
    split_sentences,
    tokenize,
)

CHUNKING_STRATEGIES = (
    "fixed200",
    "fixed400",
    "fixed200+head",
    "section",
    "para200",
)


class ChunkingService:
    """Builds chunk indexes for a named chunking strategy."""

    @staticmethod
    def build_index(posts: dict[str, dict], strategy: str) -> list[dict]:
        """Chunk every post with ``strategy``. Returns ``[{slug, text,
        tokens}]`` — the same shape as the fixed-window index builder."""
        if strategy not in CHUNKING_STRATEGIES:
            raise ValueError(f"unknown chunking strategy: {strategy!r}")
        chunks: list[dict] = []
        for slug, post in posts.items():
            title = post["title"].replace("-", " ")
            if strategy == "fixed200":
                texts = chunk_words(post["text"].split(), 200, 40)
                for t in texts:
                    chunks.append(
                        {"slug": slug, "text": t, "tokens": tokenize(t)}
                    )
            elif strategy == "fixed400":
                texts = chunk_words(post["text"].split(), 400, 80)
                for t in texts:
                    chunks.append(
                        {"slug": slug, "text": t, "tokens": tokenize(t)}
                    )
            elif strategy == "fixed200+head":
                # Map each word offset to its section heading, then
                # window over the whole post on a 160-word step.
                words, heads = [], []
                for h, body in post["sections"]:
                    bw = body.split()
                    words.extend(bw)
                    heads.extend([h or title] * len(bw))
                for i in range(0, len(words), 160):
                    c = words[i : i + 200]
                    if len(c) < 20 and chunks and i > 0:
                        break
                    head = heads[i] if i < len(heads) else title
                    text = head + " — " + " ".join(c)
                    chunks.append(
                        {"slug": slug, "text": text, "tokens": tokenize(text)}
                    )
                    if i + 200 >= len(words):
                        break
            elif strategy == "section":
                for h, body in post["sections"]:
                    head = h or title
                    for piece in split_sentences(body, 400):
                        text = head + " — " + piece if piece else head
                        chunks.append(
                            {
                                "slug": slug,
                                "text": text,
                                "tokens": tokenize(text),
                            }
                        )
            elif strategy == "para200":
                for h, body in post["sections"]:
                    head = h or title
                    # Body is already cleaned to one line; split it back
                    # into sentences and accumulate ~200 words per chunk.
                    cur: list[str] = []
                    for sent in split_into_sentences(body):
                        sw = sent.split()
                        if len(cur) + len(sw) > 200 and cur:
                            text = head + " — " + " ".join(cur)
                            chunks.append(
                                {
                                    "slug": slug,
                                    "text": text,
                                    "tokens": tokenize(text),
                                }
                            )
                            cur = sw
                        else:
                            cur.extend(sw)
                    if cur:
                        text = head + " — " + " ".join(cur)
                        chunks.append(
                            {
                                "slug": slug,
                                "text": text,
                                "tokens": tokenize(text),
                            }
                        )
        return chunks
