"""Text processing: tokenization, markdown cleaning, chunking.

Faithful port of the original experiment script. The stopword list, the
``[a-z0-9]+`` token pattern, the markdown-cleaning rules, and the chunking
windows are byte-for-byte the originals, so the API reproduces the
reference ``results.json`` exactly.
"""

import re

# Exact stopword list from the original experiment script.
STOP = set(
    """a an the and or but if then else when at by for with about against
between into through during before after above below to from up down in out
on off over under again further once here there all any both each few more
most other some such no nor not only own same so than too very can will
just should now is am are was were be been being have has had having do
does did doing would could ought i you he she it we they me him her us
them my your his its our their this that these those of as what which
who whom how why when where is it s t d ll m re ve don aren isn wasn
weren hasn haven hadn doesn didn won wouldn shan shouldn can couldn mustn
needn""".split()
)

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase ``[a-z0-9]+`` tokens, minus stopwords."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


def clean_markdown(md: str) -> str:
    """Strip frontmatter and markdown syntax, keeping the plain words.

    Rules (original): drop the ``---`` frontmatter block; skip fence lines
    (`` ``` ``), image lines (``![``), and ``:::`` block markers; turn
    table pipes into spaces; replace links with their anchor text; strip
    ``* _ ` # >`` runs; collapse whitespace.
    """
    if md.startswith("---"):
        end = md.find("\n---", 3)
        if end != -1:
            md = md[end + 4:]
    lines = []
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("```"):
            continue
        if s.startswith("!["):
            continue
        if s.startswith(":::"):
            continue
        if s.startswith("|") and s.endswith("|"):
            # keep table content as plain words
            s = s.replace("|", " ")
        lines.append(s)
    text = " ".join(lines)
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)", r"\1", text
    )  # links -> anchor text
    text = re.sub(r"[*_`#>]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_words(words: list[str], size: int, overlap: int) -> list[str]:
    """Sliding windows of ``size`` words with ``overlap`` words shared.

    The step is ``size - overlap`` (at least 1). A trailing chunk under
    20 words is dropped — unless it is the only chunk.
    """
    step = max(1, size - overlap)
    chunks: list[str] = []
    for i in range(0, len(words), step):
        chunk = words[i : i + size]
        if len(chunk) < 20 and chunks:
            break
        chunks.append(" ".join(chunk))
        if i + size >= len(words):
            break
    return chunks
