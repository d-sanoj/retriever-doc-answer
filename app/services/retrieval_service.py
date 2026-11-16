"""Retrieval: chunk-index building and ranking.

Index construction is cached per (chunk size, title prepend, retriever), so
repeated searches and evaluations reuse the same index and scorer.
"""

from functools import lru_cache

import numpy as np

from app.core.config import settings
from app.repositories.corpus_repository import CorpusRepository
from app.services.bm25_service import BM25
from app.services.text_service import chunk_words, tokenize
from app.services.tfidf_service import TFIDF

Chunk = dict
Scorer = BM25 | TFIDF


class RetrievalService:
    """Builds chunk indexes and ranks them for a query."""

    def __init__(self, repository: CorpusRepository | None = None) -> None:
        self.repository = repository or CorpusRepository()

    @staticmethod
    def build_index(
        posts: dict[str, dict], size: int, title_prepend: bool = False
    ) -> list[Chunk]:
        """Chunk every post. Returns ``[{slug, text, tokens}]``.

        ``size == 0`` means one chunk per post. Otherwise windows of
        ``size`` words with ``size // 5`` overlap. With ``title_prepend``,
        the post title (slug with hyphens as spaces) is prepended to the
        tokens only — the chunk text shown to the user is unchanged.
        """
        chunks: list[Chunk] = []
        for slug, post in posts.items():
            words = post["text"].split()
            if size == 0:  # whole post
                texts = [post["text"]]
            else:
                texts = chunk_words(words, size, size // 5)
            for t in texts:
                full = (
                    (post["title"].replace("-", " ") + " " + t)
                    if title_prepend
                    else t
                )
                chunks.append(
                    {"slug": slug, "text": t, "tokens": tokenize(full)}
                )
        return chunks

    @lru_cache(maxsize=64)
    def _index(
        self, chunk_size: int, title_prepend: bool, retriever: str
    ) -> tuple[list[Chunk], Scorer]:
        posts = self.repository.load_posts()
        chunks = self.build_index(posts, chunk_size, title_prepend)
        if retriever == "bm25":
            scorer: Scorer = BM25(
                chunks, k1=settings.bm25_k1, b=settings.bm25_b
            )
        elif retriever == "tfidf":
            scorer = TFIDF(chunks)
        else:
            raise ValueError(f"unknown retriever: {retriever!r}")
        return chunks, scorer

    def get_index(
        self, chunk_size: int, title_prepend: bool, retriever: str
    ) -> tuple[list[Chunk], Scorer]:
        """Return the cached (chunks, scorer) for this configuration."""
        return self._index(chunk_size, title_prepend, retriever)

    @staticmethod
    def rank(
        chunks: list[Chunk], scorer: Scorer, query: str
    ) -> list[tuple[int, float]]:
        """Rank all chunks for a query: ``[(chunk_index, score)]``.

        Stable descending sort, so ties keep chunk order — as in the original.
        """
        query_tokens = tokenize(query)
        scores = scorer.scores(query_tokens)
        order = np.argsort(-scores, kind="stable")
        return [(int(i), float(scores[i])) for i in order]

    def search(
        self,
        query: str,
        retriever: str = "bm25",
        chunk_size: int = 200,
        top_k: int = 5,
        title_prepend: bool = False,
    ) -> tuple[list[dict], int]:
        """Search the corpus; returns (ranked chunks, total chunk count)."""
        chunks, scorer = self.get_index(chunk_size, title_prepend, retriever)
        ranked = self.rank(chunks, scorer, query)
        results = [
            {
                "rank": pos,
                "slug": chunks[i]["slug"],
                "score": score,
                "preview": chunks[i]["text"][:140],
            }
            for pos, (i, score) in enumerate(ranked[:top_k], start=1)
        ]
        return results, len(chunks)
