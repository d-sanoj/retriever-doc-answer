"""Evaluation: grading, metrics, and the full experiment suite.

This is a faithful port of the original experiment script's ``evaluate``,
``chunk_evaluate``, ``main``, and rewording sections, so the API reproduces
``results.json`` exactly. Metric keys keep the original names (``recall@5``
etc.) in the full-experiment output.
"""

import math
from datetime import date

import numpy as np

from app.models.corpus import Question
from app.repositories.corpus_repository import CorpusRepository
from app.services.chunking_service import CHUNKING_STRATEGIES
from app.services.retrieval_service import Chunk, RetrievalService, Scorer

EXCLUDED_SLUG = "retriever-finds-the-document-not-the-answer"
CHUNK_SIZES = (100, 200, 400, 800, 0)
CHUNK_LEVEL_SIZES = (100, 200, 400)
# Blog folders excluded from the chunking snapshot's corpus, as recorded
# in data/chunking/results-chunking.json: the 2026-11-16 retriever post
# (it quotes the golden questions) and the chunking post itself.
CHUNKING_EXCLUDED_FOLDERS = ["blog-15-11-16-2026", "blog-17-11-18-2026"]


def _metrics(ranks: list[int | None]) -> dict:
    """Recall@1/3/5/10, MRR, median gold rank, missed@5, never_found.

    ``ranks`` holds the 1-based gold rank per question (``None`` = not found).
    Identical to the original script's computation.
    """
    arr = np.array([r if r is not None else 10**9 for r in ranks], dtype=float)
    out: dict = {}
    for k in (1, 3, 5, 10):
        out[f"recall@{k}"] = float((arr <= k).mean())
    out["mrr"] = float((1.0 / np.where(arr > 10**8, np.inf, arr)).mean())
    median = float(np.median(np.where(arr > 10**8, np.nan, arr)))
    out["median_gold_rank"] = None if math.isnan(median) else median
    out["gold_ranks"] = [None if r > 10**8 else int(r) for r in arr]
    out["missed_at_5"] = int((arr > 5).sum())
    out["never_found"] = int((arr > 10**8).sum())
    return out


def is_gold_chunk(
    chunk: Chunk, question: Question, min_terms: int = 2
) -> bool:
    """Chunk-level gold: from the gold post AND contains >= min_terms terms.

    Term presence is substring matching on the chunk text (not the title
    prepend) — as in the original.
    """
    if chunk["slug"] != question.gold:
        return False
    text = chunk["text"].lower()
    hits = sum(1 for t in question.answer_terms if t.lower() in text)
    return hits >= min_terms


class EvaluationService:
    """Runs evaluations and the full experiment suite."""

    def __init__(
        self,
        repository: CorpusRepository | None = None,
        retrieval: RetrievalService | None = None,
    ) -> None:
        self.repository = repository or CorpusRepository()
        self.retrieval = retrieval or RetrievalService(self.repository)

    def _gold_ranks(
        self,
        chunks: list[Chunk],
        scorer: Scorer,
        questions: list[Question],
        *,
        chunk_level: bool,
        min_terms: int = 2,
    ) -> list[int | None]:
        ranks: list[int | None] = []
        for q in questions:
            ranked = self.retrieval.rank(chunks, scorer, q.q)
            gold_rank = None
            for pos, (i, _) in enumerate(ranked, start=1):
                if chunk_level:
                    if is_gold_chunk(chunks[i], q, min_terms):
                        gold_rank = pos
                        break
                elif chunks[i]["slug"] == q.gold:
                    gold_rank = pos
                    break
            ranks.append(gold_rank)
        return ranks

    def evaluate(
        self,
        retriever: str = "bm25",
        chunk_size: int = 200,
        grading: str = "post",
        min_terms: int = 2,
        title_prepend: bool = False,
        strategy: str | None = None,
    ) -> dict:
        """Evaluate one configuration; returns metrics plus chunk count.

        With ``strategy`` set, the index is built by that chunking
        strategy instead of fixed windows; ``chunk_size`` and
        ``title_prepend`` are then ignored.
        """
        chunks, scorer = self.retrieval.get_index(
            chunk_size, title_prepend, retriever, strategy
        )
        questions = self.repository.load_questions()
        ranks = self._gold_ranks(
            chunks,
            scorer,
            questions,
            chunk_level=(grading == "chunk"),
            min_terms=min_terms,
        )
        return {"chunks": len(chunks), **_metrics(ranks)}

    def _top5(
        self,
        chunks: list[Chunk],
        scorer: Scorer,
        query: str,
        preview_len: int = 140,
    ) -> list[dict]:
        ranked = self.retrieval.rank(chunks, scorer, query)
        return [
            {
                "slug": chunks[i]["slug"],
                "score": round(score, 2),
                "preview": chunks[i]["text"][:preview_len],
            }
            for i, score in ranked[:5]
        ]

    def _per_question(
        self, chunks: list[Chunk], scorer: Scorer, questions: list[Question]
    ) -> list[dict]:
        rows = []
        for q in questions:
            ranked = self.retrieval.rank(chunks, scorer, q.q)
            post_rank = next(
                (
                    pos
                    for pos, (i, _) in enumerate(ranked, start=1)
                    if chunks[i]["slug"] == q.gold
                ),
                None,
            )
            chunk_rank = next(
                (
                    pos
                    for pos, (i, _) in enumerate(ranked, start=1)
                    if is_gold_chunk(chunks[i], q)
                ),
                None,
            )
            rows.append(
                {
                    "id": q.id,
                    "gold": q.gold,
                    "post_rank": post_rank,
                    "answer_chunk_rank": chunk_rank,
                }
            )
        return rows

    def _per_question_200w(
        self, chunks: list[Chunk], scorer: Scorer, questions: list[Question]
    ) -> list[dict]:
        return self._per_question(chunks, scorer, questions)

    def run_rewording(
        self,
        question_ids: list[int] | None = None,
        chunk_size: int = 200,
        retriever: str = "bm25",
    ) -> dict:
        """Rewording experiment: original vs variants for each question."""
        chunks, scorer = self.retrieval.get_index(chunk_size, False, retriever)
        questions = self.repository.load_questions()
        by_id = {q.id: q for q in questions}
        items = []
        for item in self.repository.load_rewordings():
            if question_ids is not None and item.id not in question_ids:
                continue
            q0 = by_id[item.id]
            rows = []
            for label, text in [("original", q0.q)] + [
                (f"variant{i + 1}", v) for i, v in enumerate(item.variants)
            ]:
                ranked = self.retrieval.rank(chunks, scorer, text)
                gold_rank = next(
                    (
                        pos
                        for pos, (i, _) in enumerate(ranked, start=1)
                        if chunks[i]["slug"] == q0.gold
                    ),
                    None,
                )
                chunk_rank = next(
                    (
                        pos
                        for pos, (i, _) in enumerate(ranked, start=1)
                        if is_gold_chunk(chunks[i], q0)
                    ),
                    None,
                )
                rows.append(
                    {
                        "label": label,
                        "text": text,
                        "gold_rank": gold_rank,
                        "answer_chunk_rank": chunk_rank,
                        "top5": self._top5(
                            chunks, scorer, text, preview_len=110
                        ),
                    }
                )
            items.append({"id": item.id, "gold": q0.gold, "rows": rows})
        return {
            "experiment": "rewording",
            "retriever": f"{retriever}@{chunk_size}w",
            "items": items,
        }

    def run_full_experiment(self) -> dict:
        """Run the full suite; return the results.json-shaped object."""
        posts = self.repository.load_posts()
        questions = self.repository.load_questions()
        total_words = sum(p["words"] for p in posts.values())

        results: dict = {
            "corpus": {
                "posts": len(posts),
                "words": total_words,
                "files": sorted(posts),
                "generated": date.today().isoformat(),
                "excluded": [EXCLUDED_SLUG],
            },
            "questions": len(questions),
            "experiments": {},
        }
        experiments = results["experiments"]

        for size in CHUNK_SIZES:
            label = "whole-post" if size == 0 else f"{size}w"
            chunks, _ = self.retrieval.get_index(size, False, "bm25")
            for name in ("bm25", "tfidf"):
                _, scorer = self.retrieval.get_index(size, False, name)
                ranks = self._gold_ranks(
                    chunks, scorer, questions, chunk_level=False
                )
                entry = {"chunks": len(chunks), **_metrics(ranks)}
                # Post-level entries carry no never_found key in results.json.
                entry.pop("never_found")
                experiments[f"{name}@{label}"] = entry
            if size == 200:
                chunks_t, scorer_t = self.retrieval.get_index(
                    size, True, "bm25"
                )
                ranks = self._gold_ranks(
                    chunks_t, scorer_t, questions, chunk_level=False
                )
                entry = {"chunks": len(chunks_t), **_metrics(ranks)}
                entry.pop("never_found")
                experiments["bm25+title@200w"] = entry

        # Chunk-level grading: does the chunk WITH the answer reach the top k?
        for size in CHUNK_LEVEL_SIZES:
            label = f"{size}w"
            chunks, _ = self.retrieval.get_index(size, False, "bm25")
            for name in ("bm25", "tfidf"):
                _, scorer = self.retrieval.get_index(size, False, name)
                metrics = self._gold_ranks(
                    chunks, scorer, questions, chunk_level=True
                )
                experiments[f"chunk-{name}@{label}"] = {
                    "chunks": len(chunks),
                    **_metrics(metrics),
                }
        chunks_t, scorer_t = self.retrieval.get_index(200, True, "bm25")
        metrics = self._gold_ranks(
            chunks_t, scorer_t, questions, chunk_level=True
        )
        experiments["chunk-bm25+title@200w"] = {
            "chunks": len(chunks_t),
            **_metrics(metrics),
        }
        # Loose gold definition (>=1 answer term): the generous reading.
        chunks, scorer = self.retrieval.get_index(200, False, "bm25")
        metrics = self._gold_ranks(
            chunks, scorer, questions, chunk_level=True, min_terms=1
        )
        experiments["chunk-loose>=1-bm25@200w"] = {
            "chunks": len(chunks),
            **_metrics(metrics),
        }

        # Per-question ranks for the headline run (BM25 @ 200w).
        results["per_question_200w"] = self._per_question_200w(
            chunks, scorer, questions
        )

        # Rewording experiment (BM25 @ 200w).
        results["rewording"] = self.run_rewording(
            question_ids=None, chunk_size=200, retriever="bm25"
        )

        # Top-5 dump for the headline run, for failure analysis.
        dump = []
        for q in questions:
            ranked = self.retrieval.rank(chunks, scorer, q.q)
            gold_rank = next(
                (
                    pos
                    for pos, (i, _) in enumerate(ranked, start=1)
                    if chunks[i]["slug"] == q.gold
                ),
                None,
            )
            dump.append(
                {
                    "id": q.id,
                    "q": q.q,
                    "gold": q.gold,
                    "gold_rank": gold_rank,
                    "top5": self._top5(chunks, scorer, q.q),
                }
            )
        results["top5_bm25_200w"] = dump
        return results

    def run_chunking_experiment(self) -> dict:
        """Run the chunking comparison; return the results-chunking.json
        shaped object.

        Port of ``measure-chunking.py`` (2026-11-18 post): all five
        strategies, BM25 only, post-level and chunk-level (strict, >=2
        answer terms) grading, per-question ranks, the fixed200-vs-section
        rewording comparison, and the top-5 dump for the section run.
        Runs on this service's repository — the chunking snapshot
        (15 posts, 22 questions) when wired to it in ``main.py``.
        """
        posts = self.repository.load_posts_with_sections()
        questions = self.repository.load_questions()
        total_words = sum(p["words"] for p in posts.values())

        results: dict = {
            "corpus": {
                "posts": len(posts),
                "words": total_words,
                "files": sorted(posts),
                "generated": date.today().isoformat(),
                "excluded_folders": sorted(CHUNKING_EXCLUDED_FOLDERS),
            },
            "questions": len(questions),
            "experiments": {},
        }
        experiments = results["experiments"]

        per_question: dict[str, list[dict]] = {}
        for strategy in CHUNKING_STRATEGIES:
            chunks, scorer = self.retrieval.get_index(
                200, False, "bm25", strategy=strategy
            )
            sizes = [len(c["text"].split()) for c in chunks]
            median_words = int(np.median(sizes))
            for level, chunk_level, prefix in (
                ("post", False, ""),
                ("chunk", True, "chunk-"),
            ):
                ranks = self._gold_ranks(
                    chunks, scorer, questions, chunk_level=chunk_level
                )
                experiments[f"{prefix}bm25@{strategy}"] = {
                    "chunks": len(chunks),
                    "median_chunk_words": median_words,
                    **_metrics(ranks),
                }
            per_question[strategy] = self._per_question(
                chunks, scorer, questions
            )
        results["per_question"] = per_question

        # Rewording: fixed200 vs section, answer-chunk rank only.
        results["rewording"] = self._chunking_rewording(questions)

        # Top-5 dump for the headline run (section), for failure analysis.
        chunks, scorer = self.retrieval.get_index(
            200, False, "bm25", strategy="section"
        )
        dump = []
        for q in questions:
            dump.append(
                {
                    "id": q.id,
                    "q": q.q,
                    "gold": q.gold,
                    "top5": self._top5(chunks, scorer, q.q),
                }
            )
        results["top5_bm25_section"] = dump
        return results

    def _chunking_rewording(self, questions: list[Question]) -> dict:
        """Rewording experiment, fixed200 vs section (answer-chunk rank)."""
        indexes: dict[str, tuple[list[Chunk], Scorer]] = {}
        for strategy in ("fixed200", "section"):
            indexes[strategy] = self.retrieval.get_index(
                200, False, "bm25", strategy=strategy
            )
        by_id = {q.id: q for q in questions}
        items = []
        for item in self.repository.load_rewordings():
            q0 = by_id[item.id]
            rows = []
            for label, text in [("original", q0.q)] + [
                (f"variant{i + 1}", v) for i, v in enumerate(item.variants)
            ]:
                row: dict = {"label": label, "text": text}
                for strategy in ("fixed200", "section"):
                    chunks, scorer = indexes[strategy]
                    ranked = self.retrieval.rank(chunks, scorer, text)
                    row[strategy] = next(
                        (
                            pos
                            for pos, (i, _) in enumerate(ranked, start=1)
                            if is_gold_chunk(chunks[i], q0)
                        ),
                        None,
                    )
                rows.append(row)
            items.append({"id": item.id, "gold": q0.gold, "rows": rows})
        return {"experiment": "rewording", "items": items}
