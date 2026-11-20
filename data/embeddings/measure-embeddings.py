#!/usr/bin/env python3
"""Measure embedding retrieval (all-MiniLM-L6-v2) against BM25/TF-IDF on the
sanoj.dev blog corpus — the sequel to the 2026-11-16 retriever post, which
left embeddings untested ("no embedding model was available in this run").

Corpus: the 16 posts in corpus/ next to this script (the 15 from the
chunking experiment plus the 2026-11-19 commit-messages post). The two posts
that quote the golden questions (the 2026-11-16 retriever post and the
2026-11-18 chunking post) are not in the corpus folder, so they cannot
match themselves. 22 questions: the 20 from 2026-11-16 plus 2 for the
jokeGPT post. Same chunking, same grading, same BM25 code as the original
measure-retrieval.py, so the numbers are comparable.

Rankers: BM25 (k1 1.5, b 0.75), embeddings (MiniLM-L6-v2, cosine),
hybrid (reciprocal rank fusion, k=60, of BM25 + embeddings). (A TF-IDF
scorer is defined below but not run here; it tied BM25 in the 2026-11-16
experiment.)

Run (needs sentence-transformers + torch, NOT in this repo's
requirements.txt — the API service stays model-free):
  pip install sentence-transformers
  python measure-embeddings.py
Writes results.json next to the script. Run from 2026-11-20.
"""
import json
import math
import re
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
BLOG = HERE / "corpus"
# The corpus folder already excludes the two posts that quote the golden
# questions (the 2026-11-16 retriever post and the 2026-11-18 chunking
# post), so there is nothing further to exclude.
EXCLUDE_FOLDERS = set()
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

STOP = set("""a an the and or but if then else when at by for with about against
between into through during before after above below to from up down in out on off
over under again further once here there all any both each few more most other some
such no nor not only own same so than too very can will just should now is am are was
were be been being have has had having do does did doing would could ought i you he
she it we they me him her us them my your his its our their this that these those
of as what which who whom how why when where is it s t d ll m re ve don aren isn wasn
weren hasn haven hadn doesn didn won wouldn shan shouldn can couldn mustn needn""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


def clean_markdown(md: str) -> str:
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
            s = s.replace("|", " ")
        lines.append(s)
    text = " ".join(lines)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_`#>]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_posts():
    posts = {}
    for path in sorted(BLOG.glob("*.md")):
        if path.parent.name in EXCLUDE_FOLDERS:
            continue
        text = clean_markdown(path.read_text())
        posts[path.stem] = {"title": path.stem, "text": text,
                            "words": len(text.split())}
    return posts


def chunk_words(words, size, overlap):
    step = max(1, size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        c = words[i:i + size]
        if len(c) < 20 and chunks:
            break
        chunks.append(" ".join(c))
        if i + size >= len(words):
            break
    return chunks


def build_index(posts, size, title_prepend=False):
    chunks = []
    for slug, post in posts.items():
        words = post["text"].split()
        if size == 0:
            texts = [post["text"]]
        else:
            texts = chunk_words(words, size, size // 5)
        for t in texts:
            full = (post["title"].replace("-", " ") + " " + t) if title_prepend else t
            chunks.append({"slug": slug, "text": t, "tokens": tokenize(full)})
    return chunks


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.n = len(chunks)
        self.docs = [c["tokens"] for c in chunks]
        self.lens = [len(d) for d in self.docs]
        self.avgdl = sum(self.lens) / max(1, self.n)
        df = {}
        for d in self.docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5))
                    for t, f in df.items()}

    def scores(self, query_tokens):
        qtf = {}
        for t in query_tokens:
            qtf[t] = qtf.get(t, 0) + 1
        out = np.zeros(self.n)
        for t, _ in qtf.items():
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, d in enumerate(self.docs):
                f = d.count(t)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
                    out[i] += idf * f * (self.k1 + 1) / denom
        return out


class TFIDF:
    def __init__(self, chunks):
        self.n = len(chunks)
        vocab = {}
        df = {}
        for c in chunks:
            for t in set(c["tokens"]):
                df[t] = df.get(t, 0) + 1
                if t not in vocab:
                    vocab[t] = len(vocab)
        self.vocab = vocab
        self.idf = np.array([math.log((self.n + 1) / (df[t] + 1)) + 1
                             for t in vocab])
        mat = np.zeros((self.n, len(vocab)))
        for i, c in enumerate(chunks):
            tf = {}
            for t in c["tokens"]:
                tf[t] = tf.get(t, 0) + 1
            for t, f in tf.items():
                mat[i, vocab[t]] = (1 + math.log(f)) * self.idf[vocab[t]]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self.mat = mat / np.where(norms == 0, 1, norms)

    def scores(self, query_tokens):
        q = np.zeros(len(self.vocab))
        for t in query_tokens:
            if t in self.vocab:
                q[self.vocab[t]] += 1
        n = np.linalg.norm(q)
        if n == 0:
            return np.zeros(self.n)
        return self.mat @ (q / n)


class Embeddings:
    """Dense retrieval with a sentence-transformer, cosine similarity."""

    def __init__(self, chunks, model):
        self.model = model
        self.n = len(chunks)
        t0 = time.time()
        self.mat = model.encode([c["text"] for c in chunks],
                                batch_size=32, normalize_embeddings=True,
                                show_progress_bar=False)
        self.encode_time = time.time() - t0

    def scores(self, query_text):
        q = self.model.encode([query_text], normalize_embeddings=True,
                              show_progress_bar=False)[0]
        return self.mat @ q


class HybridRRF:
    """Reciprocal rank fusion of two rankers (k=60). Scores are ranks."""

    def __init__(self, k=60):
        self.k = k

    def fuse(self, rank_lists):
        n = len(rank_lists[0])
        out = np.zeros(n)
        for ranks in rank_lists:
            for pos, i in enumerate(ranks, start=1):
                out[i] += 1.0 / (self.k + pos)
        return out


def rank(chunks, scorer, query, text_query=None):
    if isinstance(scorer, Embeddings):
        s = scorer.scores(text_query if text_query is not None else query)
    else:
        s = scorer.scores(tokenize(query))
    order = np.argsort(-s, kind="stable")
    return [(int(i), float(s[i])) for i in order]


def rank_indices(chunks, scorer, query, text_query=None):
    return [i for i, _ in rank(chunks, scorer, query, text_query)]


def is_gold_chunk(chunk, q, min_terms=2):
    if chunk["slug"] != q["gold"]:
        return False
    text = chunk["text"].lower()
    hits = sum(1 for t in q.get("answer_terms", []) if t.lower() in text)
    return hits >= min_terms


def evaluate_ranks(ranks_list, chunks, questions, gold_fn):
    ranks = []
    for q, r in zip(questions, ranks_list):
        gold_rank = None
        for pos, i in enumerate(r, start=1):
            if gold_fn(chunks[i], q):
                gold_rank = pos
                break
        ranks.append(gold_rank)
    arr = np.array([r if r is not None else 10 ** 9 for r in ranks], dtype=float)
    out = {}
    for k in (1, 3, 5, 10):
        out[f"recall@{k}"] = float((arr <= k).mean())
    out["mrr"] = float((1.0 / np.where(arr > 10 ** 8, np.inf, arr)).mean())
    out["median_gold_rank"] = float(np.median(np.where(arr > 10 ** 8, np.nan, arr)))
    out["gold_ranks"] = [None if r > 10 ** 8 else int(r) for r in arr]
    out["missed_at_5"] = int((arr > 5).sum())
    out["never_found"] = int((arr > 10 ** 8).sum())
    return out


def post_gold(chunk, q):
    return chunk["slug"] == q["gold"]


def main():
    posts = load_posts()
    data = json.loads((HERE / "questions.json").read_text())
    questions = data["questions"]
    total_words = sum(p["words"] for p in posts.values())
    print(f"corpus: {len(posts)} posts, {total_words:,} words, "
          f"{len(questions)} questions")
    print(f"excluded folders: {sorted(EXCLUDE_FOLDERS)}")

    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    load_time = time.time() - t0
    print(f"model {MODEL_NAME} loaded in {load_time:.1f}s")

    results = {"corpus": {"posts": len(posts), "words": total_words,
                          "files": sorted(posts),
                          "generated": "2026-11-20",
                          "excluded_folders": sorted(EXCLUDE_FOLDERS)},
               "questions": len(questions),
               "model": {"name": MODEL_NAME, "dim": int(model.get_sentence_embedding_dimension()),
                         "load_time_s": round(load_time, 1)},
               "experiments": {}}
    rrf = HybridRRF(k=60)

    for size in (200, 400):
        label = f"{size}w"
        chunks = build_index(posts, size)
        print(f"\n== chunk size {label}: {len(chunks)} chunks ==")
        # sanity: every question must have at least one gold chunk
        zero = [q["id"] for q in questions
                if not any(is_gold_chunk(c, q) for c in chunks)]
        print(f"  questions with zero gold chunks: {zero or 'none'}")

        bm25 = BM25(chunks)
        emb = Embeddings(chunks, model)
        t_q0 = time.time()
        bm25_ranks = [rank_indices(chunks, bm25, q["q"]) for q in questions]
        emb_ranks = [rank_indices(chunks, emb, q["q"], q["q"]) for q in questions]
        hyb_ranks = []
        for b, e in zip(bm25_ranks, emb_ranks):
            s = rrf.fuse([b, e])
            hyb_ranks.append([int(i) for i in np.argsort(-s, kind="stable")])
        q_time = (time.time() - t_q0) / len(questions)
        print(f"  emb index encode: {emb.encode_time:.1f}s for {len(chunks)} chunks; "
              f"avg query time (bm25+emb+rrf): {q_time*1000:.0f}ms")

        for name, ranks in (("bm25", bm25_ranks), ("emb", emb_ranks),
                            ("hybrid", hyb_ranks)):
            for level, fn, prefix in (("post", post_gold, ""),
                                      ("chunk", is_gold_chunk, "chunk-")):
                ev = evaluate_ranks(ranks, chunks, questions, fn)
                key = f"{prefix}{name}@{label}"
                results["experiments"][key] = {
                    "chunks": len(chunks), **{k: v for k, v in ev.items()}}
                print(f"  {key:18s} recall@1={ev['recall@1']:.2f} "
                      f"recall@5={ev['recall@5']:.2f} mrr={ev['mrr']:.3f} "
                      f"median_rank={ev['median_gold_rank']:.0f} "
                      f"missed@5={ev['missed_at_5']} never={ev['never_found']}")
        results["experiments"][f"emb@{label}"]["encode_time_s"] = round(emb.encode_time, 1)
        results["experiments"][f"emb@{label}"]["avg_query_ms"] = round(q_time * 1000, 1)

        # per-question chunk-level ranks for this size
        per_q = []
        for qi, q in enumerate(questions):
            def at(ranks):
                return next((pos for pos, i in enumerate(ranks, start=1)
                             if is_gold_chunk(chunks[i], q)), None)
            per_q.append({"id": q["id"], "gold": q["gold"],
                          "bm25_answer_chunk_rank": at(bm25_ranks[qi]),
                          "emb_answer_chunk_rank": at(emb_ranks[qi]),
                          "hybrid_answer_chunk_rank": at(hyb_ranks[qi])})
        results[f"per_question_{label}"] = per_q

    # Rewording experiment at 200w: same 6 items, three rankers
    chunks = build_index(posts, 200)
    bm25 = BM25(chunks)
    emb = Embeddings(chunks, model)
    rew = {"experiment": "rewording", "chunk_size": "200w", "items": []}
    print("\n== rewording (200w, answer-chunk rank) ==")
    for item in data["rewordings"]:
        q0 = next(q for q in questions if q["id"] == item["id"])
        rows = []
        for label, text in [("original", q0["q"])] + \
                [(f"variant{i+1}", v) for i, v in enumerate(item["variants"])]:
            b = rank_indices(chunks, bm25, text)
            e = rank_indices(chunks, emb, text, text)
            s = rrf.fuse([b, e])
            h = [int(i) for i in np.argsort(-s, kind="stable")]

            def at(ranks):
                return next((pos for pos, i in enumerate(ranks, start=1)
                             if is_gold_chunk(chunks[i], q0)), None)
            rows.append({"label": label, "text": text,
                         "bm25_answer_chunk_rank": at(b),
                         "emb_answer_chunk_rank": at(e),
                         "hybrid_answer_chunk_rank": at(h)})
        rew["items"].append({"id": item["id"], "gold": q0["gold"], "rows": rows})
        print(f"  Q{item['id']}: " + " | ".join(
            f"{row['label']} bm25={row['bm25_answer_chunk_rank']} "
            f"emb={row['emb_answer_chunk_rank']} hyb={row['hybrid_answer_chunk_rank']}"
            for row in rows))
    results["rewording"] = rew

    # Top-5 dump for the headline run (emb @ 200w), for failure analysis
    dump = []
    for qi, q in enumerate(questions):
        r = rank(chunks, emb, q["q"], q["q"])
        dump.append({"id": q["id"], "q": q["q"], "gold": q["gold"],
                     "top5": [{"slug": chunks[i]["slug"], "score": round(s, 3),
                               "preview": chunks[i]["text"][:140]}
                              for i, s in r[:5]]})
    results["top5_emb_200w"] = dump

    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
