#!/usr/bin/env python3
"""What does a cosine similarity score actually mean? Measure it.

Model: sentence-transformers/all-MiniLM-L6-v2 (384 dims, 22.7M params).
Corpus: the 16 sanoj.dev blog posts in ../embeddings/corpus (the same corpus
as the 2026-11-20 embeddings-vs-BM25 experiment), 1,749 sentences and 229
non-overlapping 200-word chunks; 22 golden questions from
../embeddings/questions.json.

What it measures:
  1. Cosine of sentence pairs from different posts (unrelated topics),
     same post, hand-written paraphrases, and near-duplicates.
  2. The shared component: the norm of the mean vector, and how much of
     its energy sits in its top dimensions.
  3. The same distributions after mean-centering (subtract the corpus
     mean vector, renormalize).
  4. For each golden question: the best chunk that actually answers it
     (strict rule: right post, >= 2 answer terms) vs the best chunk that
     does not — and the background level of a question against all chunks.

Run (needs sentence-transformers + torch, NOT in this repo's
requirements.txt — the API service stays model-free):
  pip install sentence-transformers
  python measure-cosine.py
Writes results.json and plot-data.json next to the script.
Run from 2026-11-26. Numbers quoted in the sanoj.dev post
"What 0.7 actually means" (2026-11-26) come from this run.
"""
import json
import os
import re
import random
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CORPUS = Path(os.environ.get("CORPUS_DIR", HERE / ".." / "embeddings" / "corpus"))
QPATH = Path(os.environ.get("QUESTIONS_JSON", HERE / ".." / "embeddings" / "questions.json"))
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RNG = random.Random(20261126)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def clean_markdown(md: str) -> str:
    if md.startswith("---"):
        end = md.find("\n---", 3)
        if end != -1:
            md = md[end + 4:]
    md = re.sub(r"```.*?```", " ", md, flags=re.S)
    md = re.sub(r"!\[.*?\]\(.*?\)", " ", md)
    md = re.sub(r"\[([^\]]*)\]\(.*?\)", r"\1", md)
    md = re.sub(r"[#>*`_]", " ", md)
    md = re.sub(r"\s+", " ", md)
    return md.strip()


def sentences_of(post_path: Path):
    out = []
    for s in SENT_SPLIT.split(clean_markdown(post_path.read_text())):
        s = s.strip()
        if 6 <= len(s.split()) <= 40 and s[0].isupper():
            out.append(s)
    return out


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def stats(vals):
    v = np.array(sorted(vals))
    return {
        "n": len(v),
        "mean": round(float(v.mean()), 4),
        "median": round(float(np.median(v)), 4),
        "min": round(float(v.min()), 4),
        "p10": round(float(np.percentile(v, 10)), 4),
        "p90": round(float(np.percentile(v, 90)), 4),
        "p99": round(float(np.percentile(v, 99)), 4),
        "max": round(float(v.max()), 4),
        "frac_above_0.5": round(float((v > 0.5).mean()), 4),
        "frac_above_0.7": round(float((v > 0.7).mean()), 4),
        "frac_above_0.8": round(float((v > 0.8).mean()), 4),
    }


def norm_stats(vecs):
    norms = np.linalg.norm(vecs, axis=1)
    return {
        "mean": round(float(norms.mean()), 4),
        "min": round(float(norms.min()), 4),
        "max": round(float(norms.max()), 4),
    }


def top_dim_share(mean_vec):
    energy = np.sort(mean_vec ** 2)[::-1]
    total = float(energy.sum())
    return {f"top_{k}": round(float(energy[:k].sum() / total), 4)
            for k in (1, 3, 5, 10)}


def hist(vals, bins):
    counts, edges = np.histogram(vals, bins=bins)
    return {"edges": [round(float(e), 3) for e in edges],
            "counts": [int(c) for c in counts]}


def main():
    from sentence_transformers import SentenceTransformer

    posts = sorted(CORPUS.glob("*.md"))
    per_post = {p.stem: sentences_of(p) for p in posts}
    all_sents = [(stem, s) for stem, ss in per_post.items() for s in ss]
    print("sentences:", len(all_sents))

    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print("model loaded in", round(time.time() - t0, 1), "s")

    texts = [s for _, s in all_sents]
    t0 = time.time()
    vecs = np.array(model.encode(texts, batch_size=64, show_progress_bar=False))
    print("encoded", len(texts), "in", round(time.time() - t0, 1), "s")

    post_indices = {}
    for gi, (stem, _) in enumerate(all_sents):
        post_indices.setdefault(stem, []).append(gi)

    cross_idx = []
    while len(cross_idx) < 600:
        i, j = RNG.randrange(len(all_sents)), RNG.randrange(len(all_sents))
        if all_sents[i][0] != all_sents[j][0]:
            cross_idx.append((i, j))
    cross = [cos(vecs[i], vecs[j]) for i, j in cross_idx]

    same_idx = []
    while len(same_idx) < 600:
        idxs = post_indices[RNG.choice(list(post_indices))]
        i, j = RNG.choice(idxs), RNG.choice(idxs)
        if i != j:
            same_idx.append((i, j))
    same = [cos(vecs[i], vecs[j]) for i, j in same_idx]

    # Hand-written paraphrases (the author wrote these; they are examples,
    # not corpus data) and near-duplicates (same sentence, casing and
    # punctuation changed).
    pairs = [
        ("The model learns by adjusting its weights a little after each mistake.",
         "After every error, the model nudges its weights slightly in the right direction."),
        ("Chunking by section beats fixed word windows for retrieval.",
         "Splitting text at section boundaries works better than fixed-size chunks."),
        ("Flaky tests pass sometimes and fail other times without any code change.",
         "A flaky test gives different results on different runs even though nothing changed."),
        ("The agent edited a file nobody asked it to touch.",
         "The agent changed a file it was never told to modify."),
        ("Cosine similarity measures the angle between two vectors.",
         "Cosine similarity tells you how close two vectors point in the same direction."),
    ]
    para = [cos(*model.encode([a, b], show_progress_bar=False)) for a, b in pairs]
    near_dup = []
    for stem, s in RNG.sample(all_sents, 20):
        v1 = model.encode([s], show_progress_bar=False)[0]
        v2 = model.encode([s.lower().replace(".", " !")], show_progress_bar=False)[0]
        near_dup.append(cos(v1, v2))

    # Shared component of sentence vectors.
    mean_vec = vecs.mean(axis=0)
    sent_shared = {
        "mean_vector_norm": round(float(np.linalg.norm(mean_vec)), 4),
        "mean_vector_norm_squared": round(float(np.linalg.norm(mean_vec) ** 2), 4),
        "top_dimension_share": top_dim_share(mean_vec),
    }

    # Mean-centering.
    centered = vecs - mean_vec
    centered = centered / np.linalg.norm(centered, axis=1, keepdims=True)
    cross_c = [cos(centered[i], centered[j]) for i, j in cross_idx]
    same_c = [cos(centered[i], centered[j]) for i, j in same_idx]

    # Question-to-chunk: gold vs best non-gold, and the background level.
    questions = json.loads(QPATH.read_text())["questions"]
    chunk_texts, chunk_posts = [], []
    for p in posts:
        words = clean_markdown(p.read_text()).split()
        for i in range(0, len(words), 200):
            chunk_texts.append(" ".join(words[i:i + 200]))
            chunk_posts.append(p.stem)
    cvecs = np.array(model.encode(chunk_texts, batch_size=64, show_progress_bar=False))
    cmean = cvecs.mean(axis=0)
    chunk_shared = {
        "mean_vector_norm": round(float(np.linalg.norm(cmean)), 4),
        "mean_vector_norm_squared": round(float(np.linalg.norm(cmean) ** 2), 4),
        "top_dimension_share": top_dim_share(cmean),
    }
    qvecs = np.array(model.encode([q["q"] for q in questions], show_progress_bar=False))
    sims = qvecs @ cvecs.T / (
        np.linalg.norm(qvecs, axis=1, keepdims=True)
        * np.linalg.norm(cvecs, axis=1, keepdims=True).T
    )
    gold_cos, nongold_cos, gold_ranks = [], [], []
    for qi, q in enumerate(questions):
        terms = [t.lower() for t in q["answer_terms"]]
        gold_idx = [ci for ci in range(len(chunk_texts))
                    if chunk_posts[ci] == q["gold"]
                    and sum(1 for t in terms if t in chunk_texts[ci].lower()) >= 2]
        best_gold = max(gold_idx, key=lambda ci: sims[qi][ci])
        gold_cos.append(round(float(sims[qi][best_gold]), 4))
        gold_ranks.append(int((np.argsort(-sims[qi]) == best_gold).nonzero()[0][0]) + 1)
        order = np.argsort(-sims[qi])
        for ci in order:
            if ci not in gold_idx:
                nongold_cos.append(round(float(sims[qi][ci]), 4))
                break
    background = [round(float(x), 4) for x in sims.flatten()]

    bins = np.linspace(-0.2, 1.0, 25)
    results = {
        "model": MODEL_NAME,
        "n_sentences": len(all_sents),
        "n_chunks_200w": len(chunk_texts),
        "n_questions": len(questions),
        "sentence_vector_norms": norm_stats(vecs),
        "chunk_vector_norms": norm_stats(cvecs),
        "cross_post_pairs": stats(cross),
        "same_post_pairs": stats(same),
        "paraphrase_pairs": [round(float(x), 4) for x in para],
        "near_duplicate_casing_punct": stats(near_dup),
        "sentence_shared_component": sent_shared,
        "chunk_shared_component": chunk_shared,
        "centered_cross_post_pairs": stats(cross_c),
        "centered_same_post_pairs": stats(same_c),
        "gold_chunk_cosine": stats(gold_cos),
        "best_nongold_chunk_cosine": stats(nongold_cos),
        "gold_chunk_cosine_values": gold_cos,
        "best_nongold_chunk_cosine_values": nongold_cos,
        "gold_ranks": gold_ranks,
        "background_question_to_all_chunks": stats(background),
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    plot_data = {
        "cross_post_pairs": [round(float(x), 4) for x in cross],
        "same_post_pairs": [round(float(x), 4) for x in same],
        "centered_cross_post_pairs": [round(float(x), 4) for x in cross_c],
        "paraphrase_pairs": [round(float(x), 4) for x in para],
        "gold_chunk_cosine_values": gold_cos,
        "best_nongold_chunk_cosine_values": nongold_cos,
        "background_hist": hist(background, bins),
        "gold_hist": hist(gold_cos, bins),
        "nongold_hist": hist(nongold_cos, bins),
    }
    (HERE / "plot-data.json").write_text(json.dumps(plot_data, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
