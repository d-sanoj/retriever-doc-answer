# Embeddings experiment

The 2026-11-20 follow-up to the retriever and chunking experiments: the
embedding model the first post could not test, run against BM25 on the same
golden set. Corpus: the 16 posts in `corpus/` (the 15 from the chunking
experiment plus the 2026-11-19 commit-messages post), 44,153 words, 22
questions. The two posts that quote the golden questions (the 2026-11-16
retriever post and the 2026-11-18 chunking post) are not in the corpus
folder, so the questions cannot match themselves.

Model: `sentence-transformers/all-MiniLM-L6-v2` — 6 layers, 384 dimensions,
22.7M parameters, ~90 MB download, trained on 1.17B sentence pairs, inputs
longer than 256 word pieces truncated (that limit matters: 70% of the
200-word chunks and all of the 400-word chunks exceed it).

Rankers: BM25 (k1 1.5, b 0.75), embeddings (cosine similarity), and hybrid
(reciprocal rank fusion of the two lists, k = 60). Same chunking, same
strict grading as the first experiment: a chunk counts only if it comes
from the right post and contains at least two of the question's answer
terms.

## Headline numbers (strict chunk-level recall@5, exact)

| Chunk size | BM25 | Embeddings | Hybrid (RRF) |
| --- | --- | --- | --- |
| 200 words | 0.73 (16/22) | **0.82 (18/22)** | 0.77 (17/22) |
| 400 words | 0.91 (20/22) | 0.91 (20/22) | 0.91 (20/22) |

Post-level recall@5: embeddings 1.00 at both chunk sizes (right post
ranked first for all 22 questions); BM25 0.95 at 200 words. MRR
(chunk-level) at 200 words: BM25 0.635, embeddings 0.748, hybrid 0.605.

What embeddings fix: the synonym failures. Q6 (nerf-watch) BM25 rank 13 →
embeddings rank 1; Q8 (weight-space) 9 → 1; Q10 (how a model learns) 5 → 1;
Q19 (token costs) 6 → 1. What they break: Q3 and Q18, both about the
claude-mem post — embeddings top the post's generic intro chunk for both,
and the answering passage sits at ranks 7 and 9.

The hybrid loses to pure embeddings (0.77 vs 0.82 at 200 words). RRF with
k = 60 is nearly flat at the top — 1/62 − 1/63 ≈ 0.00026 against
1/62 − 1/79 ≈ 0.0035 — so when the rankers disagree, the fusion follows
BM25's spread, the weaker ranker. Q8 and Q10 both go from embeddings rank 1
to hybrid rank 6.

Rewording (same 6 questions, 2 paraphrases each, 200 words): embeddings
win the two plain-words paraphrases that keep the topic's distinctive words
(Q1: 4 → 1, Q5: 5 → 1) and collapse on the two that drop them (Q11: 12 →
104, Q3: 20 → 29). Across the 12 paraphrases: better twice, tied twice,
worse eight times.

The 400-word tie has a mechanism: the model truncates past 256 word
pieces, and in Q20's worse gold chunk both answer terms sit past token 256
("rent" at token 449, "moving target" at token 451, of 535). At 400 words
the embedding is built from the first half of the chunk.

Section chunking (0.91 with BM25, on the 15-post corpus) still holds the
best number measured so far; the unmeasured combination is embeddings over
section chunks.

## Run it

Needs `sentence-transformers` + `torch` — deliberately NOT in the repo's
`requirements.txt`, so the API service stays model-free:

```bash
pip install sentence-transformers
python measure-embeddings.py
```

Writes `results.json` next to the script. Reference output is the
`results.json` in this folder (run 2026-11-20, reproduced exactly on
re-run). The script loads the model from the HuggingFace Hub on first run
(~90 MB, cached afterwards).

## Files

- `measure-embeddings.py` — the experiment (BM25, embeddings, hybrid,
  rewording comparison, top-5 dump).
- `questions.json` — the 22 questions and 6 rewording items.
- `results.json` — the reference output.
- `corpus/` — the 16 blog posts, frontmatter stripped at load time.

Blog post: ["Embeddings beat BM25 on my blog, 82% to 73% — and the hybrid
lost"](https://sanoj.dev/blog/embeddings-vs-bm25-measured/).
