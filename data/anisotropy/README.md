# Cosine experiment

The 2026-11-26 follow-up to the embeddings experiment: what does a cosine
similarity score actually mean? Same model (`sentence-transformers/
all-MiniLM-L6-v2`), same 16-post corpus, same 22 golden questions — but
instead of ranking, this experiment calibrates the scale itself.

Corpus: the 16 posts in `../embeddings/corpus/`, 1,749 sentences (6–40
words each) and 229 non-overlapping 200-word chunks; 22 questions from
`../embeddings/questions.json`. Note the chunking differs from the
embeddings experiment (overlapping windows, 281 chunks), so the ranks here
are not directly comparable to that run's 82% — everything compared inside
this experiment uses the same chunking on both sides.

## Headline numbers (exact, from `results.json`)

Sentence pairs (600 random pairs each):

| Relationship | Mean cosine | Notes |
| --- | --- | --- |
| Same sentence, casing/punctuation changed (n=20) | 0.98 | Range 0.96–1.00 |
| Paraphrase, hand-written (n=5) | 0.75 | Range 0.67–0.91 (0.67, 0.67, 0.73, 0.78, 0.91) |
| Two sentences, same post | 0.20 | p90 0.40, max 0.76 |
| Two sentences, different posts | 0.11 | p90 0.24, p99 0.38, max 0.56; 2/600 above 0.5 |

The shared component: the mean of all sentence vectors has norm 0.34
(0.49 for chunk vectors), so nearly half of every chunk vector points in
one shared direction. The squared norm predicts the unrelated-pair floor
exactly: 0.34² = 0.116 against a measured random-pair mean of 0.112
(n=2,000). The shared direction is diffuse, not a few rogue dimensions:
the single largest dimension holds 2.6% of the mean vector's energy
(2.4% for chunks), the top five hold 11%.

Mean-centering (subtract the corpus mean vector, renormalize): unrelated
pairs 0.11 → −0.01, same-post pairs 0.20 → 0.09. The scale becomes
honest (0 = unrelated) but the separation barely changes (0.09 → 0.10),
so centering fixes the scale, not the ranking.

Question → chunk (22 questions, 229 chunks): the background level of a
question against all chunks is 0.14. The best chunk that answers the
question scores 0.51 on average (range 0.33–0.72); the best chunk that
answers nothing scores 0.52 (range 0.29–0.70). The answering chunk
outscores the best wrong chunk in only 10 of 22 questions. At a 0.5
threshold, 12 answering chunks and 13 wrong chunks pass; at 0.7, 1
answering chunk and 0 wrong chunks pass. No threshold separates right
from wrong.

## Run it

Needs `sentence-transformers` + `torch`, deliberately not in
`requirements.txt` — the API service stays model-free:

```bash
pip install sentence-transformers
python data/anisotropy/measure-cosine.py
```

It writes `data/anisotropy/results.json` and `data/anisotropy/
plot-data.json` (raw distributions for the post's charts).

## Sources

- Mu, Bhat, and Viswanath, "All-but-the-Top" (ICLR 2018) — word
  vectors share a common vector with norm up to half the average norm;
  removing it improves word-similarity benchmarks.
- Ethayarajh, "How Contextual are Contextualized Word Representations?"
  (EMNLP 2019) — contextual representations are anisotropic in every
  layer of ELMo, BERT, and GPT-2.
- Timkey and van Schijndel, "All Bark and No Bite" (EMNLP 2021) — 1–5
  rogue dimensions dominate cosine similarity in BERT, GPT-2, RoBERTa,
  and XLNet.
