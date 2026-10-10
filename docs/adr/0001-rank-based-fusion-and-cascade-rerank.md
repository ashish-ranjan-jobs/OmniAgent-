# ADR-0001: Rank-based fusion (RRF) + cascade cross-encoder rerank

Date: 2026-10-10 | Status: accepted

## Context
Retrieval must combine a dense leg (pgvector + BGE cosine) and a sparse leg
(BM25). Their raw scores are on different scales and distributions: cosine in
[-1, 1] concentrated near 0.7-0.9 for related text; BM25 unbounded above 0.
A reranker is too expensive to apply to the whole corpus.

## Decision
1. Fuse with Reciprocal Rank Fusion: score(d) = sum over legs of 1/(K + rank),
   K = 60 (Cormack et al. 2009). Ranks only -- no score normalization, no unit
   mismatch, no training.
2. Cascade: cheap legs fetch top-k each, RRF fuses to top-N (FUSION_TOP_N=20),
   then a cross-encoder (ms-marco-MiniLM-L-6-v2) scores only those pairs and
   the top RERANK_TOP_K survive.

## Consequences
+ Deterministic and unit-testable fusion (pure function, tie-break by id).
+ Rerank cost is bounded by FUSION_TOP_N, independent of corpus size.
- RRF reshuffles can cost MRR when both legs are already excellent (observed:
  1.000 -> 0.958 on a 24-doc corpus); the reranker restores it. Measured in
  evals/results/.
- K is a hyperparameter; small K trusts top ranks, large K flattens.
