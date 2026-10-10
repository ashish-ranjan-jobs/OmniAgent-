# Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) is a rank-based method for combining multiple
retrieval result lists into one ranking. For a document d, the fused score is
the sum over each system s of 1 / (K + rank_s(d)), where rank starts at 1 and
K is a smoothing constant, classically 60, from Cormack et al. 2009
("Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning
Methods").

RRF's key property: it uses only ranks, never raw scores. Cosine similarities
and BM25 scores live on different scales and distributions; linearly mixing
them requires fragile normalization. RRF sidesteps the unit mismatch entirely.
A small K trusts top positions more; a large K flattens differences between
ranks. RRF is training-free and robust to outlier scores from either system.
