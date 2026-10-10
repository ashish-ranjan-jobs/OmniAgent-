# Hybrid Retrieval

Dense retrieval (embedding similarity) captures paraphrase and semantic
relatedness but often misses exact identifiers: error codes, function names,
acronyms, version strings. Sparse lexical retrieval (BM25 or Postgres
tsvector) is the exact inverse: perfect on identifiers, blind to vocabulary
mismatch.

Hybrid retrieval runs both legs over the same corpus and fuses the ranked
lists. Fusion options include score normalization plus weighted sums (fragile:
cosine and BM25 scores are on different scales and distributions) and
rank-based fusion such as Reciprocal Rank Fusion, which uses only ranks and
therefore never mixes incompatible units. The final stage in most production
stacks is a cross-encoder rerank over the fused top candidates.
