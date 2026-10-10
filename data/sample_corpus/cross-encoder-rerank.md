# Cross-Encoder Reranking

A bi-encoder encodes query and passage independently into vectors compared
by cosine similarity: fast, cacheable, and scalable to millions of passages,
but blind to token-level interactions. A cross-encoder concatenates query and
passage into one transformer input, applies full cross-attention, and emits a
relevance logit: substantially more accurate, but it must run the model once
per (query, passage) pair, so scoring the whole corpus is infeasible.

The standard production pattern is a cascade: cheap recall stage (bi-encoder
plus BM25, top 20-50 candidates), then a cross-encoder reranks only those
candidates and keeps the top 5. A common CPU-friendly reranker is
ms-marco-MiniLM-L-6-v2 (6 layers, trained on MS MARCO), which is what this
system uses behind the RERANKER_MODEL setting.
