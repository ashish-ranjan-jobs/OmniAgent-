# BGE Embeddings

BGE (bge-small-en-v1.5), released by BAAI, is a 384-dimension English text
embedding model. It is a bidirectional (BERT-style) encoder fine-tuned with
contrastive learning over large query-passage pair data, which is what makes
its query-passage similarity meaningful for retrieval.

Usage detail that matters in practice: queries should be prefixed with the
instruction "Represent this sentence for searching relevant passages:" while
documents are embedded raw. Skipping the query prefix measurably hurts
recall for short queries. The small variant runs comfortably on CPU
(~130M parameters); the base and large variants use 768 and 1024 dimensions
respectively and cost proportionally more storage and compute.
