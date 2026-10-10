# BM25 Variants and Field Weighting

Classic BM25 treats a document as one bag of terms. Field-weighted extensions (BM25F, Lucene's BM25Similarity) score title, body, and anchor text separately, then combine per-field term contributions with field-specific boosts and length normalization parameters tuned per field. Search engineers tune these per corpus via relevance judgements and interleaving experiments; defaults differ across engines and rarely transfer between collections.
