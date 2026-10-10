# Late Chunking

Late chunking delays the chunking decision: it embeds the whole document once with a long-context encoder, then pools token-level embeddings into chunk vectors afterward. Every chunk vector therefore sees the full document context, which preserves pronoun references and topical continuity that per-chunk embedding loses. It trades a larger encoder at query time for coherent chunk representations.
