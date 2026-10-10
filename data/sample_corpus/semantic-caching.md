# Semantic Caching

A semantic cache stores previous answers keyed by the embedding of the
question rather than the exact string. A new query within a similarity
threshold hits the cache and skips the model entirely, cutting latency and
cost for repeated questions. Risks: near-duplicate questions with different
intent return wrong cached answers, so thresholds must be strict and cache
entries must be invalidated when the underlying corpus changes.
