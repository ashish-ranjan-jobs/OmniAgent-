# Linear Attention Approximations

Standard attention compares every token pair, costing quadratic time and memory in sequence length. Linear attention variants rewrite the softmax kernel so keys and values combine first, yielding linear-time updates and constant-memory state per head. Quality approximates softmax attention on many tasks but degrades on long-range retrieval-style queries, so hybrids route easy contexts through the approximation and hard ones exactly.
