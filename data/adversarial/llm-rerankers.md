# LLMs as Rerankers

Instruction-tuned models rerank by prompting: pointwise (score each candidate against the query), pairwise (which of two is better), or listwise (order a slate in one pass). Log-probability scoring squeezes signal without generating text. Costs are tokens: pointwise scales linearly with candidates, listwise packs many into one context. Output-format instability is the operational headache; constrained decoding mitigates it.
