# Grounded Refusal

A grounded RAG system must be able to refuse: when retrieved evidence is
below a threshold (too few chunks, or chunks that do not address the
question), the synthesizer returns an explicit refusal instead of generating
an answer from the language model's parametric memory.

Refusal correctness is an explicit evaluation axis: unanswerable questions
in the golden set must produce refusals, and the refusal should state what
was missing. Fail-closed verification pairs with this: if the fact-checker
cannot verify a draft, the draft is treated as unsupported (triggering
reflection or flagging) rather than silently passing. Refusal plus strict
claim-level verification is the strongest anti-hallucination guardrail pair
in this architecture.
