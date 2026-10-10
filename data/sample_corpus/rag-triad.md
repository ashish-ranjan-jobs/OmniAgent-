# The RAG Triad

The RAG Triad, popularized by TruLens, evaluates a retrieval-augmented system
along three independent axes, each localizing a different failure mode:

1. Context relevance: are the retrieved chunks pertinent to the query?
   Low scores indicate a retrieval problem (embedding model, chunk size,
   top-k).
2. Groundedness: is every claim in the answer supported by the retrieved
   context? Computed claim by claim; a strict judge marks a claim supported
   only if the evidence states it explicitly. One minus groundedness is the
   hallucination rate.
3. Answer relevance: does the answer actually address the question, rather
   than merely discussing the topic?

The triad needs no reference answers, which makes it usable where labelled
ground truth is scarce. Failure localization is the point: each metric points
at a different component to fix.
