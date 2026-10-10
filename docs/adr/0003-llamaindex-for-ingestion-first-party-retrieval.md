# ADR-0003: LlamaIndex owns ingestion; retrieval stays first-party

Date: 2026-10-10 | Status: accepted

## Context
LlamaIndex provides readers (PDF/MD/TXT), the SentenceSplitter, and the
HuggingFaceEmbedding wrapper. It also provides vector stores and retriever
abstractions that would own our retrieval path.

## Decision
Use LlamaIndex for parsing/splitting/embedding (boilerplate-heavy, low design
risk). Keep the chunk table, dedupe, hybrid legs, fusion and rerank
first-party: schema in our Alembic migrations; retrieval code we can
whiteboard and ablate stage-by-stage.

## Consequences
+ The schema and the RRF/rerank logic are explainable line-by-line (interview
  requirement) and toggleable for ablation.
+ Idempotency is implemented exactly as designed (SHA-256 content hash, unique
  constraint) rather than inherited from a framework's docstore semantics.
- We maintain ~200 lines of retrieval code that LlamaIndex components could
  partially provide; accepted because ablation toggles and explainability live
  exactly there.
