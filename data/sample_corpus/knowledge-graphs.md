# Knowledge Graphs

A knowledge graph stores entities and typed relations (papers cite papers,
people author papers) that keyword and vector search cannot express.
GraphRAG augments retrieval by expanding seed entities along relation edges
to gather multi-hop context. Cost: entity extraction and relation curation
at ingest time, plus cold-start data. Best used where relations themselves
answer the question.
