# Document Expansion with Predicted Queries

Doc2Query and docTTTTTquery expand each document before indexing: a small trained model generates the questions the passage likely answers, and those generated queries join the indexed text. The expanded lexical index matches user vocabulary the original text never contained. Cost is inference at ingest time plus occasional hallucinated expansions; the win shows up on keyword-heavy queries against short documents.
