# Chunking Strategies

Chunking splits documents into retrieval units. Fixed-size token windows
with overlap are simple and predictable; sentence-boundary splitting avoids
cutting mid-thought; structural splitting uses headings and sections when
markup is available. Too-small chunks lose context, too-large chunks dilute
similarity scores and waste context window budget. Overlap guards against
answers straddling a boundary.
