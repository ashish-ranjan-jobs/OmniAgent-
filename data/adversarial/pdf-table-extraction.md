# PDF Table Extraction

Table extraction tools (Camelot, pdfplumber) recover tabular structure from PDFs: lattice mode uses ruling lines, stream mode infers columns from whitespace alignment. Merged cells, multi-page tables, and borderless layouts break naive heuristics, so extraction pipelines verify against row-count sanity checks and fall back to page images for manual review. Recovered tables usually need schema mapping before storage.
