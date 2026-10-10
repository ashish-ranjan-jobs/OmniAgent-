# Document Parsing

Parsing turns PDFs, office documents, and scans into retrievable text.
PDF text layers extract directly; scanned pages need OCR, which introduces
errors on tables, columns, and handwriting. Layout-aware parsers reconstruct
reading order and table structure. Parsing quality bounds the entire
downstream pipeline: retrieval cannot find what parsing never captured.
