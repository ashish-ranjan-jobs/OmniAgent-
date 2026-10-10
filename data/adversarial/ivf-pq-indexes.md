# IVF and Product Quantization Indexes

Inverted-file (IVF) indexes partition vectors into coarse clusters via a quantizer; a query probes the nearest nlist cells within a nprobe budget, trading recall for speed. Product quantization (PQ) compresses vectors into short codes so distance is computed over codebooks, cutting memory per vector by an order of magnitude. IVF+PQ pipelines scale to billion-vector datasets where graph indexes such as HNSW cost too much RAM, at the price of tuning partition counts.
