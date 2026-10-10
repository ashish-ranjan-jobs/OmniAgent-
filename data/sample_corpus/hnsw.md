# HNSW Indexes for Vector Search

Hierarchical Navigable Small World (HNSW) graphs are the dominant approximate
nearest neighbor index for vector databases, including pgvector. HNSW builds
multiple layers of proximity graphs: upper layers are sparse and enable long
greedy jumps; the bottom layer is dense and contains every vector. A search
starts at the top and greedily descends, giving expected O(log n) query time.

Important parameters: M (connections per node, commonly 16) trades memory
for recall, and ef_construction / ef_search control candidate list sizes for
build quality and query recall respectively. pgvector supports HNSW with
cosine distance, L2, and inner product, exposed in SQL through operators
like the cosine distance operator used by this system's dense retrieval leg.
