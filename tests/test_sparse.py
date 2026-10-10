from __future__ import annotations

import app.retrieval.sparse as sparse

# NB: rank_bm25's classic IDF ln((N - df + 0.5)/(df + 0.5)) degenerates to 0
# when N=2, df=1 -- tiny corpora are pathological for BM25. Fixtures use 6+ docs.
ROWS = [
    ("c1", "reciprocal rank fusion combines retrieval rankings"),
    ("c2", "bm25 term frequency saturation parameter k1"),
    ("c3", "vector embeddings capture semantic similarity"),
    ("c4", "cross encoders rerank candidate passages jointly"),
    ("c5", "postgres stores documents and metadata"),
    ("c6", "checkpoints persist graph state between turns"),
]


async def _install_fake(monkeypatch):
    async def fake_load_rows():
        return ROWS

    monkeypatch.setattr(sparse, "_load_rows", fake_load_rows)
    sparse._index = None
    sparse._built_count = -1


async def test_sparse_search_ranks_keyword_doc_first(monkeypatch):
    await _install_fake(monkeypatch)
    hits = await sparse.sparse_search("bm25 saturation parameter k1", k=3)
    assert hits and hits[0][0] == "c2"

    hits = await sparse.sparse_search("reciprocal rank fusion rankings", k=3)
    assert hits and hits[0][0] == "c1"


async def test_sparse_search_no_positive_score(monkeypatch):
    await _install_fake(monkeypatch)
    assert await sparse.sparse_search("unrelated zebra quantum", k=3) == []
