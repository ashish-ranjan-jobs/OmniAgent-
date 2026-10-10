"""Sparse (BM25) retrieval over the same chunk table.

In-process BM25Okapi index, rebuilt when the chunk count changes. Fine at
small/medium corpus sizes; Postgres tsvector is the drop-in upgrade path for
large corpora behind the same function signature.
"""

from __future__ import annotations

import threading

from rank_bm25 import BM25Okapi
from sqlalchemy import select

from app.db.models import Chunk
from app.db.session import get_sessionmaker

_lock = threading.Lock()
_index: BM25Okapi | None = None
_chunk_ids: list[str] = []
_built_count: int = -1


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if len(t) > 1]


async def _load_rows() -> list[tuple[str, str]]:
    sm = get_sessionmaker()
    async with sm() as session:
        rows = (await session.execute(select(Chunk.id, Chunk.content))).all()
        return [(r.id, r.content) for r in rows]


async def ensure_index() -> int:
    """(Re)build the BM25 index if stale. Returns the indexed chunk count."""
    global _index, _chunk_ids, _built_count
    rows = await _load_rows()
    if _index is not None and len(rows) == _built_count:
        return _built_count
    with _lock:
        _chunk_ids = [r[0] for r in rows]
        corpus = [_tokenize(r[1]) for r in rows]
        _index = BM25Okapi(corpus) if corpus else None
        _built_count = len(rows)
    return _built_count


async def sparse_search(query: str, k: int) -> list[tuple[str, float]]:
    """Top-k (chunk_id, bm25_score); higher = more keyword overlap."""
    await ensure_index()
    if _index is None:
        return []
    scores = _index.get_scores(_tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(_chunk_ids[i], float(scores[i])) for i in order if scores[i] > 0]
