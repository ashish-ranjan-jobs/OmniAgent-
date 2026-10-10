"""Dense retrieval: pgvector cosine distance over chunk embeddings."""

from __future__ import annotations

from sqlalchemy import select

from app.db.models import Chunk, Document
from app.db.session import get_sessionmaker


async def dense_search(query_vec: list[float], k: int) -> list[dict]:
    """Top-k chunks by cosine similarity (pgvector <=> operator)."""
    sm = get_sessionmaker()
    async with sm() as session:
        distance = Chunk.embedding.cosine_distance(query_vec).label("distance")
        stmt = (
            select(Chunk.id, Chunk.document_id, Chunk.content, Document.title, distance)
            .join(Document, Chunk.document_id == Document.id)
            .order_by(distance)
            .limit(k)
        )
        rows = (await session.execute(stmt)).all()
    return [
        {
            "chunk_id": r.id,
            "document_id": r.document_id,
            "content": r.content,
            "title": r.title,
            "distance": float(r.distance),
        }
        for r in rows
    ]
