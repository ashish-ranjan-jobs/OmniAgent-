"""Ingestion: LlamaIndex readers + SentenceSplitter -> BGE -> pgvector.

Idempotent by design: chunk content is SHA-256 hashed; existing hashes are
skipped, so re-ingesting an unchanged corpus is a no-op and an edited document
re-embeds only its changed chunks (ADR-0003: LlamaIndex owns
parsing/splitting/embedding; we own the schema, dedupe and persistence).
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from sqlalchemy import select

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import Chunk, Document, IngestionJob
from app.db.session import get_sessionmaker
from app.ingestion.embedder import embed_texts

log = get_logger(__name__)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_and_split(path: str) -> list[dict]:
    """Read PDF/MD/TXT from a directory and split into chunk dicts (CPU-bound)."""
    s = get_settings()
    reader = SimpleDirectoryReader(input_dir=str(path), recursive=True)
    documents = reader.load_data()
    splitter = SentenceSplitter(chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)
    return [
        {
            "text": node.get_content(),
            "source": node.metadata.get("file_path", str(path)),
            "title": Path(node.metadata.get("file_path", "doc")).stem,
        }
        for node in nodes
    ]


async def ingest_directory(path: str) -> dict:
    """Idempotently ingest a directory of documents. Returns a summary."""
    sm = get_sessionmaker()
    async with sm() as session:
        job = IngestionJob(status="running", started_at=datetime.now(UTC))
        session.add(job)
        await session.flush()

        try:
            chunks = await asyncio.to_thread(load_and_split, path)
            hashes = [_hash(c["text"]) for c in chunks]

            existing = set(
                await session.scalars(
                    select(Chunk.content_hash).where(Chunk.content_hash.in_(hashes))
                )
            )
            new = [(c, h) for c, h in zip(chunks, hashes, strict=True) if h not in existing]
            skipped = len(chunks) - len(new)

            created = 0
            if new:
                vectors = await asyncio.to_thread(
                    embed_texts, [c["text"] for c, _ in new]
                )

                # Upsert one Document per source file.
                doc_ids: dict[str, str] = {}
                for src in sorted({c["source"] for c, _ in new}):
                    doc = (
                        await session.scalars(
                            select(Document).where(Document.source_path == src)
                        )
                    ).first()
                    if doc is None:
                        doc = Document(
                            source_path=src, title=Path(src).stem, content_hash=_hash(src)
                        )
                        session.add(doc)
                        await session.flush()
                    doc.title = Path(src).stem
                    doc_ids[src] = doc.id

                per_source: dict[str, int] = {}
                for (chunk, h), vec in zip(new, vectors, strict=True):
                    per_source[chunk["source"]] = per_source.get(chunk["source"], 0) + 1
                    session.add(
                        Chunk(
                            document_id=doc_ids[chunk["source"]],
                            chunk_index=per_source[chunk["source"]],
                            content=chunk["text"],
                            content_hash=h,
                            embedding=vec,
                        )
                    )
                created = len(new)
                await session.commit()

            job.status = "ok"
            job.documents_seen = len({c["source"] for c in chunks})
            job.chunks_created = created
            job.chunks_skipped = skipped
            job.finished_at = datetime.now(UTC)
            await session.commit()
            summary = {
                "documents": len({c["source"] for c in chunks}),
                "chunks_created": created,
                "chunks_skipped": skipped,
            }
            log.info("ingestion_complete", **summary)
            return summary
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = datetime.now(UTC)
            await session.commit()
            log.error("ingestion_failed", error=str(exc))
            raise
