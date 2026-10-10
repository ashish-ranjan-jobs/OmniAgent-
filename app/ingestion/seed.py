"""CLI: python -m app.ingestion.seed [directory] -- idempotent corpus seed."""

from __future__ import annotations

import asyncio
import sys

from app.core.logging import configure_logging
from app.ingestion.pipeline import ingest_directory


async def main(path: str) -> None:
    summary = await ingest_directory(path)
    print(summary)


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "data/sample_corpus"))
