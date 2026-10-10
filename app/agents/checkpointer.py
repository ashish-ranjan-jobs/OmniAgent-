"""LangGraph checkpointing to Postgres: multi-turn research threads.

AsyncPostgresSaver.from_conn_string is an async context manager (it owns a
psycopg connection); we hold it open for the process lifetime via an
AsyncExitStack and close it on FastAPI shutdown. Its checkpoint tables are
created by .setup() and owned by the framework -- deliberately outside our
Alembic migrations (separate ownership boundary).

Known limitation (documented in docs/LLD.md): the saver serializes checkpoint
writes over a single connection; the scaling path is a pooled saver per
worker process.
"""

from __future__ import annotations

from contextlib import AsyncExitStack

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.core.config import get_settings

_stack: AsyncExitStack | None = None
_checkpointer: AsyncPostgresSaver | None = None


async def get_checkpointer() -> AsyncPostgresSaver:
    global _stack, _checkpointer
    if _checkpointer is None:
        _stack = AsyncExitStack()
        _checkpointer = await _stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync)
        )
        await _checkpointer.setup()
    return _checkpointer


async def close_checkpointer() -> None:
    """Called from FastAPI lifespan shutdown."""
    global _stack, _checkpointer
    if _stack is not None:
        await _stack.aclose()
        _stack = None
        _checkpointer = None
