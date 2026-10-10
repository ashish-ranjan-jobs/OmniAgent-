"""BGE embeddings via LlamaIndex HuggingFaceEmbedding (local CPU).

Model weights are cached under /opt/omniagent/hf-cache (HF_HOME) so teardown
is a single directory removal. Queries use BGE's recommended search prefix,
which measurably improves recall for short queries.
"""

from __future__ import annotations

import os

# Isolate HuggingFace caches under /opt/omniagent (teardown = rm -rf).
os.environ.setdefault("HF_HOME", "/opt/omniagent/hf-cache")

from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: E402

from app.core.config import get_settings  # noqa: E402

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_embedder: HuggingFaceEmbedding | None = None


def get_embedder() -> HuggingFaceEmbedding:
    """Lazy singleton: model loads once per process (~400MB RAM)."""
    global _embedder
    if _embedder is None:
        s = get_settings()
        _embedder = HuggingFaceEmbedding(model_name=s.embedding_model)
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed chunk texts (document side, no prefix)."""
    if not texts:
        return []
    return [list(map(float, v)) for v in get_embedder().get_text_embedding_batch(texts)]


def embed_query(query: str) -> list[float]:
    """Embed a user query with the BGE search prefix."""
    return list(map(float, get_embedder().get_text_embedding(BGE_QUERY_PREFIX + query)))
