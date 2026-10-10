"""Application configuration, sourced exclusively from .env (12-factor style)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every runtime knob lives here. No config scattered in code."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # -- app ---------------------------------------------------------------
    app_env: str = "dev"
    log_level: str = "INFO"

    # -- api ----------------------------------------------------------------
    api_host: str = "localhost"
    api_port: int = 8000
    api_key: str = ""  # optional shared secret; if set, X-API-Key header required

    # -- database ------------------------------------------------------------
    db_user: str = "omniagent"
    db_password: str = "change-me"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "omniagent"

    # -- llm backend (OpenAI-compatible endpoint: Ollama / vLLM / API) --------
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "llama3.2:3b"  # strong role: synthesis
    llm_fast_model: str = "llama3.2:3b"  # fast role: planning, verdicts
    llm_api_key: str = "ollama"
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 2

    # -- llm fallback (off by default; local-first) ---------------------------
    llm_fallback_enabled: bool = False
    llm_fallback_base_url: str = ""
    llm_fallback_api_key: str = ""
    llm_fallback_model: str = ""

    # -- local models -------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # -- ingestion -----------------------------------------------------------
    chunk_size: int = 400
    chunk_overlap: int = 50

    # -- hybrid retrieval ----------------------------------------------------
    retrieval_top_k: int = 8
    fusion_top_n: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 5
    hybrid_enabled: bool = True
    rerank_enabled: bool = True

    # -- agent graph ---------------------------------------------------------
    max_reflections: int = 2
    refusal_min_evidence: int = 3

    # -- evaluation -----------------------------------------------------------
    eval_golden_path: str = "evals/golden/queries.jsonl"

    @property
    def database_url(self) -> str:
        """Async SQLAlchemy URL (asyncpg driver)."""
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def database_url_sync(self) -> str:
        """psycopg URL for the LangGraph AsyncPostgresSaver checkpoint store."""
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor; import-time safe, test-overridable."""
    return Settings()
