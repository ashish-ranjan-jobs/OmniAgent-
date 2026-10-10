from __future__ import annotations

from app.core.config import Settings


def test_database_url_assembly():
    s = Settings(
        _env_file=None,
        db_user="u",
        db_password="p",
        db_host="h",
        db_port=5433,
        db_name="d",
    )
    assert s.database_url == "postgresql+asyncpg://u:p@h:5433/d"
    assert s.database_url_sync == "postgresql://u:p@h:5433/d"


def test_defaults_sensible():
    s = Settings(_env_file=None)
    assert s.max_reflections >= 0
    assert s.refusal_min_evidence >= 1
    assert s.rrf_k > 0
