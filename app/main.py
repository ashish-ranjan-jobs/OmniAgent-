"""Application entrypoint: uvicorn app.main:app"""

from __future__ import annotations

from app.api.app import create_app

app = create_app()


if __name__ == "__main__":
    import uvicorn

    from app.core.config import get_settings

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port)
