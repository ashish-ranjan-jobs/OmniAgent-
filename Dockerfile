FROM python:3.12-slim AS base

# Non-root runtime user (least privilege inside the container).
RUN groupadd -r app && useradd -r -g app app

WORKDIR /srv/omniagent

COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir . && pip cache purge

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
