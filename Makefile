PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: install db-up db-down migrate seed run test lint ablation eval-triad eval-reflection docker-app teardown-info

install:
	$(PIP) install -e ".[dev]"

db-up:
	docker compose up -d db && docker compose ps

db-down:
	docker compose down

migrate:
	$(PY) -m alembic upgrade head

seed:
	$(PY) -m app.ingestion.seed data/sample_corpus

run:
	$(PY) -m uvicorn app.main:app --host localhost --port 8000

test:
	$(PY) -m pytest -m "not integration" -q

lint:
	.venv/bin/ruff check .

ablation:
	$(PY) -m app.eval.runner --mode ablation

eval-triad:
	$(PY) -m app.eval.runner --mode triad

eval-reflection:
	$(PY) -m app.eval.runner --mode reflection

docker-app:
	docker compose --profile app up -d --build

teardown-info:
	@cat docs/TEARDOWN.md
