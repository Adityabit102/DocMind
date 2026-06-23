.PHONY: dev backend frontend test ingest eval lint format docker-up docker-down clean install

install:
	python -m pip install -r requirements-dev.txt

dev:
	uvicorn app.main:app --reload --port 8000 & \
	cd frontend && npm run dev

backend:
	uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

test:
	pytest tests/ -v --cov=app --cov=rag --cov-report=term-missing

ingest:
	python -c "from rag.ingestion.pipeline import ingest_directory; ingest_directory('data/uploads/')"

eval:
	python -c "from rag.evaluation.ragas_eval import run_evaluation; run_evaluation()"

lint:
	ruff check app/ rag/ tests/ && mypy app/ rag/

format:
	black app/ rag/ tests/ && isort app/ rag/ tests/

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf data/faiss_index/* data/uploads/* data/metadata.json logs/*.log
	rm -rf .ruff_cache .mypy_cache .pytest_cache .coverage
