.PHONY: help install data train serve test lint clean docker-up docker-down stream

# ============================================================
# HELP
# ============================================================
help: ## Show this help message
	@echo "Fraud Detection System - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================================
# DEVELOPMENT
# ============================================================
install: ## Install Python dependencies
	pip install -r requirements.txt

install-dev: ## Install with dev dependencies
	pip install -r requirements.txt
	pip install jupyter matplotlib seaborn

data: ## Generate synthetic transaction data
	python -m src.ingestion.simulator --output data/raw/transactions.csv --n-records 100000

data-large: ## Generate large dataset (1M records)
	python -m src.ingestion.simulator --output data/raw/transactions.csv --n-records 1000000

train: ## Train fraud detection model
	python -m src.model.trainer --data data/raw/transactions.csv --model-name fraud_model_v1

train-cv: ## Train with cross-validation
	python -m src.model.trainer --data data/raw/transactions.csv --model-name fraud_model_v1 --cv-folds 5

serve: ## Start FastAPI server locally
	uvicorn src.api.main:app --reload --port 8000

# ============================================================
# TESTING
# ============================================================
test: ## Run all tests
	pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# ============================================================
# CODE QUALITY
# ============================================================
lint: ## Run linters
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format: ## Format code
	black src/ tests/
	ruff check --fix src/ tests/

# ============================================================
# DOCKER
# ============================================================
docker-build: ## Build Docker image
	docker-compose build

docker-up: ## Start all services (Kafka + API + Monitoring)
	docker-compose up -d

docker-down: ## Stop all services
	docker-compose down

docker-logs: ## View logs
	docker-compose logs -f

docker-ps: ## Show running services
	docker-compose ps

# ============================================================
# STREAMING
# ============================================================
stream: ## Start transaction producer (simulated)
	docker-compose --profile simulate up -d fraud-producer

stream-local: ## Start local producer (requires Kafka running)
	python -m src.streaming.producer --tps 100

consume: ## Start fraud scoring consumer locally
	python -m src.streaming.consumer

# ============================================================
# MONITORING
# ============================================================
monitor: ## Open Grafana dashboard
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	@echo "API Docs: http://localhost:8000/docs"

# ============================================================
# FULL PIPELINE
# ============================================================
pipeline: install data train ## Run full pipeline (install -> data -> train)
	@echo "\nPipeline complete! Start serving with: make serve"

demo: docker-up stream ## Start full demo (Docker services + streaming)
	@echo "\nDemo running!"
	@echo "  API: http://localhost:8000/docs"
	@echo "  Grafana: http://localhost:3000"
	@echo "  Kafka UI: http://localhost:8080 (run with --profile debug)"

# ============================================================
# CLEANUP
# ============================================================
clean: ## Clean generated files
	rm -rf data/raw/*.csv data/processed/* data/models/*
	rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-docker: ## Remove Docker volumes and images
	docker-compose down -v --rmi local
