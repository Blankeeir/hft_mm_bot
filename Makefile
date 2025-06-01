.PHONY: install test lint format clean run-tests run-paper run-live

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

test-coverage:
	pytest tests/ --cov=. --cov-report=html --cov-report=term

lint:
	black --check .
	isort --check-only .
	mypy .

format:
	black .
	isort .

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf logs/*.log

run-tests: test

run-paper:
	python main.py --mode paper

run-live:
	python main.py --mode live

run-training:
	python main.py --mode training

setup-env:
	cp .env.example .env
	@echo "Please edit .env file with your API credentials"

docker-build:
	docker build -t hft-mm-bot .

docker-run:
	docker run --env-file .env hft-mm-bot

help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  test         - Run test suite"
	@echo "  test-coverage - Run tests with coverage report"
	@echo "  lint         - Check code style"
	@echo "  format       - Format code"
	@echo "  clean        - Clean temporary files"
	@echo "  run-paper    - Run in paper trading mode"
	@echo "  run-live     - Run in live trading mode"
	@echo "  run-training - Run in training mode"
	@echo "  setup-env    - Setup environment file"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run in Docker container"
