# RAG Assistant Makefile
# Convenient commands for development and deployment

.PHONY: help install setup clean run test docker-build docker-run docker-stop lint format docs

# Default target
help:
	@echo "RAG Assistant - Available Commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  install     - Run complete installation"
	@echo "  setup       - Run setup script only"
	@echo "  clean       - Clean up temporary files"
	@echo ""
	@echo "Development:"
	@echo "  run         - Start the Gradio application"
	@echo "  cli         - Start CLI mode"
	@echo "  test        - Run all tests"
	@echo "  test-quick  - Run quick tests only"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code with black"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build    - Build Docker image"
	@echo "  docker-run      - Run with Docker Compose"
	@echo "  docker-stop     - Stop Docker containers"
	@echo "  docker-clean    - Clean Docker images and volumes"
	@echo ""
	@echo "Documentation:"
	@echo "  docs        - Generate documentation"
	@echo "  readme      - View README.md"
	@echo ""
	@echo "Maintenance:"
	@echo "  backup      - Backup data and configurations"
	@echo "  logs        - View recent logs"
	@echo "  status      - Show system status"

# Variables
VENV_NAME = rag_env
PYTHON = $(VENV_NAME)/bin/python
PIP = $(VENV_NAME)/bin/pip
DOCKER_IMAGE = rag-assistant
DOCKER_TAG = latest

# Installation and Setup
install:
	@echo "🚀 Starting installation..."
	chmod +x install.sh
	./install.sh

setup:
	@echo "⚙️ Running setup..."
	$(PYTHON) setup.py

setup-dev: install
	@echo "🔧 Setting up development environment..."
	$(PIP) install -r requirements-dev.txt || echo "requirements-dev.txt not found, skipping dev dependencies"

# Environment management
venv:
	@echo "🐍 Creating virtual environment..."
	python3 -m venv $(VENV_NAME)
	$(PIP) install --upgrade pip

activate:
	@echo "To activate virtual environment, run:"
	@echo "source $(VENV_NAME)/bin/activate"

# Application running
run:
	@echo "🎯 Starting RAG Assistant..."
	$(PYTHON) app.py

run-debug:
	@echo "🐛 Starting in debug mode..."
	DEBUG=true $(PYTHON) app.py

cli:
	@echo "💻 Starting CLI mode..."
	$(PYTHON) cli.py

chat:
	@echo "💬 Starting interactive chat..."
	$(PYTHON) cli.py chat

# Testing
test:
	@echo "🧪 Running all tests..."
	$(PYTHON) test_rag.py

test-quick:
	@echo "⚡ Running quick tests..."
	$(PYTHON) test_rag.py --component requirements

test-processor:
	@echo "📄 Testing document processor..."
	$(PYTHON) test_rag.py --component processor

test-model:
	@echo "🤖 Testing model setup..."
	$(PYTHON) test_rag.py --component model

test-rag:
	@echo "🔍 Testing RAG engine..."
	$(PYTHON) test_rag.py --component rag

# Code quality
lint:
	@echo "🔍 Running linting..."
	$(PYTHON) -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics || echo "flake8 not installed"
	$(PYTHON) -m pylint *.py || echo "pylint not installed"

format:
	@echo "🎨 Formatting code..."
	$(PYTHON) -m black . || echo "black not installed"
	$(PYTHON) -m isort . || echo "isort not installed"

# Docker operations
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run:
	@echo "🐳 Starting with Docker Compose..."
	docker-compose up -d

docker-stop:
	@echo "🐳 Stopping Docker containers..."
	docker-compose down

docker-logs:
	@echo "📜 Showing Docker logs..."
	docker-compose logs -f rag-assistant

docker-shell:
	@echo "🐚 Opening shell in container..."
	docker-compose exec rag-assistant bash

docker-clean:
	@echo "🧹 Cleaning Docker images and volumes..."
	docker-compose down -v
	docker system prune -f
	docker image prune -f

# Data and model management
download-model:
	@echo "📥 Downloading model..."
	$(PYTHON) -c "from model_setup import setup_model; setup_model()"

add-docs:
	@echo "📚 Adding documents from data directory..."
	$(PYTHON) cli.py add ./data

process-samples:
	@echo "📄 Processing sample documents..."
	mkdir -p data/samples
	echo "This is a sample document for testing the RAG system." > data/samples/sample.txt
	$(PYTHON) cli.py add ./data/samples

# Maintenance
clean:
	@echo "🧹 Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -name ".DS_Store" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build

clean-logs:
	@echo "🧹 Cleaning old logs..."
	$(PYTHON) -c "from utils import cleanup_old_logs; cleanup_old_logs(7)"

clean-cache:
	@echo "🧹 Cleaning processed data cache..."
	rm -rf processed_data/*
	rm -rf vector_store/*

backup:
	@echo "💾 Creating backup..."
	mkdir -p backups
	tar -czf backups/rag-backup-$(shell date +%Y%m%d_%H%M%S).tar.gz \
		data/ models/ vector_store/ .env config.py logs/

restore-backup:
	@echo "📦 Available backups:"
	@ls -la backups/*.tar.gz 2>/dev/null || echo "No backups found"
	@echo "To restore: tar -xzf backups/backup-file.tar.gz"

# Monitoring and status
status:
	@echo "📊 System Status:"
	$(PYTHON) cli.py status

logs:
	@echo "📜 Recent logs:"
	tail -n 50 logs/*.log 2>/dev/null || echo "No log files found"

logs-follow:
	@echo "📜 Following logs..."
	tail -f logs/*.log

disk-usage:
	@echo "💾 Disk usage:"
	du -sh data/ models/ vector_store/ processed_data/ logs/ 2>/dev/null || echo "Directories not found"

system-info:
	@echo "🔧 System Information:"
	$(PYTHON) -c "from utils import create_system_info; import json; print(json.dumps(create_system_info(), indent=2))"

# Development helpers
dev-install: venv
	@echo "🔧 Installing development dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install black isort flake8 pylint pytest jupyter || echo "Some dev tools not available"

notebook:
	@echo "📓 Starting Jupyter notebook..."
	$(PYTHON) -m jupyter notebook || echo "Jupyter not installed"

requirements-update:
	@echo "📋 Updating requirements..."
	$(PIP) freeze > requirements.txt

# Docker development
docker-dev:
	@echo "🐳 Building development Docker image..."
	docker build -f Dockerfile.dev -t $(DOCKER_IMAGE):dev . || echo "Dockerfile.dev not found"

docker-test:
	@echo "🧪 Running tests in Docker..."
	docker run --rm $(DOCKER_IMAGE):$(DOCKER_TAG) python test_rag.py

# Documentation
docs:
	@echo "📚 Generating documentation..."
	$(PYTHON) -m pydoc -w . || echo "pydoc failed"

readme:
	@echo "📖 README.md:"
	@cat README.md | head -50

# Security
security-scan:
	@echo "🔒 Running security scan..."
	$(PYTHON) -m pip audit || echo "pip-audit not installed"
	$(PYTHON) -m bandit -r . || echo "bandit not installed"

# Quick start for new users
quickstart: install
	@echo "🚀 Quick start complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Edit .env file with your settings"
	@echo "2. Run 'make run' to start the application"
	@echo "3. Open http://localhost:7860 in your browser"
	@echo ""
	@echo "Or try CLI mode with 'make cli'"

# Help for specific components
help-docker:
	@echo "🐳 Docker Commands Help:"
	@echo "  make docker-build  - Build the Docker image"
	@echo "  make docker-run    - Start containers with docker-compose"
	@echo "  make docker-stop   - Stop all containers"
	@echo "  make docker-logs   - View container logs"
	@echo "  make docker-shell  - Open shell in container"

help-dev:
	@echo "🔧 Development Commands Help:"
	@echo "  make dev-install   - Setup development environment"
	@echo "  make test          - Run all tests"
	@echo "  make lint          - Check code quality"
	@echo "  make format        - Format code"
	@echo "  make notebook      - Start Jupyter notebook"

# Default make behavior
.DEFAULT_GOAL := help