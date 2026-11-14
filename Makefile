SHELL := /bin/bash
IMAGE_NAME := keystroke-auth-ml-api
PYTHON := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
PYTEST := $(shell if [ -x .venv/bin/pytest ]; then echo .venv/bin/pytest; else echo pytest; fi)
RUFF := $(shell if [ -x .venv/bin/ruff ]; then echo .venv/bin/ruff; else echo ruff; fi)

.PHONY: help install dev test lint migrate retrain docker-build compose-up compose-down clean

help:
	@echo "Comandos disponíveis:"
	@echo "  make install       - Instala dependências do projeto e de desenvolvimento"
	@echo "  make dev           - Executa o servidor Django de desenvolvimento (localhost:8000)"
	@echo "  make test          - Executa a suíte de testes automatizados (pytest)"
	@echo "  make lint          - Executa verificação estática de código (ruff)"
	@echo "  make migrate       - Executa as migrações do banco de dados"
	@echo "  make retrain       - Executa o retreinamento manual de modelos devidos"
	@echo "  make docker-build  - Constrói a imagem Docker da API"
	@echo "  make compose-up    - Inicia db, api e scheduler via Docker Compose"
	@echo "  make compose-down  - Encerra os serviços do Docker Compose"
	@echo "  make clean         - Remove artefatos temporários e caches"

install:
	@if command -v uv >/dev/null 2>&1; then \
		uv sync --extra dev; \
	else \
		python3 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -e ".[dev]"; \
	fi

dev:
	$(PYTHON) manage.py runserver 0.0.0.0:8000

test:
	DJANGO_TEST=1 $(PYTEST)

lint:
	$(RUFF) check .

migrate:
	$(PYTHON) manage.py migrate

retrain:
	$(PYTHON) manage.py retrain_models

docker-build:
	docker build -t $(IMAGE_NAME) .

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info db.sqlite3 models_data
