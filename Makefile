DC = docker compose

.DEFAULT_GOAL := help

.PHONY: help
help: ## Показать эту справку
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: up
up: ## Поднять всё (backend + frontend)
	$(DC) up -d
	@echo "Backend:  http://localhost:8000/docs"
	@echo "Frontend: http://localhost:3000"

.PHONY: down
down: ## Остановить всё
	$(DC) down

.PHONY: build
build: ## Пересобрать контейнеры без кэша
	$(DC) build --no-cache

.PHONY: logs
logs: ## Смотреть логи
	$(DC) logs -f

.PHONY: restart
restart: ## Перезапустить контейнеры
	$(DC) restart

.PHONY: clean
clean: ## Снести контейнеры и volumes
	$(DC) down -v

.PHONY: ingest
ingest: ## Собрать parquet-кеш из папки hackathon
	$(DC) exec backend python -m data_layer.loaders
	$(DC) exec backend python -m data_layer.time_join

.PHONY: train
train: ## Обучить ML-модели
	$(DC) exec backend python -m ml.training.train_all

.PHONY: test
test: ## Прогнать тесты
	$(DC) exec backend pytest