# Структура проекта

Документ описывает: как разложен код, из каких слоёв состоит система, кто чем владеет из команды, какие есть соглашения. Обязательное чтение перед первым коммитом.

## Общий взгляд

Система - веб-приложение из двух сервисов в Docker Compose:
- `backend` (Python 3.11 + FastAPI) - вся логика, ML, агенты, интеграция с GigaChat.
- `frontend` (Next.js 14 + TypeScript) - дашборд оператора.

Внутри backend работают семь слоёв, каждый строго над предыдущим:

```
┌─────────────────────────────────────────────────┐
│  app/          HTTP-роуты FastAPI, DI-контейнер │
├─────────────────────────────────────────────────┤
│  agents/       Оркестратор + 5 агентов          │
├─────────────────────────────────────────────────┤
│  llm/          MCP + GigaChat + formatter + Q&A │
├─────────────────────────────────────────────────┤
│  ml/           Модели, оптимизатор, обучение    │
├─────────────────────────────────────────────────┤
│  data_layer/   Loaders, time-join, simulator    │
├─────────────────────────────────────────────────┤
│  tracing/      SQLite-лог обмена агентов        │
├─────────────────────────────────────────────────┤
│  hackathon/    Сырьё, не коммитится             │
└─────────────────────────────────────────────────┘
```

Правило: слой видит только то, что ниже. `agents/` использует `ml/` и `data_layer/`, но не наоборот.

## Полное дерево

```
neftecode/
├── docs/                       # коммитится
│   └── tz/
│       ├── tz_ml.md
│       ├── tz_ai_agent.md
│       ├── data_reference.md
│       └── structure.md        # этот документ
│
├── hackathon/                  # gitignore, у каждого локально
│   ├── data/
│   │   ├── avt_tags.csv
│   │   └── 242000_tags.csv
│   ├── Выгрузка ПАК ...xlsx
│   ├── ЛИМСы ...xlsx
│   ├── Теги_хакатон.xlsx
│   ├── АВТ_схемы.pdf
│   └── ТЗ_нефтекод.docx
│
├── data/                       # gitignore, автогенерируется
│   ├── *.parquet               # кеш из hackathon/
│   └── tracing.db              # SQLite трейсов
│
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI-приложение
│   │   ├── config.py           # settings (env + defaults)
│   │   ├── deps.py             # DI-контейнер
│   │   └── routes/
│   │       ├── health.py
│   │       ├── recommend.py
│   │       ├── ask.py
│   │       ├── state.py
│   │       ├── trace.py
│   │       ├── history.py
│   │       └── scenarios.py
│   │
│   ├── data_layer/
│   │   ├── loaders.py          # чтение csv/xlsx -> parquet
│   │   ├── time_join.py        # синхронизация источников
│   │   ├── simulator.py        # get_state(timestamp)
│   │   └── feature_registry.yaml
│   │
│   ├── ml/
│   │   ├── types.py            # общие dataclass-типы
│   │   ├── feature_engineering.py
│   │   ├── models/
│   │   │   ├── quality_avt.py
│   │   │   ├── quality_hydro.py
│   │   │   ├── blending.py
│   │   │   └── anomaly.py
│   │   ├── optimizer/
│   │   │   └── pareto.py
│   │   ├── validation/
│   │   │   └── walk_forward.py
│   │   ├── training/
│   │   │   └── train_all.py
│   │   └── artifacts/          # gitignore, .pkl моделей
│   │
│   ├── agents/
│   │   ├── schemas.py          # pydantic-контракты
│   │   ├── data_agent.py
│   │   ├── quality_agent.py
│   │   ├── reliability_agent.py
│   │   ├── optimization_agent.py
│   │   └── orchestrator.py
│   │
│   ├── llm/
│   │   ├── gigachat_client.py
│   │   ├── mcp_server.py
│   │   ├── formatter.py
│   │   ├── qa_handler.py
│   │   └── prompts/
│   │       ├── system.md
│   │       ├── formatter_system.md
│   │       └── qa_system.md
│   │
│   ├── tracing/
│   │   ├── logger.py           # AgentTracer
│   │   └── store.py            # SQLite-обёртка
│   │
│   ├── tests/
│   │   ├── test_health.py
│   │   ├── test_schemas.py
│   │   ├── test_orchestrator_flow.py
│   │   ├── test_llm_guardrails.py
│   │   └── test_scenarios.py
│   │
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .dockerignore
│
├── frontend/
│   ├── app/                    # Next.js App Router
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   └── globals.css
│   ├── components/
│   │   ├── StatePanel.tsx
│   │   ├── RecommendationCard.tsx
│   │   ├── AgentTrace.tsx
│   │   ├── ParetoChart.tsx
│   │   ├── TimeMachine.tsx
│   │   ├── QAChat.tsx
│   │   └── TagChart.tsx
│   ├── lib/
│   │   └── api.ts              # HTTP-клиент к backend
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.mjs
│   ├── tailwind.config.ts
│   ├── postcss.config.mjs
│   ├── Dockerfile
│   └── .dockerignore
│
├── docker-compose.yml
├── Makefile
├── .env.example
├── .gitignore
└── README.md
```

## Слои backend

### `data_layer/`

Читает исходники хакатона, приводит к единому формату, отдаёт срез процесса на момент времени.

Ключевые функции:
- `loaders.load_all(hackathon_dir, cache_dir)` - конвертирует все CSV/XLSX в parquet.
- `time_join.build_master_frame(parquet_dir)` - собирает единый DataFrame с телеметрией, ПАК и ЛИМС.
- `simulator.get_state(timestamp)` - возвращает срез без утечки будущего.

Что кладётся в `data/`:
- `avt_tags.parquet`, `hydro_tags.parquet` - телеметрия.
- `pak_sulfur.parquet`, `pak_d15.parquet` - анализаторы.
- `lims_long.parquet` - ЛИМС в длинном формате (date, sampling_point, indicator, value).
- `master.parquet` - объединённый фрейм с колонками telemetry + latest LIMS + lims_age_hours.

Всё в `data/` регенерируется одной командой, руками ничего не правится.

### `ml/`

Обучаемые модели и оптимизатор. Полное ТЗ - `docs/tz/tz_ml.md`.

Артефакты обучения (`ml/artifacts/*.pkl`) в git не попадают, генерируются `make train`.

Публичный интерфейс (что импортируют agents/):
- `ml.types` - `QualityPrediction`, `Interval`, `Variant`, `OptimizationConstraints`.
- `ml.models.QualityAVTModel`, `QualityHydroModel`, `BlendingModel`, `AnomalyDetector`.
- `ml.optimizer.ParetoOptimizer`.

### `agents/`

Обёртки над ML-компонентами плюс оркестратор. Полное ТЗ - `docs/tz/tz_ai_agent.md`.

`schemas.py` - единый источник pydantic-контрактов для обмена. Меняется только по согласованию с MLщиком и тимлидом.

`orchestrator.Orchestrator.decide(timestamp)` - единственная точка входа для верхнего слоя. Возвращает `OrchestratorOutput` с полем `mode` в `{"recommend", "silent", "refuse"}`.

### `llm/`

Всё, что связано с GigaChat.

- `gigachat_client.GigaChatClient` - обёртка над SDK с кешем и явным `LLMUnavailableError`.
- `mcp_server.MCPServer` - MCP-сервер с read-only инструментами.
- `formatter.format_recommendation(decision, client)` - JSON решения в человеческий текст. Имеет fallback без LLM.
- `qa_handler.answer(decision_id, question, client)` - ответ на вопрос оператора.
- `prompts/*.md` - системные промпты, версионируются в git.

Правило: LLM не принимает управляющих решений, только формулирует.

### `tracing/`

`AgentTracer` пишет в SQLite (`data/tracing.db`). Один цикл принятия решения = один `decision_id`, все сообщения агентов связаны им.

Читается через `get_trace(decision_id)`, используется роутом `/trace/{decision_id}` и дашбордом.

### `app/`

Тонкий слой над всем остальным.

- `main.py` - FastAPI-приложение, CORS, подключение роутеров.
- `config.py` - `Settings` через `pydantic-settings`. Читает `.env`.
- `deps.py` - функции-фабрики для FastAPI Depends (модели, оркестратор, трейсер). Здесь всё инстанцируется один раз при старте.
- `routes/*.py` - по одному файлу на группу эндпоинтов, каждый экспортирует `router: APIRouter`.

Правило: в роутах никакой логики. Только вызов из `deps` и возврат pydantic-модели.

## Frontend

Next.js 14 App Router. TypeScript strict mode.

- `app/page.tsx` - главная страница дашборда, композиция компонентов.
- `app/layout.tsx` - корневая обёртка, метаданные, глобальные стили.
- `components/*.tsx` - каждый компонент в своём файле, экспорт named (не default).
- `lib/api.ts` - единственный HTTP-клиент, все запросы к backend через него.

Правило: fetch внутри компонентов - через TanStack Query, не через прямой `fetch` в JSX.

## Границы зон ответственности

| Каталог | Владелец |
|---|---|
| `backend/data_layer/` | Тимлид |
| `backend/ml/` | MLщик |
| `backend/agents/` | AI-агентщик |
| `backend/llm/` | AI-агентщик |
| `backend/tracing/` | AI-агентщик |
| `backend/app/` | Тимлид |
| `frontend/` | Тимлид |
| `docker-compose.yml`, `Dockerfile`, `Makefile`, `.env*` | Тимлид |
| `docs/tz/` | Тимлид (правки от всех) |

Правила изменений в чужой зоне:
- Мелкая правка (typo, docstring) - можно без согласования, PR с меткой `trivial`.
- Изменение интерфейса (сигнатура публичного метода) - только после согласования с владельцем.
- Добавление новой зависимости - согласование с тимлидом.

## Правила и соглашения

### Python

- Форматирование: `ruff format`, длина строки 100.
- Линтинг: `ruff check`.
- Типизация: везде, где это не создаёт лишнего шума. Возвращаемый тип публичных методов обязателен.
- Docstring: только там, где WHY не очевиден из имён. Не описывать WHAT.
- Async: HTTP-роуты, оркестратор, вызовы LLM. Обучение моделей и feature engineering - sync.
- Импорты: абсолютные внутри пакета (`from ml.types import QualityPrediction`, не `from ..types`).

### TypeScript

- Строгий режим (`strict: true`).
- Экспорт named, не default (проще рефакторить).
- Никаких `any`, использовать `unknown` там, где тип неизвестен.
- Компоненты - функциональные, никаких классов.
- Стили - Tailwind, никаких CSS-модулей и styled-components.

### Именование

- Файлы Python - snake_case.
- Классы - PascalCase.
- Функции и переменные - snake_case.
- Константы - UPPER_SNAKE.
- Файлы TSX - PascalCase, если это компонент; camelCase, если утилиты.
- Ветки git - `feature/xxx`, `fix/xxx`, `docs/xxx`.

### Логи

- Все агенты пишут в `AgentTracer` через хук в базовом классе. Ручные `print` не используются.
- Уровни: `INFO` - обычная работа, `WARNING` - что-то подозрительное, `ERROR` - падение обрабатываемое, `CRITICAL` - падение необрабатываемое.
- Ошибки LLM - `WARNING` (сработал fallback), а не `ERROR`.

## Локальный запуск

Стандартный путь:

```bash
cp .env.example .env
# заполнить GIGACHAT_CREDENTIALS
make up
```

- Frontend: http://localhost:3000
- Backend Swagger: http://localhost:8000/docs

Без Docker (для отладки одного слоя):

```bash
# backend
cd backend
python -m venv .venv
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload

# frontend
cd frontend
pnpm install
pnpm dev
```

## Инфраструктура

### Docker Compose

Два сервиса, `backend` и `frontend`, оба с bind-mount исходников для hot-reload. Порты 8000 и 3000 наружу. Папка `hackathon/` монтируется в backend только для чтения (`:ro`).

Health-check backend: pull `/health`, интервал 15 секунд.

### Makefile

Основные команды:
- `make up` - поднять всё.
- `make down` - остановить.
- `make build` - пересобрать образы без кеша.
- `make logs` - смотреть логи.
- `make restart` - перезапустить контейнеры.
- `make clean` - снести всё вместе с volumes.
- `make train` - обучить ML-модели внутри backend-контейнера.
- `make test` - прогнать pytest.

### Env

`.env` содержит секреты и локальные настройки, в git не коммитится. Шаблон - `.env.example`. При добавлении новой переменной в `Settings` (backend/app/config.py) обязательно синхронизировать `.env.example`.

## Как добавлять новое

### Новый роут FastAPI

1. Создать `backend/app/routes/новое.py` с `router = APIRouter(tags=["..."])`.
2. Определить pydantic-схемы прямо в файле (небольшие) или в `agents/schemas.py` (если контракт агентский).
3. Подключить в `backend/app/main.py`: `app.include_router(новое.router)`.

### Новая ML-модель

1. Добавить класс в `backend/ml/models/xxx.py`.
2. Добавить обучающий скрипт в `backend/ml/training/`.
3. Расширить `training/train_all.py`, чтобы модель поднималась одной командой.
4. Добавить типы в `backend/ml/types.py`, если нужны новые.
5. Обновить `metrics_report.json` схему.

### Новый агент

1. Добавить `schemas.py`: `<Agent>Input`, `<Agent>Output`.
2. Создать `backend/agents/<agent>.py` с классом и асинхронным методом.
3. Прописать вызов в `orchestrator.py`.
4. Обновить диаграмму в `docs/tz/tz_ai_agent.md`.

### Новый компонент фронта

1. Файл `frontend/components/<Name>.tsx`, named export.
2. Если нужен запрос к API - через `lib/api.ts` + TanStack Query.
3. Стили - Tailwind в className.
4. Подключить в `app/page.tsx`.

## Тестирование

Backend: pytest, лежит в `backend/tests/`. Запуск: `make test`.

Обязательные категории тестов:
- Контракты pydantic (`test_schemas.py`).
- Три демо-сценария end-to-end (`test_scenarios.py`).
- Guardrails LLM (`test_llm_guardrails.py`).
- Валидация ML без утечки времени (`ml/validation/`).

Frontend: пока без тестов. Приоритет - работающий UI.
