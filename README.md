# Neftecode

Мультиагентный советник для оператора НПЗ по цепочке АВТ, гидроочистка и блендинг.
Прогнозирует качество товарного дизеля, оценивает риск выхода за спеку, предлагает
управляющие воздействия. Если данных не хватает - отказывается от рекомендации.

## Стек

Backend: Python 3.11, FastAPI, pandas, polars, LightGBM, Optuna, pydantic v2, MCP, GigaChat.
Frontend: Next.js 14, TypeScript, Tailwind, Recharts, TanStack Query.
Инфра: Docker Compose, SQLite для трейсов.

## Запуск

```bash
cp .env.example .env
make up
```

- Backend Swagger: http://localhost:8000/docs
- Frontend: http://localhost:3000

## Структура

```
neftecode/
├── docs/               # наша документация (ТЗ, справочники)
├── hackathon/          # gitignored, папка с исходниками хакатона
├── data/               # gitignored, parquet-кеш
├── backend/
│   ├── app/            # FastAPI, роуты
│   ├── data_layer/     # loaders, time-join, simulator
│   ├── ml/             # модели, optimizer, validation
│   ├── agents/         # агенты, orchestrator
│   ├── llm/            # MCP, GigaChat, formatter
│   ├── tracing/        # логи агентов
│   └── tests/
├── frontend/
│   ├── app/
│   ├── components/
│   └── lib/
├── docker-compose.yml
├── Makefile
└── .env.example
```

## API

| Метод | Эндпоинт | Что делает |
|---|---|---|
| POST | /recommend            | timestamp -> карточка рекомендации + decision_id |
| POST | /ask                  | вопрос оператора -> ответ GigaChat |
| GET  | /state?ts=            | снимок телеметрии + свежесть ЛИМС |
| GET  | /trace/{decision_id}  | обмен агентов за один цикл |
| GET  | /history?tag=         | ряд для графика |
| GET  | /scenarios            | демо-сценарии |
| POST | /scenarios/{id}/run   | запуск сценария |

## Данные

Исходники хакатона в папку `hackathon/` (не коммитятся). Ожидаемый состав:
- `data/avt_tags.csv` - телеметрия АВТ (74 тега, 10-мин).
- `data/242000_tags.csv` - телеметрия гидроочистки (28 тегов).
- `Выгрузка ПАК ...xlsx` - поточные анализаторы.
- `ЛИМСы ...xlsx` - лабораторные анализы.
- `Теги_хакатон.xlsx` - справочник + формулы ВАК.
- `АВТ_схемы.pdf`, `ТЗ_нефтекод.docx`.

Файлы раздают организаторы хакатона, лежат в общем хранилище команды. Подробнее - в `docs/tz/data_reference.md`.

## Команды

```bash
make up        # поднять
make logs      # смотреть логи
make down      # остановить
make train     # обучить модели
make test      # тесты
make clean     # снести всё вместе с volumes
```
