# Architecture — AI-агентный слой

> Дополнение к `docs/spec.md`: диаграмма, модули, потоки данных, трейсы демо-сценариев
> и контракт интеграции с app-слоем.
> Зона: `backend/agents/`, `backend/llm/`, `backend/tracing/`.

## 1. Обзор

AI-слой — конвейер «данные → решение → текст», собранный вокруг оркестратора.
Оркестратор вызывает специализированных агентов, применяет ограничения, выбирает
лучший вариант и передаёт его Formatter'у для перевода в человеческий текст. Все
шаги пишутся в трейсер (SQLite). GigaChat используется только для формулировки
текста и Q&A, а не для принятия решений — управляющие решения принимает
детерминированный код.

## 2. Диаграмма

```
                       app/routes (HTTP)
             /recommend      /ask        /trace
                  │             │           │
                  ▼             ▼           ▼
        ┌──────────────────────────────────────────┐
        │            Orchestrator.decide()         │   backend/agents/orchestrator.py
        └──────┬───────────────┬──────────────┬────┘
               │               │              │
        ┌──────▼─────┐  ┌──────▼──────┐  ┌────▼─────────────┐
        │ DataAgent  │  │QualityAgent │  │ ReliabilityAgent │   (Quality+Reliability
        └──────┬─────┘  └──────┬──────┘  └────┬─────────────┘    через asyncio.gather)
               │               │              │
        ┌──────▼─────┐  ┌──────▼──────┐       │ limits → constraints
        │Anomaly     │  │QualityAVT    │       └──────────────────┐
        │Detector    │  │QualityHydro  │                          │
        │Simulator   │  │BlendingModel │                          ▼
        └────────────┘  └─────────────┘               ┌─────────────────────┐
                                                      │ OptimizationAgent   │
                                                      │ ParetoOptimizer     │
                                                      └──────────┬──────────┘
                                                                 ▼
                                                      ┌─────────────────────┐
                                                      │      Formatter      │──▶ GigaChatClient ──▶ text
                                                      │  (fallback при       │
                                                      │  LLMUnavailableError)│
                                                      └──────────┬──────────┘
                                                                 ▼
                                                      ┌─────────────────────┐
                                                      │     AgentTracer     │──▶ SQLite (store.py)
                                                      └─────────────────────┘

   Q&A:  /ask ──▶ qa_handler.answer() ──▶ GigaChatClient ◀──▶ MCPServer (read-only tools)
```

## 3. Компоненты

| Компонент | Файл | Ответственность | Зависимость |
|---|---|---|---|
| DataAgent | `agents/data_agent.py` | свежесть/аномалии данных | `AnomalyDetector`, `Simulator` |
| QualityAgent | `agents/quality_agent.py` | прогноз качества и спеки | `QualityAVTModel`, `QualityHydroModel`, `BlendingModel` |
| ReliabilityAgent | `agents/reliability_agent.py` | класс риска и доп. лимиты | `feature_registry.yaml` |
| OptimizationAgent | `agents/optimization_agent.py` | набор Парето-вариантов | `ParetoOptimizer` |
| Orchestrator | `agents/orchestrator.py` | цикл принятия решения | все агенты, `Formatter`, `AgentTracer` |
| GigaChatClient | `llm/gigachat_client.py` | запросы к LLM + кеш + fallback | GigaChat API |
| MCPServer | `llm/mcp_server.py` | read-only инструменты для LLM | `Simulator`, `QualityAgent`, `AgentTracer` |
| Formatter | `llm/formatter.py` | JSON-решение → текст | `GigaChatClient`, `guardrails` |
| QA handler | `llm/qa_handler.py` | ответы на вопросы оператора | `GigaChatClient`, `MCPServer` |
| Guardrails | `llm/guardrails.py` | пост-фильтр чисел в тексте LLM | — |
| AgentTracer | `tracing/logger.py` | запись шагов обмена | `store.py` (SQLite) |
| Store | `tracing/store.py` | персистентность трейса | SQLite |

## 4. Поток данных (цикл `decide`)

1. `tracer.start_decision()` → новый `decision_id`.
2. `DataAgent.check(timestamp)` → `DataAgentOutput` (stale/anomalies).
   Если `is_stale` или (`has_anomalies` и `anomaly_score > 0.7`) → `mode="refuse"`.
3. Параллельно: `QualityAgent.forecast` + `ReliabilityAgent.assess` (`asyncio.gather`).
4. Проверка «нужно ли вмешательство»: все `combined_spec_risk < 0.1` и
   `severity_class == "normal"` → `mode="silent"`.
5. Сборка `OptimizationConstraints` (hard + controllable_ranges + max_deviation_pct).
6. `OptimizationAgent.find_variants` → `list[Variant]`.
7. Фильтр `feasible` → если пусто, `mode="refuse"`.
8. Ранжирование по взвешенной сумме (safety 0.5, quality 0.3, yield 0.15, energy 0.05).
9. Выбор лучшего + 2–3 альтернативы.
10. `Formatter.format_recommendation` (с fallback и guardrail).
11. `OrchestratorOutput(mode="recommend", payload, explanation_text)`.

Каждый шаг фиксируется в `AgentTracer.record(...)` и доступен через
`get_trace(decision_id)`.

Приоритет разрешения конфликтов: безопасность > качество > экономика. Лимиты из
`ReliabilityAgent` явно попадают в `controllable_ranges` и сужают диапазон поиска
оптимизатора.

## 5. Трейсы демо-сценариев

Трейсер хранит по одному шагу на вызов агента: `(step_no, agent, input_hash,
output_summary, duration_ms, timestamp)`. `output_summary` — JSON (обрезка до 500
символов). Ниже — ожидаемая последовательность шагов для каждого сценария.

### 5.1 `stable_period` → `mode="silent"`

Данные свежие, аномалий нет, все риски < 0.1, `severity_class="normal"`.
До оптимизатора цикл не доходит.

| step_no | agent | output (кратко) |
|---|---|---|
| 1 | data | `{"timestamp": "...", "is_stale": false, "has_anomalies": false, "anomaly_score": 0.0, ...}` |
| 2 | quality | `{"combined_spec_risk": {"sulfur_over_10": 0.03}, ...}` |
| 3 | reliability | `{"severity_class": "normal", "severity_score": 0.0, "risk_factors": [], "limits": {...}}` |
| 4 | orchestrator | `{"mode": "silent", "payload": {}, "explanation_text": "Режим стабилен, вмешательство не требуется."}` |

### 5.2 `sulfur_risk` → `mode="recommend"`

Прогноз серы с риском превышения спеки (`sulfur_over_10 >= 0.1`). Проходит полный
цикл с оптимизацией.

| step_no | agent | output (кратко) |
|---|---|---|
| 1 | data | `{"is_stale": false, "has_anomalies": false, ...}` |
| 2 | quality | `{"combined_spec_risk": {"sulfur_over_10": 0.40}, ...}` |
| 3 | reliability | `{"severity_class": "normal", "limits": {"hydro_T5": [293.32, 384.39], ...}}` |
| 4 | optimization | `{"variants": [{"action": {"hydro_T5": 310.0}, "feasible": true, ...}], "infeasible_reasons": []}` |
| 5 | orchestrator | `{"mode": "recommend", "payload": {"best_variant": {...}, "alternatives": [...], "checks": {...}, "weights": {...}}, "explanation_text": "..."}` |

### 5.3 `stale_data` → `mode="refuse"`

ЛИМС старше 24 ч и аномальная телеметрия. Отказ происходит на шаге проверки данных,
до прогноза и оптимизации.

| step_no | agent | output (кратко) |
|---|---|---|
| 1 | data | `{"is_stale": true, "has_anomalies": true, "anomaly_score": 0.9, "warnings": ["Обнаружена аномалия...", "Устаревшие лабораторные значения: ..."]}` |
| 2 | orchestrator | `{"mode": "refuse", "payload": {"reason": "..."}, "explanation_text": "..."}` |

`get_trace(decision_id)` возвращает эти шаги в порядке `step_no`; фронт запрашивает
их через `GET /trace/{decision_id}`.

## 6. Соответствие критериям приёмки (`spec.md` §12)

| # | Критерий | Доказательство |
|---|---|---|
| 1 | Контракты `schemas.py` реализованы и покрыты тестами | `agents/schemas.py` + `tests/test_schemas.py` (6) |
| 2 | Пять классов агентов, у каждого асинхронный метод | `check`/`forecast`/`assess`/`find_variants`/`decide` в `agents/*.py` |
| 3 | Полный цикл `decide` + параллельный вызов Quality/Reliability | `tests/test_orchestrator_flow.py` (6), `asyncio.gather` в `orchestrator.py` |
| 4 | Три демо-сценария возвращают ожидаемый `mode` | `tests/test_scenarios.py` (3) |
| 5 | `decide` без LLM < 3 с, с LLM < 10 с | `tests/test_orchestrator_flow.py::test_decide_completes_under_3_seconds_without_llm`; LLM-часть — бюджет GigaChat (см. §7.4) |
| 6 | Fallback при недоступном GigaChat | `tests/test_formatter.py`, `tests/test_llm_client.py` |
| 7 | Guardrail-тест на выдуманные числа | `tests/test_llm_guardrails.py` (9) |
| 8 | Трейсер логирует все шаги, `get_trace` возвращает полную последовательность | `tests/test_tracer.py` (7), `tests/test_scenarios.py` (проверка состава трейса) |
| 9 | Промпты, диаграмма, `docs/architecture_agents.md` | `llm/prompts/*`, настоящий документ |

## 7. Интеграция с app-слоем (выполнена)

AI-слой подключён к HTTP-роутам в `neftecode-main`. Адаптер лежит в
`app/ai_adapter.py`, сборка агентов — в `app/deps.py`. Ниже — фактическая
реализация и контракт маппинга `OrchestratorOutput` → `RecommendationResponse`.

### 7.1 `OrchestratorOutput` → `RecommendationResponse`

`Orchestrator.decide(timestamp)` возвращает `OrchestratorOutput`:

```python
{
    "decision_id": str,
    "mode": Literal["recommend", "silent", "refuse"],
    "payload": {
        # для recommend:
        "best_variant": {...},       # ml.types.Variant (asdict)
        "alternatives": [...],
        "checks": {"combined_spec_risk": {...}, "severity_class": str},
        "weights": {"safety": 0.5, "quality": 0.3, "yield": 0.15, "energy": 0.05},
        # для refuse:
        "reason": str,
        # для silent: {}
    },
    "trace_id": str,
    "explanation_text": str,
}
```

Текущий `app/schemas.RecommendationResponse` использует форму
`variants: list[Variant]`, `default_weights`, `trace: list[TraceStep]`,
`timestamp`, `warnings`. Требуется адаптер (в `app/`):
`best_variant + alternatives → variants`, `payload.weights → default_weights`,
`get_trace(decision_id) → trace`, `decision_id → timestamp` (или добавить
`timestamp` в `payload`).

Варианты в `payload` — это `ml.types.Variant` (`action`, `expected`,
`metrics{sulfur,yield,energy,severity}`, `feasible`, `infeasible_reason`),
тогда как `app.schemas.Variant` ждёт `id`, `action`, `delta`, `metrics
{safety,yield,energy,wear}`, `predicted`. Маппинг и рендер — на стороне `app/`
(см. также §7.3 про `/trace`).

### 7.2 Объекты в `app/deps.py`

Для полного цикла нужно инстанцировать и передать в оркестратор:

```python
simulator   = get_simulator(settings.data_dir)
anomaly     = AnomalyDetector.load(ml/artifacts/anomaly.pkl)
quality_avt = QualityAVTModel.load(ml/artifacts/quality_avt.pkl)
quality_hydro = QualityHydroModel.load(ml/artifacts/quality_hydro.pkl)
blending    = BlendingModel()
optimizer   = ParetoOptimizer(quality_avt, quality_hydro, blending)

data         = DataAgent(anomaly=anomaly, simulator=simulator, stale_threshold_hours=24)
quality      = QualityAgent(avt=quality_avt, hydro=quality_hydro, blending=blending)
reliability  = ReliabilityAgent(registry_path=feature_registry.yaml)
optimization = OptimizationAgent(optimizer=optimizer)
tracer       = AgentTracer(settings.tracing_db_path)
formatter    = Formatter(client=gigachat_client)

orch = Orchestrator(data, quality, reliability, optimization,
                    formatter=formatter, tracer=tracer)
```

`app/deps.py` теперь собирает весь пайплайн лениво через `@lru_cache`:
`_models()` возвращает реальные модели при наличии артефактов (иначе `None`
→ mock-режим), а `_data_agent`/`_quality_agent`/`_reliability_agent`/
`_optimization_agent` и `_orchestrator` инстанцируют агентов. `DataAgent` всегда
получает реальный `Simulator` (state), `ReliabilityAgent` — реальный
`feature_registry.yaml`. При отсутствии артефактов агенты работают в mock-режиме,
а `Formatter` деградирует до fallback-текста.

### 7.3 `/trace/{decision_id}`

Роут возвращает `list[TraceStep]` (`agent, duration_ms, input_summary, output`)
через `trace_to_steps(tracer, decision_id)` из `app/ai_adapter.py`: `TraceEntry`
→ `TraceStep`, `output_summary` (JSON, до 500 символов) разбирается в `output`,
`input_hash` усекается до 16 символов в `input_summary`.

### 7.4 Замечание про критерий №5

`ParetoOptimizer.MAX_OPTIMIZATION_SECONDS = 4.0` — внутренний бюджет оптимизации.
Формально это больше порога «без LLM < 3 с». На практике TPESampler обычно
завершается быстрее, но для честного демо порог оркестратора (3 с) и бюджет
оптимизатора (4 с) нужно согласовать с ML.

## 8. Модульная карта

| Модуль | Состояние |
|---|---|
| `agents/schemas.py` | готово: полный набор контрактов |
| `agents/*.py` | готово: 5 агентов + оркестратор (11 шагов) |
| `llm/*.py` | готово: клиент + MCP + Formatter + Q&A + guardrails |
| `tracing/*.py` | готово: трейсер + SQLite (авто-схема, WAL) |
| `app/routes/recommend.py` | подключён: `orch.decide` + `app/ai_adapter.py`, fallback на `ml_adapter` |
| `app/ai_adapter.py` | адаптер `OrchestratorOutput` → `RecommendationResponse` + `trace_to_steps` |

## 9. Синхронизация с ML

- Типы (`QualityPrediction`, `Variant`, `OptimizationConstraints`) заморожены в `docs/api_ml.md`.
- До готовности реальных моделей используются mock-режимы обёрток; оркестратор работает параллельно.
- Перед демо: замена моков на реальные модели и совместный прогон трёх сценариев (`tests/test_scenarios.py`).
- Реальные timestamp сценариев подбираются совместно с ML после появления `master.parquet` и артефактов.


