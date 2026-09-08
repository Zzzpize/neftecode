# ТЗ: AI-агентщик

## Роль

Зона ответственности: мультиагентная архитектура, оркестратор, обмен сообщениями между агентами, интеграция с GigaChat через MCP, формулировка рекомендаций и Q&A с оператором. Всё, что превращает набор ML-моделей и данных в единого «советника», разговаривающего с пользователем.

## Контекст

ML отдаёт готовые модели-«оракулы» с фиксированным API (см. `docs/tz/tz_ml.md` и `docs/api_ml.md`). Data layer отдаёт срез процесса на момент времени. Задача AI-слоя: правильно позвать нужных агентов, разрешить конфликты между ними, применить жёсткие ограничения, сформулировать понятный текст для оператора. Все критические решения (что делать с процессом) принимает детерминированный код, GigaChat отвечает только за формулировку и Q&A.

## Состав работ

Реализуются семь компонентов:

1. Контракты сообщений между агентами (pydantic-модели).
2. Пять агентов: Data, Quality, Reliability, Optimization, плюс оркестратор.
3. Клиент GigaChat с кешем и fallback-режимом.
4. MCP-сервер с read-only инструментами для LLM.
5. Formatter: превращение JSON-решения в текст карточки рекомендации.
6. Q&A handler: ответы на вопросы оператора по последней рекомендации.
7. Трейсер обмена агентов: SQLite-хранилище, API для чтения трейса.

## Вне зоны ответственности

- Обучение ML-моделей и feature engineering. Всё это делает ML, наружу отдаются классы с фиксированным API.
- Data layer, чтение CSV/XLSX, синхронизация по времени, симулятор.
- HTTP-роуты FastAPI, интеграция с фронтом.
- Docker, инфраструктура.
- Дашборд и визуализация.

## Данные и зависимости

От ML принимается:
- Модули `backend/ml/types.py`, `backend/ml/models/`, `backend/ml/optimizer/`.
- Экземпляры классов `QualityAVTModel`, `QualityHydroModel`, `BlendingModel`, `AnomalyDetector`, `ParetoOptimizer`.
- Feature registry `backend/data_layer/feature_registry.yaml` для получения диапазонов управляемых тегов.

От data layer принимается:
- Симулятор реалтайма: функция `get_state(timestamp) -> pd.DataFrame`.
- Метаданные тегов (описания, единицы).

Наружу отдаётся:
- Класс `Orchestrator` с методом `async decide(timestamp)`, вызываемый из HTTP-роутов.
- Функция `async answer(decision_id, question)` для Q&A.
- Функция `get_trace(decision_id)` для чтения обмена агентов.

## Требуемые классы и API

### Контракты сообщений (`backend/agents/schemas.py`)

Все обмены между агентами валидируются pydantic v2. Импорт общих ML-типов из `backend/ml/types.py`.

```python
class DataAgentInput(BaseModel):
    timestamp: datetime

class DataAgentOutput(BaseModel):
    timestamp: datetime
    is_stale: bool
    has_anomalies: bool
    anomaly_score: float
    flagged_tags: list[str]
    is_out_of_envelope: bool
    stale_tags: list[str]
    warnings: list[str]

class QualityAgentOutput(BaseModel):
    avt: QualityPrediction        # из ml.types
    hydro: QualityPrediction
    blended: QualityPrediction | None
    combined_spec_risk: dict[str, float]  # {"sulfur_over_10": 0.27, ...}

class ReliabilityAgentOutput(BaseModel):
    severity_class: Literal["normal", "elevated", "heavy"]
    severity_score: float
    risk_factors: list[str]
    limits: dict[str, tuple[float, float]]  # доп. ограничения на управляемые теги

class OptimizationAgentOutput(BaseModel):
    variants: list[Variant]       # из ml.types
    infeasible_reasons: list[str]

class OrchestratorOutput(BaseModel):
    decision_id: str
    mode: Literal["recommend", "silent", "refuse"]
    payload: dict                 # для recommend: best_variant + alternatives + expected + checks
    trace_id: str
    explanation_text: str         # человеческий текст от GigaChat (или fallback)

class AskInput(BaseModel):
    decision_id: str
    question: str

class AskOutput(BaseModel):
    answer: str
    tool_calls_used: int
```

Все контракты покрываются юнит-тестами в `backend/tests/test_schemas.py`.

### Агенты

Каждый агент - тонкая обёртка над ML-компонентами плюс бизнес-логика.

```python
# backend/agents/data_agent.py
class DataAgent:
    def __init__(self, anomaly: AnomalyDetector, simulator: Simulator,
                 stale_threshold_hours: float = 24.0): ...
    async def check(self, inp: DataAgentInput) -> DataAgentOutput: ...

# backend/agents/quality_agent.py
class QualityAgent:
    def __init__(self, avt: QualityAVTModel, hydro: QualityHydroModel,
                 blending: BlendingModel): ...
    async def forecast(self, state: pd.DataFrame) -> QualityAgentOutput: ...
    async def forecast_after_action(self, state: pd.DataFrame,
                                    action: dict[str, float]) -> QualityAgentOutput: ...

# backend/agents/reliability_agent.py
class ReliabilityAgent:
    async def assess(self, state: pd.DataFrame) -> ReliabilityAgentOutput: ...

# backend/agents/optimization_agent.py
class OptimizationAgent:
    def __init__(self, optimizer: ParetoOptimizer): ...
    async def find_variants(self, state: pd.DataFrame,
                            constraints: OptimizationConstraints) -> OptimizationAgentOutput: ...

# backend/agents/orchestrator.py
class Orchestrator:
    def __init__(self, data: DataAgent, quality: QualityAgent,
                 reliability: ReliabilityAgent, optimization: OptimizationAgent,
                 formatter: Formatter, tracer: AgentTracer): ...
    async def decide(self, timestamp: datetime) -> OrchestratorOutput: ...
```

Все методы асинхронные. Вызовы независимых агентов (Quality и Reliability) внутри оркестратора идут параллельно через `asyncio.gather`.

### LLM-слой

```python
# backend/llm/gigachat_client.py

@dataclass
class ToolCall:
    name: str
    arguments: dict

@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    tokens_used: int
    cached: bool

class GigaChatClient:
    def __init__(self, credentials: str, model: str = "GigaChat-Max",
                 verify_ssl: bool = False, cache_size: int = 256): ...
    async def complete(self, system: str, messages: list[dict],
                       tools: list[dict] | None = None,
                       temperature: float = 0.0) -> LLMResponse: ...
    async def health(self) -> bool: ...
```

Требования:
- Кеш ответов по хешу `(system, messages, tools)`. LRU, размер по умолчанию 256.
- Rate limiting: не более 5 одновременных запросов.
- Fallback: при недоступности API метод `complete` бросает `LLMUnavailableError`, вызывающий код обязан обработать.

```python
# backend/llm/mcp_server.py
class MCPServer:
    def __init__(self, simulator, quality_agent, tracer): ...
    def run(self, port: int = 8765) -> None: ...
```

Экспортируемые инструменты (все read-only):
- `get_current_state(timestamp: str)` возвращает срез процесса.
- `get_forecast(timestamp: str, action: dict | None)` возвращает прогноз качества.
- `explain_recommendation(decision_id: str)` возвращает топ-факторы SHAP.
- `get_history(tag: str, ts_from: str, ts_to: str)` возвращает временной ряд.

```python
# backend/llm/formatter.py

async def format_recommendation(decision: OrchestratorOutput,
                                client: GigaChatClient) -> str: ...

def format_recommendation_fallback(decision: OrchestratorOutput) -> str: ...
```

Formatter вызывает GigaChat с промптом из `backend/llm/prompts/formatter_system.md`. При недоступности LLM автоматически используется `format_recommendation_fallback` (шаблонный текст без LLM).

```python
# backend/llm/qa_handler.py

async def answer(decision_id: str, question: str,
                 client: GigaChatClient,
                 max_tool_calls: int = 3) -> AskOutput: ...
```

Q&A цикл: LLM получает контекст последней рекомендации, доступ к MCP-инструментам, отвечает на вопрос. Ограничение на количество tool-calls жёсткое.

### Трейсер

```python
# backend/tracing/logger.py

@dataclass
class TraceEntry:
    decision_id: str
    step_no: int
    agent: str
    input_hash: str
    output_summary: str        # первые 500 символов JSON
    duration_ms: float
    timestamp: datetime

class AgentTracer:
    def __init__(self, db_path: Path): ...
    def start_decision(self) -> str: ...          # возвращает новый decision_id
    def record(self, decision_id: str, agent: str,
               input_data: dict, output: dict, duration_ms: float) -> None: ...
    def get_trace(self, decision_id: str) -> list[TraceEntry]: ...
    def list_recent(self, limit: int = 50) -> list[str]: ...  # последние decision_id
```

Хранилище - SQLite (`backend/tracing/store.py`), схема создаётся автоматически при первом запуске.

## Логика оркестратора

Метод `decide(timestamp)` выполняет следующий цикл. Все шаги логируются через трейсер.

1. Создаётся новый `decision_id` через `tracer.start_decision()`.
2. Вызывается `data_agent.check(timestamp)`. Если `is_stale=True` или `has_anomalies=True` с скором выше порога 0.7, возвращается `OrchestratorOutput(mode="refuse")` с причиной из `warnings`.
3. Параллельно (через `asyncio.gather`) вызываются `quality_agent.forecast(state)` и `reliability_agent.assess(state)`.
4. Проверка «нужно ли вмешательство»: если `combined_spec_risk` для всех показателей ниже порога 0.1 и `severity_class == "normal"`, возвращается `OrchestratorOutput(mode="silent")` с сообщением «режим стабилен, вмешательство не требуется».
5. Сборка `OptimizationConstraints`:
   - `hard` - жёсткие спеки (сера ≤ 10 мг/кг, доли блендинга = 1.0).
   - `controllable_ranges` - объединение диапазонов из feature registry и ограничений от `reliability_agent`.
   - `max_deviation_pct` - 10% от текущего значения по умолчанию.
6. Вызывается `optimization_agent.find_variants(state, constraints)`.
7. Фильтр по `feasible=True`. Если пусто, возвращается `OrchestratorOutput(mode="refuse")` с причиной «нет допустимых вариантов в рамках ограничений».
8. Ранжирование оставшихся вариантов по взвешенной сумме метрик. Веса по умолчанию: безопасность 0.5, качество 0.3, выход 0.15, энергия 0.05.
9. Выбирается лучший, к нему добавляются 2-3 альтернативы для показа на дашборде.
10. Вызывается `formatter.format_recommendation` для получения текста. При `LLMUnavailableError` используется fallback-шаблон.
11. Возвращается `OrchestratorOutput(mode="recommend", payload=..., explanation_text=...)`.

Приоритет разрешения конфликтов: безопасность важнее качества, качество важнее экономики. Если Quality советует поднять температуру, а Reliability считает это недопустимым, оптимизатор не должен предлагать этот вариант благодаря ограничениям.

## Guardrails для LLM

Правила, обязательные к соблюдению:
1. Все числа в тексте, сгенерированном LLM, должны присутствовать в исходном JSON. Пост-фильтр проверяет это регулярным выражением: извлекает числа из ответа и сверяет со значениями в исходной структуре. При обнаружении «изобретённого» числа ответ считается невалидным, применяется fallback.
2. LLM не имеет доступа ни к каким write-операциям. MCP экспортирует только read-only инструменты.
3. Максимум 3 tool-call на один вызов Q&A. Далее ответ обрезается.
4. `temperature=0` во всех вызовах для детерминированности демо.
5. Кеш LLM-ответов включён по умолчанию.
6. При недоступности GigaChat система обязана продолжать работу через fallback-текст.

Тесты в `backend/tests/test_llm_guardrails.py`:
- Проверка, что fallback вызывается при поднятом `LLMUnavailableError`.
- Проверка пост-фильтра на подставленных примерах.
- Проверка ограничения количества tool-calls.

## Демо-сценарии

Реализуются три сценария в `backend/tests/test_scenarios.py`. Каждый возвращает конкретный `mode` от оркестратора:

1. `stable_period` - выбирается временная точка, где нет риска, ожидается `mode="silent"`.
2. `sulfur_risk` - точка, где через час фактически произошло превышение серы (проверяется по истории), ожидается `mode="recommend"` с содержательным изменением параметров.
3. `stale_data` - точка, где ЛИМС старше 48 часов и телеметрия имеет аномалии, ожидается `mode="refuse"`.

Конкретные timestamp для каждого сценария подбираются совместно с ML.

## Артефакты сдачи

К моменту завершения работы в репозитории присутствуют:
- Полностью реализованные модули `backend/agents/`, `backend/llm/`, `backend/tracing/`.
- Промпты в `backend/llm/prompts/`: `system.md`, `formatter_system.md`, `qa_system.md`, few-shot примеры для трёх режимов ответа.
- Заполненный `backend/llm/prompts/README.md` с описанием роли каждого промпта.
- Документ `docs/architecture_agents.md` с диаграммой обмена агентов и примером трейса для каждого демо-сценария.
- Тесты: `test_schemas.py`, `test_orchestrator_flow.py`, `test_llm_guardrails.py`, `test_scenarios.py`. Все проходят в CI (или через `make test`).
- SQLite-схема трейсов инициализируется автоматически при первом запуске.

## Взаимодействие

Точки синхронизации с ML:
- Первый день: фиксация формата `QualityPrediction`, `Variant`, `OptimizationConstraints` в `docs/api_ml.md`. Дальше эти типы не меняются без явного согласования.
- В процессе: заглушки (mock-классы) для всех ML-компонентов, чтобы оркестратор можно было разрабатывать параллельно, до готовности реальных моделей.
- Перед демо: замена моков на реальные модели, совместный прогон трёх сценариев.

Точки синхронизации с тимлидом:
- Первый день: согласование формата `OrchestratorOutput`, поскольку он попадает в HTTP-ответ `/recommend`.
- Согласование, где именно тимлид инстанцирует все объекты (в `backend/app/deps.py`) и передаёт их в оркестратор.
- Согласование формата логов трейсера, которые фронт запрашивает через `/trace/{decision_id}`.

## Порядок работы

Приоритетность:
1. Контракты `schemas.py` и mock-агенты. Оркестратор работает end-to-end на заглушках.
2. Трейсер и SQLite-хранилище. Работает независимо от остальных компонентов.
3. Реальные обёртки агентов над ML-моделями (по мере готовности ML).
4. Клиент GigaChat и MCP-сервер.
5. Formatter с промптом и fallback.
6. Q&A handler.
7. Guardrail-пост-фильтры и тесты.
8. Три демо-сценария.

## Критерии приёмки

Работа принимается, если выполнены все пункты:
1. Все контракты в `schemas.py` реализованы и покрыты тестами.
2. Пять классов агентов реализованы, каждый имеет асинхронный метод из ТЗ.
3. Оркестратор реализует полный цикл `decide` с параллельным вызовом Quality и Reliability.
4. Три демо-сценария возвращают ожидаемый `mode` (silent, recommend, refuse).
5. Один цикл `decide` без LLM выполняется быстрее 3 секунд, с LLM - быстрее 10 секунд.
6. Fallback-режим работает: при недоступном GigaChat система возвращает шаблонный текст, не падает.
7. Guardrail-тест на выдуманные числа проходит.
8. Трейсер логирует все шаги, `get_trace(decision_id)` возвращает полную последовательность.
9. Промпты, диаграмма архитектуры и документ `docs/architecture_agents.md` готовы.

## Риски

1. Галлюцинации LLM. GigaChat может «улучшить» числа в тексте (например, округлить серу с 8.2 до 5). Обязателен пост-фильтр на все числа в ответе. Без него теряется весь смысл guardrails.
2. Latency GigaChat. Реальные ответы 3-8 секунд, что критично для пользовательского опыта. Действия: параллельный вызов агентов, кеш, «оптимистичный рендер» (JSON-факты отдаются сразу, текст LLM подгружается асинхронно).
3. Изменение API моделей после первого дня. При правках нужна общая пересборка обёрток. Мера: фиксация типов в `docs/api_ml.md` в первый день, любые изменения после - через согласование.
4. Конфликты между Quality и Reliability. Если Quality хочет поднять температуру, а Reliability против, оптимизатор должен получить корректные ограничения. Мера: `reliability_agent.limits` явно попадает в `OptimizationConstraints.controllable_ranges` и сужает диапазон.
5. Слишком много tool-call от LLM в Q&A. Оператор задаёт вопрос, LLM гоняет MCP-инструменты 10 раз, всё встаёт. Мера: жёсткий лимит 3 вызова.
6. Незакрытые SQLite-соединения. Мера: трейсер использует контекст-менеджер для каждой записи.
