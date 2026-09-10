# Контракт между ML-модулем и AI-агентами


После согласования этого документа названия классов, сигнатуры публичных методов и структура возвращаемых объектов не меняются без отдельного обсуждения. Алгоритмы, признаки и обученные артефакты могут обновляться без изменения этого контракта.

## Что предоставляет ML-модуль

ML-модуль предоставляет пять компонентов:

1. `QualityAVTModel` — прогноз качества дизельной фракции 240–350 после АВТ.
2. `QualityHydroModel` — прогноз качества дизеля после гидроочистки.
3. `BlendingModel` — расчёт свойств смеси компонентов.
4. `AnomalyDetector` — оценка аномальности текущего состояния процесса.
5. `ParetoOptimizer` — подбор вариантов изменения режима.

Публичные типы находятся в `ml.types`. Модели качества, блендинг и детектор аномалий импортируются из `ml.models`, оптимизатор — из `ml.optimizer`.

```python
from ml.models import (
    AnomalyDetector,
    BlendingModel,
    QualityAVTModel,
    QualityHydroModel,
)
from ml.optimizer import ParetoOptimizer
from ml.types import (
    AnomalyReport,
    BlendedProduct,
    Component,
    Explanation,
    Interval,
    OptimizationConstraints,
    QualityPrediction,
    Variant,
)
```

## Формат состояния процесса

Методы `predict`, `predict_after_action`, `explain`, `score` и `find_pareto` принимают `pandas.DataFrame`, содержащий ровно одну строку — состояние процесса на конкретный момент времени.

```python
import pandas as pd

state = pd.DataFrame(
    [
        {
            "date": pd.Timestamp("2025-08-15 12:00:00"),
            "avt_T55": 348.0,
            "avt_F30": 44.5,
            "hydro_T6": 315.2,
            "hydro_P13": 2.1,
            "pak_sulfur_ppm": 8.4,
        }
    ]
)
```

На практике этот срез должен приходить из data layer или симулятора через `get_state(timestamp)`. Самостоятельно склеивать телеметрию, ПАК и ЛИМС внутри агента не нужно.

Правила входного `DataFrame`:

- должна быть ровно одна строка;
- названия колонок должны совпадать с `backend/data_layer/feature_registry.yaml`;
- теги АВТ имеют префикс `avt_`, теги гидроочистки — `hydro_`;
- дополнительные колонки разрешены и могут быть проигнорированы конкретной моделью;
- пропуски разрешены, но могут снизить `confidence` и добавить сообщения в `warnings`;
- состояние должно содержать только информацию, доступную на указанный момент времени.

Если вместо `DataFrame` передан другой объект, модель выбрасывает `TypeError`. Если в `DataFrame` не одна строка, модель выбрасывает `ValueError`.

## Общие типы результата

### `Interval`

Один прогнозируемый показатель с интервалом неопределённости.

```python
@dataclass
class Interval:
    mean: float
    low: float
    high: float
    unit: str
```

- `mean` — центральный прогноз;
- `low` — нижняя граница, квантиль 0.1;
- `high` — верхняя граница, квантиль 0.9;
- `unit` — единица измерения.

В штатном прогнозе выполняется `low <= mean <= high`.

### `QualityPrediction`

Общий ответ модели качества.

```python
@dataclass
class QualityPrediction:
    predictions: dict[str, Interval]
    spec_risk: dict[str, float]
    confidence: Literal["high", "medium", "low"]
    warnings: list[str]
```

- `predictions` — прогнозы показателей качества;
- `spec_risk` — вероятности нарушения спецификаций в диапазоне от `0.0` до `1.0`;
- `confidence` — общая уверенность модели в ответе;
- `warnings` — причины, по которым прогноз нужно интерпретировать осторожно.

Значения `confidence`:

- `high` — состояние находится в знакомой области, значимых проблем с данными нет;
- `medium` — есть умеренные пропуски, устаревшие данные или небольшое отклонение от исторического режима;
- `low` — данных недостаточно, они устарели либо состояние заметно выходит за исторический диапазон.

Агент не должен скрывать `warnings`. При `confidence="low"` нельзя формулировать рекомендацию как гарантированный результат.

Пример:

```python
QualityPrediction(
    predictions={
        "sulfur_ppm": Interval(
            mean=8.6,
            low=7.4,
            high=10.3,
            unit="ppm",
        ),
        "D15": Interval(
            mean=831.2,
            low=829.8,
            high=832.7,
            unit="kg/m3",
        ),
    },
    spec_risk={"sulfur_over_10": 0.24},
    confidence="medium",
    warnings=["Последний анализ ЛИМС старше 24 часов"],
)
```

### `Explanation`

Числовое объяснение прогноза.

```python
@dataclass
class Explanation:
    top_features: list[tuple[str, float]]
    base_value: float
```

`top_features` содержит до 10 пар `(имя признака, SHAP-значение)`, отсортированных по убыванию абсолютного влияния. Положительное значение сдвигает прогноз вверх, отрицательное — вниз. `base_value` — базовое значение модели до учёта влияния признаков.

Так как один вызов модели возвращает несколько показателей, `explain` относится к главному показателю модели:

- для `QualityAVTModel` — `T50`;
- для `QualityHydroModel` — `sulfur_ppm`.

Вызывать `explain` следует с тем же состоянием, для которого перед этим был получен прогноз.

```python
prediction = model.predict(state)
explanation = model.explain(state)
```

## Модель качества АВТ

Путь к классу: `ml.models.QualityAVTModel`.

Модель прогнозирует:

- `T50`, °C;
- `T90`, °C;
- `D15`, кг/м³;
- `CFPP`, °C.

Публичный API:

```python
class QualityAVTModel:
    @classmethod
    def load(cls, path: str) -> "QualityAVTModel": ...

    def predict(
        self,
        state: pd.DataFrame,
    ) -> QualityPrediction: ...

    def predict_after_action(
        self,
        state: pd.DataFrame,
        action: dict[str, float],
    ) -> QualityPrediction: ...

    def explain(
        self,
        state: pd.DataFrame,
    ) -> Explanation: ...
```

Загрузка и прогноз:

```python
model = QualityAVTModel.load(
    "ml/artifacts/quality_avt.pkl"
)

prediction = model.predict(state)

t50 = prediction.predictions["T50"]
print(t50.mean, t50.low, t50.high, t50.unit)
```

## Модель качества гидроочистки

Путь к классу: `ml.models.QualityHydroModel`.

Модель прогнозирует:

- `sulfur_ppm`, ppm — главный показатель;
- `T50`, °C;
- `T90`, °C;
- `D15`, кг/м³.

Публичный API идентичен модели АВТ:

```python
class QualityHydroModel:
    @classmethod
    def load(cls, path: str) -> "QualityHydroModel": ...

    def predict(
        self,
        state: pd.DataFrame,
    ) -> QualityPrediction: ...

    def predict_after_action(
        self,
        state: pd.DataFrame,
        action: dict[str, float],
    ) -> QualityPrediction: ...

    def explain(
        self,
        state: pd.DataFrame,
    ) -> Explanation: ...
```

Для серы ключ риска называется `sulfur_over_10` и означает вероятность превышения лимита 10 ppm.

```python
model = QualityHydroModel.load(
    "ml/artifacts/quality_hydro.pkl"
)

prediction = model.predict(state)
risk = prediction.spec_risk["sulfur_over_10"]

if risk >= 0.5:
    print("Высокий риск превышения серы")
```

Порог, по которому агент принимает решение, не зашивается в ML-модель. Модель возвращает вероятность, а агент применяет правила своего сценария.

## Прогноз после изменения режима

`predict_after_action` позволяет оценить состояние после установки новых значений управляемых тегов.

```python
after = model.predict_after_action(
    state,
    action={
        "T55": 350.0,
        "F30": 46.0,
    },
)
```

Значения в `action` являются абсолютными уставками:

```python
{"T55": 350.0}
```

означает «установить T55 равным 350», а не «увеличить T55 на 350».

Для действий допускаются два варианта имени тега:

```python
{"T55": 350.0}       # короткое имя из справочника КИП
{"avt_T55": 350.0}   # полное имя колонки master frame
```

Канонический формат для обмена между агентами — короткое имя из справочника КИП. Префикс установки можно использовать, если без него имя неоднозначно.

Модель применяет действие к копии состояния и не изменяет исходный `DataFrame`.

Если тег неизвестен или не относится к нужной установке, ожидается `KeyError`. Если значение не является числом, ожидается `TypeError`.

## Модель блендинга

Путь к классу: `ml.models.BlendingModel`.

```python
@dataclass
class Component:
    name: str
    properties: dict[str, float]
    mass_flow: float


@dataclass
class BlendedProduct:
    properties: dict[str, float]
    total_mass: float
```

Публичный API:

```python
class BlendingModel:
    def blend(
        self,
        components: list[Component],
        fractions: list[float],
    ) -> BlendedProduct: ...
```

Пример:

```python
components = [
    Component(
        name="hydro_diesel",
        properties={
            "sulfur_ppm": 6.0,
            "D15": 831.0,
            "T50": 282.0,
            "T90": 342.0,
        },
        mass_flow=80.0,
    ),
    Component(
        name="other_component",
        properties={
            "sulfur_ppm": 12.0,
            "D15": 840.0,
            "T50": 295.0,
            "T90": 355.0,
        },
        mass_flow=20.0,
    ),
]

product = BlendingModel().blend(
    components,
    fractions=[0.8, 0.2],
)
```

Правила:

- порядок `fractions` соответствует порядку `components`;
- `fractions` — массовые доли от `0.0` до `1.0`;
- сумма долей должна равняться `1.0` с допуском `1e-6`;
- все компоненты должны содержать одинаковый набор свойств;
- `mass_flow` задаётся в т/ч;
- `total_mass` равен сумме `mass_flow` компонентов;
- сера, D15, T50 и T90 смешиваются линейно;
- вязкость и CFPP смешиваются через нелинейные индексы.

При неверном балансе долей, отрицательной доле, несовпадающем наборе свойств или некорректном значении модель выбрасывает `ValueError`.

## Детектор аномалий

Путь к классу: `ml.models.AnomalyDetector`.

```python
@dataclass
class AnomalyReport:
    is_anomaly: bool
    anomaly_score: float
    flagged_tags: list[str]
    is_out_of_envelope: bool
    stale_tags: list[str]
```

Публичный API:

```python
class AnomalyDetector:
    @classmethod
    def load(cls, path: str) -> "AnomalyDetector": ...

    def score(
        self,
        state: pd.DataFrame,
    ) -> AnomalyReport: ...
```

Пример:

```python
detector = AnomalyDetector.load(
    "ml/artifacts/anomaly.pkl"
)

report = detector.score(state)

if report.is_anomaly:
    print(report.anomaly_score)
    print(report.flagged_tags)
```

Поля результата:

- `is_anomaly` — итоговый флаг аномалии;
- `anomaly_score` — степень аномальности от `0.0` до `1.0`;
- `flagged_tags` — теги с аномальными значениями;
- `is_out_of_envelope` — состояние находится вне исторического облака режимов;
- `stale_tags` — сигналы, которые не менялись дольше допустимого времени.

## Парето-оптимизатор

Путь к классу: `ml.optimizer.ParetoOptimizer`.

```python
@dataclass
class OptimizationConstraints:
    hard: dict[str, tuple[float, float]]
    controllable_ranges: dict[str, tuple[float, float]]
    max_deviation_pct: float


@dataclass
class Variant:
    action: dict[str, float]
    expected: QualityPrediction
    metrics: dict[str, float]
    feasible: bool
    infeasible_reason: str | None
```

Публичный API:

```python
class ParetoOptimizer:
    def __init__(
        self,
        quality_avt: QualityAVTModel,
        quality_hydro: QualityHydroModel,
        blending: BlendingModel,
    ): ...

    def find_pareto(
        self,
        state: pd.DataFrame,
        constraints: OptimizationConstraints,
        n_variants: int = 10,
    ) -> list[Variant]: ...
```

Пример:

```python
optimizer = ParetoOptimizer(
    quality_avt=quality_avt,
    quality_hydro=quality_hydro,
    blending=blending,
)

constraints = OptimizationConstraints(
    hard={
        "sulfur_ppm": (0.0, 10.0),
        "D15": (820.0, 845.0),
    },
    controllable_ranges={
        "T55": (340.0, 370.0),
        "F30": (40.0, 50.0),
    },
    max_deviation_pct=5.0,
)

variants = optimizer.find_pareto(
    state,
    constraints,
    n_variants=10,
)
```

Смысл полей ограничений:

- `hard` — допустимый диапазон выходного показателя качества;
- `controllable_ranges` — абсолютные допустимые диапазоны управляемых тегов;
- `max_deviation_pct` — максимальное отклонение предлагаемой уставки от текущего значения в процентах.

Смысл результата:

- `action` — абсолютные значения предлагаемых уставок;
- `expected` — прогноз качества после действия;
- `metrics` — значения критериев оптимизации;
- `feasible` — можно ли использовать вариант;
- `infeasible_reason` — причина отклонения варианта или `None`.

Стандартные ключи `metrics`:

```python
{
    "sulfur": 6.4,
    "yield": 0.85,
    "energy": 1.11,
    "severity": 0.30,
}
```

Агент может рекомендовать оператору только варианты с `feasible=True`. Вариант с `infeasible_reason="low_historical_density"` находится в слабо представленной области исторических данных и не должен использоваться как рабочая рекомендация.

Если при заданных ограничениях невозможно получить `n_variants` допустимых вариантов, список может содержать меньше элементов. Пустой список означает, что допустимое действие не найдено.

## Преобразование результатов в JSON

ML-модуль возвращает dataclass-объекты, а не Pydantic-модели. Для передачи в API или схему агента их можно преобразовать через `dataclasses.asdict`.

```python
from dataclasses import asdict

prediction_payload = asdict(prediction)
anomaly_payload = asdict(report)
variants_payload = [asdict(variant) for variant in variants]
```

Пример результата:

```python
{
    "predictions": {
        "sulfur_ppm": {
            "mean": 8.6,
            "low": 7.4,
            "high": 10.3,
            "unit": "ppm",
        }
    },
    "spec_risk": {"sulfur_over_10": 0.24},
    "confidence": "medium",
    "warnings": [],
}
```

## Рекомендуемый порядок вызовов агентом

```python
state = simulator.get_state(timestamp)

anomaly = anomaly_detector.score(state)
current_quality = quality_hydro.predict(state)

if anomaly.is_anomaly or anomaly.is_out_of_envelope:
    # Агент учитывает повышенный риск и не выдаёт уверенную рекомендацию.
    ...

variants = optimizer.find_pareto(
    state,
    constraints,
    n_variants=10,
)

safe_variants = [
    variant
    for variant in variants
    if variant.feasible
]
```

AI-агент отвечает за выбор сценария, формулировку рекомендации и решение `recommend` / `silent` / `refuse`. ML-модуль отвечает только за числовой прогноз, оценку риска, аномалии и варианты управляющих воздействий.

## Ошибки, которые должен обрабатывать агент

- `FileNotFoundError` — артефакт модели не найден;
- `TypeError` — вместо `DataFrame` или числового значения передан объект неверного типа;
- `ValueError` — передано не одна строка состояния, некорректные доли смеси или некорректные ограничения;
- `KeyError` — неизвестный тег действия или отсутствует обязательное имя результата.

Ошибку загрузки модели нельзя заменять выдуманным прогнозом. Ошибку единичного вызова нужно записать в trace и передать оркестратору как техническую недоступность соответствующего ML-компонента.

## Пути к артефактам

При запуске backend из каталога `/app` используются пути:

```text
ml/artifacts/quality_avt.pkl
ml/artifacts/quality_hydro.pkl
ml/artifacts/anomaly.pkl
ml/artifacts/metrics_report.json
```

Если код запускается из корня репозитория без Docker, соответствующие пути начинаются с `backend/`:

```text
backend/ml/artifacts/quality_avt.pkl
backend/ml/artifacts/quality_hydro.pkl
backend/ml/artifacts/anomaly.pkl
backend/ml/artifacts/metrics_report.json
```

Пути лучше задавать через конфигурацию backend, а не дублировать строковыми литералами в каждом агенте.
