# Python API ML

Область: только ML-ядро. HTTP, агенты, UI и загрузка исходных XLSX сюда не входят.
Основные контракты из `docs/tz/tz_ml.md` сохранены; опциональные сценарии
передаются дополнительными колонками состояния, без изменения сигнатур.

## Подготовка и пути

Из корня проекта:

```sh
PYTHONPATH=backend .venv/bin/python -m ml.training.preflight
PYTHONPATH=backend .venv/bin/python -m ml.training.train_all
```

В запущенном Docker Compose эквивалент обучения — `make train`.
Вход обучения: `data/master.parquet`. Выход: `data/ml_training.parquet`
(признаки и отдельные метки), `data/ml_features.parquet` (только inference),
`backend/ml/artifacts/{quality_avt,quality_hydro,anomaly}.pkl` и
`backend/ml/artifacts/metrics_report.json`. Артефакты и parquet не коммитятся.

Актуальный справочник — `backend/ml/expert_reference.json`, версия
`expert_xlsx_2026_09_17`: теги и 17 формул из двух новых XLSX.
Старые обученные quality-артефакты без этой версии отвергаются при загрузке.
`VAKCatalog.load_expert()` использует новый справочник;
`VAKCatalog.load(path)` оставлен для совместимости.

## Состояние процесса

`state` — **одна строка** подготовленного DataFrame на момент T. Передавайте
полные имена: `hydro_T6`, `hydro_F9`, `hydro_P13`, `avt_F30` и т. д.
Нужны признаки, использованные при обучении, входы ВАК и служебные колонки;
сырой срез телеметрии не заменяет подготовленные лаги/rolling/возраст ЛИМС.

Дата ЛИМС означает отбор пробы. Общая временная зона; консервативная задержка
доступности результата — 4 часа. В inference-таблице `lims__*` уже сдвинуты
по доступности, возраст считается от отбора, `ml_lims_delay_h == 4`.
`label__*` — исходный лабораторный факт для оценки на момент отбора,
**не вход публичных quality predict/explain**. Отдельные метки нельзя включать в признаки.
Если берёте строку из training-таблицы, удалите их через `inference_frame`.
Не устанавливайте marker вручную на сырой таблице: пересоберите кеш обучением.

```python
import sys
from pathlib import Path
import pandas as pd

# Запуск из корня репозитория.
sys.path.insert(0, str(Path("backend").resolve()))
from ml.models import QualityAVTModel, QualityHydroModel, BlendingModel, AnomalyDetector
from ml.optimizer import ParetoOptimizer
from ml.availability import inference_frame
from ml.paths import DATA_DIR, ARTIFACT_DIR

avt = QualityAVTModel.load(str(ARTIFACT_DIR / "quality_avt.pkl"))
hydro = QualityHydroModel.load(str(ARTIFACT_DIR / "quality_hydro.pkl"))
anomaly = AnomalyDetector.load(str(ARTIFACT_DIR / "anomaly.pkl"))
features = pd.read_parquet(DATA_DIR / "ml_features.parquet")
timestamp = pd.Timestamp("2025-08-01 12:00:00")
state = inference_frame(features.loc[pd.to_datetime(features.date).le(timestamp)].tail(1))
if state.empty:
    raise ValueError("Нет данных на момент T")
state = state.reset_index(drop=True)
```

## Общие типы: backend/ml/types.py

Все ответы — dataclass, не HTTP/Pydantic-модели.

| Тип | Поля |
| --- | --- |
| Interval | mean, low, high, unit |
| QualityPrediction | predictions: dict[str, Interval], spec_risk: dict[str, float], confidence: high/medium/low, warnings: list[str] |
| Explanation | top_features: list[tuple[str, float]], base_value: float |
| AnomalyReport | is_anomaly, anomaly_score, flagged_tags, is_out_of_envelope, stale_tags |
| Component | name, properties: dict[str, float], mass_flow: float |
| BlendedProduct | properties: dict[str, float], total_mass: float |
| OptimizationConstraints | hard, controllable_ranges, max_deviation_pct |
| Variant | action, expected: QualityPrediction, metrics, feasible, infeasible_reason |

`Interval.mean` — имя поля по контракту; при ML это q50 (для online-серы
с причинной поправкой по прошлым ошибкам), не обязательно
арифметическое ожидание. Сырые q10/q90 охватывают **80%**.
После `train_all` low/high — conformal 90%-интервал. Для большинства
показателей он фиксируется на validation; для серы после прогрева
используются ошибки предыдущих 30 доступных измерений ПАК. Измерение в T
участвует в обновлении только после прогноза для T. До калибровки это q10/q90.
При отсутствии данных границы могут быть NaN: интервал неизвестен, не нулевой.
Поля контракта не изменены; семантика интервала уточняет противоречие в ТЗ
между q10/q90 и требованием coverage 90%.

## QualityAVTModel / QualityHydroModel

```python
@classmethod
def load(cls, path: str): ...
def predict(self, state: pd.DataFrame) -> QualityPrediction: ...
def predict_after_action(self, state: pd.DataFrame, action: dict[str, float]) -> QualityPrediction: ...
def explain(self, state: pd.DataFrame) -> Explanation: ...
```

АВТ: `T50`, `T90`, `D15`, `CFPP`.
Hydro: `sulfur_ppm`, `T50`, `T90`, `D15`, дополнительно по эксперту
`T95`, `cetane_number`. Температуры — C, плотность — kg/m3,
сера — ppm (= мг/кг), цетановое число — dimensionless.

ВАК + квантильные LightGBM на residual. При отсутствии ВАК — train-медиана,
а не выдуманная формула. Если ML не улучшает validation MAE, ML отключается.
При недостатке меток возвращается резервная оценка/ВАК и low confidence с предупреждением.
В train CFPP АВТ нет меток; для цетанового числа их недостаточно для LightGBM.

Сера обучается на **сыром ПАК**; лаборатория Hydro точки 2 — независимый
контроль в `quality_hydro_lims_reference` отчёта. При расхождении контрольный
результат — ЛИМС; автоматической подгонки ПАК под редкую лабораторию нет.
Recall по ПАК не заменяет проверку лабораторного контрольного источника.

`spec_risk["sulfur_over_10"]` — приближённая вероятность из интервала,
не отдельный калиброванный классификатор. Выбор порога бинарного риска
выполняется на validation и сохраняется в артефакте/отчёте; он не обязательно 0.5.

```python
prediction = hydro.predict(state)
sulfur = prediction.predictions["sulfur_ppm"]
print(sulfur.mean, sulfur.low, sulfur.high, prediction.confidence, prediction.warnings)
explanation = hydro.explain(state)
print(explanation.top_features)  # до 10 (имя, SHAP), по |SHAP|
```

SHAP относится к residual ML выбранного показателя (АВТ T50, Hydro сера),
не объясняет целиком ВАК, online-поправку, интервалы или причинный эффект. При выключенном ML
список может быть пуст. Методы quality логируют вход/результат; after_action также действие.

Для новых Hydro-артефактов используется причинная online-калибровка серы:
один экземпляр модели получает реальные состояния через `predict` в
хронологическом порядке. Предыдущий ПАК — наблюдение для калибратора,
не текущий target в признаках LightGBM. После разрыва истории >6 часов
или при недостатке измерений возвращается low confidence и фиксированный
validation-интервал. `predict_after_action` и synthetic feed multiplier не
обновляют историю. Загрузка одиночного позднего среза не заменяет realtime-
последовательность: для демо подайте предыдущие 90 состояний до выбранной точки.

Переобучение из готового ML-кеша без изменения data layer:
`PYTHONPATH=backend .venv/bin/python -m ml.training.train_all --reuse-features`.
Окно Hydro выбирается между 2023–2024 и 2024 только по validation MAE серы;
выбор и результаты сравнения сохраняются в отчёте. Перед заменой старые
артефакты копируются в `backend/ml/artifacts/backups/`.

### Последствия действия

Значения `action` — **абсолютные**, не приращения. Исходный state не изменяется.

```python
state["ml_action_horizon_minutes"] = 180.0
after = hydro.predict_after_action(state, {"hydro_T6": float(state.hydro_T6.iloc[0]) + 1.0})
```

Управляющие теги Hydro: T6 — температура, F9 — массовый расход,
P13 — давление. Их управляемость — рабочее предположение, требующее
подтверждения экспертом. P8/T11/F19 — не прежние управляющие теги.
Диапазоны оптимизатор берёт из `backend/data_layer/feature_registry.yaml`.

При обученном отклике серы используется constrained observational surrogate
с задержкой 0–180 минут. Он не является доказанной causal-моделью.
Остальные качества Hydro и действия АВТ — статические модельные оценки.
Транспортная задержка АВТ → Hydro оценивается отдельно и не равна горизонту действия.
Все after_action результаты явно low confidence. Прямой вызов after_action
не заменяет проверку допустимости через оптимизатор.

## BlendingModel

```python
def blend(self, components: list[Component], fractions: list[float]) -> BlendedProduct: ...
```

Массовые доли неотрицательны, сумма 1 с допуском 1e-6; иначе ValueError.
Сера, D15, температуры и обычные общие показатели — линейное массовое среднее.
Вязкость — Refutas. CFPP — текущий приближённый индекс
`(CFPP + 273.15)**2` с обратным преобразованием; это **не подтверждённый
стандартный закон ПТФ**. Для полного соответствия ТЗ закон надо согласовать
и проверить на данных смесей. Линейные T95/цетановое число также сценарное приближение.

```python
from ml.types import Component
blend = BlendingModel().blend(
    [Component("A", {"sulfur_ppm": 3.0, "T95": 330.0, "cetane_number": 53.0}, 40.0),
     Component("B", {"sulfur_ppm": 15.0, "T95": 360.0, "cetane_number": 47.0}, 40.0)],
    [0.5, 0.5],
)
print(blend.properties, blend.total_mass)
```

## AnomalyDetector

```python
@classmethod
def load(cls, path: str) -> "AnomalyDetector": ...
def score(self, state: pd.DataFrame) -> AnomalyReport: ...
```

Isolation Forest + rolling z-score + stale/envelope checks.
`anomaly.score(state)` возвращает числовой отчёт. Rolling/stale требуют
подготовленных временных признаков, одной сырой строкой историю не восстановить.
Если известные инциденты не размечены, training считает train нормальным с
предупреждением; это ограничение, а не подтверждённая чистота train.
AUC без независимой разметки инцидентов не считать доказательством качества.

## ParetoOptimizer

```python
def __init__(self, quality_avt, quality_hydro, blending): ...
def find_pareto(self, state: pd.DataFrame,
                constraints: OptimizationConstraints, n_variants: int = 10) -> list[Variant]: ...
```

```python
optimizer = ParetoOptimizer(avt, hydro, BlendingModel())
constraints = optimizer.constraints_from_registry(
    hard={"sulfur_ppm": (0., 10.), "T95": (0., 360.), "cetane_number": (51., 100.)},
    max_deviation_pct=3.,
    tags=["hydro_T6", "hydro_F9", "hydro_P13"],
)
variants = optimizer.find_pareto(state, constraints, n_variants=10)
recommendations = [v for v in variants if v.feasible]
```

Пределы T95/цетанового числа в примере — параметры демо, не утверждение о
конкретном товарном стандарте. Проверяются все заданные hard-ограничения
по low/high; отсутствующие/NaN интервалы означают отказ. Также проверяются
диапазоны и совместная плотность исторических управляющих состояний.

Четыре цели: sulfur ↓, yield ↑, energy ↓, severity ↓. Yield/energy — прокси,
не измеренный товарный выход и не полная экономическая модель.
Выдача Парето-ранжированная: могут добавляться последующие недоминируемые слои.
Поиск ограничен примерно 4 секундами, но итоговое время надо измерять.
Может быть меньше n_variants. Если допустимых нет, могут возвращаться
отклонённые варианты с feasible=False/reason; **не выдавать их за рекомендации**.
Само наличие списка не доказывает выполнение критерия «10 допустимых за <5 с».

### Опциональный сценарий блендинга

```python
from ml.scenario_blending import default_scenario
state["ml_blending_scenario"] = pd.Series([default_scenario()], index=state.index)
state["ml_feed_sulfur_multiplier"] = 1.5
state["ml_action_horizon_minutes"] = 180.
variants = optimizer.find_pareto(state, constraints)
```

`ml_blending_scenario`: True для defaults либо dict сценария; без колонки
сценарий выключен. Обязательны hard по sulfur_ppm, T95, cetane_number.
Настройки: `backend/ml/blending_scenario.json` — два синтетических резервуара,
расходы/ёмкости, эффективность присадки и стоимость.
Присадка ≤3% конечной массы, цена 100× ДТ; экономический прокси cost —
пятая цель. Действия `blend__stored_fraction`, `blend__tank_share`,
`blend__additive_fraction` — доли рецепта, не теги КИП.
Множитель серы положительный: синтетический стресс прогноза Hydro,
**не измерение серы сырья**. Эффект T95 присадки не моделируется;
цетановая эффективность задаётся предположением. Результат low confidence.

## JSON и ошибки

```python
import json
from dataclasses import asdict
from ml.demo_scenarios import clean_json
payload = json.dumps(clean_json(asdict(prediction)), ensure_ascii=False, allow_nan=False)
```

NaN/inf преобразуются в null; отсутствие интервала не скрывается.
На невалидный shape/type, неизвестный тег, неверную политику ЛИМС или
некорректные доли — исключение, не вымышленный успешный прогноз.
На недостаток обучающих данных — low confidence/warnings либо неизвестный интервал.

## Проверка перед сдачей

`notebooks/ml_report.ipynb` показывает реальные test-метрики, лабораторный
контроль, coverage, SHAP, блендинг, аномалии и три регулируемых ML-сценария.
Нужны новые артефакты. Сценарии не заменяют историческую accuracy-валидацию
или отдельный совместный end-to-end с оркестратором.
Оставшиеся отступления/критерии и порядок запуска — `docs/ml_compliance.md`.
