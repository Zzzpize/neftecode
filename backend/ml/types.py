from dataclasses import dataclass
from typing import Literal

@dataclass
class Interval:
    mean: float
    low: float          # q10 до калибровки; нижняя 90%-граница после неё
    high: float         # q90 до калибровки; верхняя 90%-граница после неё
    unit: str

@dataclass
class QualityPrediction:
    predictions: dict[str, Interval]      # например {"sulfur_ppm": Interval(...), "T50": Interval(...)}
    spec_risk: dict[str, float]           # вероятность выхода за спеку, например {"sulfur_over_10": 0.27}
    confidence: Literal["high", "medium", "low"]
    warnings: list[str]                   # человекочитаемые предупреждения

@dataclass
class Explanation:
    top_features: list[tuple[str, float]] # (имя фичи, значение SHAP), отсортировано по |value|
    base_value: float                     # базовое предсказание модели

@dataclass
class AnomalyReport:
    is_anomaly: bool
    anomaly_score: float                  # 0..1
    flagged_tags: list[str]
    is_out_of_envelope: bool
    stale_tags: list[str]                 # теги, не менявшиеся больше порога

@dataclass
class Component:
    name: str
    properties: dict[str, float]          # {"sulfur_ppm": 8.2, "D15": 831.5, "T50": 285, ...}
    mass_flow: float                      # т/ч

@dataclass
class BlendedProduct:
    properties: dict[str, float]
    total_mass: float

@dataclass
class OptimizationConstraints:
    hard: dict[str, tuple[float, float]]           # жёсткие спеки, например {"sulfur_ppm": (0, 10)}
    controllable_ranges: dict[str, tuple[float, float]]  # диапазоны управляемых тегов
    max_deviation_pct: float                       # максимальное отклонение от текущего значения

@dataclass
class Variant:
    action: dict[str, float]              # {"T55": 348.0, "F30": 44.5, ...}
    expected: QualityPrediction
    metrics: dict[str, float]             # {"sulfur": 6.4, "yield": 0.85, "energy": 1.11, "severity": 0.3}
    feasible: bool
    infeasible_reason: str | None
