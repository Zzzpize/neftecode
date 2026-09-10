import time

import pandas as pd
import pytest

from ml.models.blending import BlendingModel
from ml.optimizer.pareto import ParetoOptimizer
from ml.types import (
    Interval,
    OptimizationConstraints,
    QualityPrediction,
)


class DummyAVTModel:
    def __init__(self):
        self.artifact = {
            "feature_q01": {
                "avt_F30": 80.0,
                "avt_F32": 40.0,
                "avt_F65": 450.0,
                "hydro_T5": 290.0,
            },
            "feature_q99": {
                "avt_F30": 140.0,
                "avt_F32": 80.0,
                "avt_F65": 600.0,
                "hydro_T5": 340.0,
            },
        }

    def predict(
        self,
        state: pd.DataFrame,
    ) -> QualityPrediction:
        return QualityPrediction(
            predictions={
                "CFPP": Interval(
                    mean=-15.0,
                    low=-17.0,
                    high=-13.0,
                    unit="C",
                ),
            },
            spec_risk={},
            confidence="high",
            warnings=[],
        )


class DummyHydroModel:
    def __init__(
        self,
        *,
        historical_t5_high: float = 340.0,
    ):
        self.artifact = {
            "feature_q01": {
                "hydro_T5": 290.0,
            },
            "feature_q99": {
                "hydro_T5": historical_t5_high,
            },
        }

    def predict(
        self,
        state: pd.DataFrame,
    ) -> QualityPrediction:
        temperature = float(
            state.iloc[0]["hydro_T5"]
        )

        sulfur = (
            20.0
            - 0.4 * (temperature - 300.0)
        )

        return QualityPrediction(
            predictions={
                "sulfur_ppm": Interval(
                    mean=sulfur,
                    low=sulfur - 0.5,
                    high=sulfur + 0.5,
                    unit="ppm",
                ),
                "D15": Interval(
                    mean=830.0,
                    low=829.0,
                    high=831.0,
                    unit="kg/m3",
                ),
            },
            spec_risk={
                "sulfur_over_10": (
                    1.0 if sulfur > 10.0 else 0.0
                ),
            },
            confidence="high",
            warnings=[],
        )


@pytest.fixture
def state() -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "date": pd.Timestamp(
                "2025-08-01 12:00:00"
            ),
            "hydro_T5": 300.0,
            "avt_F30": 100.0,
            "avt_F32": 50.0,
            "avt_F65": 500.0,
        }]
    )


def make_optimizer(
    *,
    historical_t5_high: float = 340.0,
) -> ParetoOptimizer:
    return ParetoOptimizer(
        quality_avt=DummyAVTModel(),       # type: ignore[arg-type]
        quality_hydro=DummyHydroModel(     # type: ignore[arg-type]
            historical_t5_high=historical_t5_high,
        ),
        blending=BlendingModel(),
    )


def test_returns_ten_feasible_pareto_variants(
    state: pd.DataFrame,
):
    optimizer = make_optimizer()

    constraints = OptimizationConstraints(
        hard={
            "sulfur_ppm": (0.0, 10.0),
            "D15": (820.0, 845.0),
        },
        controllable_ranges={
            "hydro_T5": (300.0, 330.0),
        },
        max_deviation_pct=10.0,
    )

    variants = optimizer.find_pareto(
        state,
        constraints,
        n_variants=10,
    )

    assert len(variants) == 10
    assert all(
        variant.feasible
        for variant in variants
    )
    assert all(
        variant.infeasible_reason is None
        for variant in variants
    )

    for variant in variants:
        assert set(variant.metrics) == {
            "sulfur",
            "yield",
            "energy",
            "severity",
        }

        assert (
            300.0
            <= variant.action["hydro_T5"]
            <= 330.0
        )

        # Проверяется верхняя граница интервала,
        # а не только средний прогноз.
        assert (
            variant.expected
            .predictions["sulfur_ppm"]
            .high
            <= 10.0
        )


def test_max_deviation_limits_search_range(
    state: pd.DataFrame,
):
    optimizer = make_optimizer()

    constraints = OptimizationConstraints(
        hard={
            "sulfur_ppm": (0.0, 100.0),
        },
        controllable_ranges={
            "hydro_T5": (200.0, 500.0),
        },
        max_deviation_pct=5.0,
    )

    variants = optimizer.find_pareto(
        state,
        constraints,
        n_variants=10,
    )

    assert variants

    for variant in variants:
        # 5% от текущих 300 °C = 15 °C.
        assert (
            285.0
            <= variant.action["hydro_T5"]
            <= 315.0
        )


def test_low_density_variant_is_marked_infeasible(
    state: pd.DataFrame,
):
    optimizer = make_optimizer(
        historical_t5_high=305.0,
    )

    constraints = OptimizationConstraints(
        hard={
            "sulfur_ppm": (0.0, 100.0),
        },
        controllable_ranges={
            "hydro_T5": (320.0, 320.0),
        },
        max_deviation_pct=10.0,
    )

    variants = optimizer.find_pareto(
        state,
        constraints,
        n_variants=1,
    )

    assert len(variants) == 1
    assert variants[0].feasible is False
    assert (
        variants[0].infeasible_reason
        == "low_historical_density"
    )


def test_unknown_controllable_tag_is_rejected(
    state: pd.DataFrame,
):
    optimizer = make_optimizer()

    constraints = OptimizationConstraints(
        hard={},
        controllable_ranges={
            "UNKNOWN_TAG": (0.0, 1.0),
        },
        max_deviation_pct=10.0,
    )

    with pytest.raises(KeyError):
        optimizer.find_pareto(
            state,
            constraints,
        )


def test_optimizer_does_not_mutate_state(
    state: pd.DataFrame,
):
    optimizer = make_optimizer()
    original = state.copy(deep=True)

    constraints = OptimizationConstraints(
        hard={
            "sulfur_ppm": (0.0, 100.0),
        },
        controllable_ranges={
            "hydro_T5": (300.0, 330.0),
        },
        max_deviation_pct=10.0,
    )

    optimizer.find_pareto(
        state,
        constraints,
        n_variants=3,
    )

    pd.testing.assert_frame_equal(
        state,
        original,
    )


def test_optimizer_finishes_under_five_seconds(
    state: pd.DataFrame,
):
    optimizer = make_optimizer()

    constraints = OptimizationConstraints(
        hard={
            "sulfur_ppm": (0.0, 10.0),
        },
        controllable_ranges={
            "hydro_T5": (300.0, 330.0),
        },
        max_deviation_pct=10.0,
    )

    started = time.perf_counter()

    optimizer.find_pareto(
        state,
        constraints,
        n_variants=10,
    )

    elapsed = time.perf_counter() - started

    assert elapsed < 5.0