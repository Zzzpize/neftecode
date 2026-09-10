# Парето-оптимизатор через Optuna. Метрики: сера, выход, энергия, тяжесть режима.
from __future__ import annotations

import time
import math
from dataclasses import dataclass

import optuna
import pandas as pd

from ml.models.blending import BlendingModel
from ml.models.quality_avt import QualityAVTModel
from ml.models.quality_hydro import QualityHydroModel
from ml.types import (
    OptimizationConstraints,
    QualityPrediction,
    Variant,
)


@dataclass
class _EvaluatedVariant:
    variant: Variant
    constraint_violation: float


class ParetoOptimizer:
    """
    Подбирает управляющие воздействия и возвращает Парето-фронт.

    Направления оптимизации:
    - sulfur: minimize;
    - yield: maximize;
    - energy: minimize;
    - severity: minimize.
    """

    MAX_OPTIMIZATION_SECONDS = 4.0
    MIN_TRIALS = 64
    TRIALS_PER_VARIANT = 8

    PRODUCT_FLOW_COLUMNS = (
        "avt_F30",  # фракция 290–350 °C
        "avt_F32",  # фракция 240–290 °C
    )
    FEED_FLOW_COLUMNS = (
        "avt_F65",
    )

    def __init__(
        self,
        quality_avt: QualityAVTModel,
        quality_hydro: QualityHydroModel,
        blending: BlendingModel,
    ):
        self.quality_avt = quality_avt
        self.quality_hydro = quality_hydro
        self.blending = blending

    def find_pareto(
        self,
        state: pd.DataFrame,
        constraints: OptimizationConstraints,
        n_variants: int = 10,
    ) -> list[Variant]:
        self._validate_inputs(
            state=state,
            constraints=constraints,
            n_variants=n_variants,
        )

        search_ranges = self._build_search_ranges(
            state=state,
            constraints=constraints,
        )

        evaluated: dict[int, _EvaluatedVariant] = {}

        def constraints_func(
            trial: optuna.trial.FrozenTrial,
        ) -> tuple[float]:
            return tuple(   #type: ignore
                trial.user_attrs.get(
                    "constraint_values",
                    (0.0,),
                )
            )

        sampler = optuna.samplers.TPESampler(
            seed=42,
            multivariate=True,
            constraints_func=constraints_func,
        )

        study = optuna.create_study(
            directions=[
                "minimize",  # sulfur
                "maximize",  # yield
                "minimize",  # energy
                "minimize",  # severity
            ],
            sampler=sampler,
        )

        def objective(
            trial: optuna.Trial,
        ) -> tuple[float, float, float, float]:
            action = {
                tag: trial.suggest_float(
                    tag,
                    low,
                    high,
                )
                for tag, (low, high) in search_ranges.items()
            }

            changed_state = self._apply_action(
                state=state,
                action=action,
            )

            avt_prediction = self.quality_avt.predict(
                changed_state
            )
            hydro_prediction = self.quality_hydro.predict(
                changed_state
            )

            expected = self._combine_predictions(
                avt_prediction=avt_prediction,
                hydro_prediction=hydro_prediction,
            )

            metrics = {
                "sulfur": self._prediction_value(
                    expected,
                    "sulfur_ppm",
                    fallback=1_000_000.0,
                ),
                "yield": self._yield_proxy(changed_state),
                "energy": self._energy_proxy(
                    state=state,
                    changed_state=changed_state,
                    search_ranges=search_ranges,
                ),
                "severity": self._severity_proxy(
                    state=state,
                    changed_state=changed_state,
                    search_ranges=search_ranges,
                ),
            }

            infeasible_reason, violation = (
                self._check_hard_constraints(
                    prediction=expected,
                    constraints=constraints,
                )
            )

            if infeasible_reason is None:
                is_low_density = self._is_low_historical_density(
                    changed_state=changed_state,
                    action=action,
                )

                if is_low_density:
                    infeasible_reason = (
                        "low_historical_density"
                    )
                    violation = 1.0

            trial.set_user_attr(
                "constraint_values",
                (float(violation),),
            )

            variant = Variant(
                action={
                    tag: float(value)
                    for tag, value in action.items()
                },
                expected=expected,
                metrics=metrics,
                feasible=infeasible_reason is None,
                infeasible_reason=infeasible_reason,
            )

            evaluated[trial.number] = _EvaluatedVariant(
                variant=variant,
                constraint_violation=violation,
            )

            return (
                metrics["sulfur"],
                metrics["yield"],
                metrics["energy"],
                metrics["severity"],
            )

        deadline = (
            time.perf_counter()
            + self.MAX_OPTIMIZATION_SECONDS
        )

        max_trials = max(
            256,
            n_variants * 64,
        )

        batch_size = max(
            32,
            n_variants * 4,
        )

        while len(evaluated) < max_trials:
            feasible_count = sum(
                item.variant.feasible
                for item in evaluated.values()
            )

            if feasible_count >= n_variants:
                break

            remaining_seconds = (
                deadline - time.perf_counter()
            )

            if remaining_seconds <= 0:
                break

            remaining_trials = (
                max_trials - len(evaluated)
            )

            study.optimize(
                objective,
                n_trials=min(
                    batch_size,
                    remaining_trials,
                ),
                timeout=remaining_seconds,
                show_progress_bar=False,
                gc_after_trial=False,
            )

        feasible = [
            item.variant
            for item in evaluated.values()
            if item.variant.feasible
        ]

        if feasible:
            return self._select_pareto_ranked_variants(
                feasible,
                n_variants=n_variants,
            )

        # Если допустимых решений нет, возвращаем ближайшие
        # отклонённые варианты с заполненной причиной.
        rejected = sorted(
            evaluated.values(),
            key=lambda item: (
                item.constraint_violation,
                item.variant.metrics["severity"],
            ),
        )

        return [
            item.variant
            for item in rejected[:n_variants]
        ]

    def _build_search_ranges(
        self,
        state: pd.DataFrame,
        constraints: OptimizationConstraints,
    ) -> dict[str, tuple[float, float]]:
        result: dict[str, tuple[float, float]] = {}

        deviation_fraction = (
            constraints.max_deviation_pct / 100.0
        )

        for tag, absolute_range in (
            constraints.controllable_ranges.items()
        ):
            if len(absolute_range) != 2:
                raise ValueError(
                    f"Range for {tag!r} must contain "
                    "exactly two values"
                )

            absolute_low = float(absolute_range[0])
            absolute_high = float(absolute_range[1])

            if (
                not math.isfinite(absolute_low)
                or not math.isfinite(absolute_high)
            ):
                raise ValueError(
                    f"Range for {tag!r} must be finite"
                )

            if absolute_low > absolute_high:
                raise ValueError(
                    f"Invalid range for {tag!r}: "
                    f"{absolute_low} > {absolute_high}"
                )

            column = self._resolve_state_column(state, tag)
            current = float(state.iloc[0][column])

            if not math.isfinite(current):
                raise ValueError(
                    f"Current value for {tag!r} is not finite"
                )

            maximum_change = (
                abs(current) * deviation_fraction
            )

            deviation_low = current - maximum_change
            deviation_high = current + maximum_change

            effective_low = max(
                absolute_low,
                deviation_low,
            )
            effective_high = min(
                absolute_high,
                deviation_high,
            )

            if effective_low > effective_high:
                raise ValueError(
                    f"No valid search range for {tag!r}: "
                    "absolute limits do not include the "
                    "allowed deviation from current value"
                )

            result[tag] = (
                effective_low,
                effective_high,
            )

        return result

    def _apply_action(
        self,
        state: pd.DataFrame,
        action: dict[str, float],
    ) -> pd.DataFrame:
        changed_state = state.copy(deep=True)

        for tag, value in action.items():
            column = self._resolve_state_column(
                changed_state,
                tag,
            )
            changed_state.loc[:, column] = float(value)

        return changed_state

    @staticmethod
    def _combine_predictions(
        avt_prediction: QualityPrediction,
        hydro_prediction: QualityPrediction,
    ) -> QualityPrediction:
        """
        Формирует прогноз конечного продукта.

        Для совпадающих показателей приоритет имеет гидроочистка.
        CFPP, которого нет у Hydro-модели, берётся из AVT-модели.
        """

        predictions = dict(avt_prediction.predictions)
        predictions.update(hydro_prediction.predictions)

        spec_risk = dict(avt_prediction.spec_risk)
        spec_risk.update(hydro_prediction.spec_risk)

        confidence_order = {
            "high": 0,
            "medium": 1,
            "low": 2,
        }

        confidence = max(
            (
                avt_prediction.confidence,
                hydro_prediction.confidence,
            ),
            key=confidence_order.__getitem__,
        )

        warnings = list(
            dict.fromkeys(
                avt_prediction.warnings
                + hydro_prediction.warnings
            )
        )

        return QualityPrediction(
            predictions=predictions,
            spec_risk=spec_risk,
            confidence=confidence,
            warnings=warnings,
        )

    @staticmethod
    def _prediction_value(
        prediction: QualityPrediction,
        name: str,
        *,
        fallback: float,
    ) -> float:
        interval = prediction.predictions.get(name)

        if interval is None:
            return fallback

        value = float(interval.mean)

        if not math.isfinite(value):
            return fallback

        return value

    def _yield_proxy(
        self,
        state: pd.DataFrame,
    ) -> float:
        """
        Baseline-прокси выхода товарного дизеля:

        (F30 + F32) / F65

        Если необходимые теги отсутствуют, возвращается 0.
        """

        product_values = [
            float(state.iloc[0][column])
            for column in self.PRODUCT_FLOW_COLUMNS
            if column in state.columns
            and pd.notna(state.iloc[0][column])
        ]

        feed_values = [
            float(state.iloc[0][column])
            for column in self.FEED_FLOW_COLUMNS
            if column in state.columns
            and pd.notna(state.iloc[0][column])
        ]

        if not product_values or not feed_values:
            return 0.0

        feed = sum(feed_values)

        if feed <= 0:
            return 0.0

        return sum(product_values) / feed

    def _energy_proxy(
        self,
        *,
        state: pd.DataFrame,
        changed_state: pd.DataFrame,
        search_ranges: dict[str, tuple[float, float]],
    ) -> float:
        """
        Энергетический прокси.

        Учитывается положительное изменение температур,
        тепловых нагрузок Q и расхода пара F5.
        """

        contributions: list[float] = []

        for tag, (low, high) in search_ranges.items():
            column = self._resolve_state_column(state, tag)
            short_name = column.split("_", maxsplit=1)[-1]

            is_energy_tag = (
                short_name.startswith("T")
                or short_name.startswith("Q")
                or short_name == "F5"
            )

            if not is_energy_tag:
                continue

            width = max(high - low, 1e-12)
            before = float(state.iloc[0][column])
            after = float(changed_state.iloc[0][column])

            contributions.append(
                max(after - before, 0.0) / width
            )

        if not contributions:
            return 0.0

        return sum(contributions) / len(contributions)

    def _severity_proxy(
        self,
        *,
        state: pd.DataFrame,
        changed_state: pd.DataFrame,
        search_ranges: dict[str, tuple[float, float]],
    ) -> float:
        """
        Тяжесть режима — RMS нормализованных изменений
        всех управляющих тегов.
        """

        squared_changes: list[float] = []

        for tag, (low, high) in search_ranges.items():
            column = self._resolve_state_column(state, tag)

            width = max(high - low, 1e-12)
            before = float(state.iloc[0][column])
            after = float(changed_state.iloc[0][column])

            normalized_change = (
                (after - before) / width
            )
            squared_changes.append(
                normalized_change**2
            )

        if not squared_changes:
            return 0.0

        return math.sqrt(
            sum(squared_changes)
            / len(squared_changes)
        )

    @staticmethod
    def _check_hard_constraints(
        *,
        prediction: QualityPrediction,
        constraints: OptimizationConstraints,
    ) -> tuple[str | None, float]:
        total_violation = 0.0
        reasons: list[str] = []

        for property_name, limits in (
            constraints.hard.items()
        ):
            low = float(limits[0])
            high = float(limits[1])

            interval = prediction.predictions.get(
                property_name
            )

            if interval is None:
                reasons.append(
                    f"missing_prediction:{property_name}"
                )
                total_violation += 1.0
                continue

            # Для жёсткой проверки используем верхнюю/нижнюю
            # границы интервала, а не только среднее.
            predicted_low = float(interval.low)
            predicted_high = float(interval.high)

            if predicted_low < low:
                width = max(high - low, 1e-12)
                total_violation += (
                    low - predicted_low
                ) / width
                reasons.append(
                    f"{property_name}_below_{low:g}"
                )

            if predicted_high > high:
                width = max(high - low, 1e-12)
                total_violation += (
                    predicted_high - high
                ) / width
                reasons.append(
                    f"{property_name}_above_{high:g}"
                )

        if not reasons:
            return None, 0.0

        return (
            "hard_constraint:"
            + ",".join(reasons),
            total_violation,
        )

    def _is_low_historical_density(
        self,
        *,
        changed_state: pd.DataFrame,
        action: dict[str, float],
    ) -> bool:
        """
        Baseline-фильтр плотности.

        В качестве исторического envelope используются q01/q99,
        сохранённые в артефактах моделей качества.
        """

        artifacts = (
            self.quality_avt.artifact,
            self.quality_hydro.artifact,
        )

        for tag in action:
            column = self._resolve_state_column(
                changed_state,
                tag,
            )
            value = float(changed_state.iloc[0][column])

            bounds_found = False

            for artifact in artifacts:
                q01 = artifact.get("feature_q01", {})
                q99 = artifact.get("feature_q99", {})

                if column not in q01 or column not in q99:
                    continue

                low = float(q01[column])
                high = float(q99[column])

                if not (
                    math.isfinite(low)
                    and math.isfinite(high)
                ):
                    continue

                bounds_found = True

                if value < low or value > high:
                    return True

            # Если тег отсутствует в исторических признаках обеих
            # моделей, его плотность оценить невозможно.
            if not bounds_found:
                continue

        return False

    @classmethod
    def _pareto_front(
        cls,
        variants: list[Variant],
    ) -> list[Variant]:
        front: list[Variant] = []

        for candidate in variants:
            dominated = any(
                cls._dominates(other, candidate)
                for other in variants
                if other is not candidate
            )

            if not dominated:
                front.append(candidate)

        return front

    @staticmethod
    def _dominates(
        first: Variant,
        second: Variant,
    ) -> bool:
        first_values = (
            first.metrics["sulfur"],
            -first.metrics["yield"],
            first.metrics["energy"],
            first.metrics["severity"],
        )
        second_values = (
            second.metrics["sulfur"],
            -second.metrics["yield"],
            second.metrics["energy"],
            second.metrics["severity"],
        )

        no_worse = all(
            left <= right
            for left, right in zip(
                first_values,
                second_values,
            )
        )
        strictly_better = any(
            left < right
            for left, right in zip(
                first_values,
                second_values,
            )
        )

        return no_worse and strictly_better

    @classmethod
    def _select_pareto_ranked_variants(
        cls,
        variants: list[Variant],
        *,
        n_variants: int,
    ) -> list[Variant]:
        """
        Выбирает варианты последовательными Парето-фронтами.

        Сначала в результат попадает первый недоминируемый фронт.
        Если в нём меньше запрошенного количества вариантов,
        оставшиеся места заполняются следующим фронтом. Такой подход
        сохраняет Парето-приоритет и позволяет вернуть требуемое ТЗ
        количество вариантов, когда первый фронт мал.
        """

        remaining = list(variants)
        selected: list[Variant] = []

        while remaining and len(selected) < n_variants:
            front = cls._pareto_front(remaining)
            available_slots = n_variants - len(selected)

            selected.extend(
                cls._select_diverse_variants(
                    front,
                    n_variants=available_slots,
                )
            )

            front_ids = {
                id(variant)
                for variant in front
            }
            remaining = [
                variant
                for variant in remaining
                if id(variant) not in front_ids
            ]

        return selected[:n_variants]

    @staticmethod
    def _select_diverse_variants(
        variants: list[Variant],
        *,
        n_variants: int,
    ) -> list[Variant]:
        """
        Раскладывает результат вдоль фронта по значению серы,
        чтобы не вернуть несколько почти одинаковых вариантов.
        """

        ordered = sorted(
            variants,
            key=lambda variant: (
                variant.metrics["sulfur"],
                -variant.metrics["yield"],
            ),
        )

        if len(ordered) <= n_variants:
            return ordered

        if n_variants == 1:
            return [ordered[len(ordered) // 2]]

        indexes = {
            round(
                index
                * (len(ordered) - 1)
                / (n_variants - 1)
            )
            for index in range(n_variants)
        }

        selected = [
            ordered[index]
            for index in sorted(indexes)
        ]

        # Защита от совпадения индексов после округления.
        if len(selected) < n_variants:
            for variant in ordered:
                if variant in selected:
                    continue

                selected.append(variant)

                if len(selected) == n_variants:
                    break

        return selected[:n_variants]

    @staticmethod
    def _resolve_state_column(
        state: pd.DataFrame,
        tag: str,
    ) -> str:
        if tag in state.columns:
            return tag

        candidates = [
            candidate
            for candidate in (
                f"avt_{tag}",
                f"hydro_{tag}",
            )
            if candidate in state.columns
        ]

        if not candidates:
            raise KeyError(
                f"Unknown controllable tag: {tag}"
            )

        if len(candidates) > 1:
            raise KeyError(
                f"Ambiguous controllable tag {tag!r}; "
                "use its full avt_/hydro_ name"
            )

        return candidates[0]

    @staticmethod
    def _validate_inputs(
        *,
        state: pd.DataFrame,
        constraints: OptimizationConstraints,
        n_variants: int,
    ) -> None:
        if not isinstance(state, pd.DataFrame):
            raise TypeError(
                "state must be a pandas DataFrame"
            )

        if len(state) != 1:
            raise ValueError(
                "state must contain exactly one row"
            )

        if not isinstance(
            constraints,
            OptimizationConstraints,
        ):
            raise TypeError(
                "constraints must be "
                "OptimizationConstraints"
            )

        if n_variants <= 0:
            raise ValueError(
                "n_variants must be positive"
            )

        if constraints.max_deviation_pct < 0:
            raise ValueError(
                "max_deviation_pct must not be negative"
            )

        if not constraints.controllable_ranges:
            raise ValueError(
                "controllable_ranges must not be empty"
            )
