# Физическая модель блендинга: линейное смешение по массе + нелинейные индексы (вязкость, ПТФ).
from __future__ import annotations

import math

from ml.types import BlendedProduct, Component

VISCOSITY_PROPERTIES = {
    "viscosity",
    "viscosity_cst",
    "viscosityk",
}

CFPP_PROPERTIES = {
    "cfpp",
    "птф",
}

class BlendingModel:
    """Расчёт свойств смеси по массовым долям компонентов."""

    FRACTION_TOLERANCE = 1e-6
    CELSIUS_TO_KELVIN = 273.15

    def __init__(self, cfpp_exponent: float = 2.0):
        if not math.isfinite(cfpp_exponent) or cfpp_exponent <= 0:
            raise ValueError("cfpp_exponent must be a positive finite number")

        self.cfpp_exponent = float(cfpp_exponent)

    
    def blend(
        self,
        components: list[Component],
        fractions: list[float],
    ) -> BlendedProduct:
        """
        Смешивает компоненты по заданным массовым долям.

        fractions — массовые доли компонентов в том же порядке,
        что и components. Сумма должна быть равна 1.
        """

        self._validate_inputs(components, fractions)

        property_names = self._validate_property_sets(components)

        blended_properties: dict[str, float] = {}

        for property_name in sorted(property_names):
            values = [
                float(component.properties[property_name])
                for component in components
            ]

            self._validate_property_values(property_name, values)

            blended_properties[property_name] = self._blend_property(
                property_name=property_name,
                values=values,
                fractions=fractions,
            )

        total_mass = sum(float(component.mass_flow) for component in components)

        return BlendedProduct(
            properties=blended_properties,
            total_mass=total_mass,
        )

    def _blend_property(
        self,
        *,
        property_name: str,
        values: list[float],
        fractions: list[float],
    ) -> float:
        """Выбирает правило смешения для конкретного свойства."""

        normalized_name = property_name.strip().lower()

        if normalized_name in VISCOSITY_PROPERTIES:
            return self._blend_viscosity(values, fractions)

        if normalized_name in CFPP_PROPERTIES:
            return self._blend_cfpp(values, fractions)

        # Сера, D15, T50, T90 и остальные аддитивные свойства.
        return self._weighted_average(values, fractions)

    @staticmethod
    def _weighted_average(
        values: list[float],
        fractions: list[float],
    ) -> float:

        return sum(
            value * fraction
            for value, fraction in zip(values, fractions)
        )

    @classmethod
    def _blend_viscosity(
        cls,
        viscosities: list[float],
        fractions: list[float],
    ) -> float:
        """
        Смешивает кинематическую вязкость через индекс Refutas.

        Формулы:
            VBN = 14.534 * ln(ln(v + 0.8)) + 10.975
            v = exp(exp((VBN - 10.975) / 14.534)) - 0.8
        """

        blend_numbers = [
            cls._viscosity_to_refutas(value)
            for value in viscosities
        ]

        blended_number = cls._weighted_average(
            blend_numbers,
            fractions,
        )

        return cls._refutas_to_viscosity(blended_number)

    @staticmethod
    def _viscosity_to_refutas(viscosity: float) -> float:
        """Переводит вязкость в viscosity blending number."""

        if viscosity <= 0.2:
            raise ValueError(
                "Viscosity must be greater than 0.2 cSt "
                "for the Refutas formula"
            )

        return 14.534 * math.log(math.log(viscosity + 0.8)) + 10.975

    @staticmethod
    def _refutas_to_viscosity(blend_number: float) -> float:
        """Переводит viscosity blending number обратно в cSt."""

        return (
            math.exp(
                math.exp((blend_number - 10.975) / 14.534)
            )
            - 0.8
        )

    
    def _blend_cfpp(
        self,
        temperatures: list[float],
        fractions: list[float],
    ) -> float:
        indexes = [
            self._cfpp_to_index(value)
            for value in temperatures
        ]

        blended_index = self._weighted_average(indexes, fractions)

        return self._index_to_cfpp(blended_index)


    def _cfpp_to_index(self, temperature: float) -> float:
        temperature_kelvin = temperature + self.CELSIUS_TO_KELVIN

        if temperature_kelvin <= 0:
            raise ValueError(
                "CFPP temperature must be above absolute zero"
            )

        return temperature_kelvin**self.cfpp_exponent


    def _index_to_cfpp(self, index: float) -> float:
        return (
            index ** (1.0 / self.cfpp_exponent)
            - self.CELSIUS_TO_KELVIN
        )

    @classmethod
    def _validate_inputs(
        cls,
        components: list[Component],
        fractions: list[float],
    ) -> None:
        if not components:
            raise ValueError("components must not be empty")

        if len(components) != len(fractions):
            raise ValueError(
                "components and fractions must have the same length"
            )

        if not all(math.isfinite(float(value)) for value in fractions):
            raise ValueError("fractions must contain only finite numbers")

        if any(value < 0 for value in fractions):
            raise ValueError("fractions must not be negative")

        fraction_sum = sum(fractions)

        if not math.isclose(
            fraction_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=cls.FRACTION_TOLERANCE,
        ):
            raise ValueError(
                f"fractions must sum to 1.0, got {fraction_sum}"
            )

        for component in components:
            mass_flow = float(component.mass_flow)

            if not math.isfinite(mass_flow):
                raise ValueError(
                    f"Component {component.name!r} has non-finite mass_flow"
                )

            if mass_flow < 0:
                raise ValueError(
                    f"Component {component.name!r} has negative mass_flow"
                )

    @staticmethod
    def _validate_property_sets(
        components: list[Component],
    ) -> set[str]:
        """
        Проверяет, что свойства присутствуют у всех компонентов.

        Иначе нельзя корректно рассчитать результат смеси.
        """

        expected = set(components[0].properties)

        if not expected:
            raise ValueError("components must contain properties")

        for component in components[1:]:
            actual = set(component.properties)

            if actual != expected:
                missing = sorted(expected - actual)
                unexpected = sorted(actual - expected)

                raise ValueError(
                    f"Component {component.name!r} has incompatible "
                    f"properties: missing={missing}, unexpected={unexpected}"
                )

        return expected

    @staticmethod
    def _validate_property_values(
        property_name: str,
        values: list[float],
    ) -> None:
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"Property {property_name!r} contains non-finite values"
            )