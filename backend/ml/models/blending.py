# Физическая модель блендинга: линейное смешение по массе + нелинейные индексы (вязкость, ПТФ).
from __future__ import annotations

from ml.types import BlendedProduct, Component

class BlendingModel:
    def blend(
        self,
        components: list[Component],
        fractions: list[float],
    ) -> BlendedProduct:
        if len(components) != len(fractions):
            raise ValueError("components and fractions must have the same length")

        if abs(sum(fractions) - 1.0) > 1e-6:
            raise ValueError("fractions must sum to 1.0")

        if not components:
            raise ValueError("components must not be empty")

        properties: dict[str, float] = {}
        all_property_names = set()
        for component in components:
            all_property_names.update(component.properties.keys())
        for prop in all_property_names:
            values = []
            for component, fraction in zip(components, fractions):
                if prop not in component.properties:
                    continue
                values.append(component.properties[prop] * fraction)
            if values:
                properties[prop] = sum(values)
        total_mass = sum(component.mass_flow for component in components)
        return BlendedProduct(
            properties=properties,
            total_mass=total_mass
        )