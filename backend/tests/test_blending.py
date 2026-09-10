import pytest

from ml.models.blending import BlendingModel
from ml.types import BlendedProduct, Component


def test_linear_blending_50_50():
    components = [
        Component(
            name="component_a",
            properties={
                "sulfur_ppm": 4.0,
                "D15": 830.0,
                "T50": 280.0,
                "T90": 340.0,
            },
            mass_flow=60.0,
        ),
        Component(
            name="component_b",
            properties={
                "sulfur_ppm": 8.0,
                "D15": 840.0,
                "T50": 300.0,
                "T90": 360.0,
            },
            mass_flow=40.0,
        ),
    ]

    result = BlendingModel().blend(
        components,
        fractions=[0.5, 0.5],
    )

    assert isinstance(result, BlendedProduct)
    assert result.properties["sulfur_ppm"] == pytest.approx(6.0)
    assert result.properties["D15"] == pytest.approx(835.0)
    assert result.properties["T50"] == pytest.approx(290.0)
    assert result.properties["T90"] == pytest.approx(350.0)
    assert result.total_mass == pytest.approx(100.0)


def test_blending_component_with_itself_returns_same_properties():
    component = Component(
        name="diesel",
        properties={
            "sulfur_ppm": 7.5,
            "D15": 832.0,
            "viscosity": 3.2,
            "CFPP": -25.0,
        },
        mass_flow=50.0,
    )

    result = BlendingModel().blend(
        [component],
        fractions=[1.0],
    )

    assert result.properties == pytest.approx(component.properties)
    assert result.total_mass == pytest.approx(50.0)


def test_viscosity_is_blended_non_linearly():
    components = [
        Component(
            "light",
            {"viscosity": 2.0},
            mass_flow=50.0,
        ),
        Component(
            "heavy",
            {"viscosity": 10.0},
            mass_flow=50.0,
        ),
    ]

    result = BlendingModel().blend(
        components,
        fractions=[0.5, 0.5],
    )

    viscosity = result.properties["viscosity"]

    assert 2.0 < viscosity < 10.0
    assert viscosity != pytest.approx(6.0)


def test_cfpp_is_blended_non_linearly():
    components = [
        Component(
            "winter",
            {"CFPP": -30.0},
            mass_flow=50.0,
        ),
        Component(
            "summer",
            {"CFPP": -10.0},
            mass_flow=50.0,
        ),
    ]

    result = BlendingModel().blend(
        components,
        fractions=[0.5, 0.5],
    )

    cfpp = result.properties["CFPP"]

    assert -30.0 < cfpp < -10.0
    assert cfpp != pytest.approx(-20.0)


@pytest.mark.parametrize(
    "fractions",
    [
        [0.4, 0.4],
        [0.5, 0.6],
        [-0.1, 1.1],
        [float("nan"), float("nan")],
    ],
)
def test_invalid_fractions_are_rejected(fractions):
    components = [
        Component("a", {"D15": 830.0}, mass_flow=1.0),
        Component("b", {"D15": 840.0}, mass_flow=1.0),
    ]

    with pytest.raises(ValueError):
        BlendingModel().blend(components, fractions)


def test_missing_property_is_rejected():
    components = [
        Component(
            "a",
            {"D15": 830.0, "sulfur_ppm": 5.0},
            mass_flow=1.0,
        ),
        Component(
            "b",
            {"D15": 840.0},
            mass_flow=1.0,
        ),
    ]

    with pytest.raises(ValueError, match="incompatible properties"):
        BlendingModel().blend(
            components,
            fractions=[0.5, 0.5],
        )

def test_cfpp_is_blended_using_temperature_index():
    model = BlendingModel(cfpp_exponent=2.0)

    components = [
        Component(
            "winter",
            {"CFPP": -30.0},
            mass_flow=50.0,
        ),
        Component(
            "summer",
            {"CFPP": -10.0},
            mass_flow=50.0,
        ),
    ]

    result = model.blend(
        components,
        fractions=[0.5, 0.5],
    )

    expected = (
        (
            0.5 * (-30.0 + 273.15) ** 2
            + 0.5 * (-10.0 + 273.15) ** 2
        )
        ** 0.5
        - 273.15
    )

    assert result.properties["CFPP"] == pytest.approx(expected)

def test_cfpp_exponent_must_be_positive():
    with pytest.raises(ValueError, match="cfpp_exponent"):
        BlendingModel(cfpp_exponent=0.0)