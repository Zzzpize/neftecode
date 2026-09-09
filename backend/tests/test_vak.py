import pandas as pd
import pytest

from ml.vak import (
    VAKError,
    VAKCatalog,
    VAKFormula,
    VAKFormulaError,
    VAKMissingFeatureError,
)

def test_avt_formula_with_decimal_comma_and_cyrillic_multiplication():
    formula = VAKFormula.compile(
        name="AVT6:240-350:D15",
        block="ЭЛОУ-АВТ-6. 240-350",
        source="10,0 + 2,0хF30 - T33",
    )
    frame = pd.DataFrame(
        {
            "avt_F30": [3.0, 5.0],
            "avt_T33": [4.0, 1.0],
        }
    )

    result = formula.evaluate(frame)

    assert result.tolist() == pytest.approx([12.0, 19.0])
    assert formula.required_columns == ("avt_F30", "avt_T33")
    assert formula.output_column == "vak__avt6_240_350_d15"


def test_hydro_formula_uses_hydro_prefix():
    formula = VAKFormula.compile(
        name="24-2000:GODT:T50",
        block="24-2000. ГО ДТ",
        source="44.625 + 10.0224*P13 + 0.8052*T6",
    )
    frame = pd.DataFrame(
        {
            "hydro_P13": [2.0],
            "hydro_T6": [300.0],
        }
    )

    result = formula.evaluate(frame)

    expected = 44.625 + 10.0224 * 2.0 + 0.8052 * 300.0
    assert result.iloc[0] == pytest.approx(expected)


def test_special_lims_reference_is_mapped_to_master_column():
    formula = VAKFormula.compile(
        name="24-2000:GODT:D15",
        block="24-2000. ГО ДТ",
        source=(
            "667.881 + "
            "0.15417*LIMS:24-2000.Pipeline.D15 + "
            "0.10774*T11"
        ),
    )
    frame = pd.DataFrame(
        {
            "lims__гидроочистка__pt2__d15": [830.0],
            "hydro_T11": [300.0],
        }
    )

    result = formula.evaluate(frame)

    expected = 667.881 + 0.15417 * 830.0 + 0.10774 * 300.0
    assert result.iloc[0] == pytest.approx(expected)


def test_catalog_loads_formulas_from_parquet(tmp_path):
    source = pd.DataFrame(
        [
            {
                "block": "ЭЛОУ-АВТ-6. 240-350",
                "name": "AVT6:240-350:T50",
                "formula": "100 + 2*F30",
            },
            {
                "block": "24-2000. ГО ДТ",
                "name": "24-2000:GODT:T50",
                "formula": "200 + T6",
            },
        ]
    )
    path = tmp_path / "vac_formulas.parquet"
    source.to_parquet(path, index=False)

    catalog = VAKCatalog.load(path)

    assert catalog.names == (
        "AVT6:240-350:T50",
        "24-2000:GODT:T50",
    )


def test_catalog_evaluates_multiple_formulas():
    source = pd.DataFrame(
        [
            {
                "block": "ЭЛОУ-АВТ-6. 240-350",
                "name": "AVT6:240-350:T50",
                "formula": "100 + 2*F30",
            },
            {
                "block": "24-2000. ГО ДТ",
                "name": "24-2000:GODT:T50",
                "formula": "200 + T6",
            },
        ]
    )
    catalog = VAKCatalog.from_frame(source)
    frame = pd.DataFrame(
        {
            "avt_F30": [10.0, 20.0],
            "hydro_T6": [300.0, 310.0],
        }
    )

    result = catalog.evaluate_many(frame)

    assert result.columns.tolist() == [
        "vak__avt6_240_350_t50",
        "vak__24_2000_godt_t50",
    ]
    assert result["vak__avt6_240_350_t50"].tolist() == pytest.approx(
        [120.0, 140.0]
    )
    assert result["vak__24_2000_godt_t50"].tolist() == pytest.approx(
        [500.0, 510.0]
    )


def test_missing_feature_raises_clear_error():
    formula = VAKFormula.compile(
        name="AVT6:240-350:T50",
        block="ЭЛОУ-АВТ-6. 240-350",
        source="100 + F30 + T33",
    )
    frame = pd.DataFrame({"avt_F30": [10.0]})

    with pytest.raises(
        VAKMissingFeatureError,
        match="avt_T33",
    ):
        formula.evaluate(frame)


def test_division_by_zero_becomes_nan():
    formula = VAKFormula.compile(
        name="AVT6:240-350:D15",
        block="ЭЛОУ-АВТ-6. 240-350",
        source="F30 / F32",
    )
    frame = pd.DataFrame(
        {
            "avt_F30": [10.0],
            "avt_F32": [0.0],
        }
    )

    result = formula.evaluate(frame)

    assert pd.isna(result.iloc[0])


def test_unsafe_python_expression_is_rejected():
    with pytest.raises(VAKFormulaError, match="запрещена конструкция"):
        VAKFormula.compile(
            name="AVT6:240-350:T50",
            block="ЭЛОУ-АВТ-6. 240-350",
            source="__import__('os').system('echo unsafe')",
        )


def test_unknown_variable_is_rejected():
    with pytest.raises(VAKFormulaError, match="неизвестная переменная"):
        VAKFormula.compile(
            name="AVT6:240-350:T50",
            block="ЭЛОУ-АВТ-6. 240-350",
            source="100 + unknown_value",
        )


def test_duplicate_formula_names_are_rejected():
    source = pd.DataFrame(
        [
            {
                "block": "ЭЛОУ-АВТ-6. 240-350",
                "name": "AVT6:240-350:T50",
                "formula": "100 + F30",
            },
            {
                "block": "ЭЛОУ-АВТ-6. 240-350",
                "name": "AVT6:240-350:T50",
                "formula": "200 + F30",
            },
        ]
    )

    with pytest.raises(VAKError):
        VAKCatalog.from_frame(source)