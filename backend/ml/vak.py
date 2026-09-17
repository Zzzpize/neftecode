"""Чтение и безопасное вычисление формул виртуальных анализаторов ВАК."""

from __future__ import annotations

import ast
import json
import operator
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


class VAKError(Exception):
    """Базовая ошибка при работе с формулами ВАК."""


class VAKFormulaError(VAKError):
    """Формула имеет неподдерживаемый или небезопасный синтаксис."""


class VAKMissingFeatureError(VAKError):
    """Во входном DataFrame отсутствует тег, необходимый формуле."""


# Формулы ВАК используют собственные имена лабораторных показателей.
# Здесь они связываются с колонками, созданными data_layer/time_join.py.
#
# Нужно отдельно подтвердить у технолога, что Pipeline соответствует точке 2.
# Если это точка 1, меняется только этот словарь.
DEFAULT_REFERENCE_ALIASES: dict[str, str] = {
    "LIMS.D15": "lims__гидроочистка__pt2__d15",
    "LIMS.95%.T": "lims__гидроочистка__pt2__95_t",
    "LIMS:24-2000.Pipeline.D15": "lims__гидроочистка__pt2__d15",
    "LIMS:24-2000.Pipeline.95%.T": "lims__гидроочистка__pt2__95_t",
}


# Короткие имена технологических тегов из справочника КИП:
# T55, F30, P13, W7, L43 и так далее.
_SHORT_TAG_PATTERN = re.compile(r"\b[TFPWLDQ]\d+\b")


# Разрешённые бинарные операции.
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


# Разрешённые унарные операции: например, -F30 или +T55.
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


# Полный список разрешённых узлов синтаксического дерева.
_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.UAdd,
    ast.USub,
)


@dataclass(frozen=True)
class VAKFormula:
    """Одна подготовленная к вычислению формула ВАК."""

    name: str
    block: str | None
    source: str
    normalized: str
    output_column: str
    reference_columns: Mapping[str, str]
    _tree: ast.Expression = field(repr=False, compare=False)

    @classmethod
    def compile(
        cls,
        *,
        name: str,
        block: str | None,
        source: str,
        reference_aliases: Mapping[str, str] | None = None,
    ) -> "VAKFormula":
        """Нормализует строку и компилирует безопасное выражение."""

        installation = _detect_installation(name=name, block=block)
        aliases = dict(DEFAULT_REFERENCE_ALIASES)

        if reference_aliases:
            aliases.update(reference_aliases)

        # Compatibility with the malformed legacy XLSX. The new expert
        # workbook explicitly confirms F65/(F32+F30).
        if name == "AVT6:240-350:CFPP":
            source = source.replace(
                "F65/F32+F30))",
                "F65/(F32+F30))",
            )
        
        normalized = _normalize_expression(source)
        normalized, reference_columns = _replace_special_references(
            normalized,
            aliases,
        )
        normalized, reference_columns = _replace_short_tags(
            normalized,
            installation,
            reference_columns,
        )

        try:
            tree = ast.parse(normalized, mode="eval")
        except SyntaxError as error:
            raise VAKFormulaError(
                f"Не удалось разобрать формулу {name!r}: {source!r}"
            ) from error

        _validate_tree(
            tree=tree,
            formula_name=name,
            allowed_names=set(reference_columns),
        )

        return cls(
            name=name,
            block=block,
            source=source,
            normalized=normalized,
            output_column=_make_output_column(name),
            reference_columns=reference_columns,
            _tree=tree,
        )

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Колонки master frame, необходимые для вычисления формулы."""

        return tuple(dict.fromkeys(self.reference_columns.values()))

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        """Вычисляет формулу для всех строк переданного DataFrame."""

        missing_columns = [
            column
            for column in self.required_columns
            if column not in frame.columns
        ]

        if missing_columns:
            raise VAKMissingFeatureError(
                f"Для формулы {self.name!r} отсутствуют колонки: "
                f"{', '.join(missing_columns)}"
            )

        values = {
            safe_name: pd.to_numeric(frame[column], errors="coerce")
            for safe_name, column in self.reference_columns.items()
        }

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            result = _evaluate_node(
                node=self._tree.body,
                values=values,
            )

        if np.isscalar(result):
            result = pd.Series(
                float(result),      # type: ignore
                index=frame.index,
                dtype="float64",
            )
        else:
            result = pd.Series(result, index=frame.index, dtype="float64")

        # Деление на ноль и переполнение не должны давать бесконечности.
        # Такие результаты становятся NaN и далее учитываются как отсутствие baseline.
        result = result.replace([np.inf, -np.inf], np.nan)
        result.name = self.output_column

        return result


class VAKCatalog:
    """Коллекция формул ВАК, загруженная из parquet-кеша."""

    def __init__(self, formulas: Mapping[str, VAKFormula]):
        self._formulas = dict(formulas)

    @classmethod
    def load_expert(cls) -> "VAKCatalog":
        """Load the versioned, normalized copy of both supplied XLSX files."""
        reference = json.loads(
            Path(__file__).with_name("expert_reference.json").read_text(encoding="utf-8")
        )
        return cls.from_frame(pd.DataFrame(reference["formulas"]))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        reference_aliases: Mapping[str, str] | None = None,
    ) -> "VAKCatalog":
        """Загружает `vac_formulas.parquet`."""

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Файл формул ВАК не найден: {path}")

        table = pd.read_parquet(path)

        return cls.from_frame(
            table,
            reference_aliases=reference_aliases,
        )

    @classmethod
    def from_frame(
        cls,
        table: pd.DataFrame,
        *,
        reference_aliases: Mapping[str, str] | None = None,
    ) -> "VAKCatalog":
        """Создаёт каталог из таблицы с колонками block, name и formula."""

        required_columns = {"block", "name", "formula"}
        missing_columns = required_columns - set(table.columns)

        if missing_columns:
            raise VAKError(
                "В таблице формул отсутствуют колонки: "
                + ", ".join(sorted(missing_columns))
            )

        if table["name"].isna().any():
            raise VAKError("В таблице ВАК есть строки без имени формулы")

        duplicated_names = (
            table.loc[table["name"].duplicated(keep=False), "name"]
            .astype(str)
            .unique()
            .tolist()
        )

        if duplicated_names:
            raise VAKError(
                "Обнаружены повторяющиеся имена формул: "
                + ", ".join(sorted(duplicated_names))
            )

        formulas: dict[str, VAKFormula] = {}

        for record in table.to_dict(orient="records"):
            name = str(record["name"]).strip()
            source = str(record["formula"]).strip()

            raw_block = record["block"]
            block = None if pd.isna(raw_block) else str(raw_block).strip()

            formulas[name] = VAKFormula.compile(
                name=name,
                block=block,
                source=source,
                reference_aliases=reference_aliases,
            )

        return cls(formulas)

    @property
    def names(self) -> tuple[str, ...]:
        """Все доступные имена формул."""

        return tuple(self._formulas)

    def get(self, name: str) -> VAKFormula:
        """Возвращает формулу по её исходному имени."""

        try:
            return self._formulas[name]
        except KeyError as error:
            available = ", ".join(self.names)
            raise KeyError(
                f"Формула ВАК {name!r} не найдена. Доступны: {available}"
            ) from error

    def evaluate(
        self,
        name: str,
        frame: pd.DataFrame,
    ) -> pd.Series:
        """Вычисляет одну формулу."""

        return self.get(name).evaluate(frame)

    def evaluate_many(
        self,
        frame: pd.DataFrame,
        names: list[str] | tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """Вычисляет несколько формул и возвращает отдельный DataFrame."""

        selected_names = self.names if names is None else tuple(names)

        result = pd.DataFrame(index=frame.index)

        for name in selected_names:
            formula = self.get(name)
            result[formula.output_column] = formula.evaluate(frame)

        return result


def _normalize_expression(expression: str) -> str:
    """Приводит формулу из Excel к синтаксису Python."""

    normalized = expression.strip()

    # 791,22872 -> 791.22872
    normalized = re.sub(
        r"(?<=\d),(?=\d)",
        ".",
        normalized,
    )

    # Нормальизация Excel выражений
    normalized = normalized.translate(
        str.maketrans(
            {
                "x": "*",
                "х": "*",
                "Х": "*",
                "×": "*",
                "−": "-",
                "–": "-",
            }
        )
    )

    return normalized


def _detect_installation(
    *,
    name: str,
    block: str | None,
) -> str:
    """Определяет префикс колонок master frame."""

    normalized_name = name.upper()
    normalized_block = (block or "").upper()

    if normalized_name.startswith("AVT") or "АВТ" in normalized_block:
        return "avt"

    if normalized_name.startswith("24-2000") or "24-2000" in normalized_block:
        return "hydro"

    raise VAKFormulaError(
        f"Не удалось определить установку для формулы {name!r}, block={block!r}"
    )


def _replace_special_references(
    expression: str,
    aliases: Mapping[str, str],
) -> tuple[str, dict[str, str]]:
    """Заменяет LIMS-ссылки на допустимые Python-идентификаторы."""

    normalized = expression
    reference_columns: dict[str, str] = {}

    # Более длинные ссылки заменяем первыми, чтобы короткий alias
    # случайно не заменил часть длинного.
    ordered_aliases = sorted(
        aliases.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for raw_reference, master_column in ordered_aliases:
        if raw_reference not in normalized:
            continue

        safe_name = f"__special_{len(reference_columns)}"
        normalized = normalized.replace(raw_reference, safe_name)
        reference_columns[safe_name] = master_column

    return normalized, reference_columns


def _replace_short_tags(
    expression: str,
    installation: str,
    reference_columns: dict[str, str],
) -> tuple[str, dict[str, str]]:
    """Заменяет F30/T55 и подобные теги на безопасные имена."""

    def replace(match: re.Match[str]) -> str:
        tag = match.group(0)
        safe_name = f"__tag_{tag}"
        reference_columns[safe_name] = f"{installation}_{tag}"
        return safe_name

    normalized = _SHORT_TAG_PATTERN.sub(replace, expression)

    return normalized, reference_columns


def _validate_tree(
    *,
    tree: ast.Expression,
    formula_name: str,
    allowed_names: set[str],
) -> None:
    """Запрещает функции, атрибуты, индексацию и неизвестные переменные."""

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise VAKFormulaError(
                f"В формуле {formula_name!r} запрещена конструкция "
                f"{type(node).__name__}"
            )

        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise VAKFormulaError(
                f"В формуле {formula_name!r} неизвестная переменная "
                f"{node.id!r}"
            )

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                node.value,
                (int, float),
            ):
                raise VAKFormulaError(
                    f"В формуле {formula_name!r} разрешены только числа"
                )


def _evaluate_node(
    *,
    node: ast.AST,
    values: Mapping[str, pd.Series],
) -> float | pd.Series:
    """Рекурсивно вычисляет заранее проверенное AST-выражение."""

    if isinstance(node, ast.Constant):
        return float(node.value)    #type: ignore

    if isinstance(node, ast.Name):
        return values[node.id]

    if isinstance(node, ast.BinOp):
        operation = _BINARY_OPERATORS[type(node.op)]
        left = _evaluate_node(node=node.left, values=values)
        right = _evaluate_node(node=node.right, values=values)
        return operation(left, right)

    if isinstance(node, ast.UnaryOp):
        operation = _UNARY_OPERATORS[type(node.op)]
        operand = _evaluate_node(node=node.operand, values=values)
        return operation(operand)

    raise VAKFormulaError(
        f"Неожиданный элемент формулы: {type(node).__name__}"
    )


def _make_output_column(name: str) -> str:
    """AVT6:240-350:T50 -> vak__avt6_240_350_t50."""

    slug = name.strip().lower()
    slug = re.sub(r"[^\w]+", "_", slug, flags=re.UNICODE)
    slug = re.sub(r"_+", "_", slug).strip("_")

    return f"vak__{slug}"
