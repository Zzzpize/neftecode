"""Guardrails для LLM (раздел 9 ``spec.md``).

Главный guardrail — пост-фильтр чисел: каждое число в тексте, сгенерированном
LLM, обязано присутствовать в исходной структуре. Извлечение чисел идёт
регулярным выражением; «изобретённое» число делает ответ невалидным.
"""
from __future__ import annotations

import re
from numbers import Number
from typing import Any

# Десятичный разделитель допускается точкой или запятой (русская локаль).
# Отрицательный/положительный lookahead/behind не дают зацепить цифры внутри
# идентификаторов тегов (например, ``hydro_T5`` или ``avt_T55``).
NUMBER_RE = re.compile(r"(?<![A-Za-zА-Яа-я_])-?\d+(?:[.,]\d+)?(?![A-Za-zА-Яа-я_])")

# Допуск сравнения чисел. LLM может округлять значения до одного знака после
# запятой (например, 7.98 -> 8.0), поэтому сравниваем с запасом 0.05: ошибка
# округления <= 0.05, а выдуманные числа отличаются на порядок сильнее.
DEFAULT_TOLERANCE = 0.05


def extract_numbers(text: str) -> list[float]:
    """Извлекает все числа из текста (регулярным выражением)."""
    numbers: list[float] = []
    for match in NUMBER_RE.findall(text):
        try:
            numbers.append(float(match.replace(",", ".")))
        except ValueError:
            continue
    return numbers


def collect_numbers(value: Any) -> set[float]:
    """Рекурсивно собирает все числовые значения из вложенной структуры."""
    numbers: set[float] = set()
    if isinstance(value, bool):
        return numbers
    if isinstance(value, Number):
        numbers.add(float(value))
    elif isinstance(value, dict):
        for v in value.values():
            numbers |= collect_numbers(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            numbers |= collect_numbers(v)
    return numbers


def find_invented_numbers(
    text: str,
    source: Any,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[float]:
    """Возвращает числа из ``text``, отсутствующие в ``source``."""
    source_numbers = collect_numbers(source)
    invented: list[float] = []
    for number in extract_numbers(text):
        if not any(abs(number - known) <= tolerance for known in source_numbers):
            invented.append(number)
    return invented


def validate_numbers(
    text: str,
    source: Any,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """True, если все числа в ``text`` присутствуют в ``source``."""
    return not find_invented_numbers(text, source, tolerance)
